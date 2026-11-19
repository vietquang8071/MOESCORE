"""Offline Full9 plus Whisper to verified Full7 feature-store assembly."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.full7_score_contract import audit_train_scores, validate_score_contracts
from data.moe_feature_store import MoeFeatureStoreWriter, MoeFeatureStoreDataset, _canonical_json, validate_feature_store
from data.set_a_cached_dataset import SetACachedDataset
from data.set_b_cached_dataset import SetBCachedDataset
from model.composite_experts.set_a_residual import SET_A_COMPONENT_ORDER, feature_order_contract as e5_feature_contract
from model.composite_experts.set_b_residual import feature_order_contract as e6_feature_contract
from model.expert_adapters.registry import FULL7_EXPERT_ORDER, profile_contract, validate_and_assemble_expert_outputs
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter
from training.set_a_trainer import load_set_a_checkpoint
from training.set_b_trainer import load_set_b_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(path: Path, adapter_class: type, dataset: Any, loader: Any) -> tuple[Any, str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"trusted residual checkpoint is missing: {path}")
    # E5/E6 checkpoint formats are trusted local torch.save artifacts. Their
    # existing loaders validate the architecture, score policy and provenance.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("architecture_config"), dict):
        raise ValueError("residual checkpoint architecture is missing")
    architecture = payload["architecture_config"]
    model = adapter_class(hidden_dims=architecture["hidden_dims"], dropout=architecture["dropout"]) if adapter_class is SetAResidualAdapter else adapter_class()
    if adapter_class is SetBResidualAdapter and model.head.architecture_config != architecture:
        from model.composite_experts.set_b_residual import SetBResidualHead
        model = SetBResidualAdapter(SetBResidualHead(hidden_dims=architecture["hidden_dims"], dropout=architecture["dropout"]))
    loader(path, model=model, expected_dataset=dataset)
    model.eval().requires_grad_(False)
    return model, _sha256(path), payload


def _head_membership(split: str, sample_ids: list[str], source_manifest_sha256: str,
                     e5_payload: dict, e6_payload: dict, *, synthetic: bool) -> dict:
    if synthetic:
        return {name: {"synthetic": True, "training_membership": "SYNTHETIC_UNTRAINED_HEAD",
                       "per_row": ["SYNTHETIC_UNTRAINED_HEAD"] * len(sample_ids)}
                for name in ("expert5", "expert6")}
    state = e5_payload.get("training_state", {})
    fit, holdout = state.get("fit_sample_ids"), state.get("holdout_sample_ids")
    e5_train_source = e5_payload.get("data_provenance", {}).get("source_manifest_sha256")
    e6_train_source = e6_payload.get("dataset_provenance", {}).get("train", {}).get("full9_source_manifest_sha256")
    if (not isinstance(fit, list) or not isinstance(holdout, list) or not fit or not holdout
            or len(fit + holdout) != len(set(fit + holdout))
            or not isinstance(e5_train_source, str) or len(e5_train_source) != 64
            or not isinstance(e6_train_source, str) or len(e6_train_source) != 64):
        raise ValueError("E5/E6 real training membership provenance is incomplete")
    if split == "train" and (set(fit + holdout) != set(sample_ids)
                             or e5_train_source != source_manifest_sha256
                             or e6_train_source != source_manifest_sha256):
        raise ValueError("E5/E6 real train membership differs from Full7 source")
    if split != "train" and (e5_train_source == source_manifest_sha256
                              or e6_train_source == source_manifest_sha256
                              or set(sample_ids).intersection(fit + holdout)):
        raise ValueError("E5/E6 real inference split overlaps training membership")
    fit_set, holdout_set = set(fit), set(holdout)
    return {
        "expert5": {"synthetic": False, "train_source_manifest_sha256": e5_train_source,
                    "fit_sample_count": len(fit), "holdout_sample_count": len(holdout),
                    "per_row": ["IN_SAMPLE_FIT" if sample_id in fit_set else
                                "OUT_OF_SAMPLE_FIT_SELECTION_HOLDOUT" if sample_id in holdout_set else
                                "OUT_OF_SAMPLE" for sample_id in sample_ids]},
        "expert6": {"synthetic": False, "train_source_manifest_sha256": e6_train_source,
                    "per_row": ["IN_SAMPLE_FIT" if split == "train" else
                                "OUT_OF_SAMPLE_FIT_VALIDATION_SELECTED" if split == "validation" else
                                "OUT_OF_SAMPLE" for _ in sample_ids]},
    }


def _score_contracts(full9_manifest: dict, e5_hash: str, e6_hash: str) -> tuple[dict, bool]:
    cache = full9_manifest["per_expert_cache_identities"]
    synthetic = all(str(cache[name].get("implementation_provenance", {}).get("source", "")).startswith("synthetic")
                    for name in ("expert1", "expert2", "expert3", "expert4", "m2d_lora"))
    if not synthetic:
        required_sources = {
            "expert1": ("model/expert_adapters/expert1_adapter.py", "expert1_finetune/Train_Expert1.py"),
            "expert2": ("model/expert_adapters/expert2_adapter.py", "expert2_finetune/expert2_model.py"),
            "expert3": ("model/expert_adapters/expert3_adapter.py",
                        "expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py",
                        "expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py"),
            "expert4": ("model/expert_adapters/expert4_adapter.py", "expert4_finetune/model.py"),
            "m2d_lora": ("model/expert_adapters/m2d_lora_adapter.py",
                         "model/expert_adapters/m2d_lora_runtime.py"),
        }
        _, dimensions, _ = profile_contract("full_9")
        for name, paths in required_sources.items():
            identity = cache[name]
            if identity.get("checkpoint_provenance") != full9_manifest["checkpoint_provenance"].get(name):
                raise ValueError(f"{name} source checkpoint identity contradicts Full9 manifest")
            contract = identity.get("score_contract")
            if not isinstance(contract, dict) or any((
                contract.get("score_shape") != [None, 1], contract.get("score_dtype") != "float32",
                contract.get("feature_shape") != [None, dimensions[name]],
                contract.get("feature_dtype") != "float32",
            )):
                raise ValueError(f"{name} source score/feature contract is incomplete")
            expected_scale = "unitless_similarity_proxy" if name in {"expert1", "expert2"} else "normalized_minus1_to1"
            if contract.get("output_scale") not in (None, expected_scale):
                raise ValueError(f"{name} source score scale contradicts Full7 contract")
            if contract.get("score_semantics") not in (None, "adapter-defined pretrained/fine-tuned score",
                                                        "clap_similarity" if name == "expert1" else
                                                        "mga_similarity" if name == "expert2" else
                                                        "fine_tuned_normalized_regression" if name == "expert3" else
                                                        "tanh_regression" if name == "expert4" else
                                                        "m2d_lora_coral_regression_normalized"):
                raise ValueError(f"{name} source score semantics contradict Full7 contract")
            implementation = identity.get("implementation_provenance")
            if not isinstance(implementation, dict) or any(
                implementation.get(relative) != _sha256(Path(__file__).parents[1] / relative)
                for relative in paths
            ):
                raise ValueError(f"{name} source-code evidence is missing or differs from audited code")
    semantics = ("clap_similarity", "mga_similarity", "fine_tuned_normalized_regression",
                 "tanh_regression", "set_a_normalized_residual", "set_b_normalized_residual",
                 "m2d_lora_coral_regression_normalized")
    sources = ("expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "m2d_lora")
    evidence = (
        "model/expert_adapters/expert1_adapter.py",
        "model/expert_adapters/expert2_adapter.py",
        "expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:100",
        "model/expert_adapters/expert4_adapter.py",
        "model/composite_experts/set_a_residual.py",
        "model/composite_experts/set_b_residual.py",
        "model/expert_adapters/m2d_lora_runtime.py:213",
    )
    contracts = {}
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        identity = e5_hash if name == "expert5" else e6_hash if name == "expert6" else cache[sources[index]]["cache_id"]
        contracts[name] = {
            "source": sources[index], "semantics": semantics[index],
            "target_scale": "normalized_minus1_to1",
            "output_scale": "unitless_similarity_proxy" if index < 2 else "normalized_minus1_to1",
            "conversion": "identity", "identity": identity,
            "evidence": evidence[index],
            "audit_status": "SYNTHETIC_VERIFIED" if synthetic else "CODE_VERIFIED",
        }
    validate_score_contracts(contracts, synthetic=synthetic)
    return contracts, synthetic


def build_full7_split(full9_root: str | Path, whisper_root: str | Path,
                      e5_checkpoint: str | Path, e6_checkpoint: str | Path,
                      output_root: str | Path, *, split: str,
                      source_audio_dir: str | Path, shard_size: int = 32) -> dict:
    """Build one split without running any pretrained backbone or extractor."""
    if split not in {"train", "validation", "test"} or type(shard_size) is not int or shard_size <= 0:
        raise ValueError("invalid Full7 split or shard_size")
    roots = [Path(value).expanduser().resolve() for value in (full9_root, whisper_root, output_root)]
    full9_root, whisper_root, output_root = roots
    if output_root == full9_root or output_root == whisper_root or output_root.is_relative_to(full9_root) or output_root.is_relative_to(whisper_root):
        raise ValueError("Full7 output must be separate from input caches")
    a = SetACachedDataset(full9_root, split=split)
    b = SetBCachedDataset(full9_root, whisper_root, split=split, source_audio_dir=source_audio_dir)
    source = MoeFeatureStoreDataset(full9_root, split=split, require_labels=split != "test")
    if len(a) != len(b) or a.sample_ids != b.sample_ids or a.sample_ids != source.sample_ids:
        raise ValueError("Full9, Set A and Whisper sample order differs")
    if split == "test" and any("label_raw" in record for record in a.source_records):
        raise ValueError("test source manifest contains labels")
    e5, e5_hash, e5_payload = _head(Path(e5_checkpoint).expanduser().resolve(), SetAResidualAdapter, a, load_set_a_checkpoint)
    e6, e6_hash, e6_payload = _head(Path(e6_checkpoint).expanduser().resolve(), SetBResidualAdapter, b, load_set_b_checkpoint)
    full9_manifest = source.manifest
    contracts, synthetic = _score_contracts(full9_manifest, e5_hash, e6_hash)
    membership = _head_membership(split, a.sample_ids, full9_manifest["source_manifest_sha256"],
                                  e5_payload, e6_payload, synthetic=synthetic)
    source_models = full9_manifest["checkpoint_provenance"]
    provenance = {name: source_models[name] for name in FULL7_EXPERT_ORDER[:4]}
    provenance["expert5"] = {"checkpoint_sha256": e5_hash, "feature_contract": e5_feature_contract(),
                               "architecture_config": e5.head.architecture_config}
    provenance["expert6"] = {"checkpoint_sha256": e6_hash, "feature_contract": e6_feature_contract(),
                               "architecture_config": e6.head.architecture_config}
    provenance["expert7"] = source_models["m2d_lora"]
    shared = {
        "full9_cache_ids": {name: identity["cache_id"] for name, identity in full9_manifest["per_expert_cache_identities"].items()},
        "whisper_checkpoint_identity": b.whisper_checkpoint_identity,
        "whisper_preprocessing_identity": b.whisper_preprocessing_identity,
        "e5_checkpoint_sha256": e5_hash, "e6_checkpoint_sha256": e6_hash,
        "e5_feature_contract": e5_feature_contract(), "e6_feature_contract": e6_feature_contract(),
        "core_assembler_source_sha256": _sha256(Path(__file__)),
        "e5_adapter_source_sha256": _sha256(Path(__file__).parents[1] / "model/expert_adapters/set_a_residual_adapter.py"),
        "e6_adapter_source_sha256": _sha256(Path(__file__).parents[1] / "model/expert_adapters/set_b_residual_adapter.py"),
        "e5_head_source_sha256": _sha256(Path(__file__).parents[1] / "model/composite_experts/set_a_residual.py"),
        "e6_head_source_sha256": _sha256(Path(__file__).parents[1] / "model/composite_experts/set_b_residual.py"),
        "e5_loader_source_sha256": _sha256(Path(__file__).parents[1] / "training/set_a_trainer.py"),
        "e6_loader_source_sha256": _sha256(Path(__file__).parents[1] / "training/set_b_trainer.py"),
    }
    assembly = {
        "source_full9_store_identity": source.identity,
        "source_full9_manifest_sha256": _sha256(full9_root / split / "manifest.json"),
        "source_whisper_manifest_sha256": b.provenance["whisper_manifest_sha256"],
        "source_sample_manifest_sha256": full9_manifest["source_manifest_sha256"],
        "whisper_dataset_identity": b.provenance["whisper_dataset_identity"],
        "ordered_audio_caption_fingerprints_sha256": hashlib.sha256(_canonical_json([
            {"sample_id": record["sample_id"], "wave_file_name": record["wave_file_name"],
             "wav_sha256": record["wav_sha256"],
             "caption_sha256": hashlib.sha256(record["raw_caption"].encode()).hexdigest()}
            for record in b.records])).hexdigest(),
        "ordered_source_fingerprints": [
            {"sample_id": record["sample_id"], "wav_sha256": record["wav_sha256"].lower(),
             "caption_sha256": hashlib.sha256(record["raw_caption"].encode()).hexdigest()}
            for record in b.records],
        "score_contracts": contracts, "shared_source_contract": shared,
        "head_membership": membership,
        "synthetic_provenance": synthetic,
    }
    writer = MoeFeatureStoreWriter(output_root, split=split, expected_sample_ids=a.sample_ids,
        source_manifest_sha256=full9_manifest["source_manifest_sha256"],
        checkpoint_provenance=provenance, preprocessing_identity=shared,
        profile="full_7", assembly_identity=assembly)
    order9, dims9, _ = profile_contract("full_9")
    offsets = {name: sum(dims9[prior] for prior in order9[:position]) for position, name in enumerate(order9)}
    all_scores, all_labels = [], []
    for start in range(0, len(source), shard_size):
        stop = min(start + shard_size, len(source))
        rows = [source[index] for index in range(start, stop)]
        a_rows = [a[index] for index in range(start, stop)]
        b_rows = [b[index] for index in range(start, stop)]
        if any((row["sample_id"], row["wave_file_name"]) !=
               (arow["sample_id"], arow["wave_file_name"]) or
               (row["sample_id"], row["wave_file_name"]) !=
               (brow["sample_id"], brow["wave_file_name"])
               for row, arow, brow in zip(rows, a_rows, b_rows)):
            raise ValueError("Full7 row identity/order changed during assembly")
        a_batch = {"component_outputs": {name: {
            key: torch.stack([row["component_outputs"][name][key] for row in a_rows])
            for key in ("score", "features")} for name in SET_A_COMPONENT_ORDER}}
        b_batch = {"component_outputs": {"top2_laion": {
            key: torch.stack([row["component_outputs"]["top2_laion"][key] for row in b_rows])
            for key in ("score", "features")}},
            "whisper_audio": torch.stack([row["whisper_audio"] for row in b_rows]),
            "whisper_text": torch.stack([row["whisper_text"] for row in b_rows])}
        with torch.inference_mode():
            e5_output, e6_output = e5(a_batch), e6(b_batch)
        scores9 = torch.stack([row["scores"] for row in rows])
        features9 = torch.stack([row["features"] for row in rows])
        outputs = {name: (scores9[:, order9.index(name):order9.index(name) + 1],
                          features9[:, offsets[name]:offsets[name] + dims9[name]])
                   for name in FULL7_EXPERT_ORDER[:4]}
        outputs["expert5"], outputs["expert6"] = e5_output, e6_output
        outputs["expert7"] = (scores9[:, order9.index("m2d_lora"):order9.index("m2d_lora") + 1],
                              features9[:, offsets["m2d_lora"]:offsets["m2d_lora"] + dims9["m2d_lora"]])
        scores7, features7 = validate_and_assemble_expert_outputs(outputs, profile="full_7")
        labels = None if split == "test" else np.asarray([float(row["label_raw"]) for row in rows], dtype=np.float32)
        writer.write_shard(start // shard_size, sample_ids=[row["sample_id"] for row in rows],
                           wave_file_names=[row["wave_file_name"] for row in rows],
                           scores=scores7.numpy(), features=features7.numpy(), labels_raw=labels)
        if split == "train":
            all_scores.append(scores7.numpy())
            all_labels.append(labels)
    manifest_path = writer.finalize()
    manifest = validate_feature_store(output_root, split=split, require_labels=split != "test")
    report = {"split": split, "sample_count": len(source), "manifest": str(manifest_path),
              "store_identity_sha256": manifest["store_identity_sha256"],
              "synthetic_provenance": synthetic}
    if split == "train":
        report["score_audit"] = audit_train_scores(np.concatenate(all_scores), np.concatenate(all_labels))
    return report
