"""Verified synthetic source artifacts exercise the offline Full7 boundary."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch

from data.moe_feature_store import MoeFeatureStoreDataset, _canonical_json
from data.moe_expert_cache import build_assembly_identity, build_expert_cache_identity
from data.set_a_cached_dataset import SetACachedDataset
from data.set_b_cached_dataset import SetBCachedDataset
from model.composite_experts.set_a_residual import SET_A_COMPONENT_DIMS, SET_A_COMPONENT_ORDER
from model.composite_experts.set_a_residual import feature_order_contract as e5_contract
from model.composite_experts.set_b_residual import feature_order_contract as e6_contract
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.registry import profile_contract
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter
from training.set_a_trainer import SCORE_NORMALIZATION as E5_NORMALIZATION
from training.set_b_trainer import SCORE_NORMALIZATION as E6_NORMALIZATION, SELECTION_POLICY
from test_set_b_cached_dataset import synthetic_caches


def _sources(root, split="train", audio_payloads=None):
    full9, whisper, audio = synthetic_caches(root, split=split, labels=[5.0, 8.0],
                                             audio_payloads=audio_payloads)
    shard = full9 / split / "shards" / "shard_000000"
    features = np.load(shard / "features.npy")
    scores = np.load(shard / "scores.npy")
    scores[:, 0] = np.array([0.1, 0.9], dtype=np.float32)
    order, dims, _ = profile_contract("full_9")
    for position, name in enumerate(SET_A_COMPONENT_ORDER):
        dim = SET_A_COMPONENT_DIMS[name]
        generator = torch.Generator().manual_seed(position + 33)
        audio_features = torch.randn(2, dim, generator=generator)
        text_features = torch.randn(2, dim, generator=generator)
        block = build_paired_cross_features(audio_features, text_features).numpy()
        start = sum(dims[key] for key in order[:order.index(name)])
        features[:, start:start + dims[name]] = block
        scores[:, order.index(name)] = block[:, -1]
    for filename, value in (("features.npy", features), ("scores.npy", scores)):
        np.save(shard / filename, value)
    metadata_path = shard / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for filename, value in (("features.npy", features), ("scores.npy", scores)):
        metadata["array_content_sha256"][filename] = hashlib.sha256(value.tobytes()).hexdigest()
        metadata["file_sha256"][filename] = hashlib.sha256((shard / filename).read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = full9 / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = manifest["per_expert_cache_identities"]
    for name in SET_A_COMPONENT_ORDER:
        old = cache[name]
        contract = dict(old["score_contract"])
        contract.update({"feature_representation": "paired_cross_4d_plus_1_v2",
                         "feature_blocks": ["audio_embedding", "text_embedding", "absolute_difference_l2",
                                            "elementwise_product_l2", "cosine_l2_scalar"],
                         "feature_width_formula": "4D+1"})
        cache[name] = build_expert_cache_identity(
            expert_name=name, checkpoint_provenance=old["checkpoint_provenance"],
            implementation_provenance=old["implementation_provenance"],
            preprocessing_provenance=old["preprocessing_provenance"],
            input_contract=old["input_contract"], inference_policy=old["inference_policy"],
            score_contract=contract, relevant_dependencies=old["relevant_dependencies"])
    manifest["assembly_identity"] = build_assembly_identity(
        expert_profile="full_9", expert_order=order, expert_cache_identities=cache,
        feature_dimensions=dims, source_manifest_identity=manifest["source_manifest_sha256"])
    manifest["shards"][0]["array_content_sha256"] = metadata["array_content_sha256"]
    manifest["shards"][0]["file_sha256"] = metadata["file_sha256"]
    body = {key: value for key, value in manifest.items() if key != "store_identity_sha256"}
    manifest["store_identity_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    return full9, whisper, audio


def test_build_full7_from_trusted_synthetic_heads(tmp_path):
    from data.full7_assembler import build_full7_split

    full9, whisper, audio = _sources(
        tmp_path, audio_payloads=[b"shared train validation wav", b"train only wav"],
    )
    e5_data = SetACachedDataset(full9, split="train")
    _, _, validation_audio = _sources(
        tmp_path, split="validation",
        audio_payloads=[b"shared train validation wav", b"validation only wav"],
    )
    e5_validation = SetACachedDataset(full9, split="validation", require_labels=True)
    e6_data = SetBCachedDataset(full9, whisper, split="train", source_audio_dir=audio)
    e5 = SetAResidualAdapter()
    e6 = SetBResidualAdapter()
    e5_path, e6_path = tmp_path / "e5.pt", tmp_path / "e6.pt"
    e5_metrics = {"epoch": 1.0, "train_mse": 0.3, "validation_mse": 0.2,
                  "best_validation_mse": 0.2, "learning_rate": 0.001}
    torch.save({"format_version": 3, "feature_contract": e5_contract(),
                "architecture_config": e5.head.architecture_config,
                "score_normalization": E5_NORMALIZATION,
                "epoch": 1, "global_step": 1, "data_provenance": e5_data.provenance,
                "training_state": {"selection_policy": "official_validation_mse_v1",
                                    "train_provenance": e5_data.provenance,
                                    "validation_provenance": e5_validation.provenance,
                                    "train_sample_ids": e5_data.sample_ids,
                                    "validation_sample_ids": e5_validation.sample_ids,
                                    "best_validation_mse": 0.2, "best_epoch": 1,
                                    "epoch": 1, "global_step": 1,
                                    "metric_history": [e5_metrics]},
                "component_model_identities": e5_data.component_model_identities,
                "component_cache_identities": {name: e5_data.component_cache_identities[name]["cache_id"]
                                                for name in SET_A_COMPONENT_ORDER},
                "model_state_dict": e5.state_dict()}, e5_path)
    torch.save({"format_version": 1, "feature_contract": e6_contract(),
                "architecture_config": e6.head.architecture_config,
                "score_normalization": E6_NORMALIZATION, "selection_policy": SELECTION_POLICY,
                "dataset_provenance": {"train": e6_data.provenance},
                "model_state_dict": e6.state_dict()}, e6_path)
    report = build_full7_split(full9, whisper, e5_path, e6_path, tmp_path / "full7",
                               split="train", source_audio_dir=audio)
    store = MoeFeatureStoreDataset(tmp_path / "full7", split="train")
    assert report["sample_count"] == len(store) == 2
    assert store.manifest["schema_version"] == 3
    assert store[0]["scores"].shape == (7,)
    assert store[0]["features"].shape == (29959,)
    assert store[0]["sample_id"] == "train:0.wav"
    assert store.manifest["checkpoint_provenance"]["expert5"]["checkpoint_sha256"] == hashlib.sha256(e5_path.read_bytes()).hexdigest()
    membership = store.manifest["assembly_identity"]["head_membership"]
    assert len(membership["expert5"]["per_row"]) == 2
    assert len(membership["expert6"]["per_row"]) == 2
    assert membership["expert5"]["synthetic"] is True
    from scripts.audit_full7_scores import audit_store
    audit_report = audit_store(tmp_path / "full7")
    assert audit_report["diagnostics"]["sample_count"] == 2
    assert len(audit_report["score_contract_sha256"]) == 64
    assert len(audit_report["score_audit_sha256"]) == 64
    again = build_full7_split(full9, whisper, e5_path, e6_path, tmp_path / "full7",
                              split="train", source_audio_dir=audio)
    assert again["store_identity_sha256"] == report["store_identity_sha256"]

    for split in ("validation", "test"):
        split_audio = validation_audio if split == "validation" else _sources(
            tmp_path, split=split, audio_payloads=[b"test unique 0", b"test unique 1"],
        )[2]
        built = build_full7_split(full9, whisper, e5_path, e6_path, tmp_path / "full7",
                                  split=split, source_audio_dir=split_audio)
        assert built["sample_count"] == 2
    from scripts.build_full7_feature_store import main as build_main
    assert build_main(["--full9-root", str(full9), "--whisper-root", str(whisper),
                       "--e5-checkpoint", str(e5_path), "--e6-checkpoint", str(e6_path),
                       "--source-audio-root", str(tmp_path), "--output-root", str(tmp_path / "full7"),
                       "--split", "train", "--split", "validation", "--split", "test"]) == 0
    assert not MoeFeatureStoreDataset(tmp_path / "full7", split="test")[0].get("label_raw")

    import yaml
    from scripts.train_moe_gating import run_training
    from scripts.evaluate_moe_gating import evaluate_checkpoint
    from scripts.infer_moe_gating import infer_checkpoint
    config = yaml.safe_load((__import__("pathlib").Path(__file__).parents[1] /
                             "configs/moe_full7.example.yaml").read_text())
    config["data"]["feature_store_root"] = str(tmp_path / "full7")
    config["output_dir"] = str(tmp_path / "run")
    config["runtime"]["device"] = "cpu"
    config["training"]["max_epochs"] = 1
    config["training"]["batch_size"] = 2
    config["validation"]["batch_size"] = 2
    config["inference"]["batch_size"] = 2
    trained = run_training(config)
    assert trained["global_step"] == 1
    assert trained["history"][0]["train_rows_used"] == 2
    assert trained["score_review_policy"] == "SYNTHETIC_NOT_APPLICABLE"
    assert trained["human_review_performed"] is False
    assert trained["run_purpose"] == "research_only"
    run_manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert run_manifest["expert_profile"] == "full_7"
    assert run_manifest["score_review_policy"] == "SYNTHETIC_NOT_APPLICABLE"
    assert run_manifest["human_review_performed"] is False
    assert run_manifest["run_purpose"] == "research_only"
    assert run_manifest["train_store_identity_sha256"] == store.identity
    assert "allow_train_validation_audio_overlap" not in run_manifest
    assert "audio_overlap_diagnostic" not in run_manifest
    project_root = Path(__file__).resolve().parents[1]

    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    expected_dirty = bool(git_status.stdout.strip())

    assert isinstance(run_manifest["working_tree_dirty"], bool)
    assert run_manifest["working_tree_dirty"] == expected_dirty
    assert run_manifest["source_code_sha256"]["model/moe_gating_core.py"]
    checkpoint = trained["best_checkpoint"]
    assert checkpoint is not None
    evaluated = evaluate_checkpoint(config, checkpoint)
    assert evaluated["sample_count"] == 2
    assert "validation_metrics_warning" not in evaluated
    submitted = infer_checkpoint(config, checkpoint, str(tmp_path / "submission.csv"))
    assert submitted["sample_count"] == 2
    assert (tmp_path / "submission.csv").read_text().splitlines()[0] == "wave_file_name,pred_score"
    bad_e5 = torch.load(e5_path, weights_only=False)
    bad_e5["component_model_identities"]["top2_m2d"] = {"sha256": "f" * 64}
    bad_e5_path = tmp_path / "bad_e5.pt"
    torch.save(bad_e5, bad_e5_path)
    with pytest.raises(ValueError, match="component model provenance"):
        build_full7_split(full9, whisper, bad_e5_path, e6_path, tmp_path / "other_full7",
                          split="train", source_audio_dir=audio)
    (audio / "0.wav").write_bytes(b"tampered source")
    with pytest.raises(ValueError, match="source WAV changed"):
        build_full7_split(full9, whisper, e5_path, e6_path, tmp_path / "other_full7",
                          split="train", source_audio_dir=audio)
    altered_root = tmp_path / "altered_full7"
    shutil.copytree(tmp_path / "full7" / "test", altered_root / "test")
    manifest_path = altered_root / "test" / "manifest.json"
    altered = json.loads(manifest_path.read_text())
    altered["assembly_identity"]["score_contracts"]["expert3"]["semantics"] = "contradictory_regression"
    altered["score_contracts"] = altered["assembly_identity"]["score_contracts"]
    contract_hash = hashlib.sha256(_canonical_json(altered["assembly_identity"])).hexdigest()
    altered["core_assembly_contract_sha256"] = contract_hash
    shard_meta_path = altered_root / "test" / "shards" / "shard_000000" / "metadata.json"
    shard_meta = json.loads(shard_meta_path.read_text())
    shard_meta["assembly_contract_sha256"] = contract_hash
    shard_meta_path.write_text(json.dumps(shard_meta))
    altered["store_identity_sha256"] = hashlib.sha256(_canonical_json(
        {key: value for key, value in altered.items() if key != "store_identity_sha256"})).hexdigest()
    manifest_path.write_text(json.dumps(altered))
    altered_config = {**config, "data": {**config["data"], "feature_store_root": str(altered_root)}}
    with pytest.raises(ValueError, match="preprocessing contract"):
        infer_checkpoint(altered_config, checkpoint, str(tmp_path / "rejected.csv"))
    from data.moe_feature_store import validate_feature_store
    shard = tmp_path / "full7" / "test" / "shards" / "shard_000000" / "scores.npy"
    with shard.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_feature_store(tmp_path / "full7", split="test")


def test_full7_membership_tracks_official_validation_and_legacy_holdout(tmp_path):
    from data.full7_assembler import _head_membership

    full9, _, _ = _sources(tmp_path, split="train")
    train = SetACachedDataset(full9, split="train")
    _sources(tmp_path, split="validation")
    validation = SetACachedDataset(full9, split="validation", require_labels=True)
    _sources(tmp_path, split="test")
    test = SetACachedDataset(full9, split="test")
    train_hash = train.provenance["source_manifest_sha256"]
    validation_hash = validation.provenance["source_manifest_sha256"]
    e6_payload = {"dataset_provenance": {"train": {
        "full9_source_manifest_sha256": train_hash,
    }}}
    new_e5 = {
        "data_provenance": train.provenance,
        "training_state": {
            "selection_policy": "official_validation_mse_v1",
            "train_provenance": train.provenance,
            "validation_provenance": validation.provenance,
            "train_sample_ids": train.sample_ids,
            "validation_sample_ids": validation.sample_ids,
        },
    }
    train_membership = _head_membership(
        "train", train.sample_ids, train_hash, new_e5, e6_payload, synthetic=False
    )
    validation_membership = _head_membership(
        "validation", validation.sample_ids, validation_hash, new_e5,
        {"dataset_provenance": {"train": {"full9_source_manifest_sha256": train_hash}}},
        synthetic=False,
    )
    test_membership = _head_membership(
        "test", test.sample_ids, test.provenance["source_manifest_sha256"], new_e5,
        {"dataset_provenance": {"train": {"full9_source_manifest_sha256": train_hash}}},
        synthetic=False,
    )
    assert train_membership["expert5"]["per_row"] == ["IN_SAMPLE_FIT"] * len(train)
    assert validation_membership["expert5"]["per_row"] == [
        "OUT_OF_SAMPLE_FIT_VALIDATION_SELECTED"
    ] * len(validation)
    assert test_membership["expert5"]["per_row"] == ["OUT_OF_SAMPLE"] * len(test)

    legacy_e5 = {
        "data_provenance": train.provenance,
        "training_state": {
            "fit_sample_ids": train.sample_ids[:1],
            "holdout_sample_ids": train.sample_ids[1:],
        },
    }
    legacy = _head_membership(
        "train", train.sample_ids, train_hash, legacy_e5, e6_payload, synthetic=False
    )
    assert legacy["expert5"]["per_row"] == [
        "IN_SAMPLE_FIT", "OUT_OF_SAMPLE_FIT_SELECTION_HOLDOUT"
    ]


@pytest.mark.parametrize("duplicate_pair", [("train", "test"), ("validation", "test"),
                                             ("train", "validation")])
def test_builder_cli_accepts_duplicate_wav_across_any_official_splits(tmp_path, monkeypatch, duplicate_pair):
    from scripts import build_full7_feature_store as builder

    for split in ("train", "validation", "test"):
        if split in duplicate_pair:
            payloads = [b"shared WAV bytes", f"{split} unique".encode()]
        else:
            payloads = [f"{split} unique 0".encode(), f"{split} unique 1".encode()]
        full9, whisper, _ = _sources(tmp_path, split=split, audio_payloads=payloads)
    built = []
    monkeypatch.setattr(builder, "build_full7_split", lambda *args, **kwargs: (
        built.append(kwargs["split"]) or {"split": kwargs["split"], "sample_count": 2}
    ))
    report_path = tmp_path / "build-report.json"
    code = builder.main([
        "--full9-root", str(full9), "--whisper-root", str(whisper),
        "--e5-checkpoint", str(tmp_path / "e5.pt"), "--e6-checkpoint", str(tmp_path / "e6.pt"),
        "--output-root", str(tmp_path / "full7"), "--source-audio-root", str(tmp_path),
        "--split", "train", "--split", "validation", "--split", "test",
        "--report-json", str(report_path),
    ])
    assert code == 0
    report = json.loads(report_path.read_text())
    assert built == ["train", "validation", "test"]
    assert [row["sample_count"] for row in report["splits"]] == [2, 2, 2]
    assert "audio_overlap_diagnostic" not in report
    assert "test_overlap_diagnostic" not in report
