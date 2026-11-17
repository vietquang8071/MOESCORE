#!/usr/bin/env python3
"""Bounded, resumable FP32 extraction using the verified expert factories.

The parent hashes inputs once. Each worker owns one expert for all splits and
exits before the next expert starts. Intermediate NPZ files are bounded
verification shards; final stores use MoeFeatureStoreWriter unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.moe_config import load_moe_config, resolve_moe_paths
from data.unified_dataset import UnifiedXACLEDataset
from data.moe_feature_store import MoeFeatureStoreWriter, validate_feature_store, _atomic_json, _sha256_file
from data.moe_expert_cache import (
    build_assembly_identity,
    build_expert_cache_identity,
    build_sample_identity,
    load_expert_shard,
    resolve_manifest_wav_paths,
    save_expert_shard,
    validate_assembly_identity,
)
from model.expert_adapters.registry import (
    EXPERT_ORDER, EXPERT_FEATURE_DIMS, EXPERT_INPUT_CONTRACTS,
    EXPERT_FEATURE_DIMS_FULL9, TOP2_EXPERT_INPUT_CONTRACTS, create_expert_adapter,
    validate_and_assemble_expert_outputs,
)

EXPECTED_COUNTS = {"train": 7500, "validation": 3000, "test": 3000}
_TEXT_SNAPSHOT_CACHE = {}
_PAIRED_CROSS_EXPERTS = frozenset(
    {"expert1", "expert2", "top2_m2d", "top2_msclap", "top2_laion", "top2_mga"}
)
_PAIRED_CROSS_REPRESENTATION = {
    "feature_representation": "paired_cross_v1",
    "feature_blocks": [
        "audio_embedding",
        "text_embedding",
        "absolute_difference_l2",
        "elementwise_product_l2",
        "repeated_cosine_l2",
    ],
    "feature_width_multiplier": 5,
}
CHECKPOINT_FIELDS = {
    "expert1": ("pretrained_checkpoint", "finetuned_checkpoint"),
    "expert2": ("pretrained_checkpoint", "finetuned_checkpoint"),
    "expert3": ("base_checkpoint", "finetuned_checkpoint"),
    "expert4": ("beats_checkpoint", "clap_checkpoint", "finetuned_checkpoint"),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def job_identity(job):
    """Identity for a complete assembly request, excluding execution tuning."""
    if "assembly_ids" in job:
        return fingerprint({
            "schema_version": 2,
            "expert_profile": job["expert_profile"],
            "expert_order": job["expert_order"],
            "assembly_ids": job["assembly_ids"],
        })
    # Legacy global jobs are not accepted as cache provenance. This branch is
    # retained only for callers displaying their historical identity.
    return fingerprint({key: value for key, value in job.items() if key not in {"batch_sizes", "shard_size"}})


def assembly_store_root(output_root, job):
    """Return the immutable, profile-specific final-store destination."""
    return (
        Path(output_root) / "feature_cache" / "assemblies"
        / job["expert_profile"] / job_identity(job)
    )


_ADAPTER_SOURCE_FILES = {
    "expert1": "model/expert_adapters/expert1_adapter.py",
    "expert2": "model/expert_adapters/expert2_adapter.py",
    "expert3": "model/expert_adapters/expert3_adapter.py",
    "expert4": "model/expert_adapters/expert4_adapter.py",
    "top2_m2d": "model/expert_adapters/top2_m2d_adapter.py",
    "top2_msclap": "model/expert_adapters/top2_msclap_adapter.py",
    "top2_laion": "model/expert_adapters/top2_laion_adapter.py",
    "top2_mga": "model/expert_adapters/top2_mga_adapter.py",
    "m2d_lora": "model/expert_adapters/m2d_lora_adapter.py",
}
_PREPROCESS_METHODS = {
    "expert1": "_expert1_audio",
    "expert2": "_expert2_audio",
    "expert3": "_expert3_audio",
    "expert4": "_expert4_audio",
    "top2_m2d": "_expert3_audio",
    "top2_msclap": "_expert2_audio",
    "top2_laion": "_expert1_audio",
    "top2_mga": "_expert2_audio",
    "m2d_lora": "_expert3_audio",
}
_TEXT_RESOURCE_IDS = {
    "expert1": ("roberta-base",),
    "expert2": ("bert-base-uncased",),
    "expert3": ("google-bert/bert-base-uncased",),
    "expert4": ("roberta-base",),
    "top2_m2d": ("google-bert/bert-base-uncased",),
    "top2_msclap": ("gpt2",),
    "top2_laion": ("roberta-base",),
    "top2_mga": ("bert-base-uncased",),
    # PortableM2D's BertXEncoder defaults to this exact Hub ID.
    "m2d_lora": ("google-bert/bert-base-uncased",),
}
_DEPENDENCIES = {
    "expert1": ("torch", "numpy", "laion-clap", "transformers", "torchaudio", "torchlibrosa"),
    "expert2": ("torch", "numpy", "scipy", "transformers", "timm", "sentence-transformers", "PyYAML"),
    "expert3": ("torch", "numpy", "scipy", "soundfile", "torchaudio", "transformers", "timm", "einops", "nnAudio"),
    "expert4": ("torch", "numpy", "scipy", "torchaudio", "transformers", "laion-clap"),
    "top2_m2d": ("torch", "numpy", "scipy", "soundfile", "transformers", "timm", "einops", "nnAudio"),
    "top2_msclap": ("torch", "numpy", "msclap", "transformers", "torchaudio", "librosa"),
    "top2_laion": ("torch", "numpy", "transformers", "laion-clap", "torchaudio"),
    "top2_mga": ("torch", "numpy", "scipy", "transformers", "timm", "sentence-transformers", "PyYAML"),
    "m2d_lora": ("torch", "numpy", "scipy", "soundfile", "torchaudio", "transformers", "timm", "einops", "nnAudio"),
}


def _source_files_under(root):
    root = Path(root).expanduser().resolve(strict=True)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise NotADirectoryError(f"Implementation resource is not a file or directory: {root}")
    return sorted(
        path.resolve(strict=True)
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}
        and ".git" not in path.parts and "__pycache__" not in path.parts
        and not any(part.lower() in {"test", "tests"} for part in path.parts)
    )


def _text_snapshot_identity(model_id, runtime, snapshot_cache):
    cache_roots = []
    for candidate in (
        runtime.get("cache_dir"), os.environ.get("HF_HUB_CACHE"),
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        if candidate:
            root = Path(candidate).expanduser()
            if root.name != "hub" and root.name != "models":
                root = root / "hub"
            if root not in cache_roots:
                cache_roots.append(root)
    cache_key = (model_id, tuple(str(root.resolve(strict=False)) for root in cache_roots))
    if cache_key in snapshot_cache:
        return snapshot_cache[cache_key]
    model_dir_name = "models--" + model_id.replace("/", "--")
    cache_views = []
    found_snapshots = False
    for cache_root in cache_roots:
        model_root = cache_root / model_dir_name
        refs_root = model_root / "refs"
        refs = {
            str(path.relative_to(refs_root)): path.read_text(encoding="utf-8").strip()
            for path in sorted(refs_root.rglob("*"))
            if path.is_file()
        } if refs_root.is_dir() else {}
        snapshots = []
        for snapshot in sorted((model_root / "snapshots").glob("*")):
            if not snapshot.is_dir():
                continue
            files = {}
            for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
                files[str(path.relative_to(snapshot))] = _sha256_file(path)
            snapshots.append({"revision": snapshot.name, "files_sha256": files})
        found_snapshots = found_snapshots or bool(snapshots)
        cache_views.append({"refs": refs, "snapshots": snapshots})
    result = {"model_id": model_id, "cache_views": cache_views}
    if not found_snapshots:
        raise FileNotFoundError(
            f"Cannot establish content identity for required Hugging Face text/tokenizer "
            f"resource {model_id!r}; populate the configured local cache before extraction"
        )
    snapshot_cache[cache_key] = result
    return result


def compute_expert_cache_identity(expert, config, *, contract=None, feature_dim=None, device=None):
    """Fingerprint only resources and behavior that can affect this expert."""
    if expert not in config.get("experts", {}):
        raise ValueError(f"Configuration has no resources for expert {expert!r}")
    section = config["experts"][expert]
    if contract is None:
        contract = EXPERT_INPUT_CONTRACTS.get(expert) or TOP2_EXPERT_INPUT_CONTRACTS.get(expert)
    if contract is None:
        raise ValueError(f"No effective input contract is defined for {expert!r}")
    if feature_dim is None:
        feature_dim = EXPERT_FEATURE_DIMS.get(expert)
        if feature_dim is None:
            feature_dim = EXPERT_FEATURE_DIMS_FULL9.get(expert)
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError(f"A verified feature_dim is required to identify {expert!r}")
    selected_device = str(device if device is not None else config.get("runtime", {}).get("device", "auto"))
    if selected_device == "auto":
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_info = {
        "requested_device": selected_device,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if selected_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{expert} cache identity requested {selected_device}, but CUDA is unavailable")
        cuda_index = torch.device(selected_device).index or 0
        device_info.update({
            "name": torch.cuda.get_device_name(cuda_index),
            "capability": list(torch.cuda.get_device_capability(cuda_index)),
            "cudnn_version": torch.backends.cudnn.version(),
        })

    checkpoint_fields = CHECKPOINT_FIELDS.get(expert)
    if checkpoint_fields is None:
        checkpoint_fields = tuple(
            key for key in section if "checkpoint" in key.lower() and section[key] is not None
        )
    checkpoints = {}
    for field in checkpoint_fields:
        path = Path(section[field]).expanduser().resolve(strict=True)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"Missing {expert} checkpoint resource {field}: {path}")
        checkpoints[field] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}

    implementation_files = {
        PROJECT_ROOT / _ADAPTER_SOURCE_FILES[expert],
        PROJECT_ROOT / "model/expert_adapters/base.py",
    }
    if expert in _PAIRED_CROSS_EXPERTS:
        implementation_files.add(
            PROJECT_ROOT / "model/expert_adapters/paired_cross_features.py"
        )
    extra_implementation = {}
    if expert.startswith("top2_") or expert == "m2d_lora":
        implementation_files.add(PROJECT_ROOT / "model/expert_adapters/top2_common.py")
    batch_helper = globals().get("prepare_expert_inputs")
    if batch_helper is not None:
        extra_implementation["scripts/extract_moe_feature_store.py::prepare_expert_inputs"] = hashlib.sha256(
            inspect.getsource(batch_helper).encode()
        ).hexdigest()
    if getattr(contract, "requires_wav_paths", False):
        extra_implementation["data/moe_expert_cache.py::resolve_manifest_wav_paths"] = hashlib.sha256(
            inspect.getsource(resolve_manifest_wav_paths).encode()
        ).hexdigest()
    # Source modules are scoped to the selected expert. Registry source and
    # other adapter files are deliberately excluded; only this expert's
    # effective serialized input contract is included below.
    for key in ("source_dir", "unilm_source_dir"):
        value = section.get(key)
        if value:
            implementation_files.update(_source_files_under(value))
    for key, value in section.items():
        if any(token in key.lower() for token in ("yaml", "config", "tokenizer", "vocab")) and value:
            resource_path = Path(value).expanduser()
            if resource_path.exists():
                implementation_files.update(_source_files_under(resource_path))
    if expert == "expert2":
        implementation_files.add(PROJECT_ROOT / "expert2_finetune/expert2_model.py")
    if expert == "expert1":
        implementation_files.update(_source_files_under(PROJECT_ROOT / "expert1_finetune"))
    if expert == "expert3":
        implementation_files.update(
            _source_files_under(PROJECT_ROOT / "expert3_finetune/XACLE_2027_baselinemodel")
        )
    if expert in {"top2_m2d", "m2d_lora"}:
        implementation_files.update(
            _source_files_under(PROJECT_ROOT / "expert3_finetune/XACLE_2027_baselinemodel")
        )
    if expert == "m2d_lora":
        implementation_files.add(
            PROJECT_ROOT / "model/expert_adapters/m2d_lora_runtime.py"
        )
    if expert == "top2_mga" and not section.get("source_dir"):
        implementation_files.update(
            _source_files_under(PROJECT_ROOT / "expert2_finetune/MGA-CLAP")
        )
    if expert == "expert4":
        implementation_files.update(
            PROJECT_ROOT / "expert4_finetune" / filename
            for filename in (
                "inference.py", "trainer.py", "model.py", "audio_encoder.py",
                "text_encoder.py", "seq_coattn.py", "source_compatibility.py",
            )
        )
    missing_sources = [path for path in implementation_files if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing {expert} implementation source files: {missing_sources}")
    implementation = {
        str(path.resolve().relative_to(PROJECT_ROOT)) if path.resolve().is_relative_to(PROJECT_ROOT)
        else str(path.resolve()): _sha256_file(path)
        for path in sorted(implementation_files)
    } | extra_implementation

    from data.unified_dataset import TARGET_RATES, TARGET_SAMPLES, UnifiedXACLEDataset
    method_name = _PREPROCESS_METHODS.get(expert)
    preprocess_method = getattr(UnifiedXACLEDataset, method_name, None)
    if preprocess_method is None:
        raise ValueError(f"No preprocessing implementation is registered for {expert!r}")
    preprocess_code = inspect.getsource(preprocess_method)
    if expert in {"expert2", "expert4"}:
        preprocess_code += inspect.getsource(UnifiedXACLEDataset._mono_float)
    preprocessing = {
        "dataset_method": method_name,
        "dataset_method_sha256": hashlib.sha256(preprocess_code.encode()).hexdigest(),
        "effective_sample_rate_hz": TARGET_RATES["expert" + expert[-1]] if expert.startswith("expert") else contract.sample_rate_hz,
        "effective_sample_count": TARGET_SAMPLES["expert" + expert[-1]] if expert.startswith("expert") else contract.max_samples,
        "train_crop_policy": (
            "center_validation_mode" if method_name == "_expert1_audio" else "not_randomized"
        ),
        "contract": asdict(contract),
    }
    runtime = config.get("runtime", {})
    text_model_ids = list(_TEXT_RESOURCE_IDS.get(expert, ()))
    if expert == "top2_laion" and "checkpoint" not in section:
        configured_model = str(section.get("model_name_or_path", "laion/clap-htsat-unfused"))
        if configured_model not in text_model_ids:
            text_model_ids.append(configured_model)
    text_resources = {
        model_id: _text_snapshot_identity(model_id, runtime, _TEXT_SNAPSHOT_CACHE)
        for model_id in text_model_ids
    }
    dependencies = {}
    for distribution in _DEPENDENCIES.get(expert, ()):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    dependencies["text_encoder_tokenizer_resources"] = text_resources
    inference_policy = {
        "module_mode": "eval",
        "parameters_frozen": True,
        "grad_enabled": False,
        "output_dtype": "float32",
        "batch_size_affects_math": False,
        "device": device_info,
    }
    if expert == "top2_msclap":
        inference_policy.update({
            "audio_crop_policy": "center",
            "audio_duration_seconds": 7,
            "sample_rate_hz": 44_100,
            "short_audio_policy": "repeat_then_truncate",
        })
    score = {
        "score_shape": [None, 1],
        "score_dtype": "float32",
        "feature_shape": [None, feature_dim],
        "feature_dtype": "float32",
        "score_semantics": "adapter-defined pretrained/fine-tuned score",
    }
    if expert == "top2_msclap":
        score["score_semantics"] = "cosine(audio_embedding,text_embedding)"
    if expert in _PAIRED_CROSS_EXPERTS:
        score.update(_PAIRED_CROSS_REPRESENTATION)
    return build_expert_cache_identity(
        expert_name=expert,
        checkpoint_provenance=checkpoints or {"published_model_identities": {"text": text_resources}},
        implementation_provenance=implementation,
        preprocessing_provenance=preprocessing,
        input_contract=asdict(contract),
        inference_policy=inference_policy,
        score_contract=score,
        relevant_dependencies=dependencies,
    )


def bind_json(path, value):
    """Reject changes to an interrupted extraction's identity."""
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Provenance/manifest differs; use a new output directory: {path}")
    _atomic_json(path, value)


def checkpoint_provenance(config):
    cache = {}
    result = {}
    for expert in EXPERT_ORDER:
        result[expert] = {}
        for field in CHECKPOINT_FIELDS[expert]:
            path = Path(config["experts"][expert][field])
            if not path.is_file():
                raise FileNotFoundError(f"Missing approved checkpoint: {path}")
            if path not in cache:
                cache[path] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            result[expert][field] = cache[path]
    return result


def projected_dataset(dataset_root, split, work_root, *, limit=None):
    """Exclude test label columns before constructing UnifiedXACLEDataset.

    average_score values are never accessed for test. The label-free metadata
    file also prevents Unified Dataset's optional-label parser from seeing them.
    """
    root, work = Path(dataset_root), Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    source = root / "meta_data" / f"{split}_average.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        required = {"wav_file_name", "text"} | ({"average_score"} if split != "test" else set())
        if not required.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid average-score headers in {source}: {headers}")
        # Model metadata is deliberately omitted from model inputs.
        selected = ["wav_file_name", "text"] + (["average_score"] if split != "test" else [])
        rows = [{key: row[key] for key in selected} for row in reader]
    if limit is not None:
        rows = rows[:limit]
    path = work / f"{split}_inputs.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected)
        writer.writeheader()
        writer.writerows(rows)
    dataset = UnifiedXACLEDataset(path, root / "wav" / split, split=split,
                                  require_labels=split != "test")
    return dataset, headers


def prepare_expert_inputs(dataset, expert, records, *, start, device):
    """Build one expert's input from trusted dataset rows, preserving row order."""
    contract = EXPERT_INPUT_CONTRACTS.get(expert) or TOP2_EXPERT_INPUT_CONTRACTS.get(expert)
    if contract is None:
        raise ValueError(f"No input contract is registered for {expert!r}")
    if not records:
        raise ValueError("Cannot construct an empty expert input batch")
    waveforms = []
    valid_lengths = []
    paths = []
    captions = []
    for offset, record in enumerate(records):
        index = start + offset
        if index >= len(dataset.rows):
            raise ValueError("Requested manifest shard exceeds trusted dataset rows")
        row = dataset.rows[index]
        expected_split = record["sample_id"].split(":", 1)[0]
        if (
            row["sample_id"] != record["sample_id"]
            or row["wave_file_name"] != record["wave_file_name"]
            or row["captions_raw"] != record["raw_caption"]
            or row["split"] != expected_split
        ):
            raise ValueError("Trusted dataset row differs from the selected manifest identity/caption")
        path = Path(row["audio_path"]).resolve(strict=True)
        root = Path(dataset.audio_dir).resolve(strict=True)
        if not path.is_relative_to(root) or path.name != record["wave_file_name"]:
            raise ValueError(f"Trusted dataset WAV path escapes or mismatches its split root: {path}")
        if _sha256_file(path) != record["wav_sha256"]:
            raise ValueError(f"Source WAV changed after manifest selection: {record['sample_id']}")
        paths.append(str(path))
        captions.append(record["raw_caption"])
        method_name = _PREPROCESS_METHODS[expert]
        method = getattr(dataset, method_name)
        # Expert 1's preprocessing uses the row index only for train-mode random
        # crops. Top2 LAION deliberately shares the same method and evaluation
        # center crop, so dispatch on the method contract rather than expert ID.
        waveform = method(path, index) if method_name == "_expert1_audio" else method(path)
        if expert == "expert4":
            waveform, valid_length = waveform
            valid_lengths.append(valid_length)
        if not isinstance(waveform, torch.Tensor) or waveform.dtype != torch.float32:
            raise TypeError(f"{expert} preprocessing must return a float32 waveform tensor")
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"{expert} preprocessing returned an invalid single-sample shape {tuple(waveform.shape)}")
        if waveform.shape[1] != contract.max_samples or not torch.isfinite(waveform).all():
            raise ValueError(f"{expert} preprocessing returned an invalid length or non-finite waveform")
        waveforms.append(waveform)
    audio = torch.stack(waveforms, dim=0)
    expected_shape = (len(records), *contract.audio_shape[1:])
    if tuple(audio.shape) != expected_shape:
        raise ValueError(f"{expert} input shape {tuple(audio.shape)} differs from {expected_shape}")
    inputs = {
        contract.audio_key: audio.to(device),
        contract.caption_key: captions,
    }
    if contract.lengths_key:
        # Expert 4's source adapter consumes valid sample lengths, distinct
        # from its padded waveform width.
        inputs[contract.lengths_key] = torch.tensor(valid_lengths, dtype=torch.int64, device=device)
    if contract.requires_wav_paths:
        split = records[0]["sample_id"].split(":", 1)[0]
        inputs["wav_paths"] = resolve_manifest_wav_paths(
            dataset, records, split=split, start=start
        )
        inputs["wave_file_names"] = [record["wave_file_name"] for record in records]
        inputs["split"] = split
    return inputs


def prepare_job(config_path, dataset_root, output_root, *, batch_sizes=None, shard_size=128, limit=None, device=None):
    if shard_size <= 0 or (limit is not None and not 1 <= limit <= 4):
        raise ValueError("shard_size must be positive; smoke limit must be 1..4")
    config = resolve_moe_paths(load_moe_config(config_path), config_path)
    selected_device = str(device if device is not None else config["runtime"]["device"])
    if selected_device == "auto":
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output = Path(output_root).resolve()
    cache_root = output / "feature_cache"
    work = cache_root / "projections"
    batch_sizes = batch_sizes or dict.fromkeys(EXPERT_ORDER, 1)
    if set(batch_sizes) != set(EXPERT_ORDER) or any(type(n) is not int or n <= 0 for n in batch_sizes.values()):
        raise ValueError("Provide one positive extraction batch size per expert")
    provenance = checkpoint_provenance(config)
    manifests = {}
    for split in EXPECTED_COUNTS:
        dataset, headers = projected_dataset(dataset_root, split, work, limit=limit)
        if limit is None and len(dataset) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} rows, observed {len(dataset)}")
        records = []
        for row in tqdm(dataset.rows, desc=f"Identity {split}"):
            record = {"sample_id": row["sample_id"], "wave_file_name": row["wave_file_name"],
                      "raw_caption": row["captions_raw"], "wav_sha256": _sha256_file(row["audio_path"])}
            if split != "test":
                record["label_raw"] = row["label_value"]
            records.append(record)
        manifest = {"schema_version": 1, "split": split, "records": records}
        manifest["sha256"] = fingerprint(manifest)
        manifest_path = cache_root / "manifests" / split / f"{manifest['sha256']}.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Content-addressed source manifest is corrupt: {manifest_path}")
        if not manifest_path.exists():
            _atomic_json(manifest_path, manifest)
        manifests[split] = manifest
        print(f"{split}: {len(records)} pairs; columns={headers}; labels_consumed={split != 'test'}", flush=True)
    expert_identities = {
        expert: compute_expert_cache_identity(expert, config, device=selected_device)
        for expert in EXPERT_ORDER
    }
    assembly_identities = {}
    for split, manifest in manifests.items():
        assembly_identities[split] = build_assembly_identity(
            expert_profile="top1_4",
            expert_order=EXPERT_ORDER,
            expert_cache_identities=expert_identities,
            feature_dimensions=EXPERT_FEATURE_DIMS,
            source_manifest_identity=manifest["sha256"],
        )
    job = {
        "schema_version": 2,
        "expert_profile": "top1_4",
        "expert_order": list(EXPERT_ORDER),
        "inference_device": selected_device,
        "checkpoint_provenance": provenance,
        "expert_cache_identities": expert_identities,
        "feature_dimensions": dict(EXPERT_FEATURE_DIMS),
        "assembly_identities": assembly_identities,
        "assembly_ids": {split: identity["assembly_id"] for split, identity in assembly_identities.items()},
        "preprocessing_identity": {
            "schema_version": 2,
            "expert_profile": "top1_4",
            "expert_order": list(EXPERT_ORDER),
            "assembly_ids": {split: identity["assembly_id"] for split, identity in assembly_identities.items()},
        },
        "manifests": {key: val["sha256"] for key, val in manifests.items()},
        # These affect execution layout/resources but not expert mathematics.
        "shard_size": shard_size,
        "batch_sizes": batch_sizes,
        "smoke": limit is not None,
    }
    job_path = cache_root / "jobs" / f"{job_identity(job)}.json"
    if job_path.exists():
        previous = json.loads(job_path.read_text())
        if job_identity(previous) != job_identity(job):
            raise ValueError(f"Assembly job identity mismatch at {job_path}")
    _atomic_json(job_path, job)
    assembly_root = assembly_store_root(output, job)
    for split, identity in assembly_identities.items():
        identity_path = assembly_root / f"{split}_assembly_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"Assembly identity record is corrupt: {identity_path}")
        if not identity_path.exists():
            _atomic_json(identity_path, identity)
    return config, job, manifests


def load_chunk(path, *, expert, records, identity, sample_identity=None):
    """Load only schema-v2 per-expert shards; reject legacy global shards."""
    if not isinstance(identity, dict) or "cache_id" not in identity:
        raise ValueError("Legacy global extraction identities cannot validate or migrate a new expert shard")
    if sample_identity is None:
        raise ValueError("A per-shard sample identity is required for cache reuse")
    if [item["sample_id"] for item in records] != [
        item["sample_id"] for item in sample_identity["ordered_samples"]
    ]:
        raise ValueError("Shard records differ from their bound sample identity")
    return load_expert_shard(
        path,
        expert_identity=identity,
        sample_identity=sample_identity,
        feature_dim=EXPERT_FEATURE_DIMS[expert],
    )


def save_chunk(path, *, expert, records, identity, sample_identity, scores, features,
               factory_restoration, frozen_eval):
    if [item["sample_id"] for item in records] != [
        item["sample_id"] for item in sample_identity["ordered_samples"]
    ]:
        raise ValueError("Shard records differ from their bound sample identity")
    return save_expert_shard(
        path,
        expert_identity=identity,
        sample_identity=sample_identity,
        scores=scores,
        features=features,
        inference_metadata={
            "factory_restoration": factory_restoration,
            "frozen_eval": frozen_eval,
        },
    )


def chunk_path(root, expert, split, index, *, cache_id, sample_identity_sha256):
    if not isinstance(cache_id, str) or len(cache_id) != 64:
        raise ValueError("chunk_path requires an expert cache SHA-256 identity")
    if not isinstance(sample_identity_sha256, str) or len(sample_identity_sha256) != 64:
        raise ValueError("chunk_path requires a per-shard sample SHA-256 identity")
    return (
        Path(root) / "feature_cache" / "experts" / expert / cache_id / split
        / f"shard_{index:06d}_{sample_identity_sha256}.npz"
    )


def validate_manifests(job, manifests):
    if set(manifests) != set(EXPECTED_COUNTS):
        raise ValueError("Missing or extra extraction split manifests")
    for split, manifest in manifests.items():
        body = {key: value for key, value in manifest.items() if key != "sha256"}
        if (manifest.get("split") != split or fingerprint(body) != manifest.get("sha256")
                or manifest["sha256"] != job["manifests"][split]):
            raise ValueError(f"Extraction source manifest identity changed: {split}")
        ids = [r["sample_id"] for r in manifest["records"]]
        if len(ids) != len(set(ids)) or any(not sid.startswith(split + ":") for sid in ids):
            raise ValueError("Duplicate or mixed-split extraction sample identities")


def extract_worker(config, job, manifests, dataset_root, output_root, expert, device):
    from scripts.verify_real_expert_to_gate import memory_snapshot, _validate_real_module_origins
    validate_manifests(job, manifests)
    if str(device) != job.get("inference_device"):
        raise ValueError(
            f"Worker device {device!r} differs from prepared cache inference device "
            f"{job.get('inference_device')!r}; prepare a matching assembly job"
        )
    identity = job_identity(job)
    expert_identity = job["expert_cache_identities"][expert]
    cache_id = expert_identity["cache_id"]
    report_path = Path(output_root) / "feature_cache" / "experts" / expert / cache_id / "runtime.json"
    pending = []
    reused_shards = 0
    for split, manifest in manifests.items():
        for index, start in enumerate(range(0, len(manifest["records"]), job["shard_size"])):
            records = manifest["records"][start:start + job["shard_size"]]
            sample_identity = build_sample_identity(
                split,
                records,
                {
                    "expert_cache_id": cache_id,
                    "input_contract": expert_identity["input_contract"],
                    "preprocessing_provenance": expert_identity["preprocessing_provenance"],
                },
                job["feature_dimensions"][expert],
            )
            path = chunk_path(
                output_root, expert, split, index, cache_id=cache_id,
                sample_identity_sha256=sample_identity["sample_identity_sha256"],
            )
            if path.exists() or path.with_suffix(".json").exists():
                load_chunk(
                    path, expert=expert, records=records, identity=expert_identity,
                    sample_identity=sample_identity,
                )
                reused_shards += 1
            else:
                pending.append((split, index, start, records, path, sample_identity))
    if not pending:
        report = {
            "status": "PASS",
            "expert": expert,
            "expert_cache_id": cache_id,
            "assembly_job_identity": identity,
            "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY",
            "frozen_eval": True,
            "reused_validated_shards": True,
            "validated_shards": reused_shards,
            "model_constructed": False,
        }
        _atomic_json(report_path, report)
        return report
    # Detect changed checkpoint contents before any model is constructed.
    stats = {}
    for field, expected in expert_identity["checkpoint_provenance"].items():
        path = Path(config["experts"][expert][field])
        if _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint changed since expert cache preflight: {path}")
        stats[field] = (path.stat().st_size, path.stat().st_mtime_ns)
    measurements = [memory_snapshot("before_factory", device)]
    adapter = create_expert_adapter(expert, config, device=device)
    for field, before in stats.items():
        path = Path(config["experts"][expert][field])
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Checkpoint file changed during source factory restoration")
    if any(p.requires_grad for p in adapter.parameters()) or any(m.training for m in adapter.modules()):
        raise ValueError("Source factory must return a fully frozen/eval expert")
    origins = _validate_real_module_origins(expert, config)
    measurements.append(memory_snapshot("after_factory_and_restore", device))
    report = {"status": "IN_PROGRESS", "expert": expert, "pid": os.getpid(),
              "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY", "frozen_eval": True,
              "checkpoint_provenance": expert_identity["checkpoint_provenance"],
              "expert_cache_id": cache_id,
              "batch_size": job["batch_sizes"][expert], "dtype": "float32", "device": str(device),
              "sample_count_by_split": {s: len(m["records"]) for s, m in manifests.items()},
              "model_class": type(getattr(adapter, "original_model", getattr(adapter, "model", adapter))).__name__,
              "module_origins": origins, "memory": measurements, "assembly_job_identity": identity,
              "reused_validated_shards": reused_shards, "model_constructed": True}
    _atomic_json(report_path, report)
    datasets = {}
    for split in manifests:
        dataset, _ = projected_dataset(dataset_root, split, Path(output_root) / "feature_cache" / "projections",
                                       limit=len(manifests[split]["records"]) if job["smoke"] else None)
        # Eval-style center crop for E1; row IDs keep their original split.
        dataset.split = "validation"
        datasets[split] = dataset
    first = True
    for split, index, start, records, path, sample_identity in tqdm(pending, desc=expert):
        dataset = datasets[split]
        scores, features = [], []
        for offset in range(0, len(records), job["batch_sizes"][expert]):
            selected = records[offset:offset + job["batch_sizes"][expert]]
            inputs = prepare_expert_inputs(
                dataset, expert, selected, start=start + offset, device=device
            )
            with torch.inference_mode():
                score, feature = adapter(inputs)
            if (score.dtype != torch.float32 or feature.dtype != torch.float32
                    or tuple(score.shape) != (len(selected), 1)
                    or tuple(feature.shape) != (len(selected), job["feature_dimensions"][expert])
                    or not torch.isfinite(score).all() or not torch.isfinite(feature).all()):
                raise ValueError("Expert returned a non-FP32 output")
            scores.append(score.detach().cpu().numpy().copy())
            features.append(feature.detach().cpu().numpy().copy())
            if first:
                measurements.append(memory_snapshot("after_first_inference", device))
                first = False
            del inputs, score, feature
        save_chunk(
            path, expert=expert, records=records, identity=expert_identity,
            sample_identity=sample_identity, scores=np.concatenate(scores),
            features=np.concatenate(features),
            factory_restoration=report["factory_restoration"], frozen_eval=report["frozen_eval"],
        )
    measurements.append(memory_snapshot("after_cpu_transfer", device))
    report["status"] = "PASS"
    _atomic_json(report_path, report)
    return report


def assemble_stores(job, manifests, output_root):
    validate_manifests(job, manifests)
    final_root = assembly_store_root(output_root, job)
    reports = {}
    for split, manifest in manifests.items():
        assembly = job["assembly_identities"][split]
        validate_assembly_identity(assembly, expected_expert_order=job["expert_order"])
        records = manifest["records"]
        split_preprocessing_identity = {
            "schema_version": 2,
            "expert_profile": job["expert_profile"],
            "expert_order": job["expert_order"],
            "assembly_identity": assembly,
        }
        writer = MoeFeatureStoreWriter(final_root, split=split,
            expected_sample_ids=[r["sample_id"] for r in records], source_manifest_sha256=manifest["sha256"],
            checkpoint_provenance=job["checkpoint_provenance"], preprocessing_identity=split_preprocessing_identity)
        for index, start in enumerate(range(0, len(records), job["shard_size"])):
            selected = records[start:start + job["shard_size"]]
            outputs = {}
            for expert in EXPERT_ORDER:
                expert_identity = job["expert_cache_identities"][expert]
                sample_identity = build_sample_identity(
                    split,
                    selected,
                    {
                        "expert_cache_id": expert_identity["cache_id"],
                        "input_contract": expert_identity["input_contract"],
                        "preprocessing_provenance": expert_identity["preprocessing_provenance"],
                    },
                    job["feature_dimensions"][expert],
                )
                path = chunk_path(
                    output_root, expert, split, index,
                    cache_id=expert_identity["cache_id"],
                    sample_identity_sha256=sample_identity["sample_identity_sha256"],
                )
                arrays = load_chunk(
                    path, expert=expert, records=selected, identity=expert_identity,
                    sample_identity=sample_identity,
                )
                outputs[expert] = (torch.from_numpy(arrays["scores"]), torch.from_numpy(arrays["features"]))
            scores, features = validate_and_assemble_expert_outputs(outputs)
            labels = np.asarray([r["label_raw"] for r in selected], dtype=np.float32) if split != "test" else None
            writer.write_shard(index, sample_ids=[r["sample_id"] for r in selected],
                               wave_file_names=[r["wave_file_name"] for r in selected],
                               scores=scores, features=features, labels_raw=labels)
        writer.finalize()
        _atomic_json(final_root / split / "source_manifest.json", manifest)
        reports[split] = {"sample_count": len(records), "status": "PASS",
                          "identity": validate_feature_store(final_root, split=split)["store_identity_sha256"]}
    _atomic_json(final_root / "assembly_report.json", {
        "status": "PASS",
        "expert_profile": job["expert_profile"],
        "expert_order": job["expert_order"],
        "assembly_ids": job["assembly_ids"],
        "splits": reports,
    })
    return reports


def completed_stores_match(job, manifests, output_root):
    validate_manifests(job, manifests)
    final_root = assembly_store_root(output_root, job)
    for split, source in manifests.items():
        if not (final_root / split / "manifest.json").exists():
            return False
        existing = validate_feature_store(final_root, split=split, require_labels=split != "test")
        expected_preprocessing = {
            "schema_version": 2,
            "expert_profile": job["expert_profile"],
            "expert_order": job["expert_order"],
            "assembly_identity": job["assembly_identities"][split],
        }
        if (existing["checkpoint_provenance"] != job["checkpoint_provenance"]
                or existing["preprocessing_identity"] != expected_preprocessing
                or existing["source_manifest_sha256"] != source["sha256"]):
            raise ValueError(f"Completed {split} store provenance differs from current extraction")
        if split == "test" and existing["labels_present"]:
            raise ValueError("Test extraction store must be label-free")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,1,1,1")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=EXPERT_ORDER)
    parser.add_argument("--assembly-id", help="Schema-v2 assembly job identity used by worker subprocesses")
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    try:
        if args.worker:
            if not args.assembly_id:
                raise ValueError("Worker subprocess requires --assembly-id from its prepared assembly job")
            config = resolve_moe_paths(load_moe_config(args.config), args.config)
            job = json.loads((root / "feature_cache" / "jobs" / f"{args.assembly_id}.json").read_text())
            manifests = {
                split: json.loads(
                    (root / "feature_cache" / "manifests" / split / f"{job['manifests'][split]}.json").read_text()
                )
                for split in EXPECTED_COUNTS
            }
            extract_worker(config, job, manifests, args.dataset_root, root, args.worker, args.device)
        else:
            sizes = dict(zip(EXPERT_ORDER, map(int, args.batch_sizes.split(","))))
            _, job, manifests = prepare_job(args.config, args.dataset_root, root, batch_sizes=sizes,
                                            shard_size=args.shard_size, limit=4 if args.smoke else None,
                                            device=args.device)
            assembly_job_id = job_identity(job)
            if not completed_stores_match(job, manifests, root):
                for expert in EXPERT_ORDER:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", args.config,
                        "--dataset-root", args.dataset_root, "--output-root", args.output_root,
                        "--device", args.device, "--worker", expert,
                        "--assembly-id", assembly_job_id], check=True)
                assemble_stores(job, manifests, root)
            print(json.dumps({"status": "PASS", "smoke": args.smoke,
                              "cache_root": str(root), "assembly_id": assembly_job_id,
                              "feature_store_root": str(assembly_store_root(root, job))}))
    except Exception as exc:
        _atomic_json(root / "feature_cache/failures" / f"{args.assembly_id or 'preflight'}.json", {"status": "FAIL", "expert": args.worker,
                     "error": f"{type(exc).__name__}: {exc}", "resume": "Validated completed shards are retained"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
