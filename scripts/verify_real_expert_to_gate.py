#!/usr/bin/env python3
"""Xác minh tuần tự đầu ra expert thật đi vào gate LOCAL-10.

Mỗi expert được chạy trong một tiến trình con riêng. Artifact NPZ chỉ chứa
score/feature float32 và chuỗi định danh Unicode; không pickle mô hình hoặc
checkpoint. Chế độ mặc định không giữ nhiều expert thật cùng lúc.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.moe_config import load_moe_config, resolve_moe_paths
from data.unified_collate import collate_unified
from data.unified_dataset import UnifiedXACLEDataset
from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import (
    EXPERT_FEATURE_DIMS,
    EXPERT_ORDER,
    create_expert_adapter,
    validate_and_assemble_expert_outputs,
)
from model.moe_forward_integration import MoEForwardIntegration


EXPECTED_UNILM_COMMIT = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
EXPECTED_UNILM_FINGERPRINT = "fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487"
_ARTIFACT_KEYS = frozenset({
    "expert_name", "expert_order_index", "manifest_sha256", "sample_ids",
    "caption_sha256", "wav_sha256", "scores", "features",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_numeric_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must remain float32; got {value.dtype}")
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must remain numpy.float32; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def select_sample_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    max_samples: int,
    audio_info_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Choose deterministic rows covering duration groups when available.

    Selection keys use only ``seed`` and sample identity. Labels and model IDs
    are copied for provenance but never used to rank or select a row.
    """
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    if audio_info_fn is None:
        import soundfile as sf

        def audio_info_fn(row: Mapping[str, Any]) -> Mapping[str, Any]:
            info = sf.info(str(row["audio_path"]))
            return {"samplerate": int(info.samplerate), "frames": int(info.frames), "channels": int(info.channels)}

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        filename = row.get("wave_file_name")
        caption = row.get("captions_raw")
        path = Path(row.get("audio_path", ""))
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"row {index} has a missing sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        if not isinstance(filename, str) or not filename or filename in seen_files:
            raise ValueError(f"row {index} has a missing or duplicate wave_file_name")
        if not isinstance(caption, str) or not caption:
            raise ValueError(f"row {index} has a missing raw caption")
        if not path.is_file():
            raise FileNotFoundError(f"sample WAV is missing: {path}")
        info = dict(audio_info_fn(row))
        try:
            sample_rate = int(info["samplerate"])
            frames = int(info["frames"])
            channels = int(info["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid audio header for {sample_id}: {info}") from exc
        if sample_rate <= 0 or frames <= 0 or channels <= 0:
            raise ValueError(f"invalid audio header for {sample_id}: {info}")
        ten_second_frames = 10 * sample_rate
        group = "short_under_10s" if frames < ten_second_frames else (
            "exactly_10s" if frames == ten_second_frames else "over_10s"
        )
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "split": str(row.get("split", "train")),
            "row_index": int(row.get("row_index", index)),
            "csv_row_number": int(row.get("csv_row_number", index + 2)),
            "wave_file_name": filename,
            "wav_path": str(path.resolve()),
            "raw_caption": caption,
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "label_raw_0_to_10": row.get("label_value"),
            "model_id": row.get("model_id"),
            "model_name": row.get("model_name"),
            "source_audio": {
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "frames": frames,
                "duration_seconds": frames / sample_rate,
            },
            "duration_group": group,
            "wav_sha256": _sha256_file(path),
        }
        record["_selection_key"] = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        prepared.append(record)
        seen_ids.add(sample_id)
        seen_files.add(filename)

    if not prepared:
        raise ValueError("no valid sample rows are available")
    groups = ("short_under_10s", "exactly_10s", "over_10s")
    by_group = {
        group: sorted((item for item in prepared if item["duration_group"] == group), key=lambda item: item["_selection_key"])
        for group in groups
    }
    required_groups = [group for group in groups if by_group[group]]
    if max_samples < min(3, len(required_groups)):
        raise ValueError("max_samples is too small to preserve available short/exact/long coverage")
    selected: list[dict[str, Any]] = [by_group[group][0] for group in required_groups]
    selected_ids = {item["sample_id"] for item in selected}
    remaining = sorted((item for item in prepared if item["sample_id"] not in selected_ids), key=lambda item: item["_selection_key"])
    selected.extend(remaining[: max_samples - len(selected)])
    selected = selected[:max_samples]
    for item in selected:
        item.pop("_selection_key", None)
    return selected


def make_sample_manifest(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    split: str,
    dataset_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("LOCAL-10.1 score diagnostics require the train split")
    normalized_samples = [dict(sample) for sample in samples]
    ids = [item.get("sample_id") for item in normalized_samples]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest requires unique, non-empty sample IDs")
    payload = {
        "format_version": 1,
        "selection": "sha256(seed:sample_id), duration coverage first, then stable hash order",
        "seed": seed,
        "split": split,
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        "expert_order": list(EXPERT_ORDER),
        "samples": normalized_samples,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_sample_manifest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise TypeError("sample manifest must be a mapping")
    if manifest.get("format_version") != 1 or manifest.get("split") != "train":
        raise ValueError("unsupported manifest version or non-training split")
    if tuple(manifest.get("expert_order", ())) != EXPERT_ORDER:
        raise ValueError("manifest expert order does not match the verified registry")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must contain a non-empty sample list")
    if any(not isinstance(item, Mapping) for item in samples):
        raise ValueError("manifest sample entries must be mappings")
    ids = [item.get("sample_id") for item in samples]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest sample IDs must be unique and non-empty")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    fingerprint = _canonical_sha256(body)
    if manifest.get("manifest_sha256") != fingerprint:
        raise ValueError("manifest fingerprint does not match its contents")
    return fingerprint


def write_output_artifact(
    path: str | os.PathLike[str],
    *,
    expert: str,
    scores: Any,
    features: Any,
    sample_records: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> Path:
    if expert not in EXPERT_ORDER:
        raise ValueError(f"unknown expert identity {expert!r}")
    records = list(sample_records)
    ids = [item.get("sample_id") for item in records]
    if not records or len(set(ids)) != len(ids) or any(not isinstance(item, str) for item in ids):
        raise ValueError("artifact records need unique non-empty sample IDs")
    score_array = _safe_numeric_array(scores, "scores")
    feature_array = _safe_numeric_array(features, "features")
    expected_dim = EXPERT_FEATURE_DIMS[expert]
    if score_array.shape != (len(records), 1):
        raise ValueError(f"{expert} scores must have shape [{len(records)},1]; got {score_array.shape}")
    if feature_array.shape != (len(records), expected_dim):
        raise ValueError(f"{expert} features must have shape [{len(records)},{expected_dim}]; got {feature_array.shape}")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "expert_name": np.asarray(expert, dtype="U16"),
        "expert_order_index": np.asarray(EXPERT_ORDER.index(expert), dtype=np.int64),
        "manifest_sha256": np.asarray(manifest_sha256, dtype="U64"),
        "sample_ids": np.asarray(ids, dtype="U256"),
        "caption_sha256": np.asarray([item["caption_sha256"] for item in records], dtype="U64"),
        "wav_sha256": np.asarray([item["wav_sha256"] for item in records], dtype="U64"),
        "scores": score_array,
        "features": feature_array,
    }
    temp_target = target.with_suffix(target.suffix + ".tmp")
    with temp_target.open("wb") as stream:
        np.savez(stream, **arrays)
    temp_target.replace(target)
    return target


def load_output_artifact(
    path: str | os.PathLike[str],
    *,
    expert: str,
    expected_dim: int,
    expected_manifest_sha256: str,
    sample_records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"output artifact does not exist: {target}")
    with np.load(target, allow_pickle=False) as bundle:
        if set(bundle.files) != _ARTIFACT_KEYS:
            raise ValueError(f"artifact fields differ from the safe format: {bundle.files}")
        data = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    if data["expert_name"].dtype.kind != "U" or data["expert_name"].item() != expert:
        raise ValueError("artifact expert identity mismatch")
    if int(data["expert_order_index"].item()) != EXPERT_ORDER.index(expert):
        raise ValueError("artifact expert order index mismatch")
    if data["manifest_sha256"].dtype.kind != "U" or data["manifest_sha256"].item() != expected_manifest_sha256:
        raise ValueError("artifact manifest fingerprint mismatch")
    expected_ids = [record["sample_id"] for record in sample_records]
    if data["sample_ids"].dtype.kind != "U" or data["sample_ids"].tolist() != expected_ids:
        raise ValueError("artifact sample identity/order mismatch")
    for field in ("caption_sha256", "wav_sha256"):
        expected = [record[field] for record in sample_records]
        if data[field].dtype.kind != "U" or data[field].tolist() != expected:
            raise ValueError(f"artifact {field} provenance mismatch")
    scores, features = data["scores"], data["features"]
    if scores.dtype != np.float32 or scores.shape != (len(expected_ids), 1):
        raise ValueError(f"artifact score shape/dtype invalid: {scores.shape}, {scores.dtype}")
    if features.dtype != np.float32 or features.shape != (len(expected_ids), expected_dim):
        raise ValueError(f"artifact feature shape/dtype invalid: {features.shape}, {features.dtype}")
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("artifact outputs contain non-finite values")
    return data


def _linux_memory() -> dict[str, Any]:
    current_rss = None
    peak_rss = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                peak_rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    if peak_rss is None:
        # Linux reports ru_maxrss in KiB; other platforms may use bytes.
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss = raw * 1024 if sys.platform.startswith("linux") else raw
    mem_available = None
    swap_total = swap_free = None
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts and parts[0].isdigit():
                fields[key] = int(parts[0]) * 1024
        mem_available = fields.get("MemAvailable")
        swap_total, swap_free = fields.get("SwapTotal"), fields.get("SwapFree")
    except (OSError, ValueError):
        pass
    return {
        "process_rss_bytes": current_rss,
        "process_peak_rss_bytes": peak_rss,
        "system_ram_available_bytes": mem_available,
        "system_swap_total_bytes": swap_total,
        "system_swap_free_bytes": swap_free,
    }


def memory_snapshot(stage: str, device: str | torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage, **_linux_memory()}
    target = torch.device(device)
    cuda = {"available": False}
    if target.type == "cuda" and torch.cuda.is_available():
        index = target.index if target.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        cuda = {
            "available": True,
            "device_name": torch.cuda.get_device_name(index),
            "total_bytes": int(total_bytes),
            "free_bytes": int(free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
    result["cuda"] = cuda
    return result


def _file_metadata(path: str | os.PathLike[str], *, kind: str) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    value: dict[str, Any] = {"path": str(target), "kind": kind, "exists": target.exists(), "readable": os.access(target, os.R_OK)}
    if target.exists():
        stat = target.stat()
        value.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return value


def _resolve_config(
    config_path: Path,
    unilm_source_dir: Path,
    hf_cache_dir: Path | None = None,
) -> dict[str, Any]:
    config = load_moe_config(config_path)
    resolved = resolve_moe_paths(
        config,
        config_path,
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        resource_root=PROJECT_ROOT,
    )
    resolved["runtime"]["offline"] = True
    resolved["experts"]["expert4"]["unilm_source_dir"] = unilm_source_dir.resolve()
    if hf_cache_dir is not None:
        resolved["runtime"]["cache_dir"] = hf_cache_dir.expanduser().resolve()
    return resolved


def _preflight_expert3_text_cache(cache_dir: Path) -> dict[str, Any]:
    """Check BERT using the exact cache_dir passed by the original factory."""
    resolved_cache_dir = str(cache_dir.expanduser().resolve())
    environment_names = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in environment_names}
    os.environ.update({
        "HF_HOME": resolved_cache_dir,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=resolved_cache_dir, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=resolved_cache_dir, local_files_only=True
        )
        return {
            "status": "PASS",
            "HF_HOME": resolved_cache_dir,
            "cache_dir": resolved_cache_dir,
            "model_id": "google-bert/bert-base-uncased",
            "model_type": config.model_type,
            "hidden_size": int(config.hidden_size),
            "tokenizer_class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "network_access": False,
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _unilm_identity(source_dir: Path) -> dict[str, Any]:
    source = source_dir.resolve()
    beats = source if (source / "BEATs.py").is_file() else source / "beats"
    files = [beats / name for name in ("BEATs.py", "backbone.py", "modules.py")]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Expert 4 UniLM source file missing: {path}")
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "source_dir": str(source),
        "git_commit": commit,
        "source_sha256": digest.hexdigest(),
        "expected_commit": EXPECTED_UNILM_COMMIT,
        "expected_sha256": EXPECTED_UNILM_FINGERPRINT,
        "identity_pass": commit == EXPECTED_UNILM_COMMIT and digest.hexdigest() == EXPECTED_UNILM_FINGERPRINT,
        "files": [str(path) for path in files],
    }


def _configured_resources(resolved: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    keys = {
        "expert1": (("pretrained_checkpoint", "file"), ("finetuned_checkpoint", "file")),
        "expert2": (("source_dir", "directory"), ("architecture_yaml", "file"), ("training_config", "file"), ("pretrained_checkpoint", "file"), ("finetuned_checkpoint", "file")),
        "expert3": (("source_dir", "directory"), ("base_checkpoint", "file"), ("finetuned_checkpoint", "file"), ("model_config", "file")),
        "expert4": (("beats_checkpoint", "file"), ("clap_checkpoint", "file"), ("unilm_source_dir", "directory"), ("finetuned_checkpoint", "file")),
    }
    resources = {}
    for expert, entries in keys.items():
        section = resolved["experts"][expert]
        resources[expert] = {key: _file_metadata(section[key], kind=kind) for key, kind in entries}
    return resources


def _check_resources(resources: Mapping[str, Mapping[str, Any]]) -> None:
    missing = []
    for expert, entries in resources.items():
        for field, info in entries.items():
            target = Path(info["path"])
            expected_type_ok = target.is_dir() if info["kind"] == "directory" else target.is_file()
            if not info["exists"] or not info["readable"] or not expected_type_ok:
                missing.append(f"{expert}.{field}: {info['path']}")
    if missing:
        raise FileNotFoundError("Required local expert resources unavailable: " + "; ".join(missing))


def _prepared_batch(dataset: UnifiedXACLEDataset, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index_by_id = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    samples = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in index_by_id:
            raise ValueError(f"manifest sample is absent from Unified Dataset: {sample_id}")
        index = index_by_id[sample_id]
        row = dataset.rows[index]
        if row["wave_file_name"] != record["wave_file_name"] or row["captions_raw"] != record["raw_caption"]:
            raise ValueError(f"manifest metadata changed for {sample_id}")
        if _sha256_file(row["audio_path"]) != record["wav_sha256"]:
            raise ValueError(f"manifest WAV hash changed for {sample_id}")
        # Keep the train identity while requesting the Dataset's deterministic
        # non-training crop policy for reproducible inference inputs.
        original_split = dataset.split
        dataset.split = "validation"
        try:
            sample = dataset[index]
        finally:
            dataset.split = original_split
        sample["split"] = "train"
        if sample["sample_id"] != sample_id or sample["captions_raw"] != record["raw_caption"]:
            raise ValueError(f"Unified Dataset output identity mismatch for {sample_id}")
        samples.append(sample)
    return collate_unified(samples)


def _expert_input_batch(batch: Mapping[str, Any], expert: str, device: torch.device, *, stop: int | None = None) -> dict[str, Any]:
    end = batch["expert1_wavs"].shape[0] if stop is None else stop
    result: dict[str, Any] = {
        EXPERT_AUDIO_KEYS[expert]: batch[EXPERT_AUDIO_KEYS[expert]][:end].to(device),
        "captions_raw": list(batch["captions_raw"][:end]),
    }
    if expert == "expert4":
        result["expert4_wav_lens"] = batch["expert4_wav_lens"][:end].to(device)
    return result


EXPERT_AUDIO_KEYS = {
    "expert1": "expert1_wavs",
    "expert2": "expert2_wavs",
    "expert3": "expert3_wavs",
    "expert4": "expert4_wavs",
}


def _tensor_difference(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(reference.shape):
        return {"shape_equal": False, "actual_shape": list(actual.shape), "reference_shape": list(reference.shape), "max_abs_diff": None, "mean_abs_diff": None, "exact_equal": False}
    diff = (actual.float() - reference.float()).abs()
    return {
        "shape_equal": True,
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "exact_equal": bool(torch.equal(actual, reference)),
        "allclose_atol_1e-6_rtol_1e-5": bool(torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def _module_origins(expert: str, source_dir: Path | None = None) -> dict[str, Any]:
    prefixes = {
        "expert1": ("laion_clap",),
        "expert2": ("models", "tools"),
        "expert3": ("expert3_finetune.XACLE_2027_baselinemodel",),
        "expert4": ("expert4_finetune", "_moescore_expert4_beats_"),
    }[expert]
    found: dict[str, Any] = {}
    for module_name, module in tuple(sys.modules.items()):
        if not any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in prefixes):
            if not (expert == "expert4" and module_name.startswith("_moescore_expert4_beats_")):
                continue
        filename = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) if spec else None
        package_path = list(map(str, getattr(module, "__path__", ())))
        found[module_name] = {
            "file": str(Path(filename).resolve()) if filename else None,
            "spec_origin": str(Path(origin).resolve()) if origin and origin not in {"built-in", "frozen"} else origin,
            "package_path": package_path,
            "module_object_id": id(module),
        }
    return found


def _validate_real_module_origins(expert: str, config: Mapping[str, Any]) -> dict[str, Any]:
    origins = _module_origins(expert)
    if expert == "expert2":
        module = importlib.import_module("model.expert_adapters.expert2_adapter")
        module.validate_mga_import_namespace(config["experts"][expert]["source_dir"])
    elif expert == "expert3":
        module = importlib.import_module("model.expert_adapters.expert3_adapter")
        root = Path(config["experts"][expert]["source_dir"]).resolve()
        for module_name, value in tuple(sys.modules.items()):
            if module_name == "expert3_finetune.XACLE_2027_baselinemodel" or module_name.startswith("expert3_finetune.XACLE_2027_baselinemodel."):
                module.validate_expert3_module_origin(value, root, module_name)
    elif expert == "expert4":
        module = importlib.import_module("model.expert_adapters.expert4_adapter")
        unilm_root = Path(config["experts"][expert]["unilm_source_dir"]).resolve()
        module.validate_expert4_module_origins(unilm_root, require_beats=False)
        beats_root = unilm_root if (unilm_root / "BEATs.py").is_file() else unilm_root / "beats"
        private = {name: value for name, value in origins.items() if name.startswith("_moescore_expert4_beats_")}
        for name, value in private.items():
            files = [value.get("file"), value.get("spec_origin")]
            for filename in filter(None, files):
                if not Path(filename).resolve().is_relative_to(beats_root.resolve()):
                    raise ImportError(f"Expert 4 private BEATs module {name} resolves outside {beats_root}: {filename}")
    return origins


def _checkpoint_snapshot(config: Mapping[str, Any], expert: str) -> dict[str, Any]:
    section = config["experts"][expert]
    fields = {
        "expert1": ("pretrained_checkpoint", "finetuned_checkpoint"),
        "expert2": ("pretrained_checkpoint", "finetuned_checkpoint"),
        "expert3": ("base_checkpoint", "finetuned_checkpoint"),
        "expert4": ("beats_checkpoint", "clap_checkpoint", "finetuned_checkpoint"),
    }[expert]
    return {key: _file_metadata(section[key], kind="checkpoint") for key in fields}


def _classify_exception(exc: BaseException, stage: str) -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message:
        return "CUDA_OUT_OF_MEMORY"
    if isinstance(exc, FileNotFoundError):
        return "MISSING_LOCAL_RESOURCE"
    if any(term in message for term in ("couldn't find them in the cached files", "localentrynotfounderror", "local_files_only")):
        return "MISSING_TEXT_MODEL_CACHE"
    if isinstance(exc, ImportError):
        return "IMPORT_NAMESPACE_FAILURE" if any(key in message for key in ("namespace", "origin", "module")) else "IMPORT_FAILURE"
    if stage in {"FACTORY", "CHECKPOINT_RESTORATION"} and any(key in message for key in ("checkpoint", "state_dict", "fingerprint", "sha-256", "sha256")):
        return "CHECKPOINT_RESTORE_FAILURE"
    if stage == "INPUT_PREPARATION" or stage.startswith("INFERENCE"):
        return "INPUT_OR_SOURCE_FORWARD_FAILURE"
    if stage == "ARTIFACT_SERIALIZATION":
        return "OUTPUT_ARTIFACT_FAILURE"
    if stage == "OUTPUT_VALIDATION":
        return "OUTPUT_CONTRACT_FAILURE"
    return "RUNTIME_ERROR"


def _worker_report_path(path: str) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _worker_expert(args: argparse.Namespace) -> int:
    expert = args._worker_expert
    report_path = _worker_report_path(args._worker_report)
    report: dict[str, Any] = {
        "expert": expert,
        "status": "RUNNING",
        "last_completed_stage": "PROCESS_START",
        "failed_stage": None,
        "failure_category": None,
        "failure": None,
        "stages": [],
        "model_construction": False,
        "checkpoint_deserialization": False,
        "real_inference": False,
        "samples_processed": 0,
        "manifest_sha256": None,
    }
    _atomic_json(report_path, report)
    stage = "PREFLIGHT"
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        manifest = json.loads(Path(args._manifest).read_text(encoding="utf-8"))
        manifest_sha256 = validate_sample_manifest(manifest)
        records = manifest["samples"]
        report["manifest_sha256"] = manifest_sha256
        hf_cache_dir = Path(args.hf_home).expanduser().resolve() if args.hf_home else None
        config = _resolve_config(Path(args.config), Path(args.unilm_source_dir), hf_cache_dir)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is unavailable")
            # is_available() checks the driver but does not initialize the
            # selected CUDA context. Initialize it before the C++ peak-reset
            # call; otherwise this PyTorch build reports "Invalid device
            # argument" on a fresh worker.
            with torch.cuda.device(device):
                torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats()
        report["stages"].append(memory_snapshot("PROCESS_START", device))
        dataset = UnifiedXACLEDataset.from_dataset_root(args.dataset_root, split="train", seed=args.seed, require_labels=True)
        stage = "INPUT_PREPARATION"
        batch = _prepared_batch(dataset, records)
        report["input_contract"] = {
            "sample_ids": batch["sample_ids"],
            "captions_raw_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in batch["captions_raw"]],
            "routed_audio_key": EXPERT_AUDIO_KEYS[expert],
            "routed_audio_shape": list(batch[EXPERT_AUDIO_KEYS[expert]].shape),
            "routed_audio_dtype": str(batch[EXPERT_AUDIO_KEYS[expert]].dtype),
            "expert4_wav_lens": batch["expert4_wav_lens"].tolist() if expert == "expert4" else None,
            "labels_and_model_ids_passed": False,
        }
        snapshots_before = _checkpoint_snapshot(config, expert)
        if expert == "expert3":
            report["expert3_text_cache_preflight"] = _preflight_expert3_text_cache(Path(config["runtime"]["cache_dir"]))
        stage = "FACTORY"
        report["stages"].append(memory_snapshot("BEFORE_FACTORY", device))
        alias_names = ("BEATs", "backbone", "modules")
        alias_before = {name: sys.modules.get(name) for name in alias_names}
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        adapter = create_expert_adapter(expert, config, device=device)
        report["model_construction"] = True
        report["checkpoint_deserialization"] = True
        report["stages"].append(memory_snapshot("AFTER_FACTORY_AND_CHECKPOINT_RESTORATION", device))
        stage = "CHECKPOINT_RESTORATION"
        origins = _validate_real_module_origins(expert, config)
        alias_after = {name: sys.modules.get(name) for name in alias_names}
        alias_cleanup = all(alias_after[name] is alias_before[name] for name in alias_names)
        if expert == "expert4" and not alias_cleanup:
            raise ImportError("Expert 4 temporary BEATs top-level aliases were not restored")
        parameters = list(adapter.parameters())
        trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        model_info = {
            "adapter_class": f"{type(adapter).__module__}.{type(adapter).__name__}",
            "original_model_class": f"{type(adapter.model).__module__}.{type(adapter.model).__name__}" if hasattr(adapter, "model") else None,
            "original_model_module_origin": getattr(sys.modules.get(type(adapter.model).__module__), "__file__", None) if hasattr(adapter, "model") else None,
            "feature_dim": int(adapter.feature_dim),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": trainable_count,
            "all_parameters_frozen": trainable_count == 0,
            "all_modules_eval": all(not module.training for module in adapter.modules()),
            "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY",
            "checkpoint_metadata_before": snapshots_before,
            "checkpoint_metadata_after": _checkpoint_snapshot(config, expert),
            "module_origins": origins,
            "beats_aliases_restored": alias_cleanup if expert == "expert4" else None,
            "beats_alias_object_identity_preserved": alias_cleanup if expert == "expert4" else None,
        }
        if model_info["feature_dim"] != EXPERT_FEATURE_DIMS[expert] or not model_info["all_parameters_frozen"] or not model_info["all_modules_eval"]:
            raise RuntimeError(f"factory state contract failed: {model_info}")
        report["model"] = model_info
        report["last_completed_stage"] = "FACTORY_AND_CHECKPOINT_RESTORATION"
        stage = "INFERENCE_B1"
        one_batch = _expert_input_batch(batch, expert, device, stop=1)
        with torch.inference_mode():
            score_one, feature_one = adapter(one_batch)
        report["stages"].append(memory_snapshot("AFTER_FIRST_REAL_INFERENCE_B1", device))
        report["last_completed_stage"] = "FIRST_REAL_INFERENCE_B1"
        stage = "INFERENCE_REPEAT"
        with torch.inference_mode():
            score_repeat, feature_repeat = adapter(one_batch)
        stability = {
            "score": _tensor_difference(score_one, score_repeat),
            "features": _tensor_difference(feature_one, feature_repeat),
        }
        if not stability["score"].get("allclose_atol_1e-6_rtol_1e-5") or not stability["features"].get("allclose_atol_1e-6_rtol_1e-5"):
            raise RuntimeError(f"repeated eval inference is not stable: {stability}")
        report["last_completed_stage"] = "REPEATED_REAL_INFERENCE"
        stage = "INFERENCE_SUBSET"
        full_batch = _expert_input_batch(batch, expert, device)
        with torch.inference_mode():
            scores, features = adapter(full_batch)
        report["stages"].append(memory_snapshot("AFTER_SELECTED_REAL_SUBSET_INFERENCE", device))
        report["last_completed_stage"] = "SELECTED_REAL_SUBSET_INFERENCE"
        stage = "OUTPUT_VALIDATION"
        first_vs_batch = {
            "score": _tensor_difference(score_one, scores[:1]),
            "features": _tensor_difference(feature_one, features[:1]),
        }
        if not first_vs_batch["score"].get("allclose_atol_1e-6_rtol_1e-5") or not first_vs_batch["features"].get("allclose_atol_1e-6_rtol_1e-5"):
            raise RuntimeError(f"B=1 output differs from same sample in subset batch: {first_vs_batch}")
        if scores.dtype != torch.float32 or features.dtype != torch.float32:
            raise TypeError(f"real adapter outputs must be FP32; got {scores.dtype}, {features.dtype}")
        if tuple(scores.shape) != (len(records), 1) or tuple(features.shape) != (len(records), EXPERT_FEATURE_DIMS[expert]):
            stage = "OUTPUT_VALIDATION"
            raise ValueError(f"real output shape mismatch: scores={tuple(scores.shape)} features={tuple(features.shape)}")
        if not torch.isfinite(scores).all().item() or not torch.isfinite(features).all().item():
            stage = "OUTPUT_VALIDATION"
            raise ValueError("real adapter output contains non-finite values")
        cpu_scores = scores.detach().to(device="cpu").contiguous()
        cpu_features = features.detach().to(device="cpu").contiguous()
        report["stages"].append(memory_snapshot("AFTER_CPU_OUTPUT_TRANSFER", device))
        report["last_completed_stage"] = "CPU_OUTPUT_TRANSFER"
        stage = "ARTIFACT_SERIALIZATION"
        artifact = Path(args.artifact_dir) / f"{expert}_outputs.npz"
        write_output_artifact(
            artifact,
            expert=expert,
            scores=cpu_scores,
            features=cpu_features,
            sample_records=records,
            manifest_sha256=manifest_sha256,
        )
        report["artifact_path"] = str(artifact.resolve())
        report["artifact_sha256"] = _sha256_file(artifact)
        report["output"] = {
            "scores_shape": list(cpu_scores.shape),
            "features_shape": list(cpu_features.shape),
            "scores_dtype": str(cpu_scores.dtype),
            "features_dtype": str(cpu_features.dtype),
            "scores": cpu_scores.reshape(-1).tolist(),
            "finite": True,
            "sample_ids": [record["sample_id"] for record in records],
            "repeat_inference_stability": stability,
            "b1_vs_subset_first_sample": first_vs_batch,
        }
        report["real_inference"] = True
        report["samples_processed"] = len(records)
        report["last_completed_stage"] = "ARTIFACT_WRITTEN"
        report["stages"].append(memory_snapshot("BEFORE_PROCESS_EXIT", device))
        report["status"] = "PASS"
        _atomic_json(report_path, report)
        return 0
    except BaseException as exc:
        report["failed_stage"] = stage
        report["failure_category"] = _classify_exception(exc, stage)
        report["failure"] = {"exception_class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report["status"] = "FAIL"
        try:
            report["stages"].append(memory_snapshot(f"FAILURE_AT_{stage}", args.device))
        except BaseException:
            pass
        _atomic_json(report_path, report)
        return 1


class _UnusedExpert(ExpertAdapter):
    """Adapter placeholder for the output-only integration facade; never run."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__(feature_dim)

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("output-only integration must not invoke placeholder adapters")


def _score_statistics(scores: torch.Tensor) -> dict[str, Any]:
    values = scores.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "sample_count": int(values.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(values.size - finite.sum()),
        "finite_percentage": float(100.0 * finite.sum() / values.size) if values.size else None,
        "minimum": float(finite_values.min()) if finite_values.size else None,
        "maximum": float(finite_values.max()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
        "std_population": float(finite_values.std(ddof=0)) if finite_values.size else None,
        "count_below_minus_one": int((finite_values < -1.0).sum()),
        "count_above_plus_one": int((finite_values > 1.0).sum()),
    }


def _assemble_and_gate(args: argparse.Namespace, report: dict[str, Any]) -> None:
    manifest = json.loads(Path(args._manifest if args._manifest else Path(args.artifact_dir) / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha256 = validate_sample_manifest(manifest)
    records = manifest["samples"]
    output_path = Path(args.artifact_dir)
    expert_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    original_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output_reports = {}
    for expert in EXPERT_ORDER:
        artifact_path = output_path / f"{expert}_outputs.npz"
        data = load_output_artifact(
            artifact_path,
            expert=expert,
            expected_dim=EXPERT_FEATURE_DIMS[expert],
            expected_manifest_sha256=manifest_sha256,
            sample_records=records,
        )
        output_reports[expert] = json.loads((output_path / f"{expert}_report.json").read_text(encoding="utf-8"))
        if output_reports[expert].get("status") != "PASS" or not output_reports[expert].get("real_inference"):
            raise ValueError(f"{expert} artifact lacks a successful real factory inference report")
        expert_outputs[expert] = (
            torch.from_numpy(data["scores"].copy()),
            torch.from_numpy(data["features"].copy()),
        )
        original_arrays[expert] = (data["scores"].copy(), data["features"].copy())

    # Verify process identity order and sample provenance before concatenation.
    ids_by_expert = {name: output_reports[name]["output"]["sample_ids"] for name in EXPERT_ORDER}
    if any(ids_by_expert[name] != [item["sample_id"] for item in records] for name in EXPERT_ORDER):
        raise ValueError(f"cross-process sample identity order mismatch: {ids_by_expert}")
    scores, features = validate_and_assemble_expert_outputs(expert_outputs, batch_size=len(records))
    score_preservation = {}
    feature_preservation = {}
    boundaries = {}
    offset = 0
    for index, name in enumerate(EXPERT_ORDER):
        score_preservation[name] = bool(torch.equal(scores[:, index:index + 1], expert_outputs[name][0]))
        width = EXPERT_FEATURE_DIMS[name]
        segment = features[:, offset:offset + width]
        feature_preservation[name] = bool(torch.equal(segment, expert_outputs[name][1]))
        boundaries[name] = [offset, offset + width]
        offset += width
    if not all(score_preservation.values()) or not all(feature_preservation.values()):
        raise ValueError("registry assembly changed real scores or feature segments")

    # The existing integration facade requires adapters for its ordinary
    # forward path. These lightweight never-called placeholders let this
    # verification use its exact output-only gate boundary without retaining
    # any checkpoint-backed model in the assembly process.
    placeholders = {name: _UnusedExpert(EXPERT_FEATURE_DIMS[name]) for name in EXPERT_ORDER}
    integration = MoEForwardIntegration(placeholders).cpu().eval()
    tensors_before = {name: (pair[0].clone(), pair[1].clone()) for name, pair in expert_outputs.items()}
    prediction, weights = integration.forward_from_expert_outputs(expert_outputs)
    reference = (
        scores[:, 0:1] * weights[:, 0:1]
        + scores[:, 1:2] * weights[:, 1:2]
        + scores[:, 2:3] * weights[:, 2:3]
        + scores[:, 3:4] * weights[:, 3:4]
    )
    if not torch.isfinite(weights).all() or not torch.isfinite(prediction).all():
        raise ValueError("gating produced non-finite real-output values")
    if (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones(len(records)), atol=2e-7, rtol=0):
        raise ValueError("gating weights are not normalized per sample")
    torch.testing.assert_close(prediction, reference, atol=1e-6, rtol=1e-5)
    low, high = scores.min(1, keepdim=True).values, scores.max(1, keepdim=True).values
    if (prediction < low - 1e-6).any() or (prediction > high + 1e-6).any():
        raise ValueError("weighted prediction violated the convex-combination bounds")
    if any(not torch.equal(pair[0], tensors_before[name][0]) or not torch.equal(pair[1], tensors_before[name][1]) for name, pair in expert_outputs.items()):
        raise ValueError("gating integration mutated expert outputs")

    integration.train()
    integration.zero_grad(set_to_none=True)
    grad_prediction, grad_weights = integration.forward_from_expert_outputs(expert_outputs)
    objective = grad_prediction.square().mean() + grad_weights[:, 0].mean()
    objective.backward()
    gate_parameters = list(integration.gating_network.parameters())
    gate_gradients = [parameter.grad for parameter in gate_parameters if parameter.grad is not None]
    gradient_report = {
        "objective_value": float(objective.detach().item()),
        "trainable_gate_parameter_count": sum(parameter.numel() for parameter in gate_parameters if parameter.requires_grad),
        "gradient_tensor_count": len(gate_gradients),
        "all_gradients_finite": bool(gate_gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gate_gradients),
        "any_nonzero_gradient": any(bool(torch.count_nonzero(gradient)) for gradient in gate_gradients),
        "real_expert_outputs_require_grad": any(score.requires_grad or feature.requires_grad for score, feature in expert_outputs.values()),
        "expert_parameters_updated": False,
        "training_performed": False,
    }
    if not gradient_report["all_gradients_finite"] or not gradient_report["any_nonzero_gradient"] or gradient_report["real_expert_outputs_require_grad"]:
        raise RuntimeError(f"real-output gating gradient contract failed: {gradient_report}")

    report["sample_identity_alignment"] = {
        "manifest_sha256": manifest_sha256,
        "sample_ids_match_all_experts": True,
        "sample_count": len(records),
        "order": ids_by_expert,
        "caption_sha256_match": True,
        "wav_sha256_match": True,
    }
    report["sequential_output_extraction"] = {name: output_reports[name] for name in EXPERT_ORDER}
    report["assembled_score_shape"] = list(scores.shape)
    report["assembled_feature_shape"] = list(features.shape)
    report["feature_boundaries"] = boundaries
    report["score_preservation"] = score_preservation
    report["feature_preservation"] = feature_preservation
    report["real_score_distribution"] = {
        name: _score_statistics(expert_outputs[name][0]) for name in EXPERT_ORDER
    }
    report["gating"] = {
        "implementation": "model.moe_gating_core.GatingNetwork",
        "forward_boundary": "MoEForwardIntegration.forward_from_expert_outputs",
        "weights_shape": list(weights.shape),
        "prediction_shape": list(prediction.shape),
        "weights": weights.detach().tolist(),
        "prediction": prediction.detach().tolist(),
        "weight_sum_max_abs_error": float((weights.sum(1) - 1.0).abs().max().item()),
        "weights_nonnegative": bool((weights >= 0).all()),
        "weighted_sum_parity": {
            "allclose": bool(torch.allclose(prediction, reference, atol=1e-6, rtol=1e-5)),
            "max_abs_diff": float((prediction - reference).abs().max().item()),
            "convex_combination_bounds": True,
        },
        "sample_weight_max_pairwise_difference": float(torch.cdist(weights, weights, p=1).max().item()) if len(records) > 1 else 0.0,
        "backward_on_real_detached_outputs": gradient_report,
        "labels_or_model_metadata_used": False,
        "score_calibration_applied": False,
        "placeholder_adapters_invoked": False,
    }
    report["real_output_assembly"] = "PASS"
    report["real_output_gating_forward"] = "PASS"
    report["real_output_gating_backward"] = "PASS"
    report["real_four_expert_output_extraction"] = "PASS"
    report["artifacts"] = {
        name: {"path": output_reports[name]["artifact_path"], "sha256": output_reports[name]["artifact_sha256"]}
        for name in EXPERT_ORDER
    }


def _current_headroom(device: str) -> dict[str, Any]:
    mem = _linux_memory()
    target = torch.device(device)
    free_gpu = None
    total_gpu = None
    if target.type == "cuda" and torch.cuda.is_available():
        free_gpu, total_gpu = torch.cuda.mem_get_info(target.index or 0)
    mem["cuda_free_bytes"] = int(free_gpu) if free_gpu is not None else None
    mem["cuda_total_bytes"] = int(total_gpu) if total_gpu is not None else None
    return mem


def _worker_pair(args: argparse.Namespace) -> int:
    report_path = _worker_report_path(args._pair_report)
    report: dict[str, Any] = {"status": "RUNNING", "order": ["expert2", "expert4"], "stages": [], "samples_processed": 0}
    _atomic_json(report_path, report)
    stage = "PREFLIGHT"
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        manifest = json.loads(Path(args._manifest).read_text(encoding="utf-8"))
        fingerprint = validate_sample_manifest(manifest)
        hf_cache_dir = Path(args.hf_home).expanduser().resolve() if args.hf_home else None
        config = _resolve_config(Path(args.config), Path(args.unilm_source_dir), hf_cache_dir)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("live E2/E4 pair verification requires the available CUDA device")
        with torch.cuda.device(device):
            torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats()
        report["manifest_sha256"] = fingerprint
        report["stages"].append(memory_snapshot("PROCESS_START", device))
        dataset = UnifiedXACLEDataset.from_dataset_root(args.dataset_root, split="train", seed=args.seed, require_labels=True)
        batch = _prepared_batch(dataset, manifest["samples"][:1])
        aliases_before = {name: sys.modules.get(name) for name in ("BEATs", "backbone", "modules")}
        stage = "EXPERT2_FACTORY"
        report["stages"].append(memory_snapshot("BEFORE_EXPERT2_FACTORY", device))
        expert2 = create_expert_adapter("expert2", config, device=device)
        report["stages"].append(memory_snapshot("AFTER_EXPERT2_FACTORY", device))
        stage = "EXPERT4_FACTORY"
        expert4 = create_expert_adapter("expert4", config, device=device)
        report["stages"].append(memory_snapshot("AFTER_EXPERT4_FACTORY", device))
        alias_cleanup = all(sys.modules.get(name) is aliases_before[name] for name in aliases_before)
        if not alias_cleanup:
            raise ImportError("Expert 4 did not restore the pre-existing BEATs import bindings")
        _validate_real_module_origins("expert2", config)
        _validate_real_module_origins("expert4", config)
        stage = "PAIR_INFERENCE"
        outputs = {}
        with torch.inference_mode():
            outputs["expert2"] = expert2(_expert_input_batch(batch, "expert2", device))
            outputs["expert4"] = expert4(_expert_input_batch(batch, "expert4", device))
        report["stages"].append(memory_snapshot("AFTER_PAIR_INFERENCE", device))
        if tuple(outputs["expert2"][0].shape) != (1, 1) or tuple(outputs["expert2"][1].shape) != (1, 2048):
            raise ValueError("live E2 output contract mismatch")
        if tuple(outputs["expert4"][0].shape) != (1, 1) or tuple(outputs["expert4"][1].shape) != (1, 1024):
            raise ValueError("live E4 output contract mismatch")
        artifact_results = {}
        for name, adapter in (("expert2", expert2), ("expert4", expert4)):
            stored = load_output_artifact(
                Path(args.artifact_dir) / f"{name}_outputs.npz",
                expert=name,
                expected_dim=EXPERT_FEATURE_DIMS[name],
                expected_manifest_sha256=fingerprint,
                sample_records=manifest["samples"],
            )
            score_diff = _tensor_difference(outputs[name][0].detach().cpu(), torch.from_numpy(stored["scores"][:1].copy()))
            feature_diff = _tensor_difference(outputs[name][1].detach().cpu(), torch.from_numpy(stored["features"][:1].copy()))
            if not score_diff["allclose_atol_1e-6_rtol_1e-5"] or not feature_diff["allclose_atol_1e-6_rtol_1e-5"]:
                raise RuntimeError(f"live pair {name} differs from its independent output artifact")
            artifact_results[name] = {"score_shape": list(outputs[name][0].shape), "feature_shape": list(outputs[name][1].shape), "score_parity": score_diff, "feature_parity": feature_diff, "frozen": all(not p.requires_grad for p in adapter.parameters()), "eval": all(not m.training for m in adapter.modules())}
        report.update({
            "status": "PASS",
            "last_completed_stage": "PAIR_INFERENCE_AND_PARITY",
            "sample_id": manifest["samples"][0]["sample_id"],
            "samples_processed": 1,
            "outputs": artifact_results,
            "beats_aliases_restored": alias_cleanup,
            "module_origins": {"expert2": _module_origins("expert2"), "expert4": _module_origins("expert4")},
            "both_models_alive_during_inference": expert2 is not None and expert4 is not None,
        })
        report["stages"].append(memory_snapshot("BEFORE_PROCESS_EXIT", device))
        _atomic_json(report_path, report)
        return 0
    except BaseException as exc:
        report.update({"status": "FAIL", "failed_stage": stage, "failure_category": _classify_exception(exc, stage), "failure": {"exception_class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        try:
            report["stages"].append(memory_snapshot(f"FAILURE_AT_{stage}", args.device))
        except BaseException:
            pass
        _atomic_json(report_path, report)
        return 1


def _environment() -> dict[str, Any]:
    import importlib.metadata

    versions = {}
    for package in ("torch", "torchaudio", "numpy", "scipy", "soundfile", "laion-clap", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda = {"available": torch.cuda.is_available(), "device_name": None, "total_bytes": None}
    if cuda["available"]:
        cuda.update({"device_name": torch.cuda.get_device_name(0), "total_bytes": int(torch.cuda.get_device_properties(0).total_memory)})
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "packages": versions,
        "cuda": cuda,
        "memory": _linux_memory(),
        "offline_environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }


def _make_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], UnifiedXACLEDataset]:
    dataset = UnifiedXACLEDataset.from_dataset_root(args.dataset_root, split="train", seed=args.seed, require_labels=True)
    selected = select_sample_records(dataset.rows, seed=args.seed, max_samples=args.max_samples)
    manifest = make_sample_manifest(selected, seed=args.seed, split="train", dataset_root=args.dataset_root)
    return manifest, dataset


def _resource_pair_gate(device: str) -> dict[str, Any]:
    observed = _current_headroom(device)
    target_is_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    required_ram = 6 * 1024**3
    required_vram = 2 * 1024**3
    safe = (
        target_is_cuda
        and observed.get("system_ram_available_bytes") is not None
        and observed["system_ram_available_bytes"] >= required_ram
        and observed.get("cuda_free_bytes") is not None
        and observed["cuda_free_bytes"] >= required_vram
    )
    return {
        "status": "SAFE_TO_ATTEMPT" if safe else "NOT_ATTEMPTED_RESOURCE_RISK",
        "observed": observed,
        "thresholds": {"system_ram_available_bytes": required_ram, "cuda_free_bytes": required_vram if torch.device(device).type == "cuda" else None},
        "basis": "Ngưỡng bảo thủ so với RSS đỉnh E2/E4 khoảng 4.3 GiB và VRAM đỉnh phân bổ khoảng 1.6 GiB đã đo ở LOCAL-08.1.",
    }


def _run_worker(args: argparse.Namespace, expert: str, report: dict[str, Any]) -> dict[str, Any]:
    worker_report = Path(args.artifact_dir) / f"{expert}_report.json"
    log_path = Path(args.artifact_dir) / f"{expert}_worker.log"
    _atomic_json(worker_report, {"expert": expert, "status": "RUNNING", "last_completed_stage": "PROCESS_START"})
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--unilm-source-dir", str(Path(args.unilm_source_dir).resolve()),
        "--seed", str(args.seed), "--max-samples", str(args.max_samples),
        "--device", args.device, "--artifact-dir", str(Path(args.artifact_dir).resolve()),
        "--_manifest", str(Path(args.artifact_dir) / "manifest.json"),
        "--_worker-expert", expert,
        "--_worker-report", str(worker_report),
    ]
    if args.hf_home:
        command.extend(["--hf-cache-dir", str(Path(args.hf_home).expanduser().resolve())])
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    child = json.loads(worker_report.read_text(encoding="utf-8")) if worker_report.exists() else {
        "expert": expert, "status": "FAIL", "failure_category": "PROCESS_TERMINATION", "failure": {"message": "worker report was not written"}
    }
    child["process_return_code"] = result.returncode
    child["wall_seconds"] = time.monotonic() - started
    if result.returncode != 0:
        try:
            child["log_tail"] = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        except OSError:
            pass
    _atomic_json(worker_report, child)
    report["experts"][expert] = child
    _atomic_json(Path(args.report_json), report)
    return child


def _run_pair(args: argparse.Namespace, report: dict[str, Any]) -> None:
    gate = _resource_pair_gate(args.device)
    report["pairwise_e2_e4_resource_gate"] = gate
    if gate["status"] != "SAFE_TO_ATTEMPT":
        report["pairwise_e2_e4_runtime"] = {"status": "NOT_ATTEMPTED_RESOURCE_RISK", "resource_gate": gate}
        _atomic_json(Path(args.report_json), report)
        return
    pair_report = Path(args.artifact_dir) / "pairwise_e2_e4_report.json"
    log_path = Path(args.artifact_dir) / "pairwise_e2_e4_worker.log"
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--unilm-source-dir", str(Path(args.unilm_source_dir).resolve()),
        "--seed", str(args.seed), "--max-samples", str(args.max_samples),
        "--device", args.device, "--artifact-dir", str(Path(args.artifact_dir).resolve()),
        "--_manifest", str(Path(args.artifact_dir) / "manifest.json"),
        "--_pair-report", str(pair_report),
    ]
    if args.hf_home:
        command.extend(["--hf-cache-dir", str(Path(args.hf_home).expanduser().resolve())])
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    pair = json.loads(pair_report.read_text(encoding="utf-8")) if pair_report.exists() else {"status": "FAIL", "failure_category": "PROCESS_TERMINATION"}
    pair["process_return_code"] = result.returncode
    if result.returncode != 0:
        pair["log_tail"] = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
    report["pairwise_e2_e4_runtime"] = pair
    _atomic_json(Path(args.report_json), report)


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "RUNNING",
        "timestamp_utc": _utc_now(),
        "mode": args.mode,
        "environment": _environment(),
        "repository": {"root": str(PROJECT_ROOT), "branch": None, "working_tree_preserved": True},
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "expert_order": list(EXPERT_ORDER),
        "feature_dimensions": dict(EXPERT_FEATURE_DIMS),
        "experts": {},
        "real_four_expert_output_extraction": "NOT_VERIFIED",
        "real_output_assembly": "NOT_VERIFIED",
        "real_output_gating_forward": "NOT_VERIFIED",
        "real_output_gating_backward": "NOT_VERIFIED",
        "pairwise_e2_e4_runtime": {"status": "NOT_VERIFIED"},
        "simultaneous_four_expert_runtime": "NOT_ATTEMPTED_RESOURCE_RISK",
        "gating_training_performed": False,
        "pretrained_checkpoints_modified": False,
        "score_calibration_applied": False,
        "limitations": [
            "Sequential extraction is not simultaneous four-expert inference.",
            "Gate gradients are checked only; no gating optimization or training is performed.",
            "Score statistics describe only the selected bounded training subset.",
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/moe_local.example.yaml"))
    parser.add_argument("--dataset-root", default="/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset")
    parser.add_argument("--unilm-source-dir", default="/home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original")
    parser.add_argument(
        "--hf-cache-dir", "--hf-home", dest="hf_home", default=None,
        help="thư mục cache cấp Hub hiện có cho Expert 3 (ví dụ ~/.cache/huggingface/hub); không tải model",
    )
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "sequential", "assemble-and-gate"), default="sequential")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--artifact-dir", default="/tmp/moescore_local101")
    parser.add_argument("--report-json", default="/tmp/moescore_local101_runtime.json")
    parser.add_argument("--_manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-expert", choices=EXPERT_ORDER, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-report", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_pair-report", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _existing_verified_output(
    args: argparse.Namespace,
    expert: str,
    manifest_sha256: str,
    sample_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reuse a verified temp artifact only when its provenance still matches."""
    report_path = Path(args.artifact_dir) / f"{expert}_report.json"
    artifact_path = Path(args.artifact_dir) / f"{expert}_outputs.npz"
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") != "PASS"
            or previous.get("real_inference") is not True
            or previous.get("manifest_sha256") != manifest_sha256
            or previous.get("model", {}).get("factory_restoration") != "PASS_FROM_EXISTING_SOURCE_FACTORY"
            or previous.get("model", {}).get("checkpoint_metadata_after") != _checkpoint_snapshot(config, expert)
            or previous.get("artifact_sha256") != _sha256_file(artifact_path)
        ):
            return None
        load_output_artifact(
            artifact_path,
            expert=expert,
            expected_dim=EXPERT_FEATURE_DIMS[expert],
            expected_manifest_sha256=manifest_sha256,
            sample_records=sample_records,
        )
        if previous.get("output", {}).get("sample_ids") != [record["sample_id"] for record in sample_records]:
            return None
        previous["process_return_code"] = 0
        previous["reused_existing_artifact"] = True
        return previous
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args._worker_expert:
        return _worker_expert(args)
    if args._pair_report:
        return _worker_pair(args)
    output_dir = Path(args.artifact_dir).expanduser().resolve()
    if output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("verification artifacts must be outside the Git working tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json).expanduser().resolve()
    report = _base_report(args)
    try:
        config_path = Path(args.config).expanduser().resolve()
        unilm_dir = Path(args.unilm_source_dir).expanduser().resolve()
        hf_cache_dir = Path(args.hf_home).expanduser().resolve() if args.hf_home else None
        config = _resolve_config(config_path, unilm_dir, hf_cache_dir)
        resources = _configured_resources(config)
        _check_resources(resources)
        identity = _unilm_identity(unilm_dir)
        if not identity["identity_pass"]:
            raise ValueError(f"Expert 4 UniLM source identity mismatch: {identity}")
        report["checkpoint_provenance"] = resources
        report["expert4_unilm_source_identity"] = identity
        report["resolved_configuration"] = {
            "config_path": str(config_path), "offline": True, "device": args.device,
            "expert3_cache_dir": str(hf_cache_dir) if hf_cache_dir else str(config["runtime"]["cache_dir"]),
        }
        manifest_path = output_dir / "manifest.json"
        if args.mode == "assemble-and-gate":
            manifest_path = Path(args._manifest).expanduser().resolve() if args._manifest else manifest_path
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_sha = validate_sample_manifest(manifest)
        else:
            manifest, dataset = _make_manifest(args)
            manifest_sha = validate_sample_manifest(manifest)
            _atomic_json(manifest_path, manifest)
            report["selected_sample_manifest"] = {
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "sample_count": len(manifest["samples"]),
                "selection_policy": manifest["selection"],
                "samples": manifest["samples"],
                "csv_row_count": len(dataset),
            }
            if args.max_samples < len(manifest["samples"]):
                raise ValueError("manifest selected more samples than configured")
        report["model_construction_preflight"] = False
        report["checkpoint_deserialization_preflight"] = False
        report["simultaneous_four_expert_runtime"] = "NOT_ATTEMPTED_RESOURCE_RISK"
        if args.mode != "assemble-and-gate":
            report["expert3_text_cache_preflight"] = _preflight_expert3_text_cache(Path(config["runtime"]["cache_dir"]))
        _atomic_json(report_path, report)
        if args.mode == "preflight":
            report["status"] = "PASS_PREFLIGHT_ONLY"
            _atomic_json(report_path, report)
            print(f"Kiểm tra trước: PASS; manifest={manifest_sha}; chưa khởi tạo model.")
            return 0
        if args.mode == "sequential":
            before_stats = {name: _checkpoint_snapshot(config, name) for name in EXPERT_ORDER}
            for expert in EXPERT_ORDER:
                reused = _existing_verified_output(args, expert, manifest_sha, manifest["samples"], config)
                if reused is not None:
                    reused["reused_existing_artifact"] = True
                    report["experts"][expert] = reused
                else:
                    _run_worker(args, expert, report)
            if all(report["experts"].get(name, {}).get("status") == "PASS" for name in EXPERT_ORDER):
                report["real_four_expert_output_extraction"] = "PASS"
                try:
                    _assemble_and_gate(args, report)
                except BaseException as exc:
                    report["failure"] = {"stage": "ASSEMBLY_OR_GATE", "category": _classify_exception(exc, "OUTPUT_VALIDATION"), "exception_class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                    raise
                report["checkpoint_file_metadata_unchanged"] = before_stats == {name: _checkpoint_snapshot(config, name) for name in EXPERT_ORDER}
                if not report["checkpoint_file_metadata_unchanged"]:
                    raise RuntimeError("checkpoint file metadata changed during verification")
                _run_pair(args, report)
                report["status"] = "PASS — REAL FOUR-EXPERT OUTPUT-TO-GATING VERIFIED"
            else:
                report["status"] = "PARTIAL — REAL OUTPUT INTEGRATION INCOMPLETE"
                failed = [name for name in EXPERT_ORDER if report["experts"].get(name, {}).get("status") != "PASS"]
                report["failed_experts"] = failed
                report["real_four_expert_output_extraction"] = "FAIL"
                report["simultaneous_four_expert_runtime"] = "NOT_ATTEMPTED_RESOURCE_RISK"
            _atomic_json(report_path, report)
            print(f"LOCAL-10.1: {report['status']}; JSON={report_path}")
            return 0 if report["status"].startswith("PASS") else 2
        if args.mode == "assemble-and-gate":
            _assemble_and_gate(args, report)
            report["status"] = "PASS — OUTPUT ARTIFACT ASSEMBLY/GATE ONLY"
            _atomic_json(report_path, report)
            print(f"Ghép/gate artifact: PASS; JSON={report_path}")
            return 0
        raise ValueError(f"unsupported mode {args.mode}")
    except BaseException as exc:
        report["status"] = "BLOCKED — RESOURCE OR CHECKPOINT" if _classify_exception(exc, "PREFLIGHT") in {"MISSING_LOCAL_RESOURCE", "MISSING_TEXT_MODEL_CACHE", "CHECKPOINT_RESTORE_FAILURE"} else "FAIL — REAL OUTPUT OR GATING CONTRACT"
        report.setdefault("failure", {
            "stage": "PREFLIGHT_OR_ORCHESTRATION",
            "category": _classify_exception(exc, "PREFLIGHT"),
            "exception_class": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        _atomic_json(report_path, report)
        print(f"LOCAL-10.1 thất bại: {type(exc).__name__}: {exc}; JSON={report_path}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
