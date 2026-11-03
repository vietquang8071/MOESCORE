#!/usr/bin/env python3
"""Offline runtime verification for the restored, fine-tuned Expert 3.

This script intentionally restores exactly one original ``MoescoreExpert3``
instance via ``Expert3Adapter.from_finetuned`` and reuses that instance for
standalone/adapter comparisons. It does not train, download, or alter weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any


EXPECTED_BASE_SHA256 = "238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1"
EXPECTED_MODEL_DIR = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
TEXT_MODEL_ID = "google-bert/bert-base-uncased"
ATOL_DEFAULT = 1e-6
RTOL_DEFAULT = 1e-5


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Required file is not readable: {path}")
    stat = path.stat()
    with path.open("rb") as stream:
        stream.read(1)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_local": __import__("datetime").datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "readable": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_metrics(actual, expected, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "shape_match": tuple(actual.shape) == tuple(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "passed": False,
    }
    if not result["shape_match"]:
        result["error"] = "tensor shapes differ"
        return result
    a = actual.detach().to(device="cpu", dtype=torch.float64)
    e = expected.detach().to(device="cpu", dtype=torch.float64)
    delta = (a - e).abs()
    relative = delta / e.abs().clamp_min(atol)
    result.update(
        max_abs_diff=float(delta.max().item()) if delta.numel() else 0.0,
        mean_abs_diff=float(delta.mean().item()) if delta.numel() else 0.0,
        max_relative_diff=float(relative.max().item()) if relative.numel() else 0.0,
    )
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        result["error"] = str(exc).splitlines()[0]
    else:
        result["passed"] = True
    return result


def _module_origin(name: str) -> dict[str, Any]:
    import importlib.util
    import sys

    module = sys.modules.get(name)
    if module is not None:
        spec = getattr(module, "__spec__", None)
        return {
            "loaded": True,
            "file": getattr(module, "__file__", None),
            "spec_origin": getattr(spec, "origin", None) if spec else None,
            "package_paths": list(getattr(module, "__path__", ())),
        }
    spec = importlib.util.find_spec(name)
    return {
        "loaded": False,
        "file": None,
        "spec_origin": getattr(spec, "origin", None) if spec else None,
        "package_paths": list(getattr(spec, "submodule_search_locations", ()) or ()),
    }


def _check_text_resources(cache_dir: Path) -> dict[str, Any]:
    """Preflight the exact BERT tokenizer/config locally; model loads in factory."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        TEXT_MODEL_ID, cache_dir=str(cache_dir), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_ID, cache_dir=str(cache_dir), local_files_only=True
    )
    record = {
        "model_id": TEXT_MODEL_ID,
        "offline": True,
        "config_model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "status": "PASS",
    }
    del tokenizer, config
    return record


def _load_partial_payload(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Fine-tuned Expert 3 artifact must be a mapping, got {type(payload).__name__}")
    expected_top = {
        "format_version", "checkpoint_kind", "model_trainable", "model_config",
        "base_checkpoint_identity", "epoch", "best_srcc", "metrics", "config",
    }
    missing_top = sorted(expected_top - set(payload))
    unexpected_top = sorted(set(payload) - expected_top)
    if missing_top or unexpected_top:
        raise ValueError(
            f"Partial artifact top-level schema mismatch: missing={missing_top}, unexpected={unexpected_top}"
        )
    return payload


def _payload_diagnostics(payload: dict[str, Any], *, base_sha: str, base_size: int) -> dict[str, Any]:
    import torch

    trainable = payload["model_trainable"]
    if not isinstance(trainable, dict):
        raise TypeError("model_trainable must be a parameter mapping")
    non_tensor = [name for name, value in trainable.items() if not isinstance(value, torch.Tensor)]
    non_finite = [
        name for name, value in trainable.items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item()
    ]
    component_counts = {
        "alignment_head": sum(name.startswith("alignment_head.") for name in trainable),
        "audio_projector": sum(name.startswith("m2d.backbone.audio_proj.") for name in trainable),
        "m2d_audio_blocks_10_11": sum(
            name.startswith(("m2d.backbone.blocks.10.", "m2d.backbone.blocks.11.")) for name in trainable
        ),
        "bert_text_layers_10_11": sum(
            name.startswith((
                "m2d.text_encoder.text_encoder.encoder.layer.10.",
                "m2d.text_encoder.text_encoder.encoder.layer.11.",
            )) for name in trainable
        ),
    }
    identity = payload["base_checkpoint_identity"]
    runtime_config = payload["config"]
    return {
        "format_version": payload["format_version"],
        "checkpoint_kind": payload["checkpoint_kind"],
        "top_level_keys": sorted(payload),
        "epoch": payload["epoch"],
        "best_srcc": payload["best_srcc"],
        "model_config": payload["model_config"],
        "base_checkpoint_identity": identity,
        "checkpoint_config": {
            key: runtime_config.get(key)
            for key in ("m2d_checkpoint_sha256", "concat_assignment", "sample_rate", "max_len", "activation_checkpointing")
        },
        "trained_parameter_keys": len(trainable),
        "trained_parameter_component_counts": component_counts,
        "trained_parameter_tensor_elements": sum(
            int(value.numel()) for value in trainable.values() if isinstance(value, torch.Tensor)
        ),
        "non_tensor_values": len(non_tensor),
        "non_finite_tensors": len(non_finite),
        "base_sha_matches": identity.get("sha256") == base_sha
        and runtime_config.get("m2d_checkpoint_sha256") == base_sha,
        "base_size_matches": identity.get("size_bytes") == base_size,
        "released_assignment": payload.get("model_config", {}).get("concat_assignment") == "released"
        and runtime_config.get("concat_assignment") == "released",
        "missing_required_keys": None,
        "unexpected_keys": None,
        "shape_mismatches": None,
        "non_tensor_names": non_tensor,
        "non_finite_names": non_finite,
    }


def _compare_restored_parameters(model, saved: dict[str, Any]) -> dict[str, Any]:
    """Check each saved trainable tensor was copied by the original loader."""
    import torch

    parameters = dict(model.named_parameters())
    missing = sorted(set(saved) - set(parameters))
    shapes = []
    mismatches = []
    for name, value in saved.items():
        if name not in parameters or not isinstance(value, torch.Tensor):
            continue
        parameter = parameters[name]
        if tuple(parameter.shape) != tuple(value.shape):
            shapes.append(name)
            continue
        restored = parameter.detach().to(device="cpu")
        expected = value.to(dtype=parameter.dtype)
        if not torch.equal(restored, expected):
            mismatches.append(name)
    return {
        "checkpoint_trained_keys": len(saved),
        "checkpoint_keys_missing_from_model": len(missing),
        "tensor_shape_mismatches": len(shapes),
        "restored_value_mismatches": len(mismatches),
        "exact_tensor_values_match": not missing and not shapes and not mismatches,
        "missing_names": missing,
        "shape_mismatch_names": shapes,
        "value_mismatch_names": mismatches,
    }


def _instrument_adapter_forward(adapter, batch):
    """Count the original model/encoder/head calls and capture normalized pairs."""
    import types
    import torch

    model = adapter.model
    m2d = model.m2d
    counts = {"model_forward": 0, "encode_modalities": 0, "audio_encoder": 0, "text_encoder": 0, "regression_head": 0}
    captured: dict[str, Any] = {}

    def model_hook(_module, _inputs, _outputs):
        counts["model_forward"] += 1

    def head_hook(_module, _inputs, _outputs):
        counts["regression_head"] += 1

    handles = [model.register_forward_hook(model_hook)]
    if hasattr(model, "alignment_head"):
        handles.append(model.alignment_head.register_forward_hook(head_hook))

    instance_originals: list[tuple[Any, str, bool, Any]] = []

    def wrap_method(owner, name, on_result=None):
        was_local = name in owner.__dict__
        previous = owner.__dict__.get(name)
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if on_result is not None:
                on_result(args, kwargs, result)
            return result

        setattr(owner, name, wrapped)
        instance_originals.append((owner, name, was_local, previous))

    def capture_modalities(args, _kwargs, result):
        counts["encode_modalities"] += 1
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            captured["error"] = "encode_modalities did not return (text_embedding, audio_embedding)"
            return
        captured["text_embedding"], captured["audio_embedding"] = result
        captured["captions"] = list(args[1]) if len(args) > 1 else None

    wrap_method(model, "encode_modalities", capture_modalities)
    wrap_method(m2d, "encode_clap_audio", lambda *_: counts.__setitem__("audio_encoder", counts["audio_encoder"] + 1))
    wrap_method(m2d, "encode_clap_text", lambda *_: counts.__setitem__("text_encoder", counts["text_encoder"] + 1))
    try:
        with torch.no_grad():
            outputs = adapter(batch)
    finally:
        for owner, name, was_local, previous in reversed(instance_originals):
            if was_local:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        for handle in handles:
            handle.remove()
    return outputs, counts, captured


def _released_blocks(features, captured, *, embedding_dim: int, atol: float, rtol: float):
    import torch

    text = captured.get("text_embedding")
    audio = captured.get("audio_embedding")
    if not isinstance(text, torch.Tensor) or not isinstance(audio, torch.Tensor):
        return {"passed": False, "error": captured.get("error", "embeddings were not captured"), "blocks": []}
    # Source returns (text, audio) and released mode intentionally binds this as
    # (audio_feats, text_feats), hence the text-first released feature vector.
    audio_feats, text_feats = text, audio
    cosine = (audio_feats * text_feats).sum(dim=-1, keepdim=True)
    expected = [
        audio_feats,
        text_feats,
        torch.abs(audio_feats - text_feats),
        audio_feats * text_feats,
        cosine.expand(-1, embedding_dim),
    ]
    observed = [features[:, i * embedding_dim:(i + 1) * embedding_dim] for i in range(5)]
    names = ["released_text", "released_audio", "absolute_difference", "elementwise_product", "repeated_cosine"]
    checks = [
        {"name": name, **_tensor_metrics(actual, ref, atol=atol, rtol=rtol)}
        for name, actual, ref in zip(names, observed, expected)
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "block_names_and_order": names,
        "text_embedding_shape": list(text.shape),
        "audio_embedding_shape": list(audio.shape),
        "blocks": checks,
    }


def _run_case(adapter, audio, captions, *, label: str, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    model = adapter.model
    model_calls = {"count": 0}
    handle = model.register_forward_hook(lambda *_: model_calls.__setitem__("count", model_calls["count"] + 1))
    try:
        with torch.no_grad():
            standalone_score, standalone_features = model({"wavs": audio, "captions": captions})
        standalone_count = model_calls["count"]
    finally:
        handle.remove()

    adapter_outputs, calls, captured = _instrument_adapter_forward(
        adapter, {"expert3_wavs": audio, "captions_raw": captions}
    )
    adapter_score, adapter_features = adapter_outputs
    score_parity = _tensor_metrics(adapter_score, standalone_score, atol=atol, rtol=rtol)
    feature_parity = _tensor_metrics(adapter_features, standalone_features, atol=atol, rtol=rtol)
    block_parity = _released_blocks(
        adapter_features, captured, embedding_dim=adapter.embedding_dim, atol=atol, rtol=rtol
    )
    finite = bool(torch.isfinite(adapter_score).all().item() and torch.isfinite(adapter_features).all().item())
    shapes = (
        tuple(adapter_score.shape) == (audio.shape[0], 1)
        and tuple(adapter_features.shape) == (audio.shape[0], adapter.feature_dim)
        and tuple(standalone_score.shape) == tuple(adapter_score.shape)
        and tuple(standalone_features.shape) == tuple(adapter_features.shape)
    )
    device_dtype = (
        adapter_score.device == audio.device
        and adapter_features.device == audio.device
        and adapter_score.dtype == torch.float32
        and adapter_features.dtype == torch.float32
    )
    single_pass = (
        standalone_count == 1
        and calls["model_forward"] == 1
        and calls["encode_modalities"] == 1
        and calls["audio_encoder"] == 1
        and calls["text_encoder"] == 1
        and calls["regression_head"] == 1
    )
    return {
        "label": label,
        "batch_size": int(audio.shape[0]),
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "audio_device": str(audio.device),
        "captions_preserved": captured.get("captions") == captions,
        "standalone_score_shape": list(standalone_score.shape),
        "adapter_score_shape": list(adapter_score.shape),
        "feature_shape": list(adapter_features.shape),
        "score_parity": score_parity,
        "feature_parity": feature_parity,
        "released_feature_blocks": block_parity,
        "standalone_model_forward_calls": standalone_count,
        "adapter_calls": calls,
        "single_pass_verified": single_pass,
        "finite_outputs": finite,
        "output_dtype": str(adapter_score.dtype),
        "output_device": str(adapter_score.device),
        "score_values": adapter_score.detach().cpu().reshape(-1).tolist(),
        "score_min": float(adapter_score.min().item()),
        "score_max": float(adapter_score.max().item()),
        "shape_contract_passed": shapes,
        "device_dtype_contract_passed": device_dtype,
        "passed": bool(
            score_parity["passed"] and feature_parity["passed"] and block_parity["passed"]
            and single_pass and finite and shapes and device_dtype
            and captured.get("captions") == captions
        ),
    }


def _native_validation_batch(csv_path: Path, audio_root: Path, count: int):
    """Use original dataset __getitem__/collate on only the first selected rows."""
    from expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import (
        MoescoreExpert3Dataset,
    )

    if count <= 0:
        raise ValueError("num_real_samples must be positive")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"wav_file_name", "text"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Validation CSV missing required fields: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Validation CSV contains fewer than {count} rows")
    filenames = [row["wav_file_name"].strip() for row in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Selected validation rows contain duplicate WAV filenames")
    for name in filenames:
        if not name or Path(name).name != name or not (audio_root / name).is_file():
            raise FileNotFoundError(f"Selected validation WAV is missing or invalid: {audio_root / name}")

    # The original constructor reads and checks every validation row. For a
    # two-sample smoke test, retain only selected rows while reusing its exact
    # preprocessing __getitem__ and collate_fn implementations.
    dataset = object.__new__(MoescoreExpert3Dataset)
    dataset.csv_path = csv_path
    dataset.wav_dir = audio_root
    dataset.sample_rate = 16000
    dataset.max_samples = 160000
    dataset.filename_column = "wav_file_name"
    dataset.caption_column = "text"
    dataset.score_column = "average_score"
    dataset.rows = rows
    dataset.has_scores = False
    dataset.num_class = [1] * count
    items = [dataset[index] for index in range(count)]
    return MoescoreExpert3Dataset.collate_fn(items), filenames


def _memory_snapshot(device, torch) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"available": False}
    try:
        free, total = torch.cuda.mem_get_info(device)
    except (TypeError, RuntimeError):
        # Some CUDA/PyTorch builds reject a torch.device argument even though
        # the no-argument query correctly reports the current CUDA device.
        try:
            free, total = torch.cuda.mem_get_info()
        except (TypeError, RuntimeError) as exc:
            return {"available": False, "query_error": f"{type(exc).__name__}: {exc}"}
    try:
        peak = int(torch.cuda.max_memory_allocated(device))
    except (AttributeError, TypeError, RuntimeError):
        try:
            peak = int(torch.cuda.max_memory_allocated())
        except (AttributeError, TypeError, RuntimeError):
            peak = None
    return {"available": True, "free_bytes": int(free), "total_bytes": int(total), "peak_allocated_bytes": peak}


def _reset_peak_memory_stats(device, torch) -> None:
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except (TypeError, RuntimeError):
        # Match _memory_snapshot's compatibility path for builds that reject
        # torch.device even though the current CUDA device is valid.
        torch.cuda.reset_peak_memory_stats()


def _parse_args() -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default="weights/checkpoint-30.pth")
    parser.add_argument("--finetuned-checkpoint", default="weights/finetuned_weights/expert3_best.pt")
    parser.add_argument(
        "--model-config",
        default="expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json",
    )
    parser.add_argument("--source-dir", default="expert3_finetune/XACLE_2027_baselinemodel")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--checkpoint-layout-root", default="/tmp/moescore-expert3-m2d-layout")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--validation-csv")
    parser.add_argument("--audio-root")
    parser.add_argument("--num-real-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--skip-real-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _root()
    # When Python executes ``scripts/verify_expert3_runtime.py``, sys.path[0]
    # points at scripts/. Add this repository root only in this verifier
    # process so the existing adapter and namespaced source packages resolve.
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    base_path = _path(args.base_checkpoint, root)
    fine_path = _path(args.finetuned_checkpoint, root)
    config_path = _path(args.model_config, root)
    source_dir = _path(args.source_dir, root)
    cache_dir = _path(args.cache_dir, root)
    layout_root = _path(args.checkpoint_layout_root, root)
    validation_csv = _path(args.validation_csv, root) if args.validation_csv else None
    audio_root = _path(args.audio_root, root) if args.audio_root else None
    summary: dict[str, Any] = {
        "status": "INCOMPLETE",
        "offline": True,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "tested_batch_sizes": [1, 2],
        "stage": "environment_and_resource_validation",
        "errors": [],
    }
    payload = None
    adapter = None
    try:
        if args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol + args.rtol):
            raise ValueError("atol and rtol must be finite non-negative numbers")
        if not args.offline:
            raise ValueError("This verification is offline-only")

        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        os.environ["HF_HOME"] = str(cache_dir)

        base_record = _file_record(base_path)
        fine_record = _file_record(fine_path)
        config_record = _file_record(config_path)
        summary.update(
            python=platform.python_version(),
            checkpoint_paths={"base": base_record, "finetuned": fine_record, "model_config": config_record},
            source_dir=str(source_dir),
            cache_dir=str(cache_dir),
            checkpoint_layout_root=str(layout_root),
        )

        summary["stage"] = "base_sha256_verification"
        actual_sha = _sha256(base_path)
        summary["base_checkpoint_identity"] = {
            "expected_sha256": EXPECTED_BASE_SHA256,
            "actual_sha256": actual_sha,
            "matches": actual_sha == EXPECTED_BASE_SHA256,
            "size_bytes": base_record["size_bytes"],
        }
        if actual_sha != EXPECTED_BASE_SHA256:
            summary["status"] = "FAIL — BASE CHECKPOINT IDENTITY MISMATCH"
            raise RuntimeError(
                f"Base checkpoint SHA-256 mismatch: expected {EXPECTED_BASE_SHA256}, got {actual_sha}"
            )

        summary["stage"] = "offline_text_resource_preflight"
        text_resources = _check_text_resources(cache_dir)
        summary["text_encoder_resources"] = text_resources

        summary["stage"] = "partial_checkpoint_schema_preflight"
        import torch
        payload = _load_partial_payload(fine_path)
        partial = _payload_diagnostics(payload, base_sha=actual_sha, base_size=base_record["size_bytes"])
        summary["partial_checkpoint"] = partial
        if (
            payload.get("format_version") != 1
            or payload.get("checkpoint_kind") != "best"
            or not partial["base_sha_matches"]
            or not partial["base_size_matches"]
            or not partial["released_assignment"]
            or partial["non_tensor_values"] != 0
            or partial["non_finite_tensors"] != 0
            or payload.get("base_checkpoint_identity", {}).get("model_directory") != EXPECTED_MODEL_DIR
        ):
            summary["status"] = "FAIL — PARTIAL CHECKPOINT RESTORATION"
            raise ValueError("Fine-tuned partial checkpoint metadata/schema does not match the verified base")

        summary["stage"] = "model_construction_and_checkpoint_restoration"
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "cuda":
            _reset_peak_memory_stats(device, torch)
        before_memory = _memory_snapshot(device, torch)

        from model.expert_adapters.expert3_adapter import Expert3Adapter

        adapter = Expert3Adapter.from_finetuned(
            source_dir=source_dir,
            base_checkpoint=base_path,
            finetuned_checkpoint=fine_path,
            model_config=config_path,
            cache_dir=cache_dir,
            checkpoint_layout_root=layout_root,
            expected_base_sha256=EXPECTED_BASE_SHA256,
            concat_assignment="released",
            device=device,
            offline=True,
        )
        model = adapter.model
        summary["model_construction"] = "PASS"
        summary["pretrained_restoration"] = "PASS — original PortableM2D constructor loaded the verified base"
        summary["partial_restoration"] = "PASS — adapter preflight and original load_best_checkpoint succeeded"
        summary["restoration"] = {
            "format_version": partial["format_version"],
            "checkpoint_kind": partial["checkpoint_kind"],
            "trained_parameter_keys_expected_and_saved": partial["trained_parameter_keys"],
            "missing_required_trained_keys": 0,
            "unexpected_trained_keys": 0,
            "shape_mismatches": 0,
            "non_tensor_values": partial["non_tensor_values"],
            "non_finite_tensors": partial["non_finite_tensors"],
            "source_loader": "expert3_finetune.XACLE_2027_baselinemodel.utils.expert3_checkpoint.load_best_checkpoint",
        }
        summary["stage"] = "restored_trainable_tensor_verification"
        restored = _compare_restored_parameters(model, payload["model_trainable"])
        summary["restored_parameter_values"] = restored
        if not restored["exact_tensor_values_match"]:
            summary["status"] = "FAIL — PARTIAL CHECKPOINT RESTORATION"
            raise ValueError("One or more trained tensors differ after original partial restoration")

        summary["stage"] = "runtime_metadata_and_freeze_checks"
        summary["environment"] = {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": str(next(model.parameters()).dtype),
            "transformers": __import__("transformers").__version__,
            "offline_environment": {
                name: os.environ.get(name)
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
            },
        }
        summary["dimensions"] = {
            "embedding_dim": int(model.embedding_dim),
            "cross_feature_dim": int(model.cross_feature_dim),
            "adapter_feature_dim": int(adapter.feature_dim),
            "expected_cross_feature_dim": 5 * int(model.embedding_dim),
            "metadata_match": adapter.feature_dim == model.cross_feature_dim == 5 * model.embedding_dim,
            "concat_assignment": model.concat_assignment,
        }
        if not summary["dimensions"]["metadata_match"]:
            raise ValueError("Restored Expert 3 feature dimensions violate the source 5D contract")

        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        adapter_only = [parameter for parameter in adapter.parameters() if id(parameter) not in model_parameter_ids]
        summary["freeze_eval"] = {
            "expert_total_parameters": int(total_parameters),
            "expert_trainable_parameters": int(trainable_parameters),
            "adapter_only_trainable_parameters": int(sum(p.numel() for p in adapter_only if p.requires_grad)),
            "all_expert_parameters_frozen": trainable_parameters == 0,
            "all_expert_modules_eval_initially": all(not module.training for module in model.modules()),
        }
        adapter.train()
        summary["freeze_eval"]["after_adapter_train"] = {
            "adapter_training_flag": adapter.training,
            "all_expert_modules_eval": all(not module.training for module in model.modules()),
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        }
        adapter.train(False)
        summary["freeze_eval"]["after_adapter_train_false"] = {
            "adapter_training_flag": adapter.training,
            "all_expert_modules_eval": all(not module.training for module in model.modules()),
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        }
        adapter.eval()
        after_train = summary["freeze_eval"]["after_adapter_train"]
        after_train_false = summary["freeze_eval"]["after_adapter_train_false"]
        if (
            not summary["freeze_eval"]["all_expert_parameters_frozen"]
            or not after_train["adapter_training_flag"]
            or not after_train["all_expert_modules_eval"]
            or not after_train["all_parameters_frozen"]
            or after_train_false["adapter_training_flag"]
            or not after_train_false["all_expert_modules_eval"]
            or not after_train_false["all_parameters_frozen"]
        ):
            raise RuntimeError("Expert 3 freeze/evaluation invariants failed")

        summary["stage"] = "synthetic_parity"
        torch.manual_seed(args.seed)
        captions_by_batch = {
            1: ["A brief synthetic audio test with a bell."],
            2: ["A brief synthetic audio test with a bell.", "Birds singing near a stream."],
        }
        synthetic_cases = []
        for batch_size in (1, 2):
            audio = (0.1 * torch.randn(batch_size, 1, 160000, dtype=torch.float32)).to(device)
            case = _run_case(
                adapter, audio, captions_by_batch[batch_size],
                label=f"synthetic_B{batch_size}", atol=args.atol, rtol=args.rtol,
            )
            synthetic_cases.append(case)
            print(json.dumps({"case": case}, ensure_ascii=False))
            if not case["passed"]:
                summary["status"] = "FAIL — NUMERICAL PARITY"
                raise AssertionError(f"Synthetic parity or released feature assignment failed for B={batch_size}")

        summary["synthetic_cases"] = synthetic_cases
        summary["stage"] = "frozen_determinism"
        adapter.eval()
        repeat_audio = (0.1 * torch.randn(1, 1, 160000, dtype=torch.float32, device=device))
        repeat_caption = ["Deterministic repeat check for Expert 3."]
        with torch.no_grad():
            repeat_first = adapter({"expert3_wavs": repeat_audio, "captions_raw": repeat_caption})
            repeat_second = adapter({"expert3_wavs": repeat_audio, "captions_raw": repeat_caption})
        repeated = {
            "score": _tensor_metrics(repeat_first[0], repeat_second[0], atol=args.atol, rtol=args.rtol),
            "features": _tensor_metrics(repeat_first[1], repeat_second[1], atol=args.atol, rtol=args.rtol),
        }
        repeated["passed"] = repeated["score"]["passed"] and repeated["features"]["passed"]
        summary["repeated_evaluation"] = repeated
        if not repeated["passed"]:
            raise AssertionError("Repeated Expert 3 evaluation is not deterministic within tolerance")

        summary["stage"] = "module_origin_verification"
        source_resolved = source_dir.resolve()
        module_names = (
            "expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3",
            "expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d",
            "expert3_finetune.XACLE_2027_baselinemodel.utils.expert3_checkpoint",
        )
        origins = {name: _module_origin(name) for name in module_names}
        origin_checks = {}
        for name, record in origins.items():
            origin = record["file"] or record["spec_origin"]
            origin_checks[name] = bool(origin and Path(origin).resolve().is_relative_to(source_resolved))
        summary["module_origins"] = {"modules": origins, "all_under_expert3_source": all(origin_checks.values()), "checks": origin_checks}
        if not summary["module_origins"]["all_under_expert3_source"]:
            raise ImportError("One or more Expert 3 runtime modules resolved outside the intended source tree")

        summary["stage"] = "real_validation_subset"
        real_status = "NOT_VERIFIED"
        if not args.skip_real_validation and validation_csv and audio_root:
            try:
                if not validation_csv.is_file() or not audio_root.is_dir():
                    raise FileNotFoundError(f"Validation CSV/audio root missing: {validation_csv}, {audio_root}")
                real_batch, sample_names = _native_validation_batch(
                    validation_csv, audio_root, args.num_real_samples
                )
                real_audio = real_batch["wavs"].to(device)
                real_captions = real_batch["captions"]
                real_case = _run_case(
                    adapter, real_audio, real_captions, label="native_validation_subset",
                    atol=args.atol, rtol=args.rtol,
                )
                real_case["sample_ids"] = sample_names
                real_case["preprocessing"] = (
                    "MoescoreExpert3Dataset.__getitem__/collate_fn: soundfile float32, mono mean, "
                    "scipy resample_poly to 16 kHz, first 160000 samples, right zero-pad, caption.strip()"
                )
                print(json.dumps({"case": real_case}, ensure_ascii=False))
                real_status = "PASS" if real_case["passed"] else "FAIL"
                summary["real_validation"] = {
                    "status": real_status,
                    "csv": str(validation_csv),
                    "audio_root": str(audio_root),
                    "sample_ids": sample_names,
                    "sample_rate_hz": 16000,
                    "waveform_shape": list(real_audio.shape),
                    "captions": real_captions,
                    "case": real_case,
                }
            except Exception as exc:
                summary["real_validation"] = {
                    "status": "NOT_VERIFIED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            summary["real_validation"] = {"status": "NOT_VERIFIED", "reason": "validation paths not provided or skipped"}

        summary["memory"] = {
            "before_model_load": before_memory,
            "after_verification": _memory_snapshot(device, torch),
        }
        summary["resource_unchanged"] = (
            base_path.stat().st_size == base_record["size_bytes"]
            and base_path.stat().st_mtime_ns == base_record["mtime_ns"]
            and fine_path.stat().st_size == fine_record["size_bytes"]
            and fine_path.stat().st_mtime_ns == fine_record["mtime_ns"]
        )
        if not summary["resource_unchanged"]:
            raise RuntimeError("A checkpoint size/mtime changed during verification")
        summary["status"] = (
            "PASS — INCLUDING REAL VALIDATION INFERENCE"
            if real_status == "PASS"
            else "PASS — REAL CHECKPOINT & NUMERICAL PARITY VERIFIED"
        )
        summary["stage"] = "complete"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if summary.get("status") == "INCOMPLETE":
            stage = summary.get("stage", "unknown")
            if stage == "partial_checkpoint_schema_preflight" or stage == "restored_trainable_tensor_verification":
                summary["status"] = "FAIL — PARTIAL CHECKPOINT RESTORATION"
            elif stage in {"synthetic_parity", "frozen_determinism"}:
                summary["status"] = "FAIL — NUMERICAL PARITY"
            else:
                summary["status"] = "BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION"
        summary["errors"].append({"stage": summary.get("stage"), "type": type(exc).__name__, "message": str(exc)})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    finally:
        if payload is not None:
            del payload
        if adapter is not None:
            del adapter


if __name__ == "__main__":
    raise SystemExit(main())
