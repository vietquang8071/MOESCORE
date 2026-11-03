#!/usr/bin/env python3
"""Restore Expert 1 and verify its standalone/adapter numerical parity.

This is an explicit, heavyweight verification command. It never downloads
resources, changes checkpoints, constructs a second Expert 1, or trains a
model. Run it from the repository root with the project's ``nlp`` environment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from model.expert_adapters.expert1_adapter import Expert1Adapter


AUDIO_SAMPLES = 48_000 * 10
CAPTIONS = (
    "A dog barking in a quiet room.",
    "Rain falling on a metal roof.",
)


class MissingLocalResourceError(RuntimeError):
    """A required local model/tokenizer resource is absent in offline mode."""


def _file_resource(value: str, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{name} is not readable: {path}")
    return path


def _resource_snapshot(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
    return device


def _cuda_memory(device: torch.device) -> Optional[Dict[str, int]]:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _roberta_local_cache_status() -> Dict[str, Any]:
    """Inspect only the configured Hugging Face cache entry for roberta-base."""
    try:
        from huggingface_hub import constants as hub_constants

        cache_root = Path(hub_constants.HF_HUB_CACHE)
    except (ImportError, AttributeError) as exc:
        return {
            "cache_root": None,
            "repo_cache_exists": False,
            "snapshot_count": 0,
            "config_json_present": False,
            "model_weights_present": False,
            "inspection_error": str(exc),
        }

    repo_cache = cache_root / "models--roberta-base"
    snapshots_dir = repo_cache / "snapshots"
    snapshots = (
        sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots_dir.is_dir()
        else []
    )
    config_present = any((snapshot / "config.json").is_file() for snapshot in snapshots)
    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    weights_present = any(
        (snapshot / filename).is_file()
        for snapshot in snapshots
        for filename in weight_names
    )
    tokenizer_files = ("vocab.json", "merges.txt", "tokenizer.json")
    tokenizer_files_present = all(
        any((snapshot / filename).is_file() for snapshot in snapshots)
        for filename in tokenizer_files
    )
    return {
        "cache_root": str(cache_root),
        "repo_cache_exists": repo_cache.is_dir(),
        "snapshot_count": len(snapshots),
        "config_json_present": config_present,
        "model_weights_present": weights_present,
        "tokenizer_files_present": tokenizer_files_present,
        "required_for_model_construction_present": config_present and weights_present,
    }


def _close_metrics(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "max_relative_diff": None,
        }
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    absolute = (actual_f - expected_f).abs()
    # Treat reference values close to zero with a finite denominator so the
    # reported relative error remains interpretable and cannot overflow.
    relative = absolute / expected_f.abs().clamp_min(1e-12)
    return {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs_diff": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "max_relative_diff": float(relative.max().item()) if relative.numel() else 0.0,
    }


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> Optional[str]:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        return f"{name} did not meet atol={atol}, rtol={rtol}: {exc}"
    return None


def _capture_encoder_call(
    captured: List[Dict[str, Any]], module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any
) -> None:
    if len(inputs) != 2 or not isinstance(output, (tuple, list)) or len(output) != 2:
        captured.append({"invalid_encoder_contract": True})
        return
    audio_arg, captions_arg = inputs
    audio_embedding, text_embedding = output
    captured.append(
        {
            "input_audio": audio_arg.detach().clone(),
            "input_captions": list(captions_arg),
            "audio_embedding": audio_embedding.detach().clone(),
            "text_embedding": text_embedding.detach().clone(),
        }
    )


def _verify_case(
    *,
    adapter: Expert1Adapter,
    audio: torch.Tensor,
    captions: Sequence[str],
    label: str,
    atol: float,
    rtol: float,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    model = adapter.model
    encoder = model.model
    captured: List[Dict[str, Any]] = []
    handle = encoder.register_forward_hook(
        lambda module, inputs, output: _capture_encoder_call(
            captured, module, inputs, output
        )
    )
    case: Dict[str, Any] = {
        "label": label,
        "batch_size": int(audio.shape[0]),
        "audio_input_shape": list(audio.shape),
        "audio_input_dtype": str(audio.dtype),
        "audio_input_device": str(audio.device),
        "caption_count": len(captions),
        "caption_text_preserved": True,
        "atol": atol,
        "rtol": rtol,
        "errors": [],
    }

    try:
        captured.clear()
        with torch.no_grad():
            standalone_score = model(
                {"clap_wavs": audio, "caption_tokens": list(captions)}
            )
        standalone_calls = len(captured)
        if standalone_calls != 1 or not captured or captured[-1].get("invalid_encoder_contract"):
            raise RuntimeError(
                f"Standalone Expert1 expected one encoder call with two embeddings; got {standalone_calls}"
            )
        standalone_embeddings = captured[-1]
        if not torch.equal(standalone_embeddings["input_audio"], audio):
            raise RuntimeError("Standalone Expert1 encoder received a different waveform tensor")
        if standalone_embeddings["input_captions"] != list(captions):
            raise RuntimeError("Standalone Expert1 encoder received different captions")

        captured.clear()
        with torch.no_grad():
            adapter_score, adapter_features = adapter(
                {"expert1_wavs": audio, "captions_raw": list(captions)}
            )
        adapter_calls = len(captured)
        if adapter_calls != 1 or not captured or captured[-1].get("invalid_encoder_contract"):
            raise RuntimeError(
                f"Adapter expected exactly one encoder call with two embeddings; got {adapter_calls}"
            )
        adapter_embeddings = captured[-1]
        if not torch.equal(adapter_embeddings["input_audio"], audio):
            raise RuntimeError("Adapter encoder received a different waveform tensor")
        if adapter_embeddings["input_captions"] != list(captions):
            raise RuntimeError("Adapter encoder received different captions")

        if not isinstance(standalone_score, torch.Tensor) or standalone_score.shape != (audio.shape[0],):
            raise RuntimeError(
                "Standalone Expert1 score must have shape [B]; got "
                f"{getattr(standalone_score, 'shape', type(standalone_score).__name__)}"
            )
        expected_width = adapter.feature_dim
        if adapter_score.shape != (audio.shape[0], 1):
            raise RuntimeError(f"Adapter score must have shape [B,1]; got {tuple(adapter_score.shape)}")
        if adapter_features.shape != (audio.shape[0], expected_width):
            raise RuntimeError(
                f"Adapter features must have shape [B,{expected_width}]; got {tuple(adapter_features.shape)}"
            )

        score_metrics = _close_metrics(adapter_score.squeeze(-1), standalone_score)
        score_error = _assert_close(
            adapter_score.squeeze(-1), standalone_score,
            atol=atol, rtol=rtol, name=f"{label} score parity"
        )
        if score_error:
            case["errors"].append(score_error)

        dim = adapter.embedding_dim
        ref_audio = standalone_embeddings["audio_embedding"]
        ref_text = standalone_embeddings["text_embedding"]
        adapter_audio = adapter_embeddings["audio_embedding"]
        adapter_text = adapter_embeddings["text_embedding"]
        audio_feature = adapter_features[:, :dim]
        text_feature = adapter_features[:, dim:]

        feature_audio_metrics = _close_metrics(audio_feature, ref_audio)
        feature_text_metrics = _close_metrics(text_feature, ref_text)
        for component, actual, expected in (
            ("audio feature parity", audio_feature, ref_audio),
            ("text feature parity", text_feature, ref_text),
        ):
            error = _assert_close(actual, expected, atol=atol, rtol=rtol, name=f"{label} {component}")
            if error:
                case["errors"].append(error)

        audio_order_exact = torch.equal(audio_feature, adapter_audio)
        text_order_exact = torch.equal(text_feature, adapter_text)
        if not audio_order_exact or not text_order_exact:
            case["errors"].append("Adapter feature concatenation order/content differs from its encoder outputs")

        for name, tensor in (
            ("standalone_score", standalone_score),
            ("adapter_score", adapter_score),
            ("adapter_features", adapter_features),
            ("audio_embedding", adapter_audio),
            ("text_embedding", adapter_text),
        ):
            if not torch.isfinite(tensor).all().item():
                case["errors"].append(f"{name} contains NaN or infinity")

        audio_norms = torch.linalg.vector_norm(adapter_audio.float(), dim=-1)
        text_norms = torch.linalg.vector_norm(adapter_text.float(), dim=-1)
        unit_audio = torch.allclose(audio_norms, torch.ones_like(audio_norms), atol=1e-4, rtol=1e-4)
        unit_text = torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-4, rtol=1e-4)
        if not unit_audio or not unit_text:
            case["errors"].append("Embedding L2 norms are not within 1e-4 of one")

        case.update(
            {
                "standalone_score_shape": list(standalone_score.shape),
                "adapter_score_shape": list(adapter_score.shape),
                "feature_shape": list(adapter_features.shape),
                "audio_embedding_shape": list(adapter_audio.shape),
                "text_embedding_shape": list(adapter_text.shape),
                "audio_feature_order_exact": audio_order_exact,
                "text_feature_order_exact": text_order_exact,
                "standalone_encoder_calls": standalone_calls,
                "adapter_encoder_calls": adapter_calls,
                "score_parity": {**score_metrics, "passed": score_error is None},
                "audio_feature_parity": {
                    **feature_audio_metrics,
                    "passed": not any("audio feature parity" in e for e in case["errors"]),
                },
                "text_feature_parity": {
                    **feature_text_metrics,
                    "passed": not any("text feature parity" in e for e in case["errors"]),
                },
                "score_range": [
                    float(adapter_score.min().item()),
                    float(adapter_score.max().item()),
                ],
                "audio_embedding_norms": [float(x) for x in audio_norms.detach().cpu().tolist()],
                "text_embedding_norms": [float(x) for x in text_norms.detach().cpu().tolist()],
                "output_dtype": str(adapter_score.dtype),
                "output_device": str(adapter_score.device),
                "finite_outputs": not any("contains NaN" in e for e in case["errors"]),
            }
        )
    finally:
        handle.remove()

    case["passed"] = not case["errors"]
    summary.setdefault("cases", []).append(case)
    print(json.dumps({"case": case}, ensure_ascii=False), flush=True)
    if not case["passed"]:
        raise RuntimeError(f"Verification failed for {label}: {'; '.join(case['errors'])}")
    return case


def _read_csv_prefix(csv_path: Path, count: int) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Validation CSV has no header: {csv_path}")
        required = {"wav_file_name", "text", "average_score"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Validation CSV {csv_path} is missing required columns: {sorted(missing)}"
            )
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= count:
                break
    if not rows:
        raise ValueError(f"Validation CSV contains no sample rows: {csv_path}")
    return list(fieldnames), rows


def _load_native_validation_batch(
    csv_path: Path, audio_root: Path, count: int
) -> Tuple[torch.Tensor, List[str], List[str]]:
    if not audio_root.is_dir():
        raise NotADirectoryError(f"Expert 1 validation audio root is not a directory: {audio_root}")
    fieldnames, rows = _read_csv_prefix(csv_path, count)
    # Pass only a temporary copy of the selected metadata rows to the original
    # Dataset so its unchanged torchaudio/mono/resample/crop/pad path is used
    # without reading the full validation table or retaining generated data.
    with tempfile.TemporaryDirectory(prefix="moescore-e1-runtime-") as temporary:
        subset_csv = Path(temporary) / "validation_subset.csv"
        with subset_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        from expert1_finetune.Train_Expert1 import Expert1Dataset, collate_batch

        dataset = Expert1Dataset(subset_csv, audio_root, training=False)
        items = [dataset[index] for index in range(len(rows))]
        batch = collate_batch(items)
    sample_ids = [f"{row['wav_file_name']}#row{index}" for index, row in enumerate(rows)]
    return batch["clap_wavs"], list(batch["caption_tokens"]), sample_ids


def _norm_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def run(args: argparse.Namespace, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pretrained = _file_resource(args.pretrained_checkpoint, name="Pretrained LAION-CLAP checkpoint")
    finetuned = _file_resource(args.finetuned_checkpoint, name="Fine-tuned Expert 1 checkpoint")
    resources_before = {
        "pretrained": _resource_snapshot(pretrained),
        "finetuned": _resource_snapshot(finetuned),
    }
    device = _resolve_device(args.device)
    torch.manual_seed(2027)
    random.seed(2027)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2027)

    if summary is None:
        summary = {}
    summary.update({
        "status": "RUNNING",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "laion_clap_version": None,
        "laion_clap_source": None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_memory_before_load": _cuda_memory(device),
        "pretrained_checkpoint": resources_before["pretrained"],
        "finetuned_checkpoint": resources_before["finetuned"],
        "offline": True,
        "seed": 2027,
        "atol": args.atol,
        "rtol": args.rtol,
        "tested_batch_sizes": [],
        "cases": [],
        "real_validation": {"status": "NOT_VERIFIED"},
        "errors": [],
    })
    package_spec = importlib.util.find_spec("laion_clap")
    summary["laion_clap_source"] = package_spec.origin if package_spec else None
    try:
        summary["laion_clap_version"] = importlib.metadata.version("laion-clap")
    except importlib.metadata.PackageNotFoundError:
        summary["laion_clap_version"] = "unknown"
    summary["required_text_model"] = "roberta-base"
    summary["roberta_local_cache"] = _roberta_local_cache_status()

    if args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol) or not math.isfinite(args.rtol):
        raise ValueError("atol and rtol must be finite, non-negative values")
    paths_provided = args.validation_csv is not None or args.audio_root is not None
    if paths_provided and (args.validation_csv is None or args.audio_root is None):
        raise ValueError("--validation-csv and --audio-root must be provided together")
    if args.num_real_samples < 0:
        raise ValueError("--num-real-samples must be non-negative")
    if args.num_real_samples and not paths_provided:
        raise ValueError("--num-real-samples requires --validation-csv and --audio-root")
    validation_csv: Optional[Path] = None
    audio_root: Optional[Path] = None
    if args.num_real_samples:
        validation_csv = _file_resource(args.validation_csv, name="Expert 1 validation CSV")
        audio_root = Path(args.audio_root).expanduser()
        if not audio_root.is_absolute():
            audio_root = REPOSITORY_ROOT / audio_root
        audio_root = audio_root.resolve(strict=False)
        if not audio_root.is_dir():
            raise NotADirectoryError(
                f"Expert 1 validation audio root is not a directory: {audio_root}"
            )
    summary["stage"] = "model_construction_and_checkpoint_restoration"
    cache_status = summary["roberta_local_cache"]
    if not cache_status.get("required_for_model_construction_present", False):
        missing = []
        if not cache_status.get("config_json_present", False):
            missing.append("config.json")
        if not cache_status.get("model_weights_present", False):
            missing.append("RobertaModel weights (model.safetensors or pytorch_model.bin)")
        raise MissingLocalResourceError(
            "Expert 1's installed LAION-CLAP constructor calls "
            "RobertaModel.from_pretrained('roberta-base'), but the configured offline "
            f"cache {cache_status.get('cache_root')} is missing: {', '.join(missing)}. "
            "No download was attempted."
        )

    print("RESTORATION_START: constructing original Expert1 and strict-loading fine-tuned state", flush=True)
    captured_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            adapter = Expert1Adapter.from_finetuned(
                pretrained,
                finetuned,
                device=device,
                offline=True,
            )
    except Exception:
        loader_output = captured_stdout.getvalue().splitlines()
        if loader_output:
            summary["loader_stdout_tail"] = loader_output[-20:]
        raise
    loader_output = captured_stdout.getvalue().splitlines()
    summary["loader_stdout_line_count"] = len(loader_output)
    summary["pretrained_loader_log_evidence"] = [
        line.strip()
        for line in loader_output
        if any(token in line.lower() for token in ("load the specified checkpoint", "load checkpoint"))
    ][:5]

    model = adapter.model
    adapter.eval()
    clap_module = model.model.model
    clap_core = clap_module.model
    joint_embed_shape = int(clap_core.joint_embed_shape)
    summary.update(
        {
            "restoration": "PASS_STRICT",
            "strict_missing_keys": 0,
            "strict_unexpected_keys": 0,
            "joint_embed_shape": joint_embed_shape,
            "adapter_embedding_dim": adapter.embedding_dim,
            "adapter_feature_dim": adapter.feature_dim,
            "dimension_metadata_match": (
                adapter.embedding_dim == joint_embed_shape
                and adapter.feature_dim == 2 * joint_embed_shape
            ),
            "model_precision": str(next(model.parameters()).dtype),
            "tokenizer_name_or_path": getattr(
                getattr(clap_module, "tokenize", None), "name_or_path", None
            ),
            "tokenizer_vocab_size": getattr(
                getattr(clap_module, "tokenize", None), "vocab_size", None
            ),
            "tokenizer_model_max_length": getattr(
                getattr(clap_module, "tokenize", None), "model_max_length", None
            ),
        }
    )
    if not summary["dimension_metadata_match"]:
        raise RuntimeError(
            "Adapter dimension metadata is inconsistent with the constructed CLAP core: "
            f"joint={joint_embed_shape}, adapter_embedding={adapter.embedding_dim}, "
            f"feature={adapter.feature_dim}"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    adapter_only_trainable = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if id(parameter) not in model_parameter_ids and parameter.requires_grad
    )
    summary["parameters"] = {
        "expert_total": total_parameters,
        "expert_trainable": trainable_parameters,
        "adapter_only_trainable": adapter_only_trainable,
        "all_expert_parameters_frozen": trainable_parameters == 0,
        "adapter_has_no_extra_trainable_parameters": adapter_only_trainable == 0,
    }
    if trainable_parameters or adapter_only_trainable:
        raise RuntimeError("Expert 1 or adapter unexpectedly contains trainable parameters")

    adapter.train()
    all_child_modules_eval = all(not module.training for module in model.modules())
    remains_frozen_after_train = all(not parameter.requires_grad for parameter in adapter.parameters())
    summary["parent_train_mode_simulation"] = {
        "adapter_training_flag": bool(adapter.training),
        "all_expert_modules_eval": all_child_modules_eval,
        "all_parameters_frozen": remains_frozen_after_train,
    }
    adapter.eval()
    if not all_child_modules_eval or not remains_frozen_after_train:
        raise RuntimeError("Adapter.train() did not preserve frozen Expert 1 evaluation behavior")

    generator = torch.Generator(device="cpu").manual_seed(2027)
    synthetic_audio = (0.05 * torch.randn((2, 1, AUDIO_SAMPLES), generator=generator)).to(device)
    synthetic_captions = list(CAPTIONS)
    for batch_size in (1, 2):
        _verify_case(
            adapter=adapter,
            audio=synthetic_audio[:batch_size],
            captions=synthetic_captions[:batch_size],
            label=f"synthetic_B{batch_size}",
            atol=args.atol,
            rtol=args.rtol,
            summary=summary,
        )
        summary["tested_batch_sizes"].append(batch_size)

    # Repeated deterministic evaluation on the same B=1 synthetic input.
    with torch.no_grad():
        repeated_first = adapter(
            {"expert1_wavs": synthetic_audio[:1], "captions_raw": synthetic_captions[:1]}
        )
        repeated_second = adapter(
            {"expert1_wavs": synthetic_audio[:1], "captions_raw": synthetic_captions[:1]}
        )
    repeat_score = _close_metrics(repeated_first[0], repeated_second[0])
    repeat_features = _close_metrics(repeated_first[1], repeated_second[1])
    summary["repeated_evaluation"] = {
        "score": repeat_score,
        "features": repeat_features,
        "passed": (
            _assert_close(repeated_first[0], repeated_second[0], atol=args.atol, rtol=args.rtol,
                          name="repeated score") is None
            and _assert_close(repeated_first[1], repeated_second[1], atol=args.atol, rtol=args.rtol,
                              name="repeated features") is None
        ),
    }
    if not summary["repeated_evaluation"]["passed"]:
        raise RuntimeError("Repeated evaluation on identical input was not numerically stable")

    if args.num_real_samples:
        assert validation_csv is not None and audio_root is not None
        real_audio, real_captions, sample_ids = _load_native_validation_batch(
            validation_csv, audio_root, args.num_real_samples
        )
        real_audio = real_audio.to(device)
        summary["real_validation"]["sample_ids"] = sample_ids
        summary["real_validation"]["preprocessing"] = (
            "Expert1Dataset(training=False) + collate_batch from Train_Expert1.py; "
            "TorchAudio decode, mono mean, source-rate resample to 48 kHz, center crop / repeat-pad"
        )
        summary["real_validation"]["sample_rate_hz"] = 48_000
        summary["real_validation"]["waveform_shape"] = list(real_audio.shape)
        case = _verify_case(
            adapter=adapter,
            audio=real_audio,
            captions=real_captions,
            label="native_validation_subset",
            atol=args.atol,
            rtol=args.rtol,
            summary=summary,
        )
        summary["real_validation"].update({"status": "PASS", "case_passed": case["passed"]})

    norms_audio = [value for case in summary["cases"] for value in case.get("audio_embedding_norms", [])]
    norms_text = [value for case in summary["cases"] for value in case.get("text_embedding_norms", [])]
    all_scores = [
        score
        for case in summary["cases"]
        for score in case.get("score_range", [])
    ]
    summary["observed"] = {
        "actual_audio_embedding_shape": summary["cases"][0]["audio_embedding_shape"],
        "actual_text_embedding_shape": summary["cases"][0]["text_embedding_shape"],
        "audio_embedding_norm_min_max_mean": _norm_summary(norms_audio),
        "text_embedding_norm_min_max_mean": _norm_summary(norms_text),
        "score_min": min(all_scores),
        "score_max": max(all_scores),
        "feature_shape_examples": [case["feature_shape"] for case in summary["cases"]],
    }
    summary["cuda_memory_after_test"] = _cuda_memory(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        summary["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))

    resources_after = {
        "pretrained": _resource_snapshot(pretrained),
        "finetuned": _resource_snapshot(finetuned),
    }
    summary["resources_unchanged"] = resources_before == resources_after
    if not summary["resources_unchanged"]:
        raise RuntimeError("A checkpoint's size or modification time changed during verification")

    summary["status"] = "PASS_REAL_CHECKPOINT_AND_NUMERICAL_PARITY"
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Keep Hugging Face loading offline; this script never enables downloads.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--validation-csv")
    parser.add_argument("--audio-root")
    parser.add_argument("--num-real-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary: Dict[str, Any] = {
        "status": "FAIL",
        "requested_device": args.device,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    try:
        run(args, summary)
    except Exception as exc:
        summary.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, MissingLocalResourceError) or (
            isinstance(exc, OSError)
            and "cached files" in str(exc).lower()
            and "huggingface.co" in str(exc).lower()
        ):
            summary["status"] = "BLOCKED_BY_MISSING_LOCAL_RESOURCE"
        else:
            summary["status"] = "FAIL"
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
