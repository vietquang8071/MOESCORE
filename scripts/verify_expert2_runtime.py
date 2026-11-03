#!/usr/bin/env python3
"""Restore Expert 2 offline and verify standalone/adapter inference parity.

This is an explicit heavyweight verification command. It constructs exactly
one original MGA/ASE model, restores the real fine-tuned state through
``Expert2Adapter.from_finetuned()``, and compares two inference interfaces on
the same model instance. It never trains, downloads, or changes checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Set offline policy before importing Transformers or invoking the original
# MGA factory. The adapter also scopes these flags around model construction.
for _offline_key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_offline_key] = "1"

import torch

from model.expert_adapters.expert2_adapter import (
    Expert2Adapter,
    validate_mga_import_namespace,
)


AUDIO_SAMPLES = 32_000 * 10
CAPTIONS = (
    "A dog barking in a quiet room.",
    "Rain falling on a metal roof.",
)


class MissingLocalResourceError(RuntimeError):
    """A required local checkpoint or Hugging Face cache resource is absent."""


def _path(value: str, *, name: str, kind: str) -> Path:
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = REPOSITORY_ROOT / result
    result = result.resolve(strict=False)
    valid = result.is_dir() if kind == "directory" else result.is_file()
    if not valid:
        expected = "directory" if kind == "directory" else "file"
        raise FileNotFoundError(f"Expert 2 {name} is not a {expected}: {result}")
    if not os.access(result, os.R_OK):
        raise PermissionError(f"Expert 2 {name} is not readable: {result}")
    return result


def _snapshot(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {result}, but CUDA is unavailable")
        index = result.index if result.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {result}, but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
    return result


def _cuda_memory(device: torch.device) -> Optional[Dict[str, int]]:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _bert_offline_check() -> Dict[str, Any]:
    """Exercise tokenizer/config/model local-only loading before MGA build."""
    try:
        import transformers
        from transformers import BertConfig, BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        config = BertConfig.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        model = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        result = {
            "status": "PASS",
            "model_id": "bert-base-uncased",
            "transformers_version": transformers.__version__,
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "hidden_size": int(config.hidden_size),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "local_files_only": True,
        }
        del model, tokenizer, config
        gc.collect()
        return result
    except Exception as exc:
        raise MissingLocalResourceError(
            "Offline bert-base-uncased tokenizer/config/model initialization failed; "
            "no download was attempted. Check the local Hugging Face cache and repository ID."
        ) from exc


def _module_description(module: Any) -> Dict[str, Any]:
    spec = getattr(module, "__spec__", None)
    module_path = getattr(module, "__path__", None)
    return {
        "file": getattr(module, "__file__", None),
        "spec_origin": getattr(spec, "origin", None),
        "path": [str(path) for path in module_path] if module_path is not None else None,
    }


def _mga_namespace_report(source_dir: Path) -> Dict[str, Any]:
    """Report relevant actual vendor module origins and reject wrong roots."""
    names = ["models", "models.ase_model", "tools"]
    names.extend(
        name for name in sys.modules
        if name.startswith("models.") or name.startswith("tools.")
    )
    names = list(dict.fromkeys(names))
    report: Dict[str, Any] = {}
    source = source_dir.resolve()
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            report[name] = {"status": "NOT_IMPORTED"}
            continue
        details = _module_description(module)
        origins: List[Path] = []
        if details["file"]:
            origins.append(Path(details["file"]).resolve(strict=False))
        if details["spec_origin"] not in (None, "built-in", "frozen"):
            origins.append(Path(details["spec_origin"]).resolve(strict=False))
        if details["path"]:
            origins.extend(Path(item).resolve(strict=False) for item in details["path"])
        unexpected = [str(origin) for origin in origins if not origin.is_relative_to(source)]
        details["status"] = "PASS" if origins and not unexpected else "FAIL"
        details["conflicting_origins"] = unexpected
        report[name] = details
        if unexpected:
            raise RuntimeError(
                f"MGA vendor module {name!r} resolved outside {source}: {unexpected}"
            )
    # Reuse the production guard, which checks every loaded models/tools module.
    validate_mga_import_namespace(source)
    if report.get("models.ase_model", {}).get("status") != "PASS":
        raise RuntimeError("MGA factory did not leave models.ase_model imported from its source tree")
    return report


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "max_relative_diff": None,
            "passed": False,
        }
    delta = (actual.detach().float() - expected.detach().float()).abs()
    relative = delta / expected.detach().float().abs().clamp_min(1e-12)
    return {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_relative_diff": float(relative.max().item()) if relative.numel() else 0.0,
    }


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)


def _norms(value: torch.Tensor) -> Dict[str, float]:
    norms = torch.linalg.vector_norm(value.detach().float(), dim=-1)
    return {
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
        "mean": float(norms.mean().item()),
    }


def _capture_mga_encoder(captures: List[Dict[str, Any]], inputs: tuple, output: Any) -> None:
    if len(inputs) != 2 or not isinstance(output, (tuple, list)) or len(output) != 2:
        captures.append({"invalid_contract": True})
        return
    audio, captions = inputs
    audio_embed, text_embed = output
    captures.append(
        {
            "input_audio": audio.detach().clone(),
            "input_captions": list(captions),
            "audio_embedding": audio_embed.detach().clone(),
            "text_embedding": text_embed.detach().clone(),
        }
    )


def _verify_case(
    *,
    adapter: Expert2Adapter,
    audio: torch.Tensor,
    captions: Sequence[str],
    label: str,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    model = adapter.model
    encoder = model.model
    ase = encoder.model
    captures: List[Dict[str, Any]] = []
    counters = {"expert2": 0, "mga_encoder": 0, "audio_encoder": 0, "text_encoder": 0}

    def expert_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        counters["expert2"] += 1

    def encoder_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        counters["mga_encoder"] += 1
        _capture_mga_encoder(captures, inputs, output)

    def audio_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        counters["audio_encoder"] += 1

    def text_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        counters["text_encoder"] += 1

    hooks = [
        model.register_forward_hook(expert_hook),
        encoder.register_forward_hook(encoder_hook),
        ase.audio_encoder.register_forward_hook(audio_hook),
        ase.text_encoder.register_forward_hook(text_hook),
    ]
    batch_size = int(audio.shape[0])
    case: Dict[str, Any] = {
        "label": label,
        "batch_size": batch_size,
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "audio_device": str(audio.device),
        "caption_count": len(captions),
        "captions_preserved": True,
        "atol": atol,
        "rtol": rtol,
        "errors": [],
    }
    try:
        with torch.no_grad():
            standalone_score, standalone_features = model(
                {"mga_wavs": audio, "caption_tokens": list(captions)}
            )
        standalone_counts = dict(counters)
        if len(captures) != 1 or captures[-1].get("invalid_contract"):
            raise RuntimeError(f"Standalone MGAEncoder hook expected one embedding tuple; got {len(captures)}")
        standalone_embedding = captures[-1]
        if not torch.equal(standalone_embedding["input_audio"], audio):
            raise RuntimeError("Standalone Expert2 encoder received a changed waveform")
        if standalone_embedding["input_captions"] != list(captions):
            raise RuntimeError("Standalone Expert2 encoder received changed captions")

        captures.clear()
        for key in counters:
            counters[key] = 0
        with torch.no_grad():
            adapter_score, adapter_features = adapter(
                {"expert2_wavs": audio, "captions_raw": list(captions)}
            )
        adapter_counts = dict(counters)
        if len(captures) != 1 or captures[-1].get("invalid_contract"):
            raise RuntimeError(f"Adapter expected one MGAEncoder embedding tuple; got {len(captures)}")
        adapter_embedding = captures[-1]
        if not torch.equal(adapter_embedding["input_audio"], audio):
            raise RuntimeError("Adapter changed the input waveform before the original encoder")
        if adapter_embedding["input_captions"] != [caption.strip() for caption in captions]:
            raise RuntimeError("Adapter caption whitespace behavior differs from standalone policy")

        if standalone_score.shape != (batch_size, 1):
            raise RuntimeError(f"Standalone score shape expected {(batch_size, 1)}, got {tuple(standalone_score.shape)}")
        if adapter_score.shape != (batch_size, 1):
            raise RuntimeError(f"Adapter score shape expected {(batch_size, 1)}, got {tuple(adapter_score.shape)}")
        if standalone_features.ndim != 2 or adapter_features.ndim != 2:
            raise RuntimeError("Standalone/adapter feature output must be rank 2")
        if adapter_features.shape != (batch_size, adapter.feature_dim):
            raise RuntimeError(
                f"Adapter feature shape expected {(batch_size, adapter.feature_dim)}, got {tuple(adapter_features.shape)}"
            )

        for name, observed in (("standalone", standalone_counts), ("adapter", adapter_counts)):
            expected = {"expert2": 1, "mga_encoder": 1, "audio_encoder": 1, "text_encoder": 1}
            if observed != expected:
                raise RuntimeError(f"{name} forward invocation counts {observed}, expected {expected}")

        audio_embedding = adapter_embedding["audio_embedding"]
        text_embedding = adapter_embedding["text_embedding"]
        ref_audio = standalone_embedding["audio_embedding"]
        ref_text = standalone_embedding["text_embedding"]
        if audio_embedding.ndim != 2 or text_embedding.ndim != 2:
            raise RuntimeError("Runtime MSC embeddings must both be rank 2")
        audio_dim = int(audio_embedding.shape[-1])
        text_dim = int(text_embedding.shape[-1])
        if standalone_features.shape[1] != audio_dim + text_dim:
            raise RuntimeError("Standalone feature width does not equal audio MSC + text MSC width")
        if adapter.feature_dim != audio_dim + text_dim:
            raise RuntimeError(
                f"Adapter feature_dim={adapter.feature_dim} differs from runtime dimensions {audio_dim}+{text_dim}"
            )

        reference_formula = (audio_embedding * text_embedding).sum(dim=-1, keepdim=True)
        score_metrics = _metrics(adapter_score, standalone_score)
        feature_metrics = _metrics(adapter_features, standalone_features)
        audio_metrics = _metrics(audio_embedding, ref_audio)
        text_metrics = _metrics(text_embedding, ref_text)
        _assert_close(adapter_score, standalone_score, atol=atol, rtol=rtol, name=f"{label} score parity")
        _assert_close(adapter_features, standalone_features, atol=atol, rtol=rtol, name=f"{label} feature parity")
        _assert_close(standalone_score, reference_formula, atol=atol, rtol=rtol, name=f"{label} original dot-product formula")
        _assert_close(standalone_features[:, :audio_dim], ref_audio, atol=atol, rtol=rtol, name=f"{label} audio feature ordering")
        _assert_close(standalone_features[:, audio_dim:], ref_text, atol=atol, rtol=rtol, name=f"{label} text feature ordering")
        _assert_close(adapter_features[:, :audio_dim], audio_embedding, atol=atol, rtol=rtol, name=f"{label} adapter audio ordering")
        _assert_close(adapter_features[:, audio_dim:], text_embedding, atol=atol, rtol=rtol, name=f"{label} adapter text ordering")
        for label_tensor, tensor in (
            ("standalone score", standalone_score),
            ("adapter score", adapter_score),
            ("standalone features", standalone_features),
            ("adapter features", adapter_features),
        ):
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(f"{label_tensor} contains NaN or infinity")
        if adapter_score.device != audio.device or adapter_features.device != audio.device:
            raise RuntimeError("Expert 2 outputs are not on the input/model device")
        if adapter_score.dtype != torch.float32 or adapter_features.dtype != torch.float32:
            raise RuntimeError("Expert 2 outputs are not float32; no implicit precision changes are allowed")

        case.update({
            "standalone_score_shape": list(standalone_score.shape),
            "adapter_score_shape": list(adapter_score.shape),
            "standalone_feature_shape": list(standalone_features.shape),
            "adapter_feature_shape": list(adapter_features.shape),
            "audio_embedding_shape": list(audio_embedding.shape),
            "text_embedding_shape": list(text_embedding.shape),
            "audio_feature_dim": audio_dim,
            "text_feature_dim": text_dim,
            "feature_dim": adapter.feature_dim,
            "score_parity": {**score_metrics, "passed": True},
            "feature_parity": {**feature_metrics, "passed": True},
            "audio_embedding_parity": {**audio_metrics, "passed": True},
            "text_embedding_parity": {**text_metrics, "passed": True},
            "feature_order": "audio_MSC_then_text_MSC",
            "standalone_invocation_counts": standalone_counts,
            "adapter_invocation_counts": adapter_counts,
            "audio_embedding_norms": _norms(audio_embedding),
            "text_embedding_norms": _norms(text_embedding),
            "standalone_score_values": [float(value) for value in standalone_score.detach().cpu().flatten().tolist()],
            "adapter_score_values": [float(value) for value in adapter_score.detach().cpu().flatten().tolist()],
            "score_range": [float(adapter_score.min().item()), float(adapter_score.max().item())],
            "output_dtype": str(adapter_score.dtype),
            "output_device": str(adapter_score.device),
            "finite_outputs": True,
            "passed": True,
        })
        return case
    finally:
        for hook in hooks:
            hook.remove()


def _real_validation_batch(csv_path: Path, audio_root: Path, count: int):
    from expert2_finetune.expert2_data import Expert2Dataset, collate_expert2

    dataset = Expert2Dataset(csv_path, audio_root)
    selected = [dataset[index] for index in range(min(count, len(dataset)))]
    if not selected:
        raise ValueError("Expert 2 validation dataset contains no requested samples")
    batch = collate_expert2(selected)
    sample_ids = list(batch["wav_file_name"])
    return batch["mga_wavs"], list(batch["caption_tokens"]), sample_ids


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", default="weights/mga-clap.pt")
    parser.add_argument("--finetuned-checkpoint", default="weights/finetuned_weights/expert2_best.pt")
    parser.add_argument("--mga-source-dir", default="expert2_finetune/MGA-CLAP")
    parser.add_argument("--architecture-yaml", default="expert2_finetune/MGA-CLAP/settings/inference_example.yaml")
    parser.add_argument("--training-config", default="expert2_finetune/expert2_finetune.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--offline", action="store_true", default=True, help="Offline is mandatory; no downloads are attempted")
    parser.add_argument("--validation-csv")
    parser.add_argument("--audio-root")
    parser.add_argument("--num-real-samples", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    if not args.offline:
        raise ValueError("Offline mode is mandatory for Expert 2 runtime verification")
    if not math.isfinite(args.atol) or args.atol < 0 or not math.isfinite(args.rtol) or args.rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    if args.num_real_samples < 0:
        raise ValueError("num-real-samples must be non-negative")
    if (args.validation_csv is None) != (args.audio_root is None):
        raise ValueError("--validation-csv and --audio-root must be provided together")
    if args.num_real_samples and not args.validation_csv:
        raise ValueError("--num-real-samples requires validation CSV and audio root")

    summary["stage"] = "resource_preflight"
    source = _path(args.mga_source_dir, name="MGA source directory", kind="directory")
    architecture = _path(args.architecture_yaml, name="architecture YAML", kind="file")
    training = _path(args.training_config, name="training config", kind="file")
    pretrained = _path(args.pretrained_checkpoint, name="pretrained MGA checkpoint", kind="file")
    finetuned = _path(args.finetuned_checkpoint, name="fine-tuned Expert 2 checkpoint", kind="file")
    if not (source / "models" / "ase_model.py").is_file():
        raise FileNotFoundError(f"MGA ASE source is missing: {source / 'models' / 'ase_model.py'}")
    resources_before = {
        "pretrained": _snapshot(pretrained),
        "finetuned": _snapshot(finetuned),
        "architecture_yaml": _snapshot(architecture),
        "training_config": _snapshot(training),
    }
    validation_csv = audio_root = None
    if args.num_real_samples:
        validation_csv = _path(args.validation_csv, name="validation CSV", kind="file")
        audio_root = _path(args.audio_root, name="validation audio root", kind="directory")

    device = _device(args.device)
    random.seed(2027)
    torch.manual_seed(2027)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2027)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        # This installed PyTorch build rejects a torch.device argument here;
        # reset the current device after selecting the requested CUDA device.
        torch.cuda.reset_peak_memory_stats()
    summary.update({
        "status": "RUNNING",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "transformers": None,
        "device_requested": args.device,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_memory_before_load": _cuda_memory(device),
        "precision": "torch.float32",
        "offline": True,
        "seed": 2027,
        "atol": args.atol,
        "rtol": args.rtol,
        "resources": resources_before,
        "cases": [],
        "real_validation": {"status": "NOT_VERIFIED"},
        "errors": [],
    })
    try:
        summary["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        summary["transformers"] = "unknown"
    summary["mga_source"] = str(source)
    summary["architecture_yaml"] = str(architecture)
    summary["training_config"] = str(training)
    summary["required_text_model"] = "bert-base-uncased"

    summary["stage"] = "bert_offline_cache_check"
    summary["bert_offline"] = _bert_offline_check()

    summary["stage"] = "mga_model_construction_and_checkpoint_restoration"
    print("RESTORATION_START: original MGA factory, pretrained checkpoint, strict fine-tuned restore", flush=True)
    loader_output = io.StringIO()
    with contextlib.redirect_stdout(loader_output):
        adapter = Expert2Adapter.from_finetuned(
            source_dir=source,
            architecture_yaml=architecture,
            training_config=training,
            pretrained_checkpoint=pretrained,
            finetuned_checkpoint=finetuned,
            device=device,
            offline=True,
        )
    loader_lines = loader_output.getvalue().splitlines()
    summary["loader_stdout_line_count"] = len(loader_lines)
    summary["pretrained_loader_log_evidence"] = [
        line.strip() for line in loader_lines
        if any(token in line.lower() for token in ("load checkpoint", "pretrained", "mga"))
    ][:10]
    model = adapter.model
    adapter.eval()
    namespace = _mga_namespace_report(source)
    summary["mga_namespace_modules"] = namespace
    summary["restoration"] = {
        "mga_factory": "PASS",
        "pretrained_mga": "PASS_ORIGINAL_LOADER_RETURNED",
        "finetuned_expert2": "PASS_STRICT",
        "strict": True,
        "strict_missing_keys": 0,
        "strict_unexpected_keys": 0,
        "shape_mismatches": 0,
        "fine_tuned_format": "raw full Expert2.state_dict (strict load succeeded)",
        "pretrained_loader": "create_expert2 -> checkpoint['model'] -> load_mga_state(strict=True)",
        "position_ids_compatibility": "handled by original load_mga_state source path",
    }

    summary["stage"] = "runtime_dimension_and_freeze_checks"
    ase = model.model.model
    audio_width = int(ase.frame_proj[-1].out_features)
    text_width = int(ase.word_proj[-1].out_features)
    codebook_width = int(ase.codebook.shape[-1])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    adapter_only_trainable = sum(
        parameter.numel() for parameter in adapter.parameters()
        if id(parameter) not in model_parameter_ids and parameter.requires_grad
    )
    summary["dimensions_from_model_metadata"] = {
        "audio_projection_width": audio_width,
        "text_projection_width": text_width,
        "shared_msc_codebook_width": codebook_width,
        "adapter_feature_dim": adapter.feature_dim,
    }
    if audio_width != codebook_width or text_width != codebook_width:
        raise RuntimeError("MGA audio/text projection widths differ from the shared MSC codebook")
    if adapter.feature_dim != audio_width + text_width:
        raise RuntimeError("Adapter feature_dim does not equal actual configured MSC modality widths")
    summary["parameters"] = {
        "expert_total": total_parameters,
        "expert_trainable": trainable_parameters,
        "adapter_only_trainable": adapter_only_trainable,
        "all_expert_parameters_frozen": trainable_parameters == 0,
        "adapter_has_no_extra_trainable_parameters": adapter_only_trainable == 0,
    }
    if trainable_parameters or adapter_only_trainable:
        raise RuntimeError("Expert 2 or adapter unexpectedly has trainable parameters")

    adapter.train()
    all_modules_eval = all(not module.training for module in model.modules())
    remains_frozen = all(not parameter.requires_grad for parameter in adapter.parameters())
    summary["parent_train_mode_simulation"] = {
        "adapter_training_flag": bool(adapter.training),
        "all_expert_modules_eval": all_modules_eval,
        "all_parameters_frozen": remains_frozen,
    }
    if not all_modules_eval or not remains_frozen:
        raise RuntimeError("adapter.train() reactivated Expert 2 or enabled parameters")
    adapter.eval()

    summary["stage"] = "synthetic_numerical_parity"
    generator = torch.Generator(device="cpu").manual_seed(2027)
    synthetic_audio = (0.05 * torch.randn((2, 1, AUDIO_SAMPLES), generator=generator)).to(device)
    for batch_size in (1, 2):
        case = _verify_case(
            adapter=adapter,
            audio=synthetic_audio[:batch_size],
            captions=CAPTIONS[:batch_size],
            label=f"synthetic_B{batch_size}",
            atol=args.atol,
            rtol=args.rtol,
        )
        summary["cases"].append(case)
        print(json.dumps({"case": case}, ensure_ascii=False), flush=True)

    summary["stage"] = "repeated_evaluation"
    with torch.no_grad():
        repeat_a = adapter({"expert2_wavs": synthetic_audio[:1], "captions_raw": CAPTIONS[:1]})
        repeat_b = adapter({"expert2_wavs": synthetic_audio[:1], "captions_raw": CAPTIONS[:1]})
    repeat_score = _metrics(repeat_a[0], repeat_b[0])
    repeat_features = _metrics(repeat_a[1], repeat_b[1])
    _assert_close(repeat_a[0], repeat_b[0], atol=args.atol, rtol=args.rtol, name="repeated score determinism")
    _assert_close(repeat_a[1], repeat_b[1], atol=args.atol, rtol=args.rtol, name="repeated feature determinism")
    summary["repeated_evaluation"] = {
        "score": {**repeat_score, "passed": True},
        "features": {**repeat_features, "passed": True},
        "passed": True,
    }

    if args.num_real_samples:
        summary["stage"] = "native_validation_inference"
        assert validation_csv is not None and audio_root is not None
        real_audio, real_captions, sample_ids = _real_validation_batch(
            validation_csv, audio_root, args.num_real_samples
        )
        real_audio = real_audio.to(device)
        case = _verify_case(
            adapter=adapter,
            audio=real_audio,
            captions=real_captions,
            label="native_validation_subset",
            atol=args.atol,
            rtol=args.rtol,
        )
        case["sample_ids"] = sample_ids
        summary["cases"].append(case)
        summary["real_validation"] = {
            "status": "PASS",
            "sample_ids": sample_ids,
            "preprocessing": "Expert2Dataset + collate_expert2: SciPy decode, PCM scaling, mono mean, resample_poly to 32 kHz, first 10 s, right zero-pad, caption.strip()",
            "sample_rate_hz": 32000,
            "waveform_shape": list(real_audio.shape),
            "case_passed": True,
        }
        print(json.dumps({"case": case}, ensure_ascii=False), flush=True)

    summary["stage"] = "resource_immutability_and_memory"
    resources_after = {
        "pretrained": _snapshot(pretrained),
        "finetuned": _snapshot(finetuned),
        "architecture_yaml": _snapshot(architecture),
        "training_config": _snapshot(training),
    }
    summary["resources_unchanged"] = resources_before == resources_after
    if not summary["resources_unchanged"]:
        raise RuntimeError("A checkpoint or config file changed during runtime verification")
    summary["cuda_memory_after_test"] = _cuda_memory(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        summary["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    summary["tested_batch_sizes"] = [1, 2]
    summary["status"] = "PASS_REAL_CHECKPOINT_AND_NUMERICAL_PARITY"
    if args.num_real_samples:
        summary["status"] = "PASS_INCLUDING_REAL_VALIDATION_INFERENCE"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary: Dict[str, Any] = {
        "status": "FAIL",
        "stage": "argument_validation",
        "device_requested": args.device,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    try:
        run(args, summary)
    except Exception as exc:
        summary["errors"] = [f"{type(exc).__name__}: {exc}"]
        summary["traceback"] = traceback.format_exc()
        summary["status"] = (
            "BLOCKED_MISSING_RESOURCE_OR_ENVIRONMENT"
            if isinstance(exc, (MissingLocalResourceError, FileNotFoundError, PermissionError))
            else "FAIL_RESTORATION_OR_NUMERICAL_PARITY"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
