#!/usr/bin/env python3
"""Verify real Expert 2/4 factory coexistence in one fresh process.

Run this script once per initialization order. It delegates model creation to
the existing adapters and original checkpoint loaders, keeps both models
alive, and writes a partial JSON report after every completed stage. Importing
this module alone does not import torch or initialize an expert.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import traceback
from typing import Any, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_ORDERS = {
    ("expert2", "expert4"): "E2_TO_E4",
    ("expert4", "expert2"): "E4_TO_E2",
}
ATOL_DEFAULT = 1e-6
RTOL_DEFAULT = 1e-5
MIB = 1024 * 1024
GIB = 1024 * MIB
EXPECTED_UNILM_COMMIT = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
EXPECTED_UNILM_SHA256 = "fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487"
HISTORICAL_SOLO_PEAK_BYTES = {
    "expert1": 726_977_536,
    "expert2": 691_505_152,
    "expert3": 839_717_376,
    "expert4": 1_067_779_584,
}


def parse_order(value: str) -> tuple[str, str]:
    """Accept exactly the two required bidirectional pairwise orders."""
    if not isinstance(value, str):
        raise ValueError("order must be a comma-separated string")
    order = tuple(part.strip() for part in value.split(",") if part.strip())
    if order not in PAIR_ORDERS:
        raise ValueError(
            "order must be exactly expert2,expert4 or expert4,expert2; "
            f"received {value!r}"
        )
    return order  # type: ignore[return-value]


def classify_failure(exc: BaseException, stage: str) -> str:
    """Map runtime exceptions to the integration failure taxonomy."""
    chain: list[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    message = " ".join(
        f"{type(item).__name__}: {item}" for item in chain
    ).lower()
    names = {type(item).__name__.lower() for item in chain}

    if any("outofmemory" in name for name in names) or "cuda out of memory" in message:
        return "CUDA_OUT_OF_MEMORY"
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return "MISSING_LOCAL_RESOURCE"
    if any(token in message for token in ("fingerprint mismatch", "identity mismatch", "source differs", "checkpoint identity")):
        return "CHECKPOINT_IDENTITY_MISMATCH"
    if any(token in message for token in ("origin mismatch", "origin conflict", "resolves outside", "namespace.*conflict")):
        return "IMPORT_ORIGIN_MISMATCH"
    if isinstance(exc, ImportError):
        return "IMPORT_COLLISION"
    lowered_stage = stage.lower()
    if "config" in lowered_stage or "configuration" in lowered_stage:
        return "CONFIGURATION_ERROR"
    if "input" in lowered_stage:
        return "INPUT_CONTRACT_MISMATCH"
    if "output" in lowered_stage:
        return "OUTPUT_CONTRACT_MISMATCH"
    if "parity" in lowered_stage or isinstance(exc, AssertionError):
        return "NUMERICAL_PARITY_FAILURE"
    if "freeze" in lowered_stage or "eval" in lowered_stage:
        return "FREEZE_OR_EVAL_FAILURE"
    if any(token in message for token in ("checkpoint", "state_dict", "state dict", "strict restoration", "load state")):
        return "CHECKPOINT_RESTORATION_ERROR"
    if "factory" in lowered_stage:
        return "FACTORY_INITIALIZATION_ERROR"
    return "UNEXPECTED_ERROR"


def _finite_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all().item())


def compare_tensors(
    actual: Any,
    expected: Any,
    *,
    atol: float = ATOL_DEFAULT,
    rtol: float = RTOL_DEFAULT,
) -> dict[str, Any]:
    """Return numerical parity metrics without hiding shape/nonfinite errors."""
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("Numerical comparison requires two torch.Tensor values")
    result: dict[str, Any] = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "shape_match": tuple(actual.shape) == tuple(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_device": str(actual.device),
        "expected_device": str(expected.device),
        "all_finite": _finite_tensor(actual) and _finite_tensor(expected),
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if not result["shape_match"] or not result["all_finite"]:
        result.update(
            {
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "max_relative_diff": None,
                "passed": False,
            }
        )
        return result
    left = actual.detach().to(device="cpu", dtype=torch.float64)
    right = expected.detach().to(device="cpu", dtype=torch.float64)
    delta = (left - right).abs()
    relative = delta / right.abs().clamp_min(1e-12)
    result.update(
        {
            "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
            "max_relative_diff": float(relative.max().item()) if relative.numel() else 0.0,
            "passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        }
    )
    return result


def _read_kib_value(path: Path, key: str) -> Optional[int]:
    try:
        for line in path.read_text().splitlines():
            if line.startswith(key + ":"):
                parts = line.split()
                return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _system_memory() -> dict[str, Optional[int]]:
    values: dict[str, Optional[int]] = {
        "total_bytes": None,
        "available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }
    try:
        entries = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            parts = raw.split()
            if parts:
                entries[key] = int(parts[0]) * 1024
        values.update(
            {
                "total_bytes": entries.get("MemTotal"),
                "available_bytes": entries.get("MemAvailable"),
                "swap_total_bytes": entries.get("SwapTotal"),
                "swap_free_bytes": entries.get("SwapFree"),
            }
        )
    except (OSError, ValueError, IndexError):
        pass
    return values


def memory_snapshot(device: str = "auto") -> dict[str, Any]:
    """Measure current process/system memory; unavailable metrics are null."""
    status = Path("/proc/self/status")
    rss = _read_kib_value(status, "VmRSS")
    hwm = _read_kib_value(status, "VmHWM")
    try:
        peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        peak_rss = peak_raw if platform.system() == "Darwin" else peak_raw * 1024
    except (AttributeError, OSError, ValueError):
        peak_rss = hwm
    system = _system_memory()

    cuda_report: dict[str, Any] = {
        "available": False,
        "device": None,
        "device_name": None,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "free_bytes": None,
        "total_bytes": None,
        "error": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            requested = device if device not in {"auto", "cpu"} else "cuda:0"
            selected = torch.device(requested)
            if selected.type == "cuda":
                torch.cuda.set_device(selected)
            torch.cuda.synchronize(selected)
            free, total = torch.cuda.mem_get_info(selected)
            cuda_report.update(
                {
                    "available": True,
                    "device": str(selected),
                    "device_name": torch.cuda.get_device_name(selected),
                    "allocated_bytes": int(torch.cuda.memory_allocated(selected)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(selected)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                }
            )
    except Exception as exc:
        cuda_report["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "cpu": {
            "process_rss_bytes": rss,
            "process_peak_rss_bytes": peak_rss,
            "process_high_water_rss_bytes": hwm,
            "system_memory_total_bytes": system["total_bytes"],
            "system_memory_available_bytes": system["available_bytes"],
            "system_swap_total_bytes": system["swap_total_bytes"],
            "system_swap_free_bytes": system["swap_free_bytes"],
        },
        "cuda": cuda_report,
    }


def reset_cuda_peak(device: str) -> None:
    import torch

    selected = torch.device(device)
    if selected.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(selected)
        torch.cuda.synchronize(selected)
        torch.cuda.reset_peak_memory_stats()


def inspect_module_origins(
    expected_roots: Mapping[str, str | os.PathLike[str]],
    *,
    modules: Optional[Mapping[str, Any]] = None,
    required_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Report module identity/origins and enforce expected namespace roots."""
    from model.expert_adapters.registry import validate_module_origins

    registry = sys.modules if modules is None else modules
    required_missing = [name for name in required_names if name not in registry]
    if required_missing:
        raise ImportError(f"Required runtime modules were not imported: {required_missing}")
    prefixes = tuple(expected_roots)
    records = {}
    for name, module in tuple(registry.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        spec = getattr(module, "__spec__", None)
        module_path = getattr(module, "__path__", None)
        records[name] = {
            "object_id": id(module),
            "file": getattr(module, "__file__", None),
            "spec_origin": getattr(spec, "origin", None),
            "path": [str(item) for item in module_path] if module_path is not None else None,
        }
    validated = validate_module_origins(expected_roots, modules=registry)
    return {"status": "PASS", "modules": records, "validated_origins": validated}


def assess_four_expert_resource_safety(
    *,
    pairwise_snapshot: Optional[Mapping[str, Any]],
    additional_gpu_peak_bytes: int,
    additional_host_bytes: int,
    safety_margin_bytes: int = GIB,
) -> dict[str, Any]:
    """Conservative pre-attempt gate; it never claims memory feasibility."""
    if pairwise_snapshot is None:
        return {
            "status": "NOT_ATTEMPTED_RESOURCE_RISK",
            "reason": "No measured pairwise process memory is available.",
        }
    cpu = pairwise_snapshot.get("cpu", {})
    cuda = pairwise_snapshot.get("cuda", {})
    free_gpu = cuda.get("free_bytes")
    available_host = cpu.get("system_memory_available_bytes")
    if free_gpu is None or available_host is None:
        return {
            "status": "NOT_ATTEMPTED_RESOURCE_RISK",
            "reason": "Pairwise GPU-free or system-RAM-available measurement is unavailable.",
            "observed_free_gpu_bytes": free_gpu,
            "observed_host_available_bytes": available_host,
        }
    required_gpu = int(additional_gpu_peak_bytes) + int(safety_margin_bytes)
    required_host = int(additional_host_bytes) + int(safety_margin_bytes)
    gpu_ok = int(free_gpu) >= required_gpu
    host_ok = int(available_host) >= required_host
    return {
        "status": "SAFE_TO_ATTEMPT" if gpu_ok and host_ok else "NOT_ATTEMPTED_RESOURCE_RISK",
        "reason": "Conservative headroom check against prior E1/E3 peaks and their checkpoint payload sizes.",
        "observed_free_gpu_bytes": int(free_gpu),
        "required_gpu_headroom_bytes": required_gpu,
        "observed_host_available_bytes": int(available_host),
        "required_host_headroom_bytes": required_host,
        "safety_margin_bytes": int(safety_margin_bytes),
        "gpu_headroom_pass": gpu_ok,
        "host_headroom_pass": host_ok,
        "does_not_establish_joint_memory_feasibility": True,
    }


class ScenarioReport:
    """Incremental JSON report that survives factory and inference failures."""

    def __init__(self, order: tuple[str, str], *, device: str, seed: int):
        self.summary: dict[str, Any] = {
            "status": "IN_PROGRESS",
            "scenario": PAIR_ORDERS[order],
            "initialization_order": list(order),
            "device_requested": device,
            "seed": int(seed),
            "precision": "torch.float32; autocast/quantization disabled",
            "tolerance": {"atol": ATOL_DEFAULT, "rtol": RTOL_DEFAULT},
            "last_completed_stage": None,
            "failed_stage": None,
            "stages": [],
            "errors": [],
        }

    def record_stage(
        self,
        name: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        memory: Optional[Mapping[str, Any]] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "name": name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
        }
        if details is not None:
            entry["details"] = dict(details)
        if memory is not None:
            entry["memory"] = dict(memory)
            self.summary["final_memory"] = dict(memory)
        self.summary["stages"].append(entry)
        self.summary["last_completed_stage"] = name

    def fail(
        self,
        stage: str,
        exc: BaseException,
        *,
        memory: Optional[Mapping[str, Any]] = None,
    ) -> None:
        category = classify_failure(exc, stage)
        self.summary.update(
            {
                "status": category,
                "failed_stage": stage,
                "failure": {
                    "category": category,
                    "exception_class": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                },
            }
        )
        if memory is not None:
            self.summary["failure"]["memory"] = dict(memory)
            self.summary["final_memory"] = dict(memory)

    def finish(self, status: str) -> None:
        self.summary["status"] = status
        self.summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    def write(self, path: str | os.PathLike[str]) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=False)
    exists = resolved.is_file() or resolved.is_dir()
    readable = exists and os.access(resolved, os.R_OK)
    record: dict[str, Any] = {
        "path": str(resolved),
        "exists": exists,
        "readable": readable,
        "kind": "directory" if resolved.is_dir() else "file" if resolved.is_file() else None,
    }
    if resolved.is_file():
        stat = resolved.stat()
        record.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return record


def _selected_resources(config: Mapping[str, Any], order: tuple[str, str]) -> dict[str, dict[str, Any]]:
    fields = {
        "expert2": ("source_dir", "architecture_yaml", "training_config", "pretrained_checkpoint", "finetuned_checkpoint"),
        "expert4": ("beats_checkpoint", "clap_checkpoint", "unilm_source_dir", "finetuned_checkpoint"),
    }
    return {
        expert: {
            field: _file_record(Path(config["experts"][expert][field]))
            for field in fields[expert]
        }
        for expert in order
    }


def _check_resource_inventory(resources: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    missing = []
    for expert, fields in resources.items():
        for name, item in fields.items():
            if not item["exists"] or not item["readable"]:
                missing.append(f"{expert}.{name}={item['path']}")
    if missing:
        raise FileNotFoundError("Missing or unreadable local resources: " + "; ".join(missing))


def _verify_unilm_identity(source_dir: Path) -> dict[str, Any]:
    from expert4_finetune.trainer import _source_identity

    observed = _source_identity(source_dir)
    expected = {"git_commit": EXPECTED_UNILM_COMMIT, "sha256": EXPECTED_UNILM_SHA256}
    if observed != expected:
        raise ValueError(f"UniLM source identity mismatch: expected={expected}, observed={observed}")
    return {"status": "PASS", "path": str(source_dir), "expected": expected, "observed": observed}


def _module_record(module: Any) -> dict[str, Any]:
    spec = getattr(module, "__spec__", None)
    module_path = getattr(module, "__path__", None)
    return {
        "object_id": id(module),
        "file": getattr(module, "__file__", None),
        "spec_origin": getattr(spec, "origin", None),
        "path": [str(item) for item in module_path] if module_path is not None else None,
    }


def _namespace_modules(prefixes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        name: _module_record(module)
        for name, module in tuple(sys.modules.items())
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }


def _select_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {selected} was requested but CUDA is unavailable")
    if selected.type == "cuda":
        index = torch.cuda.current_device() if selected.index is None else selected.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
    return selected


def _seed_everything(seed: int, device: Any) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats()


def _make_inputs(expert: str, batch_size: int, *, seed: int, device: Any, mixed_lengths: bool = False):
    import torch

    if expert == "expert2":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        wavs = (0.05 * torch.randn((batch_size, 1, 320_000), generator=generator, dtype=torch.float32)).to(device)
        captions = [f"  A rhythmic percussion recording for sample {index}.  " for index in range(batch_size)]
        return {"expert2_wavs": wavs, "captions_raw": captions}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    wavs = torch.randn((batch_size, 1, 160_000), generator=generator, dtype=torch.float32)
    if mixed_lengths:
        lengths_values = [80_000, 160_000]
        if batch_size != 2:
            raise ValueError("Expert 4 mixed-length verification requires batch_size=2")
    else:
        lengths_values = [160_000] * batch_size
    lengths = torch.tensor(lengths_values, dtype=torch.long)
    for index, valid in enumerate(lengths_values):
        if valid < wavs.shape[-1]:
            wavs[index, :, valid:] = 0.0
    captions = [f"  A sound event caption for sample {index}.  " for index in range(batch_size)]
    return {
        "expert4_wavs": wavs.to(device),
        "expert4_wav_lens": lengths.to(device),
        "captions_raw": captions,
    }


def _original_batch(expert: str, unified: Mapping[str, Any]) -> dict[str, Any]:
    if expert == "expert2":
        return {
            "mga_wavs": unified["expert2_wavs"],
            "caption_tokens": [caption.strip() for caption in unified["captions_raw"]],
        }
    return {
        "wavs": unified["expert4_wavs"],
        "wav_lens": unified["expert4_wav_lens"],
        "captions": [caption.strip() for caption in unified["captions_raw"]],
    }


def _clone_outputs(output: Any) -> tuple[Any, Any]:
    import torch

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Expert adapter/original forward must return a (score, features) tuple")
    score, features = output
    if not isinstance(score, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise TypeError("Expert score and features must both be tensors")
    return score.detach().to("cpu").clone(), features.detach().to("cpu").clone()


def _adapter_output(adapter: Any, inputs: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    with torch.no_grad():
        return _clone_outputs(adapter(inputs))


def _standalone_output(adapter: Any, expert: str, inputs: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    with torch.no_grad():
        return _clone_outputs(adapter.model(_original_batch(expert, inputs)))


def _parity_metrics(
    adapter: Any,
    expert: str,
    inputs: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    model = adapter.model
    counter = {"model_forward": 0}

    def count_forward(_module, _inputs, _output):
        counter["model_forward"] += 1

    handle = model.register_forward_hook(count_forward)
    try:
        standalone = _standalone_output(adapter, expert, inputs)
        counter["model_forward"] = 0
        adapted = _adapter_output(adapter, inputs)
        adapter_forward_count = counter["model_forward"]
    finally:
        handle.remove()
    expected_score = standalone[0]
    actual_score = adapted[0]
    if expert == "expert4" and actual_score.ndim == 2 and expected_score.ndim == 1:
        actual_score = actual_score.squeeze(-1)
    score_metrics = compare_tensors(actual_score, expected_score, atol=atol, rtol=rtol)
    feature_metrics = compare_tensors(adapted[1], standalone[1], atol=atol, rtol=rtol)
    if not score_metrics["passed"] or not feature_metrics["passed"] or adapter_forward_count != 1:
        raise AssertionError(
            f"{expert} standalone/adapter parity failed: score={score_metrics}, "
            f"features={feature_metrics}, adapter_model_forward_count={adapter_forward_count}"
        )
    if not _finite_tensor(adapted[0]) or not _finite_tensor(adapted[1]):
        raise ValueError(f"{expert} produced non-finite adapter outputs")
    return {
        "expert": expert,
        "input_shapes": {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        },
        "valid_lengths": inputs.get("expert4_wav_lens").detach().cpu().tolist()
        if expert == "expert4"
        else None,
        "standalone_score_shape": list(standalone[0].shape),
        "adapter_score_shape": list(adapted[0].shape),
        "feature_shape": list(adapted[1].shape),
        "score_values": standalone[0].reshape(-1).tolist(),
        "standalone_adapter_score": score_metrics,
        "standalone_adapter_features": feature_metrics,
        "adapter_original_forward_calls": adapter_forward_count,
        "all_outputs_finite": True,
        "passed": True,
    }


def _adapter_state(adapter: Any) -> dict[str, Any]:
    model = adapter.model
    all_frozen = all(not parameter.requires_grad for parameter in model.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    adapter_only_trainable = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if id(parameter) not in model_parameter_ids and parameter.requires_grad
    )
    adapter.train()
    modules_eval_after_train = all(not module.training for module in model.modules())
    frozen_after_train = all(not parameter.requires_grad for parameter in model.parameters())
    adapter.eval()
    if not all_frozen or trainable_parameters or adapter_only_trainable or not modules_eval_after_train or not frozen_after_train:
        raise RuntimeError("Expert adapter violated frozen/eval behavior after parent train()")
    return {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "feature_dim": int(adapter.feature_dim),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "adapter_only_trainable_parameters": int(adapter_only_trainable),
        "all_parameters_frozen": all_frozen,
        "all_model_modules_eval_after_adapter_train": modules_eval_after_train,
        "frozen_after_adapter_train": frozen_after_train,
        "model_device": str(next(model.parameters()).device),
    }


def _parameter_bytes(adapter: Any) -> int:
    seen = set()
    result = 0
    for parameter in adapter.model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result += parameter.numel() * parameter.element_size()
    for buffer in adapter.model.buffers():
        if id(buffer) in seen:
            continue
        seen.add(id(buffer))
        result += buffer.numel() * buffer.element_size()
    return result


def _mga_namespace_report(source_dir: Path) -> dict[str, Any]:
    from model.expert_adapters.expert2_adapter import validate_mga_import_namespace

    validate_mga_import_namespace(source_dir)
    modules = _namespace_modules(("models", "tools"))
    if "models" not in modules or "models.ase_model" not in modules:
        raise ImportError("MGA factory did not leave expected models namespace modules loaded")
    for name, record in modules.items():
        origin = record.get("file") or record.get("spec_origin")
        if origin is None or not Path(origin).resolve(strict=False).is_relative_to(source_dir.resolve()):
            raise ImportError(f"MGA module {name} has unexpected origin: {record}")
    return {"status": "PASS", "modules": modules, "objects": {name: item["object_id"] for name, item in modules.items()}}


def _expert4_private_module_report(adapter: Any, unilm_source: Path) -> dict[str, Any]:
    from model.expert_adapters.expert4_adapter import validate_expert4_module_origins

    validate_expert4_module_origins(unilm_source, require_beats=False)
    beats_instance = adapter.model.audio_encoder.model
    qualified = type(beats_instance).__module__
    private_root = qualified.partition(".")[0]
    expected_dir = unilm_source if (unilm_source / "BEATs.py").is_file() else unilm_source / "beats"
    expected_files = {
        f"{private_root}.BEATs": expected_dir / "BEATs.py",
        f"{private_root}.backbone": expected_dir / "backbone.py",
        f"{private_root}.modules": expected_dir / "modules.py",
    }
    records = {}
    for name, expected in expected_files.items():
        module = sys.modules.get(name)
        if module is None:
            raise ImportError(f"Expert 4 private BEATs module is missing after construction: {name}")
        record = _module_record(module)
        origin = record["file"] or record["spec_origin"]
        if origin is None or Path(origin).resolve(strict=False) != expected.resolve(strict=False):
            raise ImportError(f"Expert 4 private module {name} expected {expected}, observed {record}")
        records[name] = record
    clap = sys.modules.get("laion_clap")
    clap_record = _module_record(clap) if clap is not None else None
    return {
        "status": "PASS",
        "private_package": private_root,
        "beats_instance_class_module": qualified,
        "private_module_records": records,
        "laion_clap": clap_record,
    }


def _assert_module_snapshot_unchanged(before: Mapping[str, Any], after: Mapping[str, Any], label: str) -> None:
    if before != after:
        raise ImportError(f"{label} module objects/origins changed after the other factory: before={before}, after={after}")


def _run_mixed_e4(adapter: Any, *, seed: int, atol: float, rtol: float, device: Any) -> dict[str, Any]:
    import torch

    from scripts.verify_expert4_runtime import ForwardProbe, make_synthetic, run_parity_case

    wavs, lengths = make_synthetic(2, [80_000, 160_000], seed=seed, device=device)
    captions = ["A hand drum plays a short rhythm.", "A bird sings beside a flowing river."]
    probe = ForwardProbe(adapter.model)
    try:
        result = run_parity_case(
            adapter.model,
            adapter,
            probe,
            label="pairwise_mixed_length_B2",
            wavs=wavs,
            lengths=lengths,
            captions=captions,
            atol=atol,
            rtol=rtol,
        )
    finally:
        probe.close()
    result["status"] = "PASS"
    result["lengths_unchanged"] = result["valid_lengths"] == [80_000, 160_000]
    result["padding_masks_and_invocation"] = result["adapter_masking_and_counts"]
    if not result["lengths_unchanged"]:
        raise AssertionError(f"Expert 4 valid lengths changed: {result['valid_lengths']}")
    return result


def _run_scenario(args: argparse.Namespace, report: ScenarioReport) -> dict[str, Any]:
    import torch

    from configs.moe_config import resolve_moe_paths
    from model.expert_adapters.registry import create_expert_adapter

    device = _select_device(args.device)
    _seed_everything(args.seed, device)
    report.summary["device"] = str(device)
    report.summary["environment"] = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "offline": True,
        "offline_environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }
    memory = memory_snapshot(str(device))
    report.record_stage("PROCESS_START", details={"working_directory": str(Path.cwd())}, memory=memory)
    report.write(args.report_json)

    current_stage = "configuration_preflight"
    raw_config_path = Path(args.config).expanduser()
    if not raw_config_path.is_absolute():
        raw_config_path = REPO_ROOT / raw_config_path
    from configs.moe_config import load_moe_config

    raw_config = load_moe_config(raw_config_path)
    resolved = resolve_moe_paths(
        raw_config,
        raw_config_path,
        project_root=REPO_ROOT,
        data_root=REPO_ROOT,
        resource_root=REPO_ROOT,
        environ={},
    )
    resolved["runtime"]["device"] = str(device)
    resolved["runtime"]["offline"] = True
    if args.unilm_source_dir:
        unilm = Path(args.unilm_source_dir).expanduser()
        if not unilm.is_absolute():
            unilm = REPO_ROOT / unilm
        unilm = unilm.resolve(strict=True)
        resolved["experts"]["expert4"]["unilm_source_dir"] = unilm
    else:
        unilm = Path(resolved["experts"]["expert4"]["unilm_source_dir"])
    resources = _selected_resources(resolved, report.summary["initialization_order"])
    _check_resource_inventory(resources)
    report.summary["configuration"] = {
        "path": str(raw_config_path.resolve(strict=True)),
        "schema_version": raw_config["schema_version"],
        "expert_order": list(raw_config["gate"]["expert_order"]),
        "resolved_resources": resources,
    }
    current_stage = "unilm_source_identity"
    source_identity = _verify_unilm_identity(unilm)
    report.summary["expert4_unilm_identity"] = source_identity
    first_name, second_name = report.summary["initialization_order"]
    source_dir = Path(resolved["experts"]["expert2"]["source_dir"])
    first_module_snapshot: dict[str, Any] = {}
    second_import_aliases_before: dict[str, Any] = {}
    second_import_aliases_after: dict[str, Any] = {}
    second_private_snapshot: dict[str, Any] = {}
    report.record_stage(
        "RESOURCE_PREFLIGHT",
        details={
            "status": "PASS",
            "selected_experts": [first_name, second_name],
            "resources": resources,
            "unilm_identity": source_identity,
        },
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)

    first_inputs = _make_inputs(first_name, args.batch_size, seed=args.seed + 10, device=device)
    second_inputs = _make_inputs(second_name, args.batch_size, seed=args.seed + 20, device=device)
    report.summary["input_contracts"] = {
        "expert2": {"audio_shape": list(first_inputs["expert2_wavs"].shape) if first_name == "expert2" else None, "sample_rate_hz": 32000, "caption_policy": "strip"},
        "expert4": {
            "audio_shape": list(first_inputs["expert4_wavs"].shape) if first_name == "expert4" else list(second_inputs["expert4_wavs"].shape) if second_name == "expert4" else None,
            "sample_rate_hz": 16000,
            "valid_lengths_key": "expert4_wav_lens",
            "length_dtype": "torch.int64",
            "caption_policy": "strip",
        },
        "preprocessing": "Deterministic FP32 waveform tensors at each source adapter boundary; no resampling/cropping/padding in the integration verifier, except explicit right-zero-padding for E4 mixed-length case.",
    }

    current_stage = "first_factory"
    reset_cuda_peak(str(device))
    report.summary["current_stage"] = current_stage
    report.record_stage(
        "BEFORE_FIRST_FACTORY",
        details={"expert": first_name},
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)
    first_adapter = create_expert_adapter(first_name, resolved, device=device)
    first_state = _adapter_state(first_adapter)
    first_param_bytes = _parameter_bytes(first_adapter)
    report.summary["experts"] = {
        first_name: {
            "factory_status": "PASS",
            "factory_api": "existing registry -> adapter.from_config -> original pretrained/fine-tuned loader",
            "model": first_state,
            "parameter_and_buffer_bytes": first_param_bytes,
        }
    }
    if first_name == "expert2":
        from model.expert_adapters.expert2_adapter import validate_mga_import_namespace

        validate_mga_import_namespace(source_dir)
        first_module_snapshot = _namespace_modules(("models", "tools"))
        report.summary["experts"][first_name]["restoration"] = {
            "pretrained_mga": "PASS_ORIGINAL_FACTORY_RETURNED",
            "finetuned_checkpoint": "PASS_STRICT_LOAD_RETURNED",
            "pretrained_loader": "create_expert2 -> checkpoint['model'] -> load_mga_state(strict=True)",
        }
        report.summary["experts"][first_name]["module_origins"] = _mga_namespace_report(source_dir)
    else:
        aliases = ("BEATs", "backbone", "modules")
        second_import_aliases_before = {name: sys.modules.get(name) for name in aliases}
        report.summary["experts"][first_name]["restoration"] = {
            "pretrained_beats": "PASS_ORIGINAL_STRICT_LOADER_RETURNED",
            "pretrained_clap": "PASS_ORIGINAL_TEXT_PROJECTION_LOADER_RETURNED",
            "finetuned_checkpoint": "PASS_ORIGINAL_PARTIAL_LOADER_AND_TENSOR_VALIDATION_RETURNED",
        }
        report.summary["experts"][first_name]["module_origins"] = _expert4_private_module_report(first_adapter, unilm)
        first_private_names = tuple(report.summary["experts"][first_name]["module_origins"]["private_module_records"])
        second_private_snapshot = {name: _module_record(sys.modules[name]) for name in first_private_names}
        report.summary["experts"][first_name]["beats_alias_cleanup"] = {
            "preexisting_alias_objects": {name: id(module) if module is not None else None for name, module in second_import_aliases_before.items()},
            "post_factory_aliases": {name: id(sys.modules.get(name)) if sys.modules.get(name) is not None else None for name in aliases},
            "top_level_aliases_restored": all(sys.modules.get(name) is prior for name, prior in second_import_aliases_before.items()),
            "constructed_private_modules_remain_loaded": True,
        }
        if not report.summary["experts"][first_name]["beats_alias_cleanup"]["top_level_aliases_restored"]:
            raise ImportError("BEATs temporary aliases were not restored after first factory")

    report.record_stage(
        "AFTER_FIRST_FACTORY",
        details={
            "expert": first_name,
            "model_class": first_state["model_class"],
            "feature_dim": first_state["feature_dim"],
            "parameters_frozen": first_state["all_parameters_frozen"],
        },
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)

    first_origin_snapshot = (
        _namespace_modules(("models", "tools")) if first_name == "expert2" else dict(second_private_snapshot)
    )
    first_clap_obj = sys.modules.get("laion_clap")
    first_baseline = _adapter_output(first_adapter, first_inputs)
    if not _finite_tensor(first_baseline[0]) or not _finite_tensor(first_baseline[1]):
        raise ValueError(f"{first_name} baseline inference produced non-finite tensors")
    report.summary["experts"][first_name]["baseline_inference"] = {
        "score_shape": list(first_baseline[0].shape),
        "feature_shape": list(first_baseline[1].shape),
        "score_values": first_baseline[0].reshape(-1).tolist(),
        "device": str(first_baseline[0].device),
        "dtype": str(first_baseline[0].dtype),
    }
    current_memory = memory_snapshot(str(device))
    report.record_stage(
        "AFTER_FIRST_INFERENCE",
        details={"first_expert": first_name, "outputs_finite": True},
        memory=current_memory,
    )
    report.write(args.report_json)

    current_stage = "second_factory"
    report.summary["current_stage"] = current_stage
    second_import_aliases_before = {name: sys.modules.get(name) for name in ("BEATs", "backbone", "modules")}
    reset_cuda_peak(str(device))
    report.record_stage(
        "BEFORE_SECOND_FACTORY",
        details={"expert": second_name, "first_expert_remains_instantiated": True},
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)
    second_adapter = create_expert_adapter(second_name, resolved, device=device)
    second_state = _adapter_state(second_adapter)
    second_param_bytes = _parameter_bytes(second_adapter)
    report.summary["experts"][second_name] = {
        "factory_status": "PASS",
        "factory_api": "existing registry -> adapter.from_config -> original pretrained/fine-tuned loader",
        "model": second_state,
        "parameter_and_buffer_bytes": second_param_bytes,
    }
    second_import_aliases_after = {name: sys.modules.get(name) for name in second_import_aliases_before}
    if second_name == "expert2":
        from model.expert_adapters.expert2_adapter import validate_mga_import_namespace

        validate_mga_import_namespace(source_dir)
        second_module_snapshot = _namespace_modules(("models", "tools"))
        report.summary["experts"][second_name]["module_origins"] = _mga_namespace_report(source_dir)
        report.summary["experts"][second_name]["restoration"] = {
            "pretrained_mga": "PASS_ORIGINAL_FACTORY_RETURNED",
            "finetuned_checkpoint": "PASS_STRICT_LOAD_RETURNED",
            "pretrained_loader": "create_expert2 -> checkpoint['model'] -> load_mga_state(strict=True)",
        }
    else:
        second_module_snapshot = _expert4_private_module_report(second_adapter, unilm)["private_module_records"]
        report.summary["experts"][second_name]["module_origins"] = _expert4_private_module_report(second_adapter, unilm)
        report.summary["experts"][second_name]["restoration"] = {
            "pretrained_beats": "PASS_ORIGINAL_STRICT_LOADER_RETURNED",
            "pretrained_clap": "PASS_ORIGINAL_TEXT_PROJECTION_LOADER_RETURNED",
            "finetuned_checkpoint": "PASS_ORIGINAL_PARTIAL_LOADER_AND_TENSOR_VALIDATION_RETURNED",
        }
        cleanup = all(second_import_aliases_after[name] is second_import_aliases_before[name] for name in second_import_aliases_before)
        report.summary["experts"][second_name]["beats_alias_cleanup"] = {
            "top_level_aliases_restored": cleanup,
            "preexisting_alias_objects": {name: id(module) if module is not None else None for name, module in second_import_aliases_before.items()},
            "post_factory_aliases": {name: id(module) if module is not None else None for name, module in second_import_aliases_after.items()},
            "private_modules": second_module_snapshot,
        }
        if not cleanup:
            raise ImportError("BEATs temporary aliases were not restored after second factory")

    if first_name == "expert2":
        post_first_modules = _namespace_modules(("models", "tools"))
        _assert_module_snapshot_unchanged(first_origin_snapshot, post_first_modules, "MGA")
        from model.expert_adapters.expert4_adapter import validate_expert4_module_origins

        validate_expert4_module_origins(unilm, require_beats=False)
        coexistence_origins = {
            "mga_module_objects_unchanged": True,
            "mga_modules": post_first_modules,
            "beats_top_level_aliases_restored": all(sys.modules.get(name) is second_import_aliases_before[name] for name in second_import_aliases_before),
            "beats_private_modules_retained": second_module_snapshot,
            "laion_clap_loaded": "laion_clap" in sys.modules,
        }
    else:
        current_private = {name: _module_record(sys.modules[name]) for name in second_private_snapshot}
        _assert_module_snapshot_unchanged(second_private_snapshot, current_private, "Expert 4 private BEATs")
        from model.expert_adapters.expert4_adapter import validate_expert4_module_origins

        validate_expert4_module_origins(unilm, require_beats=False)
        if first_clap_obj is not sys.modules.get("laion_clap"):
            raise ImportError("Expert 2 factory replaced the pre-existing LAION-CLAP module object")
        coexistence_origins = {
            "beats_private_module_objects_unchanged": True,
            "beats_private_modules": current_private,
            "beats_top_level_aliases_restored": all(sys.modules.get(name) is second_import_aliases_before[name] for name in second_import_aliases_before),
            "laion_clap_object_unchanged": True,
            "mga_modules": _mga_namespace_report(source_dir),
        }
    report.summary["coexistence_module_checks"] = {"status": "PASS", **coexistence_origins}
    if first_name == "expert2":
        report.summary["experts"][first_name]["module_objects_unchanged_after_second_factory"] = True
    else:
        report.summary["experts"][first_name]["module_objects_unchanged_after_second_factory"] = True
    report.record_stage(
        "AFTER_SECOND_FACTORY",
        details={
            "first_model_still_alive": first_adapter.model is not None,
            "second_model_created": second_adapter.model is not None,
            "coexistence_module_checks": "PASS",
        },
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)

    current_stage = "before_after_inference_parity"
    report.summary["current_stage"] = current_stage
    first_after = _adapter_output(first_adapter, first_inputs)
    before_after_score = compare_tensors(first_after[0], first_baseline[0], atol=args.atol, rtol=args.rtol)
    before_after_features = compare_tensors(first_after[1], first_baseline[1], atol=args.atol, rtol=args.rtol)
    if not before_after_score["passed"] or not before_after_features["passed"]:
        raise AssertionError(f"{first_name} outputs changed after constructing {second_name}: {before_after_score}, {before_after_features}")
    report.summary["experts"][first_name]["before_after_second_factory_parity"] = {
        "score": before_after_score,
        "features": before_after_features,
        "passed": True,
    }
    report.record_stage(
        "AFTER_FIRST_EXPERT_REINFERENCE",
        details={
            "expert": first_name,
            "score_max_abs_diff": before_after_score["max_abs_diff"],
            "feature_max_abs_diff": before_after_features["max_abs_diff"],
            "passed": True,
        },
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)

    current_stage = "standalone_adapter_parity"
    report.summary["current_stage"] = current_stage
    parity_first = _parity_metrics(first_adapter, first_name, first_inputs, atol=args.atol, rtol=args.rtol)
    parity_second = _parity_metrics(second_adapter, second_name, second_inputs, atol=args.atol, rtol=args.rtol)
    report.summary["experts"][first_name]["standalone_adapter_parity"] = parity_first
    report.summary["experts"][second_name]["standalone_adapter_parity"] = parity_second
    report.summary["experts"][second_name]["inference_after_both_factories"] = {
        "score_shape": parity_second["adapter_score_shape"],
        "feature_shape": parity_second["feature_shape"],
        "score_values": parity_second["score_values"],
        "finite": parity_second["all_outputs_finite"],
    }
    report.record_stage(
        "AFTER_SECOND_EXPERT_INFERENCE",
        details={
            "expert": second_name,
            "standalone_adapter_score_max_abs_diff": parity_second["standalone_adapter_score"]["max_abs_diff"],
            "standalone_adapter_feature_max_abs_diff": parity_second["standalone_adapter_features"]["max_abs_diff"],
            "passed": True,
        },
        memory=memory_snapshot(str(device)),
    )
    report.write(args.report_json)

    current_stage = "expert4_mixed_length_mask_parity"
    report.summary["current_stage"] = current_stage
    mixed_result: dict[str, Any]
    if device.type == "cuda" and memory_snapshot(str(device))["cuda"].get("free_bytes") is not None and memory_snapshot(str(device))["cuda"]["free_bytes"] < 512 * MIB:
        mixed_result = {
            "status": "NOT_VERIFIED_RESOURCE_LIMIT",
            "reason": "Less than 512 MiB CUDA headroom remained after both model factories; skipped B=2 mixed-length inference to avoid an OOM.",
            "memory": memory_snapshot(str(device)),
        }
    else:
        e4_adapter = first_adapter if first_name == "expert4" else second_adapter
        mixed_result = _run_mixed_e4(e4_adapter, seed=args.seed + 70, atol=args.atol, rtol=args.rtol, device=device)
    report.summary["expert4_mixed_length_masking"] = mixed_result
    if mixed_result["status"] == "NOT_VERIFIED_RESOURCE_LIMIT":
        report.summary["mixed_length_status"] = mixed_result["status"]

    current_stage = "repeated_eval_and_freeze"
    report.summary["current_stage"] = current_stage
    freeze_state = {}
    for name, adapter in ((first_name, first_adapter), (second_name, second_adapter)):
        before = _adapter_output(adapter, first_inputs if name == first_name else second_inputs)
        repeated = _adapter_output(adapter, first_inputs if name == first_name else second_inputs)
        repeated_score = compare_tensors(before[0], repeated[0], atol=args.atol, rtol=args.rtol)
        repeated_features = compare_tensors(before[1], repeated[1], atol=args.atol, rtol=args.rtol)
        if not repeated_score["passed"] or not repeated_features["passed"]:
            raise AssertionError(f"{name} repeated eval output changed")
        freeze_state[name] = {
            "model_state": _adapter_state(adapter),
            "repeated_score": repeated_score,
            "repeated_features": repeated_features,
        }
    report.summary["freeze_eval_determinism"] = freeze_state

    current_stage = "four_expert_resource_assessment"
    report.summary["current_stage"] = current_stage
    all_raw = load_moe_config_for_assessment(args.config)
    all_resolved = resolve_moe_paths(
        all_raw,
        Path(args.config).resolve(),
        project_root=REPO_ROOT,
        data_root=REPO_ROOT,
        resource_root=REPO_ROOT,
        environ={},
    )
    extra_host_bytes = 0
    for expert, field in (
        ("expert1", "pretrained_checkpoint"),
        ("expert1", "finetuned_checkpoint"),
        ("expert3", "base_checkpoint"),
        ("expert3", "finetuned_checkpoint"),
    ):
        extra_host_bytes += Path(all_resolved["experts"][expert][field]).stat().st_size
    extra_gpu_peak = HISTORICAL_SOLO_PEAK_BYTES["expert1"] + HISTORICAL_SOLO_PEAK_BYTES["expert3"]
    full_risk = assess_four_expert_resource_safety(
        pairwise_snapshot=memory_snapshot(str(device)),
        additional_gpu_peak_bytes=extra_gpu_peak,
        additional_host_bytes=extra_host_bytes,
    )
    full_risk.update(
        {
            "estimated_additional_gpu_peak_from_independent_runs_bytes": extra_gpu_peak,
            "additional_checkpoint_payload_bytes": extra_host_bytes,
            "attempted": False,
            "status": "NOT_ATTEMPTED_RESOURCE_RISK" if full_risk["status"] != "SAFE_TO_ATTEMPT" else "SAFE_TO_ATTEMPT_REQUIRES_FRESH_PROCESS",
        }
    )
    report.summary["four_expert_coexistence"] = full_risk
    checkpoints_before = {
        f"{expert}.{field}": item
        for expert, fields in resources.items()
        for field, item in fields.items()
        if item["kind"] == "file"
    }
    checkpoints_after = {
        f"{expert}.{field}": _file_record(Path(item["path"]))
        for expert, fields in resources.items()
        for field, item in fields.items()
        if item["kind"] == "file"
    }
    immutable = all(
        checkpoints_before[name]["size_bytes"] == checkpoints_after[name]["size_bytes"]
        and checkpoints_before[name]["mtime_ns"] == checkpoints_after[name]["mtime_ns"]
        for name in checkpoints_before
    )
    report.summary["checkpoint_file_metadata_unchanged"] = {
        "passed": immutable,
        "before": checkpoints_before,
        "after": checkpoints_after,
    }
    if not immutable:
        raise RuntimeError("A selected checkpoint/config file changed during multi-expert verification")
    report.record_stage(
        "SCENARIO_COMPLETE",
        details={
            "both_real_factories_succeeded": True,
            "both_instances_remain_alive": True,
            "before_after_parity": "PASS",
            "standalone_adapter_parity": "PASS",
            "expert4_mixed_length_status": mixed_result["status"],
            "four_expert_attempted": False,
        },
        memory=memory_snapshot(str(device)),
    )
    report.summary["restoration"] = {
        "expert2": "PASS" if "expert2" in report.summary["experts"] else "NOT_IN_SCENARIO",
        "expert4": "PASS" if "expert4" in report.summary["experts"] else "NOT_IN_SCENARIO",
        "expert4_unilm_identity": source_identity["status"],
    }
    report.finish("PASS — REAL PAIRWISE COEXISTENCE & PARITY VERIFIED")
    return report.summary


def load_moe_config_for_assessment(config_path: str):
    from configs.moe_config import load_moe_config

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return load_moe_config(path.resolve(strict=True))


def assess_pairwise_reports(paths: list[str], output_path: str) -> dict[str, Any]:
    """Combine two fresh-process snapshots for a conservative four-model gate."""
    if len(paths) != 2:
        raise ValueError("Exactly two pairwise scenario JSON reports are required")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    expected = {"E2_TO_E4", "E4_TO_E2"}
    observed = {item.get("scenario") for item in reports}
    if observed != expected:
        raise ValueError(f"Both completed pairwise orders are required; observed={observed}")
    if any(item.get("status") != "PASS — REAL PAIRWISE COEXISTENCE & PARITY VERIFIED" for item in reports):
        raise ValueError("Four-expert resource assessment requires both pairwise scenarios to pass")
    final_memory = [item.get("final_memory", {}) for item in reports]
    free_values = [item.get("cuda", {}).get("free_bytes") for item in final_memory]
    host_values = [item.get("cpu", {}).get("system_memory_available_bytes") for item in final_memory]
    if any(value is None for value in free_values + host_values):
        pairwise_snapshot = None
        available_free = None
        available_host = None
    else:
        available_free = min(int(value) for value in free_values)
        available_host = min(int(value) for value in host_values)
        pairwise_snapshot = {
            "cuda": {"free_bytes": available_free},
            "cpu": {"system_memory_available_bytes": available_host},
        }
    extra_host_bytes = sum(
        int(report.get("four_expert_coexistence", {}).get("additional_checkpoint_payload_bytes", 0))
        for report in reports
    ) // 2
    extra_gpu_peak = HISTORICAL_SOLO_PEAK_BYTES["expert1"] + HISTORICAL_SOLO_PEAK_BYTES["expert3"]
    decision = assess_four_expert_resource_safety(
        pairwise_snapshot=pairwise_snapshot,
        additional_gpu_peak_bytes=extra_gpu_peak,
        additional_host_bytes=extra_host_bytes,
    )
    result = {
        "status": "NOT_ATTEMPTED_RESOURCE_RISK" if decision["status"] != "SAFE_TO_ATTEMPT" else "SAFE_TO_ATTEMPT_REQUIRES_FRESH_PROCESS",
        "scenario_reports": [str(Path(path).resolve()) for path in paths],
        "pairwise_orders": sorted(observed),
        "measured_pairwise_free_vram_bytes": free_values,
        "measured_pairwise_host_available_bytes": host_values,
        "additional_gpu_peak_reference_bytes": extra_gpu_peak,
        "additional_checkpoint_payload_bytes": extra_host_bytes,
        "safety_decision": decision,
        "attempted": False,
        "reason": "This assessment does not construct models; it decides whether a separate fresh-process four-expert run is safe to attempt.",
    }
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/moe_local.example.yaml")
    parser.add_argument("--unilm-source-dir")
    parser.add_argument("--order", help="expert2,expert4 or expert4,expert2")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--offline", action="store_true", help="required; all model/cache operations remain offline")
    parser.add_argument("--report-json")
    parser.add_argument("--assess-four-expert-resources", nargs=2, metavar=("E2_E4_JSON", "E4_E2_JSON"))
    parser.add_argument("--four-expert-report-json", default="/tmp/moescore_local081_four_experts.json")
    args = parser.parse_args(argv)
    if args.assess_four_expert_resources:
        if not args.report_json:
            parser.error("--report-json is required with --assess-four-expert-resources")
        return args
    if not args.offline:
        parser.error("--offline is required; network access is not supported")
    if not args.order:
        parser.error("--order is required for a pairwise scenario")
    try:
        args.order_tuple = parse_order(args.order)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.report_json:
        parser.error("--report-json is required")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if "expert4" in args.order_tuple and not args.unilm_source_dir:
        parser.error("--unilm-source-dir is required for any scenario containing Expert 4")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.assess_four_expert_resources:
        try:
            result = assess_pairwise_reports(args.assess_four_expert_resources, args.report_json)
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 0
        except Exception as exc:
            result = {
                "status": classify_failure(exc, "four_expert_resource_assessment"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            Path(args.report_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(args.report_json).expanduser().write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
            return 1

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    report = ScenarioReport(args.order_tuple, device=args.device, seed=args.seed)
    report.summary["tolerance"] = {"atol": args.atol, "rtol": args.rtol}
    report_path = Path(args.report_json).expanduser()
    current_stage = "startup"
    try:
        # Store stage in the summary before performing any expensive operation.
        report.summary["current_stage"] = current_stage
        _run_scenario(args, report)
        report.write(report_path)
        print(json.dumps(report.summary, indent=2, ensure_ascii=False, default=str), flush=True)
        return 0
    except BaseException as exc:
        current_stage = report.summary.get("current_stage", current_stage)
        try:
            snapshot = memory_snapshot(report.summary.get("device", args.device))
        except Exception:
            snapshot = None
        report.fail(current_stage, exc, memory=snapshot)
        report.write(report_path)
        print(json.dumps(report.summary, indent=2, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
