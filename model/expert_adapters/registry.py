"""Lazy registry and output-contract helpers for the four MOESCORE experts.

This module does not construct models or load checkpoints. Factory lookup
imports only the requested lightweight adapter module; its ``from_config``
method remains responsible for restoring that expert's original model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter, validate_expert_output


EXPERT_ORDER = ("expert1", "expert2", "expert3", "expert4")
EXPERT_FEATURE_DIMS = MappingProxyType(
    {"expert1": 1024, "expert2": 2048, "expert3": 3840, "expert4": 1024}
)
EXPECTED_FEATURE_WIDTH = sum(EXPERT_FEATURE_DIMS.values())


@dataclass(frozen=True)
class ExpertInputContract:
    """Source-grounded tensor boundary metadata; preprocessing stays per expert."""

    audio_key: str
    audio_shape: tuple[Optional[int], ...]
    sample_rate_hz: int
    max_samples: int
    audio_dtype: str = "torch.float32"
    caption_key: str = "captions_raw"
    caption_policy: str = "strip"
    lengths_key: Optional[str] = None
    lengths_dtype: Optional[str] = None


EXPERT_INPUT_CONTRACTS = MappingProxyType(
    {
        "expert1": ExpertInputContract(
            audio_key="expert1_wavs",
            audio_shape=(None, 1, 480_000),
            sample_rate_hz=48_000,
            max_samples=480_000,
            caption_policy="preserve",
        ),
        "expert2": ExpertInputContract(
            audio_key="expert2_wavs",
            audio_shape=(None, 1, 320_000),
            sample_rate_hz=32_000,
            max_samples=320_000,
        ),
        "expert3": ExpertInputContract(
            audio_key="expert3_wavs",
            audio_shape=(None, 1, 160_000),
            sample_rate_hz=16_000,
            max_samples=160_000,
        ),
        "expert4": ExpertInputContract(
            audio_key="expert4_wavs",
            audio_shape=(None, 1, 160_000),
            sample_rate_hz=16_000,
            max_samples=160_000,
            lengths_key="expert4_wav_lens",
            lengths_dtype="torch.int64",
        ),
    }
)


@dataclass(frozen=True)
class ExpertRegistration:
    adapter_module: str
    adapter_class: str
    feature_dim: int


EXPERT_REGISTRY = MappingProxyType(
    {
        "expert1": ExpertRegistration(
            "model.expert_adapters.expert1_adapter", "Expert1Adapter", 1024
        ),
        "expert2": ExpertRegistration(
            "model.expert_adapters.expert2_adapter", "Expert2Adapter", 2048
        ),
        "expert3": ExpertRegistration(
            "model.expert_adapters.expert3_adapter", "Expert3Adapter", 3840
        ),
        "expert4": ExpertRegistration(
            "model.expert_adapters.expert4_adapter", "Expert4Adapter", 1024
        ),
    }
)


def _registration(expert_name: str) -> ExpertRegistration:
    try:
        return EXPERT_REGISTRY[expert_name]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Unknown expert {expert_name!r}; expected one of {EXPERT_ORDER}"
        ) from exc


def resolve_expert_adapter_class(expert_name: str) -> type[ExpertAdapter]:
    """Import and return one adapter class without constructing its model."""
    registration = _registration(expert_name)
    module = importlib.import_module(registration.adapter_module)
    adapter_class = getattr(module, registration.adapter_class, None)
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, ExpertAdapter):
        raise TypeError(
            f"Factory registration for {expert_name} must resolve to an ExpertAdapter subclass; "
            f"got {adapter_class!r}"
        )
    factory = getattr(adapter_class, "from_config", None)
    if not callable(factory):
        raise TypeError(f"Registered {expert_name} adapter has no callable from_config factory")
    return adapter_class


def _scoped_factory_config(config: Mapping[str, Any], expert_name: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("Resolved MOESCORE configuration must be a mapping")
    if "experts" not in config:
        if not isinstance(config, Mapping):
            raise TypeError(f"{expert_name} factory configuration must be a mapping")
        return config
    experts = config.get("experts")
    if not isinstance(experts, Mapping) or expert_name not in experts:
        raise ValueError(f"Resolved configuration is missing experts.{expert_name}")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise TypeError("Resolved MOESCORE runtime configuration must be a mapping")
    section = experts[expert_name]
    if not isinstance(section, Mapping):
        raise TypeError(f"experts.{expert_name} must be a mapping")
    # Preserve the established adapter from_config API while preventing one
    # expert factory from inspecting or requiring other experts' resources.
    return {"experts": {expert_name: section}, "runtime": runtime}


def create_expert_adapter(
    expert_name: str,
    resolved_config: Mapping[str, Any],
    *,
    device: Optional[str | torch.device] = None,
) -> ExpertAdapter:
    """Create exactly one requested expert through its existing config factory."""
    adapter_class = resolve_expert_adapter_class(expert_name)
    factory_config = _scoped_factory_config(resolved_config, expert_name)
    kwargs = {} if device is None else {"device": device}
    adapter = adapter_class.from_config(factory_config, **kwargs)
    if not isinstance(adapter, ExpertAdapter):
        raise TypeError(
            f"{expert_name} from_config returned {type(adapter).__name__}, not ExpertAdapter"
        )
    expected_dim = _registration(expert_name).feature_dim
    if adapter.feature_dim != expected_dim:
        raise ValueError(
            f"{expert_name} runtime feature_dim {adapter.feature_dim} does not match "
            f"the verified integration contract {expected_dim}"
        )
    return adapter


def _module_origins(module: Any) -> tuple[Path, ...]:
    origins: list[Path] = []
    filename = getattr(module, "__file__", None)
    if filename:
        origins.append(Path(filename).expanduser().resolve(strict=False))
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and spec_origin not in {"built-in", "frozen"}:
        origins.append(Path(spec_origin).expanduser().resolve(strict=False))
    for package_path in getattr(module, "__path__", ()):
        origins.append(Path(package_path).expanduser().resolve(strict=False))
    return tuple(dict.fromkeys(origins))


def validate_module_origins(
    expected_roots: Mapping[str, str | os.PathLike[str]],
    *,
    modules: Optional[Mapping[str, Any]] = None,
) -> dict[str, tuple[str, ...]]:
    """Validate loaded namespace origins without changing ``sys.modules``.

    Each key is a top-level module name such as ``models`` or ``BEATs`` and
    each value is its expected source directory. Loaded modules with that name
    or prefix must have every inspectable origin beneath the expected root.
    """
    if not isinstance(expected_roots, Mapping) or not expected_roots:
        raise TypeError("expected_roots must be a non-empty module-name to source-root mapping")
    registry = sys.modules if modules is None else modules
    roots: dict[str, Path] = {}
    for name, value in expected_roots.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(f"Expected namespace key must be a top-level module name; got {name!r}")
        root = Path(value).expanduser()
        if not root.is_absolute():
            raise ValueError(f"Expected origin root for {name!r} must be absolute; got {root}")
        roots[name] = root.resolve(strict=False)

    observed: dict[str, tuple[str, ...]] = {}
    for namespace, expected_root in roots.items():
        for module_name, module in tuple(registry.items()):
            if module_name != namespace and not module_name.startswith(namespace + "."):
                continue
            origins = _module_origins(module)
            if not origins:
                raise ImportError(
                    f"Module namespace {namespace!r} entry {module_name!r} has no inspectable origin; "
                    f"expected root={expected_root}"
                )
            conflicts = tuple(origin for origin in origins if not origin.is_relative_to(expected_root))
            if conflicts:
                raise ImportError(
                    f"Module namespace {namespace!r} entry {module_name!r} origin conflict: "
                    f"expected root={expected_root}; observed={list(origins)}"
                )
            observed[module_name] = tuple(str(origin) for origin in origins)
    return observed


def validate_and_assemble_expert_outputs(
    expert_outputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: Optional[int] = None,
    feature_dims: Optional[Mapping[str, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate four adapter output pairs and concatenate in ``EXPERT_ORDER``.

    This function only checks and concatenates outputs. It does not score,
    average, normalize, vote, or apply gating weights.
    """
    if not isinstance(expert_outputs, Mapping):
        raise TypeError("expert_outputs must be a mapping keyed by expert name")
    actual_names, expected_names = set(expert_outputs), set(EXPERT_ORDER)
    missing, unexpected = expected_names - actual_names, actual_names - expected_names
    if missing or unexpected:
        raise ValueError(
            f"Expert output registry mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    dimensions = EXPERT_FEATURE_DIMS if feature_dims is None else feature_dims
    if not isinstance(dimensions, Mapping) or set(dimensions) != expected_names:
        raise ValueError(f"feature_dims must contain exactly {EXPERT_ORDER}")
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError(f"batch_size must be a positive integer or None; got {batch_size!r}")

    scores: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    expected_device: Optional[torch.device] = None
    expected_dtype: Optional[torch.dtype] = None
    resolved_batch_size = batch_size

    # Check cross-expert placement before the shared validator scans tensor
    # values. Besides producing a clearer integration error, this avoids
    # invoking value operations on a tensor from a different device when the
    # placement contract is already violated.
    for name in EXPERT_ORDER:
        pair = expert_outputs[name]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{name} output must be a (score, features) tuple")
        score, feature = pair
        if not isinstance(score, torch.Tensor) or not isinstance(feature, torch.Tensor):
            continue  # validate_expert_output supplies the shared type diagnostic.
        if score.device != feature.device:
            raise ValueError(
                f"{name} score and features must share a device; got {score.device} and {feature.device}"
            )
        if score.dtype != feature.dtype:
            raise TypeError(f"{name} score/features dtype mismatch: {score.dtype} != {feature.dtype}")
        if expected_device is None:
            expected_device, expected_dtype = score.device, score.dtype
        elif score.device != expected_device:
            raise ValueError(
                f"{name} outputs are on {score.device}; expected the shared expert device {expected_device}"
            )
        elif score.dtype != expected_dtype:
            raise TypeError(
                f"{name} outputs use dtype {score.dtype}; expected the shared expert dtype {expected_dtype}"
            )

    expected_device = None
    expected_dtype = None
    for name in EXPERT_ORDER:
        pair = expert_outputs[name]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{name} output must be a (score, features) tuple")
        score, feature = validate_expert_output(
            pair[0], pair[1], feature_dim=dimensions[name], expert_name=name
        )
        if score.dtype != feature.dtype:
            raise TypeError(f"{name} score/features dtype mismatch: {score.dtype} != {feature.dtype}")
        if resolved_batch_size is None:
            resolved_batch_size = int(score.shape[0])
        if score.shape[0] != resolved_batch_size:
            raise ValueError(
                f"{name} output batch size {score.shape[0]} differs from expected {resolved_batch_size}"
            )
        if expected_device is None:
            expected_device, expected_dtype = score.device, score.dtype
        if score.device != expected_device or feature.device != expected_device:
            raise ValueError(
                f"{name} outputs are on {score.device}/{feature.device}; expected {expected_device}"
            )
        if score.dtype != expected_dtype or feature.dtype != expected_dtype:
            raise TypeError(
                f"{name} outputs use dtype {score.dtype}/{feature.dtype}; expected {expected_dtype}"
            )
        scores.append(score)
        features.append(feature)

    return torch.cat(scores, dim=1), torch.cat(features, dim=1)


__all__ = [
    "EXPERT_FEATURE_DIMS",
    "EXPERT_INPUT_CONTRACTS",
    "EXPERT_ORDER",
    "EXPERT_REGISTRY",
    "EXPECTED_FEATURE_WIDTH",
    "ExpertInputContract",
    "ExpertRegistration",
    "create_expert_adapter",
    "resolve_expert_adapter_class",
    "validate_and_assemble_expert_outputs",
    "validate_module_origins",
]
