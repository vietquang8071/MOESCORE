"""Portable YAML configuration loading for Set A training and inference."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml


_UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}")


def _expand(value: Any, *, field: str) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if _UNRESOLVED_ENV.search(expanded):
            raise ValueError(f"configuration field {field} contains an unset environment variable: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, field=f"{field}.{key}") for key, item in value.items()}
    return value


def _path(value: Any, *, field: str, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{field} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def load_set_a_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load YAML, expand environment paths, and validate training/model options."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read Set A YAML config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("Set A config root must be a mapping")
    config = _expand(dict(raw), field="config")
    for required in ("feature_store_root", "output_dir", "model", "training", "runtime"):
        if required not in config:
            raise ValueError(f"Set A config is missing {required}")

    base = config_path.parent
    result = dict(config)
    result["config_path"] = config_path
    result["feature_store_root"] = _path(
        config["feature_store_root"], field="feature_store_root", base=base
    )
    result["output_dir"] = _path(config["output_dir"], field="output_dir", base=base)

    manifests = config.get("source_manifest_paths", {})
    if not isinstance(manifests, Mapping) or set(manifests) - {"train", "validation", "test"}:
        raise ValueError("source_manifest_paths must map known splits to source manifest files")
    result["source_manifest_paths"] = {
        split: _path(value, field=f"source_manifest_paths.{split}", base=base)
        for split, value in manifests.items()
    }

    model = config["model"]
    if not isinstance(model, Mapping):
        raise TypeError("model must be a mapping")
    hidden_dims = model.get("hidden_dims", [256, 64])
    if (
        not isinstance(hidden_dims, list)
        or len(hidden_dims) != 2
        or any(type(value) is not int or value <= 0 for value in hidden_dims)
    ):
        raise ValueError("model.hidden_dims must contain exactly two positive integers")
    dropout = model.get("dropout", 0.2)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    result["model"] = {"hidden_dims": list(hidden_dims), "dropout": float(dropout)}

    training = config["training"]
    if not isinstance(training, Mapping):
        raise TypeError("training must be a mapping")
    epochs = _positive_int(training.get("epochs", 20), field="training.epochs")
    batch_size = _positive_int(training.get("batch_size", 8), field="training.batch_size")
    num_workers = _positive_int(training.get("num_workers", 0), field="training.num_workers", allow_zero=True)
    learning_rate = training.get("learning_rate", 1e-3)
    weight_decay = training.get("weight_decay", 1e-4)
    gradient_clip = training.get("gradient_clip", 1.0)
    seed = _positive_int(training.get("seed", 42), field="training.seed", allow_zero=True)
    holdout_fraction = training.get("holdout_fraction", 0.15)
    if (
        isinstance(holdout_fraction, bool)
        or not isinstance(holdout_fraction, (int, float))
        or not math.isfinite(holdout_fraction)
        or not 0 < holdout_fraction < 1
    ):
        raise ValueError("training.holdout_fraction must be strictly between 0 and 1")
    early_stopping_patience = training.get("early_stopping_patience", 5)
    if early_stopping_patience is not None and (
        type(early_stopping_patience) is not int or early_stopping_patience <= 0
    ):
        raise ValueError("training.early_stopping_patience must be a positive integer or null")
    min_delta = training.get("min_delta", 0.0)
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("training.min_delta must be finite and non-negative")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if gradient_clip is not None and (
        isinstance(gradient_clip, bool)
        or not isinstance(gradient_clip, (int, float))
        or gradient_clip <= 0
    ):
        raise ValueError("training.gradient_clip must be positive or null")
    result["training"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": None if gradient_clip is None else float(gradient_clip),
        "seed": seed,
        "holdout_fraction": float(holdout_fraction),
        "early_stopping_patience": early_stopping_patience,
        "min_delta": float(min_delta),
    }

    runtime = config["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    device = runtime.get("device", "auto")
    if not isinstance(device, str) or not device:
        raise ValueError("runtime.device must be a non-empty device string")
    result["runtime"] = {"device": device}

    resume = config.get("resume_from")
    result["resume_from"] = None if resume is None else _path(resume, field="resume_from", base=base)
    return result


__all__ = ["load_set_a_config"]
