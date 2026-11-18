"""Validated portable E6 configuration."""

from pathlib import Path
import math
import os

import yaml

from training.set_a_config import _expand, _path


def load_set_b_config(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"feature_store_root", "whisper_cache_root", "output_dir", "source_manifest_paths", "source_audio_dirs", "runtime", "model", "training"}:
        raise ValueError("E6 config has missing root or unsupported keys")
    raw = _expand(raw, field="config")
    for key in ("feature_store_root", "whisper_cache_root", "output_dir"):
        if key not in raw:
            raise ValueError(f"E6 config missing {key}")
    result = {key: _path(raw[key], field=key, base=path.parent) for key in ("feature_store_root", "whisper_cache_root", "output_dir")}
    for key in ("source_manifest_paths", "source_audio_dirs"):
        mapping = raw.get(key, {})
        if not isinstance(mapping, dict) or set(mapping) - {"train", "validation", "test"}:
            raise ValueError(f"{key} must map known splits")
        result[key] = {split: _path(value, field=f"{key}.{split}", base=path.parent) for split, value in mapping.items()}
    model = raw.get("model", {})
    if set(model) - {"hidden_dims", "dropout"}:
        raise ValueError("unsupported E6 model setting")
    hidden = model.get("hidden_dims", [256, 64])
    dropout = model.get("dropout", 0.2)
    if not isinstance(hidden, list) or len(hidden) != 2 or any(type(v) is not int or v <= 0 for v in hidden):
        raise ValueError("model.hidden_dims must be two positive integers")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0,1)")
    result["model"] = {"hidden_dims": hidden, "dropout": float(dropout)}
    defaults = {"epochs": 30, "batch_size": 8, "num_workers": 0, "learning_rate": 0.001,
                "weight_decay": 0.0001, "gradient_clip": 1.0, "seed": 42,
                "selection_metric": "official_validation_mse"}
    training = raw.get("training", {})
    if set(training) - set(defaults):
        raise ValueError("unsupported E6 training setting")
    settings = {**defaults, **training}
    if settings["epochs"] != 30 or settings["selection_metric"] != "official_validation_mse":
        raise ValueError("E6 requires 30 epochs and official validation MSE selection")
    for key in ("batch_size", "num_workers", "seed"):
        if type(settings[key]) is not int or settings[key] < (1 if key == "batch_size" else 0):
            raise ValueError(f"training.{key} invalid")
    for key in ("learning_rate", "weight_decay", "gradient_clip"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < (0 if key == "weight_decay" else 1e-30):
            raise ValueError(f"training.{key} invalid")
        settings[key] = float(value)
    result["training"] = settings
    runtime = raw.get("runtime", {})
    if set(runtime) - {"device"} or runtime.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, or cuda")
    result["runtime"] = {"device": runtime.get("device", "auto")}
    return result
