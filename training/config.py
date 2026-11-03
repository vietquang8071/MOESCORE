"""Path-portable configuration for gating-only training and inference."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml


class MoeTrainingConfigError(ValueError):
    """Raised for an invalid gating pipeline configuration."""


_DEVICE = re.compile(r"^cuda:\d+$")


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MoeTrainingConfigError(f"{field} must be a mapping")
    return value


def _keys(value: dict[str, Any], *, field: str, allowed: set[str], required: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise MoeTrainingConfigError(f"{field} contains unknown field(s): {unknown}")
    if missing:
        raise MoeTrainingConfigError(f"{field} is missing required field(s): {missing}")


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> None:
    if type(value) is not int or (value < 0 if allow_zero else value <= 0):
        raise MoeTrainingConfigError(f"{field} must be a {'non-negative' if allow_zero else 'positive'} integer")


def _finite_number(value: Any, field: str, *, positive: bool = False, lower: float | None = None, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MoeTrainingConfigError(f"{field} must be a finite number")
    if positive and float(value) <= 0:
        raise MoeTrainingConfigError(f"{field} must be positive")
    if lower is not None and float(value) < lower:
        raise MoeTrainingConfigError(f"{field} must be at least {lower}")
    if upper is not None and float(value) > upper:
        raise MoeTrainingConfigError(f"{field} must be at most {upper}")


def _resolve_relative(value: Any, *, base: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MoeTrainingConfigError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve(strict=False))


def load_moe_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MoeTrainingConfigError(f"cannot read training config {config_path}: {exc}") from exc
    config = _require_mapping(config, "configuration")
    _keys(
        config,
        field="configuration",
        allowed={"schema_version", "runtime", "data", "output_dir", "training", "validation", "inference"},
        required={"schema_version", "runtime", "data", "output_dir", "training", "validation", "inference"},
    )
    if config["schema_version"] != 1 or type(config["schema_version"]) is not int:
        raise MoeTrainingConfigError("schema_version must be integer 1")

    runtime = _require_mapping(config["runtime"], "runtime")
    _keys(runtime, field="runtime", allowed={"device", "seed"}, required={"device", "seed"})
    device = runtime["device"]
    if not isinstance(device, str) or not (device in {"auto", "cpu", "cuda"} or _DEVICE.fullmatch(device)):
        raise MoeTrainingConfigError("runtime.device must be auto, cpu, cuda, or cuda:<index>")
    _positive_int(runtime["seed"], "runtime.seed", allow_zero=True)

    data = _require_mapping(config["data"], "data")
    _keys(data, field="data", allowed={"feature_store_root"}, required={"feature_store_root"})
    data["feature_store_root"] = _resolve_relative(data["feature_store_root"], base=config_path.parent, field="data.feature_store_root")
    config["output_dir"] = _resolve_relative(config["output_dir"], base=config_path.parent, field="output_dir")

    training = _require_mapping(config["training"], "training")
    _keys(
        training,
        field="training",
        allowed={"batch_size", "max_epochs", "learning_rate", "weight_decay", "diversity_weight", "early_stop_patience", "checkpoint_interval_steps", "loss"},
        required={"batch_size", "max_epochs", "learning_rate", "weight_decay", "diversity_weight", "early_stop_patience", "checkpoint_interval_steps", "loss"},
    )
    for field in ("batch_size", "max_epochs", "checkpoint_interval_steps"):
        _positive_int(training[field], f"training.{field}")
    _positive_int(training["early_stop_patience"], "training.early_stop_patience", allow_zero=True)
    _finite_number(training["learning_rate"], "training.learning_rate", positive=True)
    _finite_number(training["weight_decay"], "training.weight_decay", lower=0.0)
    _finite_number(training["diversity_weight"], "training.diversity_weight")
    if float(training["diversity_weight"]) != 0.2:
        raise MoeTrainingConfigError("training.diversity_weight must preserve the original coefficient 0.2")

    loss = _require_mapping(training["loss"], "training.loss")
    _keys(loss, field="training.loss", allowed={"loss_weights", "clipped_mse", "contrastive"}, required={"loss_weights", "clipped_mse", "contrastive"})
    if not isinstance(loss["loss_weights"], list) or len(loss["loss_weights"]) != 2:
        raise MoeTrainingConfigError("training.loss.loss_weights must contain two values")
    for index, value in enumerate(loss["loss_weights"]):
        _finite_number(value, f"training.loss.loss_weights[{index}]")
    for section, fields in (
        ("clipped_mse", {"tau", "mode", "beta", "cbl"}),
        ("contrastive", {"margin", "beta", "cbl"}),
    ):
        values = _require_mapping(loss[section], f"training.loss.{section}")
        _keys(values, field=f"training.loss.{section}", allowed=fields, required=fields)
        for number_field in fields - {"mode", "cbl"}:
            _finite_number(values[number_field], f"training.loss.{section}.{number_field}")
        if type(values["cbl"]) is not bool:
            raise MoeTrainingConfigError(f"training.loss.{section}.cbl must be boolean")
    if loss["clipped_mse"]["mode"] not in {"utt", "frame"}:
        raise MoeTrainingConfigError("training.loss.clipped_mse.mode must be utt or frame")

    for section in ("validation", "inference"):
        values = _require_mapping(config[section], section)
        _keys(values, field=section, allowed={"batch_size"}, required={"batch_size"})
        _positive_int(values["batch_size"], f"{section}.batch_size")
    config["config_path"] = str(config_path)
    return config


__all__ = ["MoeTrainingConfigError", "load_moe_training_config"]
