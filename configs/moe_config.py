"""Safe YAML configuration loading and explicit MOESCORE path validation.

Schema validation is independent of filesystem resources. Call
``resolve_moe_paths`` to obtain deterministic absolute paths, then call
``validate_moe_resources`` explicitly before model or data construction.
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


PathValue = Union[str, os.PathLike]
EXPERT_ORDER = ("expert1", "expert2", "expert3", "expert4")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DEVICE_RE = re.compile(r"^cuda:\d+$")


class MoeConfigError(ValueError):
    """Raised for malformed configuration or missing configured resources."""


def _mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise MoeConfigError(f"{field} must be a mapping; got {type(value).__name__}")
    return value


def _check_keys(
    value: Mapping,
    *,
    field: str,
    allowed: set,
    required: set = frozenset(),
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MoeConfigError(f"{field} contains unknown field(s): {', '.join(map(str, unknown))}")
    missing = sorted(required - set(value))
    if missing:
        raise MoeConfigError(f"{field} is missing required field(s): {', '.join(missing)}")


def _path_field(mapping: Mapping, key: str, field: str, *, required: bool = True) -> None:
    if key not in mapping:
        if required:
            raise MoeConfigError(f"{field} is required")
        return
    value = mapping[key]
    if value is None and not required:
        return
    if not isinstance(value, str) or not value.strip():
        raise MoeConfigError(f"{field} must be a non-empty path string")


def _validate_expert_paths(
    mapping: Mapping,
    *,
    section: str,
    required_fields: set,
    path_fields: set,
    allowed: set,
) -> None:
    _check_keys(mapping, field=section, allowed=allowed, required=required_fields)
    for key in path_fields:
        _path_field(mapping, key, f"{section}.{key}")


def validate_moe_config(config: Any) -> None:
    """Validate the LOCAL-03 schema without checking whether paths exist."""
    config = _mapping(config, "configuration")
    _check_keys(
        config,
        field="configuration",
        allowed={"schema_version", "runtime", "paths", "data", "experts", "gate"},
        required={"schema_version", "runtime", "data", "experts", "gate"},
    )

    version = config["schema_version"]
    if type(version) is not int or version != 1:
        raise MoeConfigError(f"schema_version must be the supported integer 1; got {version!r}")

    runtime = _mapping(config["runtime"], "runtime")
    _check_keys(
        runtime,
        field="runtime",
        allowed={"device", "offline", "cache_dir"},
        required={"device", "offline"},
    )
    device = runtime["device"]
    if not isinstance(device, str) or not (
        device in {"auto", "cpu", "cuda"} or _DEVICE_RE.fullmatch(device)
    ):
        raise MoeConfigError("runtime.device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
    if type(runtime["offline"]) is not bool:
        raise MoeConfigError("runtime.offline must be a boolean")
    _path_field(runtime, "cache_dir", "runtime.cache_dir", required=False)

    if "paths" in config:
        paths = _mapping(config["paths"], "paths")
        root_fields = {"project_root", "data_root", "resource_root"}
        _check_keys(paths, field="paths", allowed=root_fields)
        for key in root_fields:
            if key in paths:
                _path_field(paths, key, f"paths.{key}", required=False)

    data = _mapping(config["data"], "data")
    required_data_fields = {"train_csv", "validation_csv", "audio_root"}
    data_fields = required_data_fields | {"test_csv"}
    _check_keys(data, field="data", allowed=data_fields, required=required_data_fields)
    for key in required_data_fields:
        _path_field(data, key, f"data.{key}")
    _path_field(data, "test_csv", "data.test_csv", required=False)

    experts = _mapping(config["experts"], "experts")
    if set(experts) != set(EXPERT_ORDER):
        raise MoeConfigError(
            "experts must contain exactly expert1, expert2, expert3, and expert4"
        )

    expert1 = _mapping(experts["expert1"], "experts.expert1")
    e1_fields = {"pretrained_checkpoint", "finetuned_checkpoint", "enable_fusion"}
    _validate_expert_paths(
        expert1,
        section="experts.expert1",
        required_fields=e1_fields,
        path_fields={"pretrained_checkpoint", "finetuned_checkpoint"},
        allowed=e1_fields,
    )
    if type(expert1.get("enable_fusion")) is not bool or expert1["enable_fusion"] is not False:
        raise MoeConfigError("experts.expert1.enable_fusion must be false")

    expert2 = _mapping(experts["expert2"], "experts.expert2")
    e2_paths = {
        "source_dir",
        "architecture_yaml",
        "training_config",
        "pretrained_checkpoint",
        "finetuned_checkpoint",
    }
    e2_fields = e2_paths | {"freeze_policy"}
    _validate_expert_paths(
        expert2,
        section="experts.expert2",
        required_fields=e2_fields,
        path_fields=e2_paths,
        allowed=e2_fields,
    )
    if not isinstance(expert2["freeze_policy"], str) or expert2["freeze_policy"] not in {
        "partial",
        "full",
    }:
        raise MoeConfigError("experts.expert2.freeze_policy must be 'partial' or 'full'")

    expert3 = _mapping(experts["expert3"], "experts.expert3")
    e3_paths = {"source_dir", "base_checkpoint", "finetuned_checkpoint", "model_config"}
    e3_fields = e3_paths | {"expected_base_sha256", "concat_assignment"}
    _validate_expert_paths(
        expert3,
        section="experts.expert3",
        required_fields=e3_fields,
        path_fields=e3_paths,
        allowed=e3_fields,
    )
    if not isinstance(expert3["concat_assignment"], str) or expert3[
        "concat_assignment"
    ] not in {"released", "semantic"}:
        raise MoeConfigError(
            "experts.expert3.concat_assignment must be 'released' or 'semantic'"
        )
    expected_sha = expert3["expected_base_sha256"]
    if expected_sha is not None and (
        not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha)
    ):
        raise MoeConfigError(
            "experts.expert3.expected_base_sha256 must be null or 64 hexadecimal characters"
        )

    expert4 = _mapping(experts["expert4"], "experts.expert4")
    e4_fields = {
        "beats_checkpoint",
        "clap_checkpoint",
        "unilm_source_dir",
        "finetuned_checkpoint",
    }
    _validate_expert_paths(
        expert4,
        section="experts.expert4",
        required_fields=e4_fields,
        path_fields=e4_fields,
        allowed=e4_fields,
    )

    gate = _mapping(config["gate"], "gate")
    gate_fields = {"expert_order", "hidden_dim", "dropout"}
    _check_keys(gate, field="gate", allowed=gate_fields, required=gate_fields)
    order = gate["expert_order"]
    if not isinstance(order, (list, tuple)) or tuple(order) != EXPERT_ORDER:
        raise MoeConfigError(
            "gate.expert_order must be exactly [expert1, expert2, expert3, expert4]"
        )
    hidden_dim = gate["hidden_dim"]
    if type(hidden_dim) is not int or hidden_dim <= 0:
        raise MoeConfigError("gate.hidden_dim must be a positive integer")
    dropout = gate["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise MoeConfigError("gate.dropout must be a finite number in [0, 1)")


def load_moe_config(path: PathValue) -> Dict[str, Any]:
    """Load a YAML mapping safely and validate its schema, but not its resources."""
    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise MoeConfigError(f"Unable to load YAML config {config_path}: {exc}") from exc
    validate_moe_config(config)
    return config


def _root_path(value: PathValue, *, relative_to: Path, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise MoeConfigError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=False)


def _select_root(
    *,
    field: str,
    explicit: Optional[PathValue],
    env_name: str,
    environ: Mapping,
    configured: Optional[str],
    config_dir: Path,
    fallback: Optional[Path],
) -> Path:
    if explicit is not None:
        return _root_path(explicit, relative_to=Path.cwd(), field=f"{field} override")
    if env_name in environ:
        return _root_path(environ[env_name], relative_to=Path.cwd(), field=env_name)
    if configured is not None:
        return _root_path(configured, relative_to=config_dir, field=f"paths.{field}")
    if fallback is not None:
        return fallback.resolve(strict=False)
    return config_dir.resolve(strict=False)


def _resolve_path_value(value: Optional[str], *, root: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def resolve_moe_paths(
    config: Mapping,
    config_path: PathValue,
    *,
    project_root: Optional[PathValue] = None,
    data_root: Optional[PathValue] = None,
    resource_root: Optional[PathValue] = None,
    environ: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Return a copy with all configured paths resolved to absolute ``Path`` objects.

    For each root the precedence is explicit argument, matching environment
    variable, ``paths`` config value, then the config-file directory (or the
    resolved project root for data/resource roots). Absolute individual path
    fields remain absolute and are never rebased.
    """
    validate_moe_config(config)
    environment = os.environ if environ is None else environ
    config_file = Path(config_path).expanduser().resolve(strict=False)
    config_dir = config_file.parent
    configured_roots = config.get("paths", {})

    resolved_project_root = _select_root(
        field="project_root",
        explicit=project_root,
        env_name="MOESCORE_ROOT",
        environ=environment,
        configured=configured_roots.get("project_root"),
        config_dir=config_dir,
        fallback=None,
    )
    resolved_data_root = _select_root(
        field="data_root",
        explicit=data_root,
        env_name="MOESCORE_DATA_ROOT",
        environ=environment,
        configured=configured_roots.get("data_root"),
        config_dir=config_dir,
        fallback=resolved_project_root,
    )
    resolved_resource_root = _select_root(
        field="resource_root",
        explicit=resource_root,
        env_name="MOESCORE_RESOURCE_ROOT",
        environ=environment,
        configured=configured_roots.get("resource_root"),
        config_dir=config_dir,
        fallback=resolved_project_root,
    )

    output = copy.deepcopy(dict(config))
    output["paths"] = {
        "project_root": resolved_project_root,
        "data_root": resolved_data_root,
        "resource_root": resolved_resource_root,
    }
    for key in ("train_csv", "validation_csv", "audio_root"):
        output["data"][key] = _resolve_path_value(output["data"][key], root=resolved_data_root)
    output["data"]["test_csv"] = _resolve_path_value(
        output["data"].get("test_csv"), root=resolved_data_root
    )
    output["runtime"]["cache_dir"] = _resolve_path_value(
        output["runtime"].get("cache_dir"), root=resolved_resource_root
    )
    expert_path_fields = {
        "expert1": ("pretrained_checkpoint", "finetuned_checkpoint"),
        "expert2": (
            "source_dir",
            "architecture_yaml",
            "training_config",
            "pretrained_checkpoint",
            "finetuned_checkpoint",
        ),
        "expert3": ("source_dir", "base_checkpoint", "finetuned_checkpoint", "model_config"),
        "expert4": (
            "beats_checkpoint",
            "clap_checkpoint",
            "unilm_source_dir",
            "finetuned_checkpoint",
        ),
    }
    for expert_name, fields in expert_path_fields.items():
        expert = output["experts"][expert_name]
        for field in fields:
            expert[field] = _resolve_path_value(expert[field], root=resolved_resource_root)
    return output


_FILE_RESOURCES = (
    ("data.train_csv", ("data", "train_csv"), "training CSV file"),
    ("data.validation_csv", ("data", "validation_csv"), "validation CSV file"),
    ("data.test_csv", ("data", "test_csv"), "test CSV file"),
    ("experts.expert1.pretrained_checkpoint", ("experts", "expert1", "pretrained_checkpoint"), "Expert 1 pretrained checkpoint file"),
    ("experts.expert1.finetuned_checkpoint", ("experts", "expert1", "finetuned_checkpoint"), "Expert 1 fine-tuned checkpoint file"),
    ("experts.expert2.architecture_yaml", ("experts", "expert2", "architecture_yaml"), "Expert 2 architecture YAML file"),
    ("experts.expert2.training_config", ("experts", "expert2", "training_config"), "Expert 2 training configuration file"),
    ("experts.expert2.pretrained_checkpoint", ("experts", "expert2", "pretrained_checkpoint"), "Expert 2 pretrained checkpoint file"),
    ("experts.expert2.finetuned_checkpoint", ("experts", "expert2", "finetuned_checkpoint"), "Expert 2 fine-tuned checkpoint file"),
    ("experts.expert3.base_checkpoint", ("experts", "expert3", "base_checkpoint"), "Expert 3 pretrained M2D checkpoint file"),
    ("experts.expert3.finetuned_checkpoint", ("experts", "expert3", "finetuned_checkpoint"), "Expert 3 fine-tuned checkpoint file"),
    ("experts.expert3.model_config", ("experts", "expert3", "model_config"), "Expert 3 model configuration file"),
    ("experts.expert4.beats_checkpoint", ("experts", "expert4", "beats_checkpoint"), "Expert 4 pretrained BEATs checkpoint file"),
    ("experts.expert4.clap_checkpoint", ("experts", "expert4", "clap_checkpoint"), "Expert 4 pretrained CLAP checkpoint file"),
    ("experts.expert4.finetuned_checkpoint", ("experts", "expert4", "finetuned_checkpoint"), "Expert 4 fine-tuned checkpoint file"),
)
_DIRECTORY_RESOURCES = (
    ("data.audio_root", ("data", "audio_root"), "dataset audio directory"),
    ("experts.expert2.source_dir", ("experts", "expert2", "source_dir"), "Expert 2 MGA source directory"),
    ("experts.expert3.source_dir", ("experts", "expert3", "source_dir"), "Expert 3 source directory"),
    ("experts.expert4.unilm_source_dir", ("experts", "expert4", "unilm_source_dir"), "Expert 4 UniLM source directory"),
)


def _get_nested(mapping: Mapping, keys: tuple) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def validate_moe_resources(config: Mapping) -> None:
    """Explicitly verify resolved local resources; no files are deserialized.

    Pass the mapping returned by :func:`resolve_moe_paths`. A missing Expert 3
    base hash is an error here because a runtime restoration contract needs a
    trusted identity even though a template may remain schema-valid.
    """
    failures = []
    for field, keys, description in _FILE_RESOURCES + _DIRECTORY_RESOURCES:
        try:
            value = _get_nested(config, keys)
        except (KeyError, TypeError):
            if field == "data.test_csv":
                continue
            failures.append(f"{field}: missing resolved path")
            continue
        if field == "data.test_csv" and value is None:
            continue
        path = Path(value) if isinstance(value, (str, os.PathLike)) else None
        if path is None or not path.is_absolute():
            failures.append(f"{field}: path is not resolved; call resolve_moe_paths first")
            continue
        if not path.exists():
            failures.append(
                f"{field}: expected resource type {description} at {path}; path does not exist"
            )
        elif description.endswith("directory"):
            if not path.is_dir():
                failures.append(
                    f"{field}: expected resource type {description} at {path}; found a file"
                )
            elif not os.access(path, os.R_OK):
                failures.append(f"{field}: expected readable {description} at {path}")
        elif not path.is_file():
            failures.append(
                f"{field}: expected resource type {description} at {path}; found a directory"
            )
        elif not os.access(path, os.R_OK):
            failures.append(f"{field}: expected readable {description} at {path}")

    expected_sha = config.get("experts", {}).get("expert3", {}).get("expected_base_sha256")
    if expected_sha is None:
        failures.append(
            "experts.expert3.expected_base_sha256: trusted SHA-256 is required before "
            "Expert 3 restoration; replace null with the verified base-checkpoint identity"
        )
    if failures:
        raise MoeConfigError("MOESCORE resource validation failed:\n- " + "\n- ".join(failures))


__all__ = [
    "EXPERT_ORDER",
    "MoeConfigError",
    "load_moe_config",
    "resolve_moe_paths",
    "validate_moe_config",
    "validate_moe_resources",
]
