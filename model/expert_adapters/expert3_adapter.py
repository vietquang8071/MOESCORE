"""Thin adapter for the source-preserving, partially fine-tuned M2D-CLAP Expert 3.

The M2D vendor implementation is imported only by ``from_finetuned``. Importing
this module never constructs a model, reads a checkpoint, or changes import
paths. Expert 3's released feature assignment is part of its checkpoint ABI and
is therefore enforced as ``released`` here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional, Union

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter


_EXPERT_NAME = "Expert 3"
_AUDIO_SAMPLES = 160000
_SAMPLE_RATE = 16000
_DURATION_SECONDS = 10
_FORMAT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PAYLOAD_KEYS = frozenset({
    "format_version",
    "checkpoint_kind",
    "model_trainable",
    "model_config",
    "base_checkpoint_identity",
    "epoch",
    "best_srcc",
    "metrics",
    "config",
})
_BASE_IDENTITY_KEYS = frozenset({"path", "size_bytes", "sha256", "model_directory"})
_SOURCE_PACKAGE = "expert3_finetune.XACLE_2027_baselinemodel"
_EXPECTED_SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "expert3_finetune"
    / "XACLE_2027_baselinemodel"
)

PathValue = Union[str, os.PathLike]
DeviceValue = Union[str, torch.device]


class Expert3CheckpointError(ValueError):
    """Raised when an Expert 3 partial checkpoint violates its source contract."""


def validate_expert3_module_origin(module: Any, source_dir: PathValue, module_name: str) -> None:
    """Reject an imported Expert 3 module whose file is outside its vendor tree."""
    source = Path(source_dir).expanduser().resolve()
    origin = getattr(module, "__file__", None)
    if not origin:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) if spec is not None else None
    if origin in {None, "built-in", "frozen"}:
        locations = tuple(getattr(module, "__path__", ()))
        if locations and all(Path(item).expanduser().resolve().is_relative_to(source) for item in locations):
            return
        raise ImportError(
            f"Expert 3 module {module_name!r} has no verifiable file origin under {source}; "
            f"observed origin={origin!r}, paths={locations!r}"
        )
    resolved_origin = Path(origin).expanduser().resolve()
    if not resolved_origin.is_relative_to(source):
        raise ImportError(
            f"Expert 3 module {module_name!r} resolved to conflicting origin "
            f"{resolved_origin}; expected a file under {source}"
        )


def _resolved_path(value: PathValue, *, field: str, kind: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expert 3 {field} must be a path; got {type(value).__name__}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Expert 3 {field} must already be an absolute resolved path; got {path}")
    path = path.resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Expert 3 {kind} does not exist: {path}")
    if kind == "source directory":
        valid = path.is_dir()
    else:
        valid = path.is_file()
    if not valid:
        expected = "directory" if kind == "source directory" else "regular file"
        raise ValueError(f"Expert 3 {field} must be a {expected}: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Expert 3 {field} is not readable: {path}")
    return path


def _cache_path(value: PathValue, *, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expert 3 {field} must be a path; got {type(value).__name__}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Expert 3 {field} must already be absolute; got {path}")
    path = path.resolve(strict=False)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Expert 3 {field} must be a directory: {path}")
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Expert 3 {field} is not writable: {path}")
    else:
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            raise PermissionError(
                f"Expert 3 {field} cannot be created because its nearest existing parent "
                f"is not a writable directory: {parent}"
            )
    return path


def _target_device(device: DeviceValue) -> torch.device:
    if isinstance(device, torch.device):
        target = device
    elif isinstance(device, str) and device in {"auto", "cpu", "cuda", "mps"}:
        if device == "auto":
            target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target = torch.device(device)
    elif isinstance(device, str):
        try:
            target = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Expert 3 device is invalid: {device!r}") from exc
    else:
        raise TypeError(f"Expert 3 device must be str or torch.device; got {type(device).__name__}")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Expert 3 CUDA device {target} was requested but CUDA is unavailable")
    if target.type == "cuda" and target.index is not None and target.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Expert 3 CUDA device index {target.index} is unavailable; "
            f"visible device count is {torch.cuda.device_count()}"
        )
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Expert 3 MPS was requested but MPS is unavailable")
    return target


@contextmanager
def _offline_transformer_environment(cache_dir: Path):
    """Temporarily direct Hugging Face resolution to the requested local cache."""
    names = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_model_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream, object_pairs_hook=_unique_json_pairs)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Expert 3 could not read model configuration {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"Expert 3 model configuration must be a JSON object: {path}")
    for field, expected in (("sample_rate", _SAMPLE_RATE), ("max_len", _DURATION_SECONDS)):
        if type(config.get(field)) is not int or config[field] != expected:
            raise ValueError(
                f"Expert 3 model configuration {path} must set {field}={expected}; "
                f"got {config.get(field)!r}"
            )
    if type(config.get("activation_checkpointing")) is not bool:
        raise ValueError(
            f"Expert 3 model configuration {path} must set boolean activation_checkpointing"
        )
    if config.get("concat_assignment") != "released":
        raise ValueError(
            f"Expert 3 model configuration must use concat_assignment='released'; "
            f"got {config.get('concat_assignment')!r}"
        )
    digest = config.get("m2d_checkpoint_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(
            f"Expert 3 model configuration must contain a trusted 64-character "
            "m2d_checkpoint_sha256"
        )
    return dict(config)


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"Expert 3 {field} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _model_metadata(model: nn.Module) -> tuple[int, int, dict]:
    config = getattr(model, "model_config", None)
    config = dict(config) if isinstance(config, Mapping) else {}
    embedding_dim = getattr(model, "embedding_dim", config.get("embedding_dim"))
    cross_feature_dim = getattr(model, "cross_feature_dim", config.get("cross_feature_dim"))
    if type(embedding_dim) is not int or embedding_dim <= 0:
        raise ValueError(
            "Expert3Adapter requires a positive embedding_dim from the constructed model metadata"
        )
    expected = 5 * embedding_dim
    if type(cross_feature_dim) is not int or cross_feature_dim != expected:
        raise ValueError(
            "Expert3Adapter model metadata must declare cross_feature_dim == "
            f"5 * embedding_dim ({expected}); got {cross_feature_dim!r}"
        )
    assignment = getattr(model, "concat_assignment", config.get("concat_assignment"))
    if assignment != "released":
        raise ValueError(
            "Expert3Adapter requires concat_assignment='released' to preserve fine-tuned "
            f"feature/head compatibility; got {assignment!r}"
        )
    if config:
        if config.get("embedding_dim") != embedding_dim:
            raise ValueError("Expert3Adapter model_config embedding_dim conflicts with model metadata")
        if config.get("cross_feature_dim") != cross_feature_dim:
            raise ValueError("Expert3Adapter model_config cross_feature_dim conflicts with model metadata")
        if config.get("concat_assignment") != assignment:
            raise ValueError("Expert3Adapter model_config concat_assignment conflicts with model metadata")
    return embedding_dim, cross_feature_dim, config


def _validate_model_config_against_model(model: nn.Module, checkpoint_model_config: Mapping) -> None:
    embedding_dim, feature_dim, model_config = _model_metadata(model)
    if not model_config:
        raise Expert3CheckpointError("Expert 3 original model does not expose model_config metadata")
    if dict(model.model_config) != dict(checkpoint_model_config):
        raise Expert3CheckpointError(
            "Expert 3 fine-tuned model_config does not exactly match the constructed original model"
        )
    if feature_dim != 5 * embedding_dim:
        raise Expert3CheckpointError("Expert 3 cross feature dimension is not 5 * embedding_dim")


def _validate_source_run_settings(model: nn.Module, run_config: Mapping) -> None:
    embedding_dim, feature_dim, model_config = _model_metadata(model)
    if not model_config:
        raise ValueError("Expert 3 original model does not expose model_config metadata")
    if getattr(model, "concat_assignment", None) != run_config["concat_assignment"]:
        raise ValueError("Expert 3 constructed model concat_assignment differs from model configuration")
    if model_config.get("activation_checkpointing") != run_config["activation_checkpointing"]:
        raise ValueError(
            "Expert 3 constructed model activation_checkpointing differs from model configuration"
        )
    if feature_dim != 5 * embedding_dim:
        raise ValueError("Expert 3 constructed model feature dimension is not 5 * embedding_dim")


def _load_partial_payload(path: Path) -> Mapping:
    """Safely inspect the structured best artifact before source restoration."""
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Mirrors the original loader on old PyTorch releases.
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise Expert3CheckpointError(
            f"Expert 3 could not safely read fine-tuned partial checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Expert3CheckpointError(
            f"Expert 3 fine-tuned checkpoint must be a structured best artifact, got {type(payload).__name__}"
        )
    return payload


def _validate_partial_payload(
    model: nn.Module,
    payload: Mapping,
    *,
    expected_base_sha256: str,
) -> None:
    missing_top = sorted(_PAYLOAD_KEYS - set(payload))
    extra_top = sorted(set(payload) - _PAYLOAD_KEYS)
    if missing_top or extra_top:
        raise Expert3CheckpointError(
            f"Expert 3 partial checkpoint top-level keys mismatch; "
            f"missing={missing_top}, unexpected={extra_top}"
        )
    if type(payload["format_version"]) is not int or payload["format_version"] != _FORMAT_VERSION:
        raise Expert3CheckpointError(
            f"Expert 3 unsupported partial checkpoint format_version={payload['format_version']!r}"
        )
    if payload["checkpoint_kind"] != "best":
        raise Expert3CheckpointError(
            "Expert 3 expects checkpoint_kind='best'; resume checkpoints are not inference artifacts"
        )

    model_config = payload["model_config"]
    runtime_config = payload["config"]
    saved_identity = payload["base_checkpoint_identity"]
    saved_trainable = payload["model_trainable"]
    if not isinstance(model_config, Mapping):
        raise Expert3CheckpointError("Expert 3 checkpoint model_config must be a mapping")
    if not isinstance(runtime_config, Mapping):
        raise Expert3CheckpointError("Expert 3 checkpoint config must be a mapping")
    if not isinstance(saved_identity, Mapping):
        raise Expert3CheckpointError("Expert 3 checkpoint base_checkpoint_identity must be a mapping")
    if set(saved_identity) != _BASE_IDENTITY_KEYS:
        raise Expert3CheckpointError(
            "Expert 3 checkpoint base identity keys mismatch; "
            f"expected={sorted(_BASE_IDENTITY_KEYS)}, got={sorted(saved_identity)}"
        )
    if not isinstance(saved_trainable, Mapping):
        raise Expert3CheckpointError("Expert 3 checkpoint model_trainable must be a mapping")
    if not isinstance(saved_identity["path"], str) or not saved_identity["path"]:
        raise Expert3CheckpointError("Expert 3 checkpoint base identity path must be a non-empty string")
    if type(saved_identity["size_bytes"]) is not int or saved_identity["size_bytes"] <= 0:
        raise Expert3CheckpointError("Expert 3 checkpoint base identity size_bytes must be positive")
    saved_digest = _validate_sha256(
        saved_identity["sha256"], field="checkpoint base identity sha256"
    )
    if not isinstance(saved_identity["model_directory"], str) or not saved_identity["model_directory"]:
        raise Expert3CheckpointError("Expert 3 checkpoint model_directory must be a non-empty string")

    _validate_model_config_against_model(model, model_config)
    if model_config.get("concat_assignment") != "released":
        raise Expert3CheckpointError("Expert 3 checkpoint model_config must use concat_assignment='released'")
    if runtime_config.get("concat_assignment") != "released":
        raise Expert3CheckpointError("Expert 3 checkpoint config must use concat_assignment='released'")
    if runtime_config.get("m2d_checkpoint_sha256") != expected_base_sha256:
        raise Expert3CheckpointError(
            "Expert 3 checkpoint config M2D SHA-256 does not match the selected trusted base"
        )
    if runtime_config.get("sample_rate") != _SAMPLE_RATE or runtime_config.get("max_len") != _DURATION_SECONDS:
        raise Expert3CheckpointError("Expert 3 checkpoint config must use 16 kHz audio with max_len=10")
    if runtime_config.get("activation_checkpointing") != model_config.get("activation_checkpointing"):
        raise Expert3CheckpointError(
            "Expert 3 checkpoint config activation_checkpointing conflicts with model_config"
        )

    current_identity = getattr(model, "base_checkpoint_identity", None)
    if not isinstance(current_identity, Mapping):
        raise Expert3CheckpointError("Constructed Expert 3 does not expose base_checkpoint_identity")
    current_digest = _validate_sha256(
        current_identity.get("sha256"), field="constructed base checkpoint sha256"
    )
    current_size = current_identity.get("size_bytes")
    current_model_dir = current_identity.get("model_directory")
    if saved_digest != expected_base_sha256 or current_digest != expected_base_sha256:
        raise Expert3CheckpointError(
            "Expert 3 fine-tuned checkpoint base SHA-256 does not match the verified configured base"
        )
    if saved_digest != current_digest:
        raise Expert3CheckpointError("Expert 3 checkpoint base identity SHA-256 differs from constructed model")
    if type(current_size) is not int or saved_identity["size_bytes"] != current_size:
        raise Expert3CheckpointError("Expert 3 checkpoint base identity size does not match constructed model")
    if saved_identity["model_directory"] != current_model_dir:
        raise Expert3CheckpointError(
            "Expert 3 checkpoint base model_directory does not match constructed model"
        )
    # The original training path may be machine-specific. The content identity,
    # byte length and model family are checked; the absolute path is not compared.

    epoch = payload["epoch"]
    if type(epoch) is not int or epoch < 0:
        raise Expert3CheckpointError("Expert 3 best checkpoint epoch must be a non-negative integer")
    best_srcc = payload["best_srcc"]
    if isinstance(best_srcc, bool) or not isinstance(best_srcc, (int, float)) or not math.isfinite(best_srcc):
        raise Expert3CheckpointError("Expert 3 best_srcc must be a finite number")
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or "SRCC" not in metrics:
        raise Expert3CheckpointError("Expert 3 checkpoint metrics must contain SRCC")
    if isinstance(metrics["SRCC"], bool) or not isinstance(metrics["SRCC"], (int, float)) or not math.isfinite(metrics["SRCC"]):
        raise Expert3CheckpointError("Expert 3 checkpoint metrics.SRCC must be finite")

    expected_trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(expected_trainable) - set(saved_trainable))
    unexpected = sorted(set(saved_trainable) - set(expected_trainable))
    if missing or unexpected:
        raise Expert3CheckpointError(
            f"Expert 3 trainable partial checkpoint keys mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in expected_trainable.items():
        value = saved_trainable[name]
        if not isinstance(value, torch.Tensor):
            raise Expert3CheckpointError(
                f"Expert 3 trained component {name!r} must be a tensor, got {type(value).__name__}"
            )
        if tuple(value.shape) != tuple(parameter.shape):
            raise Expert3CheckpointError(
                f"Expert 3 trained component {name!r} shape mismatch: "
                f"checkpoint={tuple(value.shape)}, model={tuple(parameter.shape)}"
            )
        if not torch.isfinite(value).all().item():
            raise Expert3CheckpointError(f"Expert 3 trained component {name!r} contains non-finite values")


def _construct_original_expert3(
    source_dir: Path,
    base_checkpoint: Path,
    *,
    cache_dir: Path,
    checkpoint_layout_root: Path,
    activation_checkpointing: bool,
    concat_assignment: str,
    expected_base_sha256: str,
) -> nn.Module:
    module_name = f"{_SOURCE_PACKAGE}.models.moescore_expert3"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Expert 3 could not import the original model factory {module_name}") from exc
    validate_expert3_module_origin(module, source_dir, module_name)
    model_class = getattr(module, "MoescoreExpert3", None)
    if model_class is None:
        raise ImportError(f"Expert 3 source module {module_name} has no MoescoreExpert3 class")
    return model_class(
        checkpoint_path=base_checkpoint,
        cache_dir=cache_dir,
        checkpoint_layout_root=checkpoint_layout_root,
        activation_checkpointing=activation_checkpointing,
        concat_assignment=concat_assignment,
        expected_checkpoint_sha256=expected_base_sha256,
    )


def _original_best_loader(source_dir: Path):
    module_name = f"{_SOURCE_PACKAGE}.utils.expert3_checkpoint"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Expert 3 could not import the original partial-checkpoint loader {module_name}") from exc
    validate_expert3_module_origin(module, source_dir, module_name)
    loader = getattr(module, "load_best_checkpoint", None)
    if not callable(loader):
        raise ImportError(f"Expert 3 source module {module_name} has no load_best_checkpoint function")
    return loader


def _validate_loaded_expert3_origins(source_dir: Path) -> dict[str, str]:
    names = (
        f"{_SOURCE_PACKAGE}.models.moescore_expert3",
        f"{_SOURCE_PACKAGE}.models.m2d.vendor.portable_m2d",
        f"{_SOURCE_PACKAGE}.utils.expert3_checkpoint",
    )
    origins = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        validate_expert3_module_origin(module, source_dir, name)
        origins[name] = str(Path(module.__file__).resolve())
    return origins


class Expert3Adapter(ExpertAdapter):
    """Adapt original M2D-CLAP Expert 3 outputs to MOESCORE's shared contract.

    ``model`` is an already constructed original ``MoescoreExpert3``. Its
    ``embedding_dim`` and ``cross_feature_dim`` metadata determine the adapter
    width; an optional explicit ``feature_dim`` is only accepted when it agrees
    with that model metadata.
    """

    def __init__(self, model: nn.Module, *, feature_dim: Optional[int] = None) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expert3Adapter model must be an nn.Module; got {type(model).__name__}")
        embedding_dim, model_feature_dim, _ = _model_metadata(model)
        if feature_dim is not None:
            if type(feature_dim) is not int or feature_dim <= 0:
                raise ValueError(f"Expert3Adapter feature_dim must be a positive integer; got {feature_dim!r}")
            if feature_dim != model_feature_dim:
                raise ValueError(
                    "Expert3Adapter feature_dim conflicts with original model metadata: "
                    f"{feature_dim} != {model_feature_dim} (5 * embedding_dim={embedding_dim})"
                )
        super().__init__(model_feature_dim)
        self.model = model
        self.embedding_dim = embedding_dim
        self._freeze_model()

    @classmethod
    def from_finetuned(
        cls,
        *,
        source_dir: PathValue,
        base_checkpoint: PathValue,
        finetuned_checkpoint: PathValue,
        model_config: PathValue,
        cache_dir: PathValue,
        expected_base_sha256: Optional[str] = None,
        concat_assignment: str = "released",
        device: DeviceValue = "cpu",
        offline: bool = True,
        checkpoint_layout_root: Optional[PathValue] = None,
    ) -> "Expert3Adapter":
        """Construct the original M2D model and restore its structured best file.

        The base loader and source ``load_best_checkpoint`` function are reused.
        Before calling the source loader, the partial artifact is safely read to
        reject missing/unexpected trained names, shape errors and metadata
        mismatches (the source loader checks names but delegates shapes to
        ``copy_``). The trusted checkpoint is consequently deserialized twice
        during restoration, but no duplicate model is created.
        """
        if type(offline) is not bool:
            raise TypeError(f"Expert 3 offline must be a boolean; got {offline!r}")
        if not offline:
            raise ValueError("Expert 3 restoration is offline-only; set runtime.offline: true")
        if concat_assignment != "released":
            raise ValueError(
                "Expert3Adapter requires concat_assignment='released' to match the fine-tuned head"
            )

        source_path = _resolved_path(source_dir, field="source_dir", kind="source directory")
        expected_source = _EXPECTED_SOURCE_DIR.resolve()
        if source_path != expected_source:
            raise ValueError(
                "Expert 3 source_dir must identify the authoritative checked-in source "
                f"{expected_source}; got {source_path}"
            )
        for relative in ("models/moescore_expert3.py", "utils/expert3_checkpoint.py"):
            if not (source_path / relative).is_file():
                raise FileNotFoundError(f"Expert 3 authoritative source is missing {relative}: {source_path}")

        base_path = _resolved_path(base_checkpoint, field="base_checkpoint", kind="pretrained base checkpoint")
        fine_path = _resolved_path(
            finetuned_checkpoint, field="finetuned_checkpoint", kind="fine-tuned partial checkpoint"
        )
        config_path = _resolved_path(model_config, field="model_config", kind="model configuration")
        model_run_config = _read_model_config(config_path)
        if concat_assignment != model_run_config["concat_assignment"]:
            raise ValueError("Expert 3 configured concat_assignment conflicts with model_config")

        source_digest = _validate_sha256(
            model_run_config["m2d_checkpoint_sha256"], field="model_config m2d_checkpoint_sha256"
        )
        if expected_base_sha256 is None:
            trusted_digest = source_digest
        else:
            trusted_digest = _validate_sha256(expected_base_sha256, field="expected_base_sha256")
            if trusted_digest != source_digest:
                raise ValueError(
                    "Expert 3 expected_base_sha256 conflicts with the authoritative source model_config"
                )

        cache_path = _cache_path(cache_dir, field="cache_dir")
        layout_path = (
            _cache_path(checkpoint_layout_root, field="checkpoint_layout_root")
            if checkpoint_layout_root is not None
            else (cache_path / "m2d_layout")
        )
        target_device = _target_device(device)

        # Check for stale/wrong cached source modules before model construction;
        # never purge sys.modules or change sys.path to force a resolution.
        _validate_loaded_expert3_origins(source_path)
        try:
            with _offline_transformer_environment(cache_path):
                model = _construct_original_expert3(
                    source_path,
                    base_path,
                    cache_dir=cache_path,
                    checkpoint_layout_root=layout_path,
                    activation_checkpointing=model_run_config["activation_checkpointing"],
                    concat_assignment=concat_assignment,
                    expected_base_sha256=trusted_digest,
                )
        except Exception as exc:
            raise RuntimeError(
                "Expert 3 original M2D construction or pretrained restoration failed; "
                "verify the trusted base checkpoint, local text-model cache and source dependencies. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expert 3 original MoescoreExpert3 constructor returned {type(model).__name__}, not nn.Module"
            )
        _validate_loaded_expert3_origins(source_path)
        _validate_source_run_settings(model, model_run_config)
        identity = getattr(model, "base_checkpoint_identity", {})
        if _validate_sha256(identity.get("sha256"), field="constructed base SHA-256") != trusted_digest:
            raise Expert3CheckpointError(
                "Expert 3 constructed model base SHA-256 differs from the configured trusted identity"
            )

        payload = _load_partial_payload(fine_path)
        _validate_partial_payload(model, payload, expected_base_sha256=trusted_digest)
        del payload
        try:
            loader = _original_best_loader(source_path)
            loader(model, fine_path)
        except Exception as exc:
            raise Expert3CheckpointError(
                f"Expert 3 original load_best_checkpoint failed for {fine_path}: {exc}"
            ) from exc
        _validate_loaded_expert3_origins(source_path)

        adapter = cls(model)
        adapter.to(target_device)
        adapter.eval()
        return adapter

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        *,
        device: Optional[DeviceValue] = None,
    ) -> "Expert3Adapter":
        """Build from a LOCAL-03 resolved config, validating Expert 3 only.

        Accepts either the full resolved mapping or its ``experts.expert3``
        section. It intentionally does not call the global all-expert resource
        validator, which would require unrelated checkpoints.
        """
        if not isinstance(config, Mapping):
            raise TypeError("Expert 3 config must be a mapping")
        if "experts" in config:
            try:
                expert_config = config["experts"]["expert3"]
            except (KeyError, TypeError) as exc:
                raise ValueError("Expert 3 config is missing experts.expert3") from exc
            runtime = config.get("runtime", {})
        else:
            expert_config = config
            runtime = {}
        if not isinstance(expert_config, Mapping):
            raise TypeError("Expert 3 config section must be a mapping")
        if not isinstance(runtime, Mapping):
            raise TypeError("Expert 3 runtime config must be a mapping")
        required = (
            "source_dir", "base_checkpoint", "finetuned_checkpoint", "model_config", "concat_assignment"
        )
        missing = [key for key in required if key not in expert_config]
        if missing:
            raise ValueError(f"Expert 3 config is missing required field(s): {', '.join(missing)}")
        cache_dir = runtime.get("cache_dir")
        if cache_dir is None:
            raise ValueError("Expert 3 resolved config requires runtime.cache_dir for local text resources")
        return cls.from_finetuned(
            source_dir=expert_config["source_dir"],
            base_checkpoint=expert_config["base_checkpoint"],
            finetuned_checkpoint=expert_config["finetuned_checkpoint"],
            model_config=expert_config["model_config"],
            cache_dir=cache_dir,
            expected_base_sha256=expert_config.get("expected_base_sha256"),
            concat_assignment=expert_config["concat_assignment"],
            device=device if device is not None else runtime.get("device", "cpu"),
            offline=runtime.get("offline", True),
        )

    def _freeze_model(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "Expert3Adapter":
        """Keep the wrapped, possibly custom-train model frozen and in eval mode."""
        super().train(mode)
        self._freeze_model()
        return self

    def _model_device(self) -> Optional[torch.device]:
        tensors = list(self.model.parameters())
        if not tensors:
            tensors = list(self.model.buffers())
        if not tensors:
            return None
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError(f"Expert 3 wrapped model spans multiple devices: {sorted(map(str, devices))}")
        return next(iter(devices))

    def _validate_batch(self, batch: Any) -> tuple[torch.Tensor, list[str]]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expert 3 batch must be a mapping; got {type(batch).__name__}")
        if "expert3_wavs" not in batch:
            raise KeyError("Expert 3 batch is missing required field 'expert3_wavs'")
        if "captions_raw" not in batch:
            raise KeyError("Expert 3 batch is missing required field 'captions_raw'")
        audio = batch["expert3_wavs"]
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"Expert 3 expert3_wavs must be a torch.Tensor; got {type(audio).__name__}")
        if audio.ndim != 3:
            raise ValueError(
                f"Expert 3 expert3_wavs must have shape [B,1,{_AUDIO_SAMPLES}]; got {tuple(audio.shape)}"
            )
        if audio.shape[0] <= 0:
            raise ValueError("Expert 3 expert3_wavs batch must be non-empty")
        if audio.shape[1] != 1:
            raise ValueError(f"Expert 3 expert3_wavs must have one channel; got {audio.shape[1]}")
        if audio.shape[2] != _AUDIO_SAMPLES:
            raise ValueError(
                f"Expert 3 expert3_wavs must contain {_AUDIO_SAMPLES} samples "
                f"(10 seconds at {_SAMPLE_RATE} Hz); got {audio.shape[2]}"
            )
        if audio.dtype != torch.float32:
            raise TypeError(f"Expert 3 expert3_wavs must use torch.float32; got {audio.dtype}")
        model_device = self._model_device()
        if model_device is not None and audio.device != model_device:
            raise ValueError(
                "Expert 3 expert3_wavs and wrapped model must be on the same device; "
                f"got {audio.device} and {model_device}"
            )
        if not torch.isfinite(audio).all().item():
            raise ValueError("Expert 3 expert3_wavs must contain only finite values")

        captions = batch["captions_raw"]
        if isinstance(captions, (str, bytes, torch.Tensor)) or not isinstance(captions, Sequence):
            raise TypeError(
                "Expert 3 captions_raw must be a sequence of raw strings; "
                f"got {type(captions).__name__}"
            )
        if len(captions) != audio.shape[0]:
            raise ValueError(
                f"Expert 3 captions_raw must contain exactly {audio.shape[0]} caption(s); got {len(captions)}"
            )
        if any(not isinstance(caption, str) for caption in captions):
            raise TypeError("Expert 3 every caption must be a string")
        processed = [caption.strip() for caption in captions]
        if any(not caption for caption in processed):
            raise ValueError("Expert 3 captions_raw cannot contain empty text after standalone whitespace stripping")
        return audio, processed

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = self._validate_batch(batch)
        original_batch = {"wavs": audio, "captions": captions}
        # The original forward returns the native regression score and the exact
        # cross-feature fed to its head. Preserve both with one model invocation.
        return self.model(original_batch)


__all__ = [
    "Expert3Adapter",
    "Expert3CheckpointError",
    "validate_expert3_module_origin",
]
