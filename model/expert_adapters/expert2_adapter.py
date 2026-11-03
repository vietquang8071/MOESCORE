"""Adapter for the original fine-tuned MGA-CLAP Expert 2 implementation.

This module imports no MGA vendor modules at import time. The original factory
and its ``models``/``tools`` namespaces are resolved only when restoration is
explicitly requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

import torch
from torch import nn
import yaml

from model.expert_adapters.base import ExpertAdapter


PathValue = Union[str, os.PathLike]
_AUDIO_SAMPLES = 32_000 * 10
_MGA_NAMESPACE_ROOTS = {"models", "tools"}


class MGAImportNamespaceError(ImportError):
    """Raised when a loaded top-level vendor namespace belongs to another tree."""


def _resolved_resource_path(
    value: Any,
    *,
    field: str,
    resource_type: str,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expert 2 {field} must be a resolved filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Expert 2 {field} must already be resolved to an absolute path; got {path}"
        )
    path = path.resolve(strict=False)
    if resource_type == "directory":
        if not path.is_dir():
            raise NotADirectoryError(
                f"Expert 2 {field} expected a source directory at {path}"
            )
    elif not path.is_file():
        raise FileNotFoundError(
            f"Expert 2 {field} expected a {resource_type} file at {path}"
        )
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Expert 2 {field} is not readable: {path}")
    return path


def _target_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Expert 2 requested {target}, but CUDA is unavailable")
        index = target.index if target.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Expert 2 requested {target}, but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
    return target


@contextmanager
def _huggingface_offline(offline: bool):
    """Temporarily ensure Transformers can use local cache files only."""
    if type(offline) is not bool:
        raise TypeError(f"Expert 2 offline must be a boolean; got {offline!r}")
    if not offline:
        raise ValueError(
            "Expert 2 restoration is offline-only; populate the local tokenizer/config cache first"
        )

    environment_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    for key in environment_keys:
        os.environ[key] = "1"

    offline_modules = (
        "huggingface_hub.constants",
        "transformers.utils.hub",
    )
    previous_flags = {}
    for module_name in offline_modules:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "HF_HUB_OFFLINE"):
            previous_flags[module_name] = module.HF_HUB_OFFLINE
            module.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for module_name, value in previous_flags.items():
            module = sys.modules.get(module_name)
            if module is not None:
                module.HF_HUB_OFFLINE = value
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _module_origins(module: Any) -> list[Path]:
    origins = []
    file_path = getattr(module, "__file__", None)
    if file_path:
        origins.append(Path(file_path).resolve(strict=False))
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and spec_origin not in {"built-in", "frozen"}:
        origins.append(Path(spec_origin).resolve(strict=False))
    for package_path in getattr(module, "__path__", ()):
        origins.append(Path(package_path).resolve(strict=False))
    return list(dict.fromkeys(origins))


def validate_mga_import_namespace(
    source_dir: PathValue,
    *,
    modules: Optional[Mapping[str, Any]] = None,
) -> None:
    """Reject loaded ``models``/``tools`` modules resolved outside MGA-CLAP.

    ``modules`` is an injectable module registry for lightweight tests; normal
    restoration checks the current process's ``sys.modules``. This function
    observes module origins only and never removes or replaces module entries.
    """
    source = Path(source_dir).expanduser().resolve(strict=False)
    registry = sys.modules if modules is None else modules
    for module_name, module in tuple(registry.items()):
        namespace = module_name.partition(".")[0]
        if namespace not in _MGA_NAMESPACE_ROOTS:
            continue
        origins = _module_origins(module)
        if not origins:
            raise MGAImportNamespaceError(
                f"MGA vendor namespace {namespace!r} module {module_name!r} is loaded "
                "but its origin cannot be inspected"
            )
        unexpected = [origin for origin in origins if not origin.is_relative_to(source)]
        if unexpected:
            raise MGAImportNamespaceError(
                f"MGA vendor namespace {namespace!r} module {module_name!r} resolves "
                f"outside MGA source {source}; origins={origins}; conflicting={unexpected}"
            )


def _last_linear_output(module: Any) -> Optional[int]:
    if not isinstance(module, nn.Module):
        return None
    for child in reversed(tuple(module.modules())):
        if isinstance(child, nn.Linear):
            return int(child.out_features)
    return None


def _msc_feature_dimensions(model: nn.Module) -> Optional[tuple[int, int]]:
    """Derive each MSC vector width from ASE projections and its shared codebook."""
    encoder = getattr(model, "model", None)
    ase = getattr(encoder, "model", None)
    if not isinstance(ase, nn.Module):
        return None

    audio_dim = _last_linear_output(getattr(ase, "frame_proj", None))
    text_dim = _last_linear_output(getattr(ase, "word_proj", None))
    codebook = getattr(ase, "codebook", None)
    if audio_dim is None and text_dim is None and not isinstance(codebook, torch.Tensor):
        return None
    if audio_dim is None or text_dim is None or not isinstance(codebook, torch.Tensor):
        raise ValueError(
            "Expert 2 ASE feature metadata is incomplete: expected frame_proj, word_proj, "
            "and a shared codebook"
        )
    if codebook.ndim != 2 or codebook.shape[1] <= 0:
        raise ValueError(
            f"Expert 2 ASE codebook must have shape [codes, D]; got {tuple(codebook.shape)}"
        )
    codebook_dim = int(codebook.shape[1])
    if audio_dim != codebook_dim or text_dim != codebook_dim:
        raise ValueError(
            "Expert 2 audio/text projection widths must match the shared MSC codebook: "
            f"frame_proj={audio_dim}, word_proj={text_dim}, codebook={codebook_dim}"
        )
    return audio_dim, text_dim


def _create_original_expert2(
    source_dir: Path,
    architecture_config: Mapping,
    pretrained_checkpoint: Path,
    device: torch.device,
    freeze_policy: str,
) -> nn.Module:
    """Call the original Expert 2 factory without importing MGA at module load."""
    from expert2_finetune.expert2_model import create_expert2

    return create_expert2(
        source_dir,
        architecture_config,
        pretrained_checkpoint,
        device,
        freeze_policy=freeze_policy,
    )


class Expert2Adapter(ExpertAdapter):
    """Expose the original MGA-CLAP score and post-MSC audio/text features.

    Args:
        model: Already constructed original ``expert2_model.Expert2`` (or a
            lightweight compatible ``nn.Module`` for contract tests).
        feature_dim: Optional trusted total feature width. When the ASE model
            exposes its two projection layers and shared MSC codebook, their
            dimensions are used and any explicit value must agree.

    The wrapped expert is registered as a child module, fully frozen, and kept
    in evaluation mode even if an outer model calls ``adapter.train()``.
    """

    def __init__(self, model: nn.Module, *, feature_dim: Optional[int] = None) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expert 2 model must be an nn.Module; got {type(model).__name__}")
        if feature_dim is not None and (type(feature_dim) is not int or feature_dim <= 0):
            raise ValueError(
                f"Expert 2 feature_dim must be a positive integer; got {feature_dim!r}"
            )

        dimensions = _msc_feature_dimensions(model)
        metadata_dim = sum(dimensions) if dimensions is not None else None
        if feature_dim is None:
            feature_dim = metadata_dim
        elif metadata_dim is not None and feature_dim != metadata_dim:
            raise ValueError(
                "Expert 2 explicit feature_dim does not match ASE projection/codebook "
                f"metadata: {feature_dim} != {metadata_dim}"
            )
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError(
                "Expert 2 feature_dim is unavailable; construct the original ASE model "
                "or provide a positive width from trusted architecture metadata"
            )

        super().__init__(feature_dim=feature_dim)
        self.model = model
        self._freeze_model()

    @staticmethod
    def _restore_finetuned_state_dict(model: nn.Module, checkpoint: Path) -> None:
        """Safely load the raw full Expert 2 best state dict with strict keys."""
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Expert 2 could not safely read fine-tuned checkpoint {checkpoint}: {exc}"
            ) from exc
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                f"Expert 2 fine-tuned checkpoint {checkpoint} must be a non-empty raw tensor state_dict"
            )
        if any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError(
                f"Expert 2 fine-tuned checkpoint {checkpoint} must be a raw tensor state_dict, "
                "not a structured resume bundle"
            )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Expert 2 strict fine-tuned state restoration failed for {checkpoint}: {exc}"
            ) from exc

    @classmethod
    def from_finetuned(
        cls,
        *,
        source_dir: PathValue,
        architecture_yaml: PathValue,
        training_config: PathValue,
        pretrained_checkpoint: PathValue,
        finetuned_checkpoint: PathValue,
        freeze_policy: Optional[str] = None,
        feature_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        offline: bool = True,
    ) -> "Expert2Adapter":
        """Build through the original MGA factory and strictly restore Expert2.

        All paths must already be resolved absolute paths, as returned by
        ``configs.moe_config.resolve_moe_paths``. The MGA pretrained loader is
        reused unchanged, including its nested ``checkpoint['model']`` and
        legacy BERT ``position_ids`` handling. Its pretrained loader currently
        uses unrestricted ``torch.load``; only trusted local artifacts may be
        supplied. The fine-tuned best state is tensor-only and loaded with
        ``weights_only=True`` plus ``strict=True``. There is no pretrained-only
        fallback.
        """
        if type(offline) is not bool:
            raise TypeError(f"Expert 2 offline must be a boolean; got {offline!r}")
        if not offline:
            raise ValueError("Expert 2 restoration is offline-only; set runtime.offline: true")
        target_device = _target_device(device)
        source_path = _resolved_resource_path(
            source_dir, field="source_dir", resource_type="directory"
        )
        architecture_path = _resolved_resource_path(
            architecture_yaml, field="architecture_yaml", resource_type="architecture YAML"
        )
        training_path = _resolved_resource_path(
            training_config, field="training_config", resource_type="training config"
        )
        pretrained_path = _resolved_resource_path(
            pretrained_checkpoint,
            field="pretrained_checkpoint",
            resource_type="pretrained checkpoint",
        )
        finetuned_path = _resolved_resource_path(
            finetuned_checkpoint,
            field="finetuned_checkpoint",
            resource_type="fine-tuned checkpoint",
        )

        ase_source = source_path / "models" / "ase_model.py"
        if not ase_source.is_file():
            raise FileNotFoundError(
                f"Expert 2 MGA source directory {source_path} is missing models/ase_model.py"
            )

        architecture = cls._load_yaml_mapping(architecture_path, field="architecture_yaml")
        cls._validate_architecture_config(architecture, architecture_path)
        training = cls._load_yaml_mapping(training_path, field="training_config")
        configured_policy = training.get("freeze_policy")
        if configured_policy not in {"partial", "full"}:
            raise ValueError(
                f"Expert 2 training config {training_path} must specify freeze_policy 'partial' or 'full'"
            )
        if freeze_policy is None:
            freeze_policy = configured_policy
        elif freeze_policy not in {"partial", "full"}:
            raise ValueError("Expert 2 freeze_policy must be 'partial' or 'full'")
        elif freeze_policy != configured_policy:
            raise ValueError(
                "Expert 2 freeze_policy argument does not match the original training config: "
                f"{freeze_policy!r} != {configured_policy!r}"
            )

        validate_mga_import_namespace(source_path)
        with _huggingface_offline(offline):
            try:
                model = _create_original_expert2(
                    source_path,
                    architecture,
                    pretrained_path,
                    target_device,
                    freeze_policy,
                )
            except Exception as exc:
                try:
                    # A conflicting vendor namespace may have been imported
                    # immediately before construction failed. Prefer that
                    # actionable origin diagnostic over a generic factory error.
                    validate_mga_import_namespace(source_path)
                except MGAImportNamespaceError as namespace_error:
                    raise namespace_error from exc
                raise RuntimeError(
                    "Expert 2 original MGA create_expert2 factory failed; check local "
                    "tokenizer/config resources and vendor import diagnostics"
                ) from exc
        validate_mga_import_namespace(source_path)
        if not isinstance(model, nn.Module):
            raise TypeError(
                "Expert 2 original create_expert2 factory must return an nn.Module; "
                f"got {type(model).__name__}"
            )
        cls._restore_finetuned_state_dict(model, finetuned_path)

        adapter = cls(model, feature_dim=feature_dim)
        return adapter.to(target_device).eval()

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Expert2Adapter":
        """Construct from a validated and path-resolved LOCAL-03 config.

        Accepts a full mapping returned by ``resolve_moe_paths`` or the
        ``experts.expert2`` section. Only Expert 2's five resources are checked;
        unavailable resources belonging to Experts 1, 3, and 4 are irrelevant.
        """
        if not isinstance(config, Mapping):
            raise TypeError("Expert 2 config must be a mapping")
        if "experts" in config:
            try:
                expert_config = config["experts"]["expert2"]
            except (KeyError, TypeError) as exc:
                raise ValueError("Expert 2 config is missing experts.expert2") from exc
            runtime_config = config.get("runtime", {})
        else:
            expert_config = config
            runtime_config = {}
        if not isinstance(expert_config, Mapping):
            raise TypeError("Expert 2 config section must be a mapping")
        if not isinstance(runtime_config, Mapping):
            raise TypeError("Expert 2 runtime config must be a mapping")
        try:
            resources = {
                key: expert_config[key]
                for key in (
                    "source_dir",
                    "architecture_yaml",
                    "training_config",
                    "pretrained_checkpoint",
                    "finetuned_checkpoint",
                )
            }
            configured_policy = expert_config["freeze_policy"]
        except KeyError as exc:
            raise ValueError(f"Expert 2 config is missing {exc.args[0]}") from exc
        selected_device = device if device is not None else runtime_config.get("device", "cpu")
        offline = runtime_config.get("offline", True)
        if type(offline) is not bool:
            raise TypeError("Expert 2 runtime.offline must be a boolean")
        return cls.from_finetuned(
            **resources,
            freeze_policy=configured_policy,
            device=selected_device,
            offline=offline,
        )

    @staticmethod
    def _load_yaml_mapping(path: Path, *, field: str) -> Mapping:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Expert 2 could not read {field} {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"Expert 2 {field} {path} must contain a YAML mapping")
        return value

    @staticmethod
    def _validate_architecture_config(config: Mapping, path: Path) -> None:
        required = {"embed_size", "audio_encoder_args", "text_encoder_args", "training"}
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(
                f"Expert 2 architecture config {path} is missing required fields: {missing}"
            )
        embedding_dim = config["embed_size"]
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(
                f"Expert 2 architecture config {path} embed_size must be a positive integer"
            )
        for field in ("audio_encoder_args", "text_encoder_args", "training"):
            if not isinstance(config[field], Mapping):
                raise ValueError(
                    f"Expert 2 architecture config {path} field {field!r} must be a mapping"
                )

    def _freeze_model(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "Expert2Adapter":
        """Preserve parent mode while forcing the entire frozen expert to eval."""
        super().train(mode)
        self.model.eval()
        return self

    def _validate_batch(self, batch: Any) -> tuple[torch.Tensor, list[str]]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expert 2 batch must be a mapping; got {type(batch).__name__}")
        if "expert2_wavs" not in batch:
            raise KeyError("Expert 2 batch is missing required field 'expert2_wavs'")
        if "captions_raw" not in batch:
            raise KeyError("Expert 2 batch is missing required field 'captions_raw'")

        audio = batch["expert2_wavs"]
        if not isinstance(audio, torch.Tensor):
            raise TypeError(
                f"Expert 2 expert2_wavs must be a torch.Tensor; got {type(audio).__name__}"
            )
        if audio.ndim != 3:
            raise ValueError(
                f"Expert 2 expert2_wavs must have rank 3 [B,1,320000]; got {tuple(audio.shape)}"
            )
        if audio.shape[0] <= 0:
            raise ValueError("Expert 2 expert2_wavs batch must be non-empty")
        if audio.shape[1] != 1:
            raise ValueError(
                f"Expert 2 expert2_wavs must have one channel; got {audio.shape[1]}"
            )
        if audio.shape[2] != _AUDIO_SAMPLES:
            raise ValueError(
                f"Expert 2 expert2_wavs must have {_AUDIO_SAMPLES} samples (10 seconds at "
                f"32 kHz); got {audio.shape[2]}"
            )
        if audio.dtype != torch.float32:
            raise TypeError(
                f"Expert 2 expert2_wavs must use float32; got {audio.dtype}"
            )
        model_device = self._model_device()
        if model_device is not None and audio.device != model_device:
            raise ValueError(
                "Expert 2 expert2_wavs and wrapped model must be on the same device; "
                f"got {audio.device} and {model_device}"
            )
        if not torch.isfinite(audio).all().item():
            raise ValueError("Expert 2 expert2_wavs must contain only finite values")

        captions = batch["captions_raw"]
        if isinstance(captions, (str, bytes, torch.Tensor)) or not isinstance(
            captions, Sequence
        ):
            raise TypeError(
                "Expert 2 captions_raw must be a sequence of raw strings; "
                f"got {type(captions).__name__}"
            )
        if len(captions) != audio.shape[0]:
            raise ValueError(
                f"Expert 2 captions_raw must contain exactly {audio.shape[0]} caption(s); "
                f"got {len(captions)}"
            )
        if any(not isinstance(caption, str) for caption in captions):
            raise TypeError("Expert 2 every caption must be a string")
        processed_captions = [caption.strip() for caption in captions]
        if any(not caption for caption in processed_captions):
            raise ValueError(
                "Expert 2 captions_raw cannot contain text that is empty after Expert 2 whitespace stripping"
            )
        return audio, processed_captions

    def _model_device(self) -> Optional[torch.device]:
        for tensor in self.model.parameters():
            return tensor.device
        for tensor in self.model.buffers():
            return tensor.device
        return None

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = self._validate_batch(batch)
        original_batch = {"mga_wavs": audio, "caption_tokens": captions}
        # Original Expert2 already returns its cosine score [B,1] and ordered
        # post-MSC [audio,text] features. Preserve both unchanged in one call.
        return self.model(original_batch)


__all__ = [
    "Expert2Adapter",
    "MGAImportNamespaceError",
    "validate_mga_import_namespace",
]
