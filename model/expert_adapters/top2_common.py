"""Shared validation and output helpers for standalone pretrained Top 2 adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import nn


def adapter_config(config: Mapping, section_name: str) -> tuple[Mapping, Mapping]:
    """Accept one adapter section or a future full config with ``experts``."""
    if not isinstance(config, Mapping):
        raise TypeError(f"{section_name} config must be a mapping")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise TypeError(f"{section_name} runtime config must be a mapping")
    if "experts" in config:
        try:
            section = config["experts"][section_name]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Config is missing experts.{section_name}") from exc
    elif section_name in config:
        section = config[section_name]
    else:
        section = config
    if not isinstance(section, Mapping):
        raise TypeError(f"{section_name} config section must be a mapping")
    return section, runtime


def target_device(value: Any, *, adapter_name: str) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"{adapter_name} device is invalid: {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{adapter_name} supports CPU or CUDA, got {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{adapter_name} requested {device}, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{adapter_name} requested {device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )
    return device


def resolved_resource(
    value: Any,
    *,
    field: str,
    adapter_name: str,
    directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{adapter_name} {field} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{adapter_name} {field} must be an absolute resolved path; got {path}")
    path = path.resolve(strict=False)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        expected = "directory" if directory else "file"
        error_type = NotADirectoryError if directory else FileNotFoundError
        raise error_type(f"{adapter_name} {field} expected a readable {expected}: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{adapter_name} {field} is not readable: {path}")
    return path


def model_device(module: nn.Module) -> Optional[torch.device]:
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors:
        return None
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"Wrapped model spans multiple devices: {sorted(map(str, devices))}")
    return next(iter(devices))


def validate_audio_batch(
    batch: Any,
    *,
    audio_key: str,
    sample_count: int,
    adapter_name: str,
    model: Optional[nn.Module] = None,
    require_model_device: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    if not isinstance(batch, Mapping):
        raise TypeError(f"{adapter_name} batch must be a mapping; got {type(batch).__name__}")
    if audio_key not in batch:
        raise KeyError(f"{adapter_name} batch is missing required field {audio_key!r}")
    if "captions_raw" not in batch:
        raise KeyError(f"{adapter_name} batch is missing required field 'captions_raw'")
    audio = batch[audio_key]
    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"{adapter_name} {audio_key} must be a torch.Tensor")
    if audio.ndim != 3 or audio.shape[1:] != (1, sample_count):
        raise ValueError(
            f"{adapter_name} {audio_key} must have shape [B,1,{sample_count}]; "
            f"got {tuple(audio.shape)}"
        )
    if audio.shape[0] <= 0:
        raise ValueError(f"{adapter_name} batch must be non-empty")
    if audio.dtype != torch.float32:
        raise TypeError(f"{adapter_name} {audio_key} must use torch.float32; got {audio.dtype}")
    if not torch.isfinite(audio).all().item():
        raise ValueError(f"{adapter_name} {audio_key} contains non-finite values")

    captions = batch["captions_raw"]
    if isinstance(captions, (str, bytes, torch.Tensor)) or not isinstance(captions, Sequence):
        raise TypeError(f"{adapter_name} captions_raw must be a sequence of strings")
    if len(captions) != audio.shape[0]:
        raise ValueError(
            f"{adapter_name} needs one caption per audio item; "
            f"got {len(captions)} for batch size {audio.shape[0]}"
        )
    if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
        raise ValueError(f"{adapter_name} captions_raw must contain non-empty strings")

    if model is not None and require_model_device:
        selected_device = model_device(model)
        if selected_device is not None and audio.device != selected_device:
            raise ValueError(
                f"{adapter_name} audio and model must be on the same device; "
                f"got {audio.device} and {selected_device}"
            )
    return audio, [caption.strip() for caption in captions]


def infer_module_output_dim(module: Any) -> Optional[int]:
    """Read a projection width from common PyTorch projection layer forms."""
    if not isinstance(module, nn.Module):
        return None
    for layer in reversed(tuple(module.modules())):
        if isinstance(layer, nn.Linear):
            return int(layer.out_features)
        normalized_shape = getattr(layer, "normalized_shape", None)
        if isinstance(normalized_shape, int) and normalized_shape > 0:
            return int(normalized_shape)
        if (
            isinstance(normalized_shape, Sequence)
            and normalized_shape
            and type(normalized_shape[-1]) is int
            and normalized_shape[-1] > 0
        ):
            return int(normalized_shape[-1])
    for name in ("embedding_dim", "output_dim", "hidden_size", "d_proj"):
        value = getattr(module, name, None)
        if type(value) is int and value > 0:
            return value
    return None


def pack_embeddings(
    audio_embedding: Any,
    text_embedding: Any,
    *,
    batch_size: int,
    embedding_dim: int,
    output_device: torch.device,
    adapter_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(embedding_dim) is not int or embedding_dim <= 0:
        raise ValueError(f"{adapter_name} embedding_dim must be a positive integer")
    if not isinstance(audio_embedding, torch.Tensor):
        try:
            audio_embedding = torch.as_tensor(audio_embedding)
        except Exception as exc:
            raise TypeError(f"{adapter_name} audio embedding is not tensor-like") from exc
    if not isinstance(text_embedding, torch.Tensor):
        try:
            text_embedding = torch.as_tensor(text_embedding)
        except Exception as exc:
            raise TypeError(f"{adapter_name} text embedding is not tensor-like") from exc
    audio_embedding = audio_embedding.to(device=output_device, dtype=torch.float32)
    text_embedding = text_embedding.to(device=output_device, dtype=torch.float32)
    for name, embedding in (("audio", audio_embedding), ("text", text_embedding)):
        if embedding.ndim != 2:
            raise ValueError(
                f"{adapter_name} {name} embedding must have shape [B,D]; "
                f"got {tuple(embedding.shape)}"
            )
        if embedding.shape[0] != batch_size:
            raise ValueError(
                f"{adapter_name} {name} embedding batch mismatch: "
                f"{embedding.shape[0]} != {batch_size}"
            )
        if embedding.shape[1] != embedding_dim:
            raise ValueError(
                f"{adapter_name} {name} embedding width mismatch: "
                f"expected {embedding_dim}, got {embedding.shape[1]}"
            )
        if not torch.isfinite(embedding).all().item():
            raise ValueError(f"{adapter_name} {name} embedding contains non-finite values")
    if audio_embedding.shape != text_embedding.shape:
        raise ValueError(
            f"{adapter_name} audio/text embedding shapes differ: "
            f"{tuple(audio_embedding.shape)} != {tuple(text_embedding.shape)}"
        )

    audio_normalized = F.normalize(audio_embedding, p=2, dim=-1)
    text_normalized = F.normalize(text_embedding, p=2, dim=-1)
    score = torch.sum(audio_normalized * text_normalized, dim=-1, keepdim=True)
    features = torch.cat((audio_embedding, text_embedding), dim=-1)
    return score, features


def freeze_module(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


@contextmanager
def huggingface_environment(*, offline: bool, cache_dir: Any = None) -> Iterator[None]:
    """Temporarily set Hugging Face offline/cache flags for lazy source loaders."""
    if type(offline) is not bool:
        raise TypeError(f"runtime.offline must be a boolean; got {offline!r}")
    if not offline and cache_dir is None:
        yield
        return
    keys = (
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "HF_MODULES_CACHE",
        "TRANSFORMERS_CACHE",
    )
    previous = {key: os.environ.get(key) for key in keys}
    cache_root = None
    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser().resolve(strict=False)
        cache_paths = {
            "HF_HOME": cache_root,
            "HF_HUB_CACHE": cache_root / "hub",
            "HF_ASSETS_CACHE": cache_root / "assets",
            "HF_XET_CACHE": cache_root / "xet",
            "HF_MODULES_CACHE": cache_root / "modules",
            "TRANSFORMERS_CACHE": cache_root / "hub",
        }
        for key, path in cache_paths.items():
            os.environ[key] = str(path)
    cache_keys = {
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "HF_MODULES_CACHE",
        "TRANSFORMERS_CACHE",
    }
    if offline:
        for key in keys[1:]:
            if key in cache_keys:
                continue
            os.environ[key] = "1"
    elif cache_root is not None:
        for key in keys[1:4]:
            os.environ.pop(key, None)
    module_overrides: dict[str, dict[str, Any]] = {
        "huggingface_hub.constants": {"HF_HUB_OFFLINE": offline},
        "transformers.utils.hub": {"HF_HUB_OFFLINE": offline},
    }
    if cache_root is not None:
        module_overrides["huggingface_hub.constants"].update(
            {
                "HF_HOME": str(cache_root),
                "HF_HUB_CACHE": str(cache_root / "hub"),
                "HF_ASSETS_CACHE": str(cache_root / "assets"),
                "HF_XET_CACHE": str(cache_root / "xet"),
            }
        )
        module_overrides["transformers.utils.hub"].update(
            {
                "HF_MODULES_CACHE": str(cache_root / "modules"),
                "TRANSFORMERS_CACHE": str(cache_root / "hub"),
                "default_cache_path": str(cache_root / "hub"),
            }
        )
    previous_attributes: dict[tuple[str, str], Any] = {}
    for module_name, attributes in module_overrides.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, value in attributes.items():
            if hasattr(module, name):
                previous_attributes[(module_name, name)] = getattr(module, name)
                setattr(module, name, value)
    try:
        yield
    finally:
        for (module_name, name), value in previous_attributes.items():
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, name, value)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


__all__ = [
    "adapter_config",
    "freeze_module",
    "huggingface_environment",
    "infer_module_output_dim",
    "model_device",
    "pack_embeddings",
    "resolved_resource",
    "target_device",
    "validate_audio_batch",
]
