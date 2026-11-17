"""Pretrained MGA-CLAP adapter with a private vendor import namespace."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
import types
from typing import Any, Optional
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.top2_common import (
    adapter_config,
    freeze_module,
    huggingface_environment,
    infer_module_output_dim,
    model_device,
    pack_embeddings,
    resolved_resource,
    target_device,
    validate_audio_batch,
)


_AUDIO_SAMPLES = 320_000
_SAMPLE_RATE = 32_000
_ADAPTER_NAME = "Top2MGAAdapter"
_DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "expert2_finetune" / "MGA-CLAP"
_SPLITS = frozenset({"train", "validation", "test"})


def _ase_embedding_dim(model: nn.Module) -> Optional[int]:
    codebook = getattr(model, "codebook", None)
    codebook_dim = (
        int(codebook.shape[-1])
        if isinstance(codebook, torch.Tensor) and codebook.ndim == 2
        else None
    )
    frame_dim = infer_module_output_dim(getattr(model, "frame_proj", None))
    word_dim = infer_module_output_dim(getattr(model, "word_proj", None))
    widths = [value for value in (codebook_dim, frame_dim, word_dim) if value is not None]
    if widths and any(value != widths[0] for value in widths):
        raise ValueError(f"{_ADAPTER_NAME} ASE projection/codebook widths disagree: {widths}")
    return widths[0] if widths else None


def _private_vendor_package(source_dir: Path) -> str:
    suffix = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:12]
    package_name = f"_moescore_top2_mga_{suffix}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        locations = tuple(Path(value).resolve(strict=False) for value in getattr(existing, "__path__", ()))
        if locations != (source_dir,):
            raise ImportError(
                f"{_ADAPTER_NAME} private namespace {package_name} is already bound "
                f"to a different path: {locations}"
            )
        return package_name

    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(source_dir)]
    package.__file__ = None
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [str(source_dir)]
    sys.modules[package_name] = package
    return package_name


def _import_ase(source_dir: Path) -> type[nn.Module]:
    # Reuse MOESCORE's read-only namespace-origin guard, then import under a
    # path-specific private package instead of adding generic `models`/`tools`
    # names or mutating sys.path.
    try:
        from model.expert_adapters.expert2_adapter import validate_mga_import_namespace

        validate_mga_import_namespace(source_dir)
    except ImportError as exc:
        raise ImportError(f"{_ADAPTER_NAME} found a conflicting MGA import namespace: {exc}") from exc

    package_name = _private_vendor_package(source_dir)
    try:
        module = importlib.import_module(f"{package_name}.models.ase_model")
    except Exception as exc:
        raise ImportError(
            f"{_ADAPTER_NAME} could not import ASE from {source_dir}/models/ase_model.py"
        ) from exc
    origin = Path(module.__file__).resolve(strict=False)
    if not origin.is_relative_to(source_dir):
        raise ImportError(
            f"{_ADAPTER_NAME} ASE resolved outside its configured vendor tree: {origin}"
        )
    ase_class = getattr(module, "ASE", None)
    if not isinstance(ase_class, type) or not issubclass(ase_class, nn.Module):
        raise TypeError(f"{_ADAPTER_NAME} source module does not expose torch.nn.Module ASE")
    return ase_class


class Top2MGAAdapter(ExpertAdapter):
    """Expose pretrained ASE's multi-grained audio/text embeddings and cosine."""

    def __init__(
        self,
        model: nn.Module,
        *,
        embedding_dim: Optional[int] = None,
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} model must be an nn.Module")
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(
                f"{_ADAPTER_NAME} requires {_SAMPLE_RATE} Hz audio; got {sample_rate}"
            )
        metadata_dim = _ase_embedding_dim(model)
        if embedding_dim is None:
            embedding_dim = metadata_dim
        elif type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(f"{_ADAPTER_NAME} embedding_dim must be a positive integer")
        elif metadata_dim is not None and embedding_dim != metadata_dim:
            raise ValueError(
                f"{_ADAPTER_NAME} embedding_dim conflicts with ASE metadata: "
                f"{embedding_dim} != {metadata_dim}"
            )
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(f"{_ADAPTER_NAME} cannot determine the ASE projection width")
        super().__init__(feature_dim=2 * embedding_dim)
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.model = model
        self.checkpoint_load_report: Optional[dict[str, Any]] = None
        self._freeze_model()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "Top2MGAAdapter":
        section, runtime = adapter_config(config, "top2_mga")
        source_dir = resolved_resource(
            section.get("source_dir", _DEFAULT_SOURCE),
            field="source_dir",
            adapter_name=_ADAPTER_NAME,
            directory=True,
        )
        architecture_value = section.get(
            "architecture_yaml", source_dir / "settings" / "inference_example.yaml"
        )
        architecture_yaml = resolved_resource(
            architecture_value,
            field="architecture_yaml",
            adapter_name=_ADAPTER_NAME,
        )
        if "pretrained_checkpoint" not in section:
            raise ValueError("top2_mga config is missing pretrained_checkpoint")
        pretrained_checkpoint = resolved_resource(
            section["pretrained_checkpoint"],
            field="pretrained_checkpoint",
            adapter_name=_ADAPTER_NAME,
        )
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_ADAPTER_NAME,
        )
        offline = runtime.get("offline", True)
        cache_dir = runtime.get("cache_dir")
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(f"{_ADAPTER_NAME} requires PyYAML to read ASE configuration") from exc
        try:
            with architecture_yaml.open("r", encoding="utf-8") as stream:
                architecture = yaml.safe_load(stream)
        except Exception as exc:
            raise ValueError(
                f"{_ADAPTER_NAME} could not read MGA architecture YAML {architecture_yaml}: {exc}"
            ) from exc
        if not isinstance(architecture, Mapping):
            raise ValueError(f"{_ADAPTER_NAME} architecture YAML must contain a mapping")
        architecture = copy.deepcopy(dict(architecture))
        audio_config = architecture.get("audio_args")
        if not isinstance(audio_config, Mapping) or audio_config.get("sr") != _SAMPLE_RATE:
            raise ValueError(
                f"{_ADAPTER_NAME} architecture must set audio_args.sr={_SAMPLE_RATE}"
            )
        if audio_config.get("max_length") != 10:
            raise ValueError(f"{_ADAPTER_NAME} architecture must set audio_args.max_length=10")
        audio_encoder_config = architecture.get("audio_encoder_args")
        if not isinstance(audio_encoder_config, Mapping):
            raise ValueError(f"{_ADAPTER_NAME} architecture is missing audio_encoder_args")
        if audio_encoder_config.get("pretrained") is True:
            raise ValueError(
                f"{_ADAPTER_NAME} requires audio_encoder_args.pretrained=false; "
                "the supplied checkpoint provides the pretrained weights"
            )
        architecture["device"] = str(selected_device)

        try:
            with huggingface_environment(offline=offline, cache_dir=cache_dir):
                ase_class = _import_ase(source_dir)
                model = ase_class(architecture)
                checkpoint = torch.load(
                    pretrained_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
        except Exception as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} failed to construct ASE or read pretrained checkpoint "
                f"{pretrained_checkpoint}; verify local BERT resources and the source YAML. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
            raise ValueError(
                f"{_ADAPTER_NAME} checkpoint {pretrained_checkpoint} must contain a model state dictionary"
            )
        try:
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} source-compatible ASE checkpoint restoration failed: {exc}"
            ) from exc
        state_size = len(model.state_dict())
        matched = state_size - len(incompatible.missing_keys)
        if state_size == 0 or matched < 0.9 * state_size:
            raise RuntimeError(
                f"{_ADAPTER_NAME} checkpoint architecture mismatch: matched "
                f"{matched}/{state_size} tensors; missing={incompatible.missing_keys[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        load_report = {
            "matched_tensors": matched,
            "model_tensors": state_size,
            "missing_keys": tuple(incompatible.missing_keys),
            "unexpected_keys": tuple(incompatible.unexpected_keys),
        }
        if incompatible.missing_keys or incompatible.unexpected_keys:
            warnings.warn(
                f"{_ADAPTER_NAME} source-compatible checkpoint load matched "
                f"{matched}/{state_size} tensors; missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}",
                RuntimeWarning,
                stacklevel=2,
            )

        model.to(selected_device)
        adapter = cls(model, embedding_dim=section.get("embedding_dim"))
        adapter.checkpoint_load_report = load_report
        return adapter.eval()

    def _freeze_model(self) -> None:
        freeze_module(self.model)

    def train(self, mode: bool = True) -> "Top2MGAAdapter":
        super().train(mode)
        self._freeze_model()
        return self

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = validate_audio_batch(
            batch,
            audio_key="expert2_wavs",
            sample_count=_AUDIO_SAMPLES,
            adapter_name=_ADAPTER_NAME,
            model=self.model,
        )
        selected_device = model_device(self.model) or audio.device
        with torch.inference_mode():
            _, frame_embeddings = self.model.encode_audio(audio.squeeze(1))
            audio_embedding = self.model.msc(frame_embeddings, self.model.codebook)
            audio_embedding = F.normalize(audio_embedding, dim=-1)

            _, word_embeddings, attention_mask = self.model.encode_text(captions)
            text_embedding = self.model.msc(
                word_embeddings,
                self.model.codebook,
                attention_mask,
            )
            text_embedding = F.normalize(text_embedding, dim=-1)
        return pack_embeddings(
            audio_embedding,
            text_embedding,
            batch_size=audio.shape[0],
            embedding_dim=self.embedding_dim,
            output_device=selected_device,
            adapter_name=_ADAPTER_NAME,
        )


__all__ = ["Top2MGAAdapter"]
