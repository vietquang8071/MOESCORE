"""Pretrained LAION-CLAP adapter using paired audio/text projection heads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.paired_cross_features import build_paired_cross_features
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


_AUDIO_SAMPLES = 480_000
_SAMPLE_RATE = 48_000
_DEFAULT_MODEL_ID = "laion/clap-htsat-unfused"
_ADAPTER_NAME = "Top2LAIONAdapter"


def _projection_dim(model: nn.Module, explicit: Optional[int]) -> int:
    config = getattr(model, "config", None)
    metadata_dim = getattr(config, "projection_dim", None)
    audio_dim = infer_module_output_dim(getattr(model, "audio_projection", None))
    text_dim = infer_module_output_dim(getattr(model, "text_projection", None))
    native_core = getattr(model, "model", None)
    native_dim = getattr(native_core, "joint_embed_shape", None)
    if type(native_dim) is not int or native_dim <= 0:
        native_dim = None
    widths = [
        width
        for width in (metadata_dim, audio_dim, text_dim, native_dim)
        if type(width) is int and width > 0
    ]
    if widths and any(width != widths[0] for width in widths):
        raise ValueError(f"{_ADAPTER_NAME} projection dimensions disagree: {widths}")
    resolved = widths[0] if widths else explicit
    if type(resolved) is not int or resolved <= 0:
        raise ValueError(
            f"{_ADAPTER_NAME} cannot determine paired projection_dim from model metadata"
        )
    if explicit is not None and explicit != resolved:
        raise ValueError(
            f"{_ADAPTER_NAME} embedding_dim conflicts with model projection metadata: "
            f"{explicit} != {resolved}"
        )
    return resolved


def _processor_sample_rate(processor: Any) -> Optional[int]:
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None:
        feature_extractor = getattr(processor, "audio_processor", None)
    value = getattr(feature_extractor, "sampling_rate", None)
    return value if type(value) is int and value > 0 else None


class Top2LAIONAdapter(ExpertAdapter):
    """Expose paired, projected LAION-CLAP embeddings and their cosine.

    Unlike the source feature script's independent pooled hidden states from
    fused/unfused checkpoints, this adapter uses one Hugging Face ``ClapModel``
    checkpoint for both modalities. It returns that model's normalized
    projected embeddings (the checkpoint's shared CLAP space), whose width is
    ``config.projection_dim`` rather than the source's 768-wide hidden states.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any = None,
        *,
        embedding_dim: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} model must be an nn.Module")
        self._native_api = callable(
            getattr(model, "get_audio_embedding_from_data", None)
        ) and callable(getattr(model, "get_text_embedding", None))
        if not self._native_api and not callable(processor):
            raise TypeError(f"{_ADAPTER_NAME} processor must be callable")
        if self._native_api and processor is not None:
            raise ValueError(f"{_ADAPTER_NAME} native CLAP API does not take a separate processor")
        resolved_dim = _projection_dim(model, embedding_dim)
        processor_rate = _processor_sample_rate(processor)
        resolved_rate = (
            (_SAMPLE_RATE if self._native_api else processor_rate)
            if sample_rate is None
            else sample_rate
        )
        if type(resolved_rate) is not int or resolved_rate <= 0:
            raise ValueError(f"{_ADAPTER_NAME} processor does not expose a valid sampling_rate")
        if processor_rate is not None and processor_rate != resolved_rate:
            raise ValueError(
                f"{_ADAPTER_NAME} sample_rate conflicts with processor metadata: "
                f"{resolved_rate} != {processor_rate}"
            )
        if resolved_rate != _SAMPLE_RATE:
            raise ValueError(
                f"{_ADAPTER_NAME} requires the 48 kHz MOESCORE expert1_wavs input; "
                f"processor expects {resolved_rate} Hz"
            )
        super().__init__(feature_dim=4 * resolved_dim + 1)
        self.embedding_dim = resolved_dim
        self.sample_rate = resolved_rate
        self.model = model
        self.processor = processor
        self._freeze_model()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "Top2LAIONAdapter":
        section, runtime = adapter_config(config, "top2_laion")
        model_id = section.get("model_name_or_path", _DEFAULT_MODEL_ID)
        if not isinstance(model_id, (str, Path)) or not str(model_id).strip():
            raise TypeError("top2_laion model_name_or_path must be a model ID or local directory")
        model_id = str(Path(model_id).expanduser()) if Path(model_id).is_absolute() else str(model_id)
        if Path(model_id).is_absolute() and not Path(model_id).is_dir():
            raise FileNotFoundError(f"{_ADAPTER_NAME} local model directory does not exist: {model_id}")
        offline = runtime.get("offline", True)
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_ADAPTER_NAME,
        )
        cache_dir = runtime.get("cache_dir")
        if "checkpoint" in section:
            checkpoint = resolved_resource(
                section["checkpoint"],
                field="checkpoint",
                adapter_name=_ADAPTER_NAME,
                )
            try:
                with huggingface_environment(offline=offline, cache_dir=cache_dir):
                    from laion_clap import CLAP_Module

                    model = CLAP_Module(
                        enable_fusion=False,
                        device=str(selected_device),
                        amodel="HTSAT-tiny",
                    )
                    model.load_ckpt(ckpt=str(checkpoint), verbose=False)
            except ImportError as exc:
                raise RuntimeError(
                    f"{_ADAPTER_NAME} requires the laion-clap package for its native checkpoint"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"{_ADAPTER_NAME} failed to load native LAION-CLAP checkpoint {checkpoint}; "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            model.to(selected_device)
            adapter = cls(
                model,
                embedding_dim=section.get("embedding_dim"),
                sample_rate=_SAMPLE_RATE,
            )
            adapter.checkpoint_path = str(checkpoint)
            return adapter.eval()

        try:
            with huggingface_environment(offline=offline, cache_dir=cache_dir):
                from transformers import ClapModel, ClapProcessor

                options = {"local_files_only": offline}
                if cache_dir is not None:
                    options["cache_dir"] = str(cache_dir)
                model = ClapModel.from_pretrained(model_id, **options)
                processor = ClapProcessor.from_pretrained(model_id, **options)
        except Exception as exc:
            mode = "local cache/files" if offline else "model repository"
            raise RuntimeError(
                f"{_ADAPTER_NAME} could not load a matched model and processor from "
                f"{model_id!r} using {mode}; paired LAION audio/text projection weights are required. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        model.to(selected_device)
        adapter = cls(
            model,
            processor,
            embedding_dim=section.get("embedding_dim"),
        )
        return adapter.eval()

    def _freeze_model(self) -> None:
        freeze_module(self.model)

    def train(self, mode: bool = True) -> "Top2LAIONAdapter":
        super().train(mode)
        self._freeze_model()
        return self

    @staticmethod
    def _move_processor_values(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
        values = dict(inputs.items()) if hasattr(inputs, "items") else dict(inputs)
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = validate_audio_batch(
            batch,
            audio_key="expert1_wavs",
            sample_count=_AUDIO_SAMPLES,
            adapter_name=_ADAPTER_NAME,
            model=self.model,
        )
        selected_device = model_device(self.model) or audio.device
        if self._native_api:
            with torch.inference_mode():
                audio_embedding = self.model.get_audio_embedding_from_data(
                    audio[:, 0, :], use_tensor=True
                )
                text_embedding = self.model.get_text_embedding(
                    captions, use_tensor=True
                )
            score, original_features = pack_embeddings(
                audio_embedding,
                text_embedding,
                batch_size=audio.shape[0],
                embedding_dim=self.embedding_dim,
                output_device=selected_device,
                adapter_name=_ADAPTER_NAME,
            )
            audio_embedding, text_embedding = original_features.chunk(2, dim=-1)
            return score, build_paired_cross_features(audio_embedding, text_embedding)

        audio_values = audio[:, 0, :].float().detach().cpu().numpy()
        try:
            audio_inputs = self.processor(
                audio=[item for item in audio_values],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            text_inputs = self.processor(
                text=captions,
                padding=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError(f"{_ADAPTER_NAME} processor failed on the unified batch: {exc}") from exc
        audio_inputs = self._move_processor_values(audio_inputs, selected_device)
        text_inputs = self._move_processor_values(text_inputs, selected_device)
        if "input_features" not in audio_inputs or "input_ids" not in text_inputs:
            raise ValueError(
                f"{_ADAPTER_NAME} processor must return input_features and input_ids"
            )

        model_inputs = {
            "input_features": audio_inputs["input_features"],
            "input_ids": text_inputs["input_ids"],
        }
        if "is_longer" in audio_inputs:
            model_inputs["is_longer"] = audio_inputs["is_longer"]
        if "attention_mask" in text_inputs:
            model_inputs["attention_mask"] = text_inputs["attention_mask"]
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
        audio_embedding = getattr(outputs, "audio_embeds", None)
        text_embedding = getattr(outputs, "text_embeds", None)
        if audio_embedding is None or text_embedding is None:
            raise RuntimeError(
                f"{_ADAPTER_NAME} ClapModel output lacks paired audio_embeds/text_embeds"
            )
        score, original_features = pack_embeddings(
            audio_embedding,
            text_embedding,
            batch_size=audio.shape[0],
            embedding_dim=self.embedding_dim,
            output_device=selected_device,
            adapter_name=_ADAPTER_NAME,
        )
        audio_embedding, text_embedding = original_features.chunk(2, dim=-1)
        return score, build_paired_cross_features(audio_embedding, text_embedding)


__all__ = ["Top2LAIONAdapter"]
