"""Pretrained M2D-CLAP adapter using the repository's PortableM2D source."""

from __future__ import annotations

from collections.abc import Mapping
import os
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
    target_device,
    validate_audio_batch,
)


_AUDIO_SAMPLES = 160_000
_SAMPLE_RATE = 16_000
_ADAPTER_NAME = "Top2M2DAdapter"


def _m2d_embedding_dims(model: nn.Module) -> tuple[Optional[int], Optional[int]]:
    backbone = getattr(model, "backbone", None)
    audio_dim = infer_module_output_dim(getattr(backbone, "audio_proj", None))
    text_dim = infer_module_output_dim(getattr(backbone, "text_proj", None))
    if text_dim is None:
        text_encoder = getattr(model, "text_encoder", None)
        text_dim = infer_module_output_dim(text_encoder)
        text_model = getattr(text_encoder, "model", None)
        if text_dim is None:
            text_dim = infer_module_output_dim(text_model)
        config = getattr(text_model, "config", None)
        if text_dim is None:
            value = getattr(config, "hidden_size", None)
            if type(value) is int and value > 0:
                text_dim = value
        sentence_dim = getattr(text_model, "get_sentence_embedding_dimension", None)
        if text_dim is None and callable(sentence_dim):
            value = sentence_dim()
            if type(value) is int and value > 0:
                text_dim = value
    return audio_dim, text_dim


class Top2M2DAdapter(ExpertAdapter):
    """Expose pretrained M2D-CLAP audio/text embeddings and their cosine.

    ``from_config`` expects ``checkpoint`` to be inside the model-named
    directory used by PortableM2D (the directory name carries its architecture
    and sample-rate metadata). The adapter uses MOESCORE's 16 kHz
    ``expert3_wavs`` tensor and does not load the fine-tuned Expert 3 state.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        embedding_dim: Optional[int] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} model must be an nn.Module")
        audio_dim, text_dim = _m2d_embedding_dims(model)
        if audio_dim is not None and text_dim is not None and audio_dim != text_dim:
            raise ValueError(
                f"{_ADAPTER_NAME} audio/text projection widths differ: {audio_dim} != {text_dim}"
            )
        metadata_dim = audio_dim if audio_dim is not None else text_dim
        if embedding_dim is None:
            embedding_dim = metadata_dim
        elif type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(f"{_ADAPTER_NAME} embedding_dim must be a positive integer")
        elif metadata_dim is not None and embedding_dim != metadata_dim:
            raise ValueError(
                f"{_ADAPTER_NAME} embedding_dim conflicts with model projection metadata: "
                f"{embedding_dim} != {metadata_dim}"
            )
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(
                f"{_ADAPTER_NAME} cannot determine the M2D-CLAP embedding width; "
                "restore the source checkpoint or provide a trusted embedding_dim"
            )
        super().__init__(feature_dim=5 * embedding_dim)
        self.embedding_dim = embedding_dim
        self.model = model
        self._freeze_model()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "Top2M2DAdapter":
        section, runtime = adapter_config(config, "top2_m2d")
        if "checkpoint" not in section:
            raise ValueError("top2_m2d config is missing checkpoint")
        # PortableM2D parses architecture metadata from the *logical* parent
        # directory name. Keep a symlink's model-named parent intact while
        # validating that its target is a readable checkpoint file.
        checkpoint_value = section["checkpoint"]
        if not isinstance(checkpoint_value, (str, os.PathLike)):
            raise TypeError(f"{_ADAPTER_NAME} checkpoint must be a filesystem path")
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            raise ValueError(
                f"{_ADAPTER_NAME} checkpoint must be an absolute path; got {checkpoint}"
            )
        checkpoint = Path(os.path.abspath(checkpoint))
        if not checkpoint.is_file() or not os.access(checkpoint, os.R_OK):
            raise FileNotFoundError(
                f"{_ADAPTER_NAME} checkpoint expected a readable file: {checkpoint}"
            )
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_ADAPTER_NAME,
        )
        offline = runtime.get("offline", True)
        cache_dir = runtime.get("cache_dir")

        if "-" not in checkpoint.parent.name:
            raise ValueError(
                f"{_ADAPTER_NAME} checkpoint parent directory must contain the PortableM2D "
                "model identifier and architecture metadata; set checkpoint under the "
                "published model directory"
            )
        try:
            with huggingface_environment(offline=offline, cache_dir=cache_dir):
                from expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d import (
                    PortableM2D,
                )

                model = PortableM2D(weight_file=str(checkpoint), flat_features=True)
                model.get_clap_text_encoder()
        except ImportError as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} could not import the checked-in PortableM2D source; "
                "install its torch, timm, einops, nnAudio and Transformers dependencies"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} failed to construct pretrained PortableM2D from "
                f"{checkpoint}; verify the checkpoint directory name and local text-encoder resources. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        model_rate = getattr(getattr(model, "cfg", None), "sample_rate", None)
        if model_rate != _SAMPLE_RATE:
            raise ValueError(
                f"{_ADAPTER_NAME} checkpoint expects sample_rate={model_rate!r}; "
                f"MOESCORE expert3_wavs is {_SAMPLE_RATE} Hz"
            )
        model.to(selected_device)
        adapter = cls(model, embedding_dim=section.get("embedding_dim"))
        return adapter.eval()

    def _freeze_model(self) -> None:
        freeze_module(self.model)

    def train(self, mode: bool = True) -> "Top2M2DAdapter":
        super().train(mode)
        self._freeze_model()
        return self

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = validate_audio_batch(
            batch,
            audio_key="expert3_wavs",
            sample_count=_AUDIO_SAMPLES,
            adapter_name=_ADAPTER_NAME,
            model=self.model,
        )
        selected_device = model_device(self.model) or audio.device
        with torch.inference_mode():
            audio_embedding = self.model.encode_clap_audio(audio.squeeze(1))
            text_embedding = self.model.encode_clap_text(captions)
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


__all__ = ["Top2M2DAdapter"]
