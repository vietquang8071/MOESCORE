"""E5-only MGA extraction path matching the Top 2 source waveform contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.top2_common import freeze_module, model_device, pack_embeddings


_AUDIO_KEY = "expert3_wavs"
_CAPTION_KEY = "captions_raw"
_SAMPLE_RATE_HZ = 16_000
_AUDIO_SAMPLES = 160_000
_EMBEDDING_DIM = 1024


class SetAMGASourceAdapter(ExpertAdapter):
    """Use source Top 2's 16 kHz/160,000-sample loader contract for MGA-CLAP.

    This E5-only adapter reuses the same checked-in ASE class, checkpoint
    factory, codebook ``msc`` aggregation, and source normalization as
    ``Top2MGAAdapter``. It intentionally consumes ``expert3_wavs`` (the 16 kHz
    waveform emitted by the original Top 2 dataset), not MOESCORE's resampled
    32 kHz ``expert2_wavs`` input. It is for feature extraction only; residual
    training and inference use cached outputs through ``SetAResidualAdapter``.
    """

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("SetAMGASourceAdapter model must be an nn.Module")
        super().__init__(feature_dim=4 * _EMBEDDING_DIM + 1)
        self.model = model
        self._freeze_model()
        self.eval()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, device: Any = None) -> "SetAMGASourceAdapter":
        from model.expert_adapters.top2_mga_adapter import Top2MGAAdapter

        restored = Top2MGAAdapter.from_config(config, device=device)
        adapter = cls(restored.model)
        adapter.checkpoint_load_report = restored.checkpoint_load_report
        return adapter.eval()

    def _freeze_model(self) -> None:
        freeze_module(self.model)

    def train(self, mode: bool = True) -> "SetAMGASourceAdapter":
        super().train(mode)
        self._freeze_model()
        return self

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("SetAMGASourceAdapter batch must be a mapping")
        audio = batch.get(_AUDIO_KEY)
        captions = batch.get(_CAPTION_KEY)
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"SetAMGASourceAdapter batch[{_AUDIO_KEY!r}] must be a tensor")
        if audio.ndim != 3 or audio.shape[1:] != (1, _AUDIO_SAMPLES):
            raise ValueError(
                f"SetAMGASourceAdapter expects {_AUDIO_KEY} shape [B,1,{_AUDIO_SAMPLES}] "
                f"at {_SAMPLE_RATE_HZ} Hz; got {tuple(audio.shape)}"
            )
        if audio.shape[0] <= 0:
            raise ValueError("SetAMGASourceAdapter audio batch cannot be empty")
        if audio.dtype != torch.float32:
            raise TypeError("SetAMGASourceAdapter audio must use float32")
        if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
            raise TypeError("SetAMGASourceAdapter captions_raw must be a sequence of strings")
        if len(captions) != audio.shape[0]:
            raise ValueError("SetAMGASourceAdapter captions batch size differs from audio")
        if any(not isinstance(caption, str) for caption in captions):
            raise TypeError("SetAMGASourceAdapter captions_raw entries must be strings")
        if not torch.isfinite(audio).all().item():
            raise ValueError("SetAMGASourceAdapter audio must contain only finite values")

        selected_device = model_device(self.model) or audio.device
        waveform = audio.to(device=selected_device).squeeze(1)
        with torch.inference_mode():
            _, frame_embeddings = self.model.encode_audio(waveform)
            audio_embedding = self.model.msc(frame_embeddings, self.model.codebook)
            audio_embedding = F.normalize(audio_embedding, dim=-1)
            _, word_embeddings, attention_mask = self.model.encode_text(list(captions))
            text_embedding = self.model.msc(
                word_embeddings,
                self.model.codebook,
                attention_mask,
            )
            text_embedding = F.normalize(text_embedding, dim=-1)
        score, paired_embeddings = pack_embeddings(
            audio_embedding,
            text_embedding,
            batch_size=audio.shape[0],
            embedding_dim=_EMBEDDING_DIM,
            output_device=selected_device,
            adapter_name=type(self).__name__,
        )
        audio_embedding, text_embedding = paired_embeddings.chunk(2, dim=-1)
        features = build_paired_cross_features(audio_embedding, text_embedding)
        return score, features


__all__ = ["SetAMGASourceAdapter"]
