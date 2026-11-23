"""Frozen inference adapter for the E8 M2D-CLAP audio/text LoRA CORAL expert."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.top2_common import (
    adapter_config,
    freeze_module,
    huggingface_environment,
    model_device,
    resolved_resource,
    target_device,
)
from model.expert_adapters.expert8_m2d_clap_runtime import (
    CORAL_FEATURE_DIM,
    CoralHead,
    Expert8CheckpointError,
    encode_pair,
    inject_expert8_lora,
    inter_feats,
    load_expert8_checkpoint,
    load_expert8_head,
    restore_ordered_expert8_lora,
    validate_base_m2d_checkpoint,
    expert8_interaction_features,
)


_ADAPTER_NAME = "Expert8M2DCLAPAdapter"
_AUDIO_SAMPLES = 160_000
_SAMPLE_RATE = 16_000


def _construct_model(base_checkpoint: Path, payload: Mapping[str, Any], *, device: torch.device,
                     offline: bool, cache_dir: Any, text_model_dir: Path) -> tuple[nn.Module, CoralHead]:
    try:
        from expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import (
            prepare_m2d_checkpoint,
        )
        from expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d import (
            PortableM2D,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{_ADAPTER_NAME} cannot import MOESCORE's PortableM2D runtime; "
            "install the repository's timm, einops and nnAudio dependencies"
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="moescore-m2d-lora-coral-layout-") as temporary:
            compatible = prepare_m2d_checkpoint(base_checkpoint, temporary)
            with huggingface_environment(offline=offline, cache_dir=cache_dir):
                model = PortableM2D(str(compatible), flat_features=True)
                model.get_clap_text_encoder(text_model_dir=str(text_model_dir))
                model.requires_grad_(False)
                targets = inject_expert8_lora(model, payload["cfg"])
                restore_ordered_expert8_lora(model, payload["lora"], targets=targets)
    except Expert8CheckpointError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{_ADAPTER_NAME} could not restore the E8 architecture from {base_checkpoint}; "
            "verify the M2D checkpoint, BERT resource and compatible runtime. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    sample_rate = getattr(getattr(model, "cfg", None), "sample_rate", None)
    if sample_rate != _SAMPLE_RATE:
        raise Expert8CheckpointError(
            f"E8 M2D backbone must use {_SAMPLE_RATE} Hz; found {sample_rate!r}"
        )
    head = CoralHead()
    load_expert8_head(head, payload["head"])
    model.to(device)
    head.to(device)
    freeze_module(model)
    freeze_module(head)
    return model, head


class Expert8M2DCLAPAdapter(ExpertAdapter):
    """Restore E8 list LoRA weights and emit the original CORAL score/features."""

    def __init__(self, model: nn.Module, head: nn.Module) -> None:
        if not isinstance(model, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} requires model and head nn.Module instances")
        super().__init__(feature_dim=CORAL_FEATURE_DIM)
        self.model = model
        self.head = head
        freeze_module(self.model)
        freeze_module(self.head)
        self.eval()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "Expert8M2DCLAPAdapter":
        section, runtime = adapter_config(config, "expert8")
        for field in ("base_checkpoint", "finetuned_checkpoint", "text_model_dir"):
            if field not in section:
                raise ValueError(f"expert8 config is missing required {field}")
        text_model_dir = resolved_resource(
            section["text_model_dir"], field="text_model_dir", adapter_name=_ADAPTER_NAME,
            directory=True,
        )
        base_checkpoint = resolved_resource(
            section["base_checkpoint"], field="base_checkpoint", adapter_name=_ADAPTER_NAME
        )
        validate_base_m2d_checkpoint(base_checkpoint)
        finetuned_checkpoint = resolved_resource(
            section["finetuned_checkpoint"], field="finetuned_checkpoint", adapter_name=_ADAPTER_NAME
        )
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"), adapter_name=_ADAPTER_NAME
        )
        offline = runtime.get("offline", True)
        if type(offline) is not bool:
            raise TypeError(f"{_ADAPTER_NAME} runtime.offline must be boolean")
        cache_dir = runtime.get("cache_dir")
        payload = load_expert8_checkpoint(finetuned_checkpoint)
        model, head = _construct_model(
            base_checkpoint, payload, device=selected_device, offline=offline, cache_dir=cache_dir,
            text_model_dir=text_model_dir,
        )
        return cls(model, head)

    def train(self, mode: bool = True) -> "Expert8M2DCLAPAdapter":
        # Expert models remain frozen/eval when this adapter sits in a trainable MoE.
        super().train(False)
        freeze_module(self.model)
        freeze_module(self.head)
        return self

    @torch.inference_mode()
    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"{_ADAPTER_NAME} batch must be a mapping")
        if "expert3_wavs" not in batch:
            raise KeyError(f"{_ADAPTER_NAME} batch is missing required field 'expert3_wavs'")
        if "captions_raw" not in batch:
            raise KeyError(f"{_ADAPTER_NAME} batch is missing required field 'captions_raw'")
        wavs = batch["expert3_wavs"]
        if not isinstance(wavs, torch.Tensor):
            raise TypeError(f"{_ADAPTER_NAME} expert3_wavs must be a torch.Tensor")
        if wavs.ndim != 3 or tuple(wavs.shape[1:]) != (1, _AUDIO_SAMPLES):
            raise ValueError(
                f"{_ADAPTER_NAME} expert3_wavs must have shape [B,1,{_AUDIO_SAMPLES}]; "
                f"got {tuple(wavs.shape)}"
            )
        if wavs.shape[0] == 0 or wavs.dtype != torch.float32:
            raise ValueError(f"{_ADAPTER_NAME} expert3_wavs must be non-empty float32 audio")
        if not torch.isfinite(wavs).all().item():
            raise ValueError(f"{_ADAPTER_NAME} expert3_wavs contains non-finite values")
        captions = batch["captions_raw"]
        if isinstance(captions, (str, bytes, torch.Tensor)) or not isinstance(captions, Sequence):
            raise TypeError(f"{_ADAPTER_NAME} captions_raw must be a sequence of strings")
        if len(captions) != wavs.shape[0]:
            raise ValueError(f"{_ADAPTER_NAME} requires one caption per audio sample")
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError(f"{_ADAPTER_NAME} captions_raw must contain non-empty strings")

        selected_device = model_device(self.model) or wavs.device
        wavs = wavs.squeeze(1).to(device=selected_device, dtype=torch.float32)
        audio, text = encode_pair(self.model, wavs, list(captions))
        if tuple(audio.shape) != (wavs.shape[0], 768) or tuple(text.shape) != (wavs.shape[0], 768):
            raise ValueError(f"{_ADAPTER_NAME} encoders must return audio/text embeddings [B,768]")
        features = expert8_interaction_features(audio, text)
        if tuple(features.shape) != (wavs.shape[0], CORAL_FEATURE_DIM):
            raise ValueError(f"{_ADAPTER_NAME} E8 features must have shape [B,{CORAL_FEATURE_DIM}]")
        logits = self.head(features)
        if tuple(logits.shape) != (wavs.shape[0], 10):
            raise ValueError(f"{_ADAPTER_NAME} CORAL head must return [B,10]")
        raw_score = torch.sigmoid(logits.float()).sum(dim=-1, keepdim=True)
        normalized_score = ((raw_score - 5.0) / 5.0).float()
        return normalized_score, features


__all__ = ["Expert8M2DCLAPAdapter"]
