"""Frozen inference adapter for the E31 M2D-CLAP audio/text LoRA CORAL expert."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import os
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
from model.expert_adapters.m2d_lora_coral_runtime import (
    CORAL_FEATURE_DIM,
    CoralHead,
    M2DLoRACoralCheckpointError,
    encode_pair,
    inject_coral_lora,
    inter_feats,
    load_coral_checkpoint,
    load_coral_head,
    restore_ordered_lora,
)


_ADAPTER_NAME = "M2DLoRACoralAdapter"
_AUDIO_SAMPLES = 160_000
_SAMPLE_RATE = 16_000


def _text_cache_path(cache_dir: Any = None) -> Path | None:
    value = cache_dir if cache_dir is not None else os.environ.get("HF_HOME")
    if value is None:
        return None
    root = Path(value).expanduser().resolve(strict=False)
    hub = root / "hub"
    if hub.is_dir():
        return hub
    if root.is_dir() and any(root.glob("models--*")):
        return root
    return root


def _construct_model(base_checkpoint: Path, payload: Mapping[str, Any], *, device: torch.device,
                     offline: bool, cache_dir: Any) -> tuple[nn.Module, CoralHead]:
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
                text_cache = _text_cache_path(cache_dir)
                if text_cache is not None:
                    # E31's BertXEncoder accepts a direct cache_dir; HF_HOME normally
                    # points one directory above the actual hub snapshot directory.
                    os.environ["HF_HOME"] = str(text_cache)
                model = PortableM2D(str(compatible), flat_features=True)
                model.get_clap_text_encoder()
                model.requires_grad_(False)
                targets = inject_coral_lora(model, payload["cfg"])
                restore_ordered_lora(model, payload["lora"], targets=targets)
    except M2DLoRACoralCheckpointError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{_ADAPTER_NAME} could not restore the E31 architecture from {base_checkpoint}; "
            "verify the M2D checkpoint, BERT resource and compatible runtime. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    sample_rate = getattr(getattr(model, "cfg", None), "sample_rate", None)
    if sample_rate != _SAMPLE_RATE:
        raise M2DLoRACoralCheckpointError(
            f"E31 M2D backbone must use {_SAMPLE_RATE} Hz; found {sample_rate!r}"
        )
    head = CoralHead()
    load_coral_head(head, payload["head"])
    model.to(device)
    head.to(device)
    freeze_module(model)
    freeze_module(head)
    return model, head


class M2DLoRACoralAdapter(ExpertAdapter):
    """Restore E31 list LoRA weights and emit the original CORAL score/features."""

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
    def from_config(cls, config: Mapping, *, device: Any = None) -> "M2DLoRACoralAdapter":
        section, runtime = adapter_config(config, "m2d_lora_coral")
        for field in ("base_checkpoint", "finetuned_checkpoint"):
            if field not in section:
                raise ValueError(f"m2d_lora_coral config is missing required {field}")
        base_checkpoint = resolved_resource(
            section["base_checkpoint"], field="base_checkpoint", adapter_name=_ADAPTER_NAME
        )
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
        payload = load_coral_checkpoint(finetuned_checkpoint)
        model, head = _construct_model(
            base_checkpoint, payload, device=selected_device, offline=offline, cache_dir=cache_dir
        )
        return cls(model, head)

    def train(self, mode: bool = True) -> "M2DLoRACoralAdapter":
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
        features = inter_feats(audio, text).float()
        if tuple(features.shape) != (wavs.shape[0], CORAL_FEATURE_DIM):
            raise ValueError(f"{_ADAPTER_NAME} E31 features must have shape [B,{CORAL_FEATURE_DIM}]")
        logits = self.head(features)
        if tuple(logits.shape) != (wavs.shape[0], 10):
            raise ValueError(f"{_ADAPTER_NAME} CORAL head must return [B,10]")
        raw_score = torch.sigmoid(logits.float()).sum(dim=-1, keepdim=True)
        normalized_score = ((raw_score - 5.0) / 5.0).float()
        return normalized_score, features


__all__ = ["M2DLoRACoralAdapter"]
