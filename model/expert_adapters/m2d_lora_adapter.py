"""Standalone, frozen inference adapter for the E34 M2D-LoRA expert."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

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
from model.expert_adapters.m2d_lora_runtime import (
    EMBEDDING_DIM,
    E34CheckpointError,
    FEATURE_DIM,
    Head,
    checkpoint_inference_config,
    encode_pair,
    inject_lora,
    inter_feats,
    load_e34_payload,
    restore_lora_parameters,
    to_moescore_scale,
    to_score,
)


_ADAPTER_NAME = "M2DLoRAAdapter"
_AUDIO_SAMPLES = 160_000
_SAMPLE_RATE = 16_000


def _text_encoder_cache_dir(cache_dir: Any = None) -> Path | None:
    """Adapt an HF_HOME root to the explicit cache_dir used by E34's BERT loader."""
    value = cache_dir if cache_dir is not None else os.environ.get("HF_HOME")
    if value is None:
        return None  # Transformers then uses its standard ~/.cache/huggingface/hub path.
    root = Path(value).expanduser().resolve(strict=False)
    hub_cache = root / "hub"
    if hub_cache.is_dir():
        return hub_cache
    if root.is_dir() and any(root.glob("models--*")):
        return root
    # A not-yet-populated directory remains a valid Transformers cache target.
    return root


@contextmanager
def _compatible_checkpoint_path(
    checkpoint: Path,
    layout_dir: Path | None,
) -> Iterator[Path]:
    """Give PortableM2D the model-directory name parsed by its source loader."""
    from expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import (
        prepare_m2d_checkpoint,
    )

    if layout_dir is not None:
        yield prepare_m2d_checkpoint(checkpoint, layout_dir)
        return
    with tempfile.TemporaryDirectory(prefix="moescore-m2d-lora-layout-") as temporary:
        yield prepare_m2d_checkpoint(checkpoint, temporary)


def _construct_e34_model(
    base_checkpoint: Path,
    payload: Mapping[str, Any],
    *,
    lora_config: Mapping[str, int],
    device: torch.device,
    offline: bool,
    cache_dir: Any,
    layout_dir: Path | None,
) -> tuple[nn.Module, Head]:
    """Restore backbone, text encoder, named LoRA tensors and the original head."""
    try:
        from expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d import (
            PortableM2D,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{_ADAPTER_NAME} cannot import MOESCORE's checked-in PortableM2D runtime; "
            "the source dependencies timm, einops and nnAudio must be installed"
        ) from exc

    try:
        with _compatible_checkpoint_path(base_checkpoint, layout_dir) as compatible_checkpoint:
            with huggingface_environment(offline=offline, cache_dir=cache_dir):
                text_cache = _text_encoder_cache_dir(cache_dir)
                if text_cache is not None:
                    # The notebook's BertXEncoder passes HF_HOME as the direct
                    # Transformers cache_dir, while standard HF_HOME stores files
                    # below its ``hub`` child.
                    os.environ["HF_HOME"] = str(text_cache)
                model = PortableM2D(str(compatible_checkpoint), flat_features=True)
                model.get_clap_text_encoder()
                model.requires_grad_(False)
                inject_lora(model, lora_config)
                restore_lora_parameters(model, payload["adapter"])
    except E34CheckpointError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{_ADAPTER_NAME} failed to restore the original E34 M2D architecture from "
            f"{base_checkpoint}; check PortableM2D dependencies and local BERT resources. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    model_rate = getattr(getattr(model, "cfg", None), "sample_rate", None)
    if model_rate != _SAMPLE_RATE:
        raise E34CheckpointError(
            f"E34 M2D backbone expects sample_rate={model_rate!r}; "
            f"MOESCORE expert3_wavs is {_SAMPLE_RATE} Hz"
        )

    head = Head()
    try:
        head.load_state_dict(payload["head"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise E34CheckpointError(
            f"E34 prediction head state does not match the original 3073-to-11 Head: {exc}"
        ) from exc
    model.to(device)
    head.to(device)
    freeze_module(model)
    freeze_module(head)
    return model, head


class M2DLoRAAdapter(ExpertAdapter):
    """Use the trained E34 CORAL/regression head and 3073 E34 interaction features.

    ``from_config`` requires explicit ``base_checkpoint`` and
    ``finetuned_checkpoint`` paths. Importing this module does not load M2D,
    Transformers, tokenizer resources or checkpoint tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        *,
        label_stats: tuple[float, float],
        select_on: str = "mix",
        mix: float = 0.5,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} model must be an nn.Module")
        if not isinstance(head, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} head must be an nn.Module")
        if len(label_stats) != 2:
            raise ValueError(f"{_ADAPTER_NAME} label_stats must contain mean and standard deviation")
        mean, std = (float(label_stats[0]), float(label_stats[1]))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"{_ADAPTER_NAME} label_stats must be finite with positive std")
        if select_on not in {"coral", "reg", "mix"}:
            raise ValueError(f"{_ADAPTER_NAME} select_on must be coral, reg or mix")
        if not math.isfinite(float(mix)) or not 0.0 <= float(mix) <= 1.0:
            raise ValueError(f"{_ADAPTER_NAME} mix must be finite and in [0,1]")
        super().__init__(feature_dim=FEATURE_DIM)
        self.model = model
        self.head = head
        self.label_stats = (mean, std)
        self.select_on = select_on
        self.mix = float(mix)
        self._freeze_components()
        self.eval()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "M2DLoRAAdapter":
        section, runtime = adapter_config(config, "m2d_lora")
        for field in ("base_checkpoint", "finetuned_checkpoint"):
            if field not in section:
                raise ValueError(f"m2d_lora config is missing required {field}")
        base_checkpoint = resolved_resource(
            section["base_checkpoint"], field="base_checkpoint", adapter_name=_ADAPTER_NAME
        )
        finetuned_checkpoint = resolved_resource(
            section["finetuned_checkpoint"],
            field="finetuned_checkpoint",
            adapter_name=_ADAPTER_NAME,
        )
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_ADAPTER_NAME,
        )
        offline = runtime.get("offline", True)
        if type(offline) is not bool:
            raise TypeError(f"{_ADAPTER_NAME} runtime.offline must be boolean")
        cache_dir = runtime.get("cache_dir")
        layout_value = runtime.get("checkpoint_layout_dir")
        layout_dir = None
        if layout_value is not None:
            layout_dir = resolved_resource(
                layout_value,
                field="runtime.checkpoint_layout_dir",
                adapter_name=_ADAPTER_NAME,
                directory=True,
            )
            if not os.access(layout_dir, os.W_OK):
                raise PermissionError(f"{_ADAPTER_NAME} checkpoint layout directory is not writable: {layout_dir}")

        payload = load_e34_payload(str(finetuned_checkpoint))
        lora_config, stats, select_on, mix = checkpoint_inference_config(payload)
        model, head = _construct_e34_model(
            base_checkpoint,
            payload,
            lora_config=lora_config,
            device=selected_device,
            offline=offline,
            cache_dir=cache_dir,
            layout_dir=layout_dir,
        )
        return cls(model, head, label_stats=stats, select_on=select_on, mix=mix)

    def _freeze_components(self) -> None:
        freeze_module(self.model)
        freeze_module(self.head)

    def train(self, mode: bool = True) -> "M2DLoRAAdapter":
        # This wrapper is inference-only; keep the parent and both children eval.
        super().train(False)
        self._freeze_components()
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
            raise ValueError(
                f"{_ADAPTER_NAME} needs one caption per audio item; got {len(captions)} "
                f"for batch size {wavs.shape[0]}"
            )
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError(f"{_ADAPTER_NAME} captions_raw must contain non-empty strings")

        device = model_device(self.model) or wavs.device
        wavs = wavs.squeeze(1).to(device=device, dtype=torch.float32)
        audio_embedding, text_embedding = encode_pair(self.model, wavs, list(captions))
        if audio_embedding.shape != text_embedding.shape:
            raise ValueError(
                f"{_ADAPTER_NAME} normalized audio/text embeddings differ: "
                f"{tuple(audio_embedding.shape)} != {tuple(text_embedding.shape)}"
            )
        if audio_embedding.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"{_ADAPTER_NAME} expects {EMBEDDING_DIM}-wide embeddings; "
                f"got {audio_embedding.shape[1]}"
            )
        features = inter_feats(audio_embedding, text_embedding)
        if features.shape != (wavs.shape[0], FEATURE_DIM):
            raise ValueError(
                f"{_ADAPTER_NAME} E34 interaction features must be [B,{FEATURE_DIM}]; "
                f"got {tuple(features.shape)}"
            )
        head_output = self.head(features)
        if head_output.shape != (wavs.shape[0], 11):
            raise ValueError(
                f"{_ADAPTER_NAME} prediction head must return [B,11]; "
                f"got {tuple(head_output.shape)}"
            )
        score_raw = to_score(
            head_output,
            self.label_stats,
            select_on=self.select_on,
            mix=self.mix,
        )
        score = to_moescore_scale(score_raw).unsqueeze(-1).float()
        return score, features.float()


__all__ = ["M2DLoRAAdapter"]
