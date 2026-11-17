"""Inference-only E34 model pieces copied from the authoritative notebook.

The M2D implementation itself is imported lazily from MOESCORE's checked-in
PortableM2D runtime. Its PortableM2D, BERT text encoder and relevant helpers
were compared against the notebook definitions and have identical ASTs.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


N_LEVELS = 10
EMBEDDING_DIM = 768
FEATURE_DIM = 4 * EMBEDDING_DIM + 1
_DEFAULT_LORA_CONFIG = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "audio_lora_blocks": 4,
    "text_lora_layers": 2,
}


class E34CheckpointError(ValueError):
    """Raised when a structured E34 checkpoint cannot restore this model."""


class LoRALinear(nn.Module):
    """The notebook's original linear LoRA wrapper, including checkpoint ABI."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def _get(module: nn.Module, name: str) -> nn.Module:
    return module[int(name)] if name.isdigit() and not hasattr(module, name) else getattr(module, name)


def _set(module: nn.Module, name: str, value: nn.Module) -> None:
    if name.isdigit() and not hasattr(module, name):
        module[int(name)] = value
    else:
        setattr(module, name, value)


def wrap_linear(root: nn.Module, path: str, r: int, alpha: int) -> str:
    """Replace one named Linear using the same traversal as the E34 notebook."""
    *parents, last = path.split(".")
    module = root
    for part in parents:
        module = _get(module, part)
    child = _get(module, last)
    if not isinstance(child, nn.Linear):
        raise TypeError(f"Expected Linear at {path}")
    _set(module, last, LoRALinear(child, r, alpha))
    return path


def _validated_lora_config(config: Mapping[str, Any] | None) -> dict[str, int]:
    values = dict(_DEFAULT_LORA_CONFIG if config is None else config)
    result: dict[str, int] = {}
    for name in _DEFAULT_LORA_CONFIG:
        value = values.get(name)
        if type(value) is not int or value <= 0:
            raise E34CheckpointError(f"E34 config {name} must be a positive integer; got {value!r}")
        result[name] = value
    if result["lora_alpha"] <= 0:
        raise E34CheckpointError("E34 config lora_alpha must be positive")
    return result


def inject_lora(model: nn.Module, config: Mapping[str, Any] | None = None) -> list[str]:
    """Inject adapters into the exact last audio/text layers selected by E34."""
    lora = _validated_lora_config(config)
    done = []
    blocks = model.backbone.blocks
    for index in range(max(0, len(blocks) - lora["audio_lora_blocks"]), len(blocks)):
        for target in ("attn.qkv", "attn.proj"):
            done.append(
                "audio." + wrap_linear(
                    model.backbone,
                    f"blocks.{index}.{target}",
                    lora["lora_rank"],
                    lora["lora_alpha"],
                )
            )
    bert = model.text_encoder.text_encoder
    layers = len(bert.encoder.layer)
    for index in range(max(0, layers - lora["text_lora_layers"]), layers):
        for target in ("attention.self.query", "attention.self.value"):
            done.append(
                "text." + wrap_linear(
                    bert,
                    f"encoder.layer.{index}.{target}",
                    lora["lora_rank"],
                    lora["lora_alpha"],
                )
            )
    return done


def restore_lora_parameters(model: nn.Module, saved: Mapping[str, Any]) -> None:
    """Restore every E34 trainable parameter, rejecting partial or foreign state."""
    if not isinstance(saved, Mapping):
        raise E34CheckpointError("E34 checkpoint 'adapter' must be a parameter mapping")
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    expected, actual = set(parameters), set(saved)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing LoRA parameters: {missing[:8]}")
        if unexpected:
            details.append(f"unexpected adapter parameters: {unexpected[:8]}")
        raise E34CheckpointError("E34 checkpoint parameter names do not match: " + "; ".join(details))

    for name, parameter in parameters.items():
        value = saved[name]
        if not isinstance(value, torch.Tensor):
            raise E34CheckpointError(f"E34 checkpoint parameter {name} must be a tensor")
        if tuple(value.shape) != tuple(parameter.shape):
            raise E34CheckpointError(
                f"E34 checkpoint parameter {name} has shape {tuple(value.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise E34CheckpointError(f"E34 checkpoint parameter {name} must be finite floating-point data")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))


class Head(nn.Module):
    """The notebook's 10-logit CORAL plus one regression prediction head."""

    def __init__(self, in_dim=4 * 768 + 1, hidden=1024, mid=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, N_LEVELS + 1),
        )
        with torch.no_grad():
            self.net[-1].weight[N_LEVELS].mul_(0.1)
            self.net[-1].bias[N_LEVELS].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def inter_feats(audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """The original E34 standardized audio/text interaction features."""
    audio = (audio - audio.mean(-1, keepdim=True)) / (audio.std(-1, keepdim=True) + 1e-6)
    text = (text - text.mean(-1, keepdim=True)) / (text.std(-1, keepdim=True) + 1e-6)
    return torch.cat(
        [audio, text, audio * text, (audio - text).abs(), (audio * text).sum(-1, keepdim=True)],
        -1,
    ).float()


def score_parts(
    output: torch.Tensor,
    stats: tuple[float, float],
    *,
    mix: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Produce the notebook's standardized CORAL, regression and mix values."""
    mu, sd = stats
    output = output.float()
    coral = (torch.sigmoid(output[:, :N_LEVELS]).sum(-1) - mu) / sd
    regression = output[:, N_LEVELS]
    return {
        "coral": coral,
        "reg": regression,
        "mix": (1 - mix) * coral + mix * regression,
    }


def to_score(
    output: torch.Tensor,
    stats: tuple[float, float],
    *,
    select_on: str = "mix",
    mix: float = 0.5,
) -> torch.Tensor:
    """Restore the E34 raw 0--10-scale prediction using saved label statistics."""
    if select_on not in {"coral", "reg", "mix"}:
        raise E34CheckpointError(f"E34 select_on must be coral, reg or mix; got {select_on!r}")
    mu, sd = stats
    return mu + sd * score_parts(output, stats, mix=mix)[select_on]


def to_moescore_scale(score_raw: torch.Tensor) -> torch.Tensor:
    """Convert an E34 raw score to MOESCORE's ``(score - 5) / 5`` target."""
    return (score_raw.float() - 5.0) / 5.0


def encode_pair(
    model: nn.Module,
    wavs: torch.Tensor,
    captions: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the exact normalized E34 audio and CLAP text embeddings."""
    features = model.to_normalized_feature(wavs)
    if wavs.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            frames = model.encode_lms(features)
    else:
        frames = model.encode_lms(features)
    audio_embedding = model.backbone.audio_proj(frames.float())
    text_embedding = model.encode_clap_text(captions, truncate=True)
    if audio_embedding.ndim != 2 or tuple(audio_embedding.shape) != (len(captions), EMBEDDING_DIM):
        raise ValueError(
            "E34 audio projection must return [B,768]; "
            f"got {tuple(audio_embedding.shape)}"
        )
    if text_embedding.ndim != 2 or tuple(text_embedding.shape) != (len(captions), EMBEDDING_DIM):
        raise ValueError(
            "E34 text encoder must return [B,768]; "
            f"got {tuple(text_embedding.shape)}"
        )
    if not torch.isfinite(audio_embedding).all().item() or not torch.isfinite(text_embedding).all().item():
        raise ValueError("E34 audio/text embeddings contain non-finite values")
    return F.normalize(audio_embedding.float(), dim=-1), F.normalize(text_embedding.float(), dim=-1)


def load_e34_payload(path: str) -> Mapping[str, Any]:
    """Safely load and validate the structured E34 checkpoint container."""
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older PyTorch: the configured local artifact is trusted.
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise E34CheckpointError(f"Could not load E34 fine-tuned checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E34CheckpointError(f"E34 checkpoint must contain a mapping, got {type(payload).__name__}")
    required = {"adapter", "head", "config", "label_stats"}
    missing = sorted(required - set(payload))
    if missing:
        raise E34CheckpointError(f"E34 checkpoint is missing required entries: {missing}")
    if not isinstance(payload["config"], Mapping):
        raise E34CheckpointError("E34 checkpoint config must be a mapping")
    if not isinstance(payload["head"], Mapping):
        raise E34CheckpointError("E34 checkpoint head must be a state-dict mapping")
    return payload


def checkpoint_inference_config(payload: Mapping[str, Any]) -> tuple[dict[str, int], tuple[float, float], str, float]:
    """Recover LoRA construction and score selection settings from the artifact."""
    config = payload["config"]
    lora = _validated_lora_config(config)
    stats = payload["label_stats"]
    if not isinstance(stats, (list, tuple)) or len(stats) != 2:
        raise E34CheckpointError("E34 label_stats must contain [mean, standard_deviation]")
    try:
        mu, sd = float(stats[0]), float(stats[1])
        mix = float(config["mix"])
    except (TypeError, ValueError, KeyError) as exc:
        raise E34CheckpointError("E34 checkpoint has invalid label_stats or mix setting") from exc
    if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 0:
        raise E34CheckpointError(f"E34 label_stats must be finite with positive std; got {stats!r}")
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise E34CheckpointError(f"E34 config mix must be finite and in [0,1]; got {mix!r}")
    select_on = config.get("select_on")
    if select_on not in {"coral", "reg", "mix"}:
        raise E34CheckpointError(f"E34 config select_on must be coral, reg or mix; got {select_on!r}")
    return lora, (mu, sd), select_on, mix


__all__ = [
    "EMBEDDING_DIM",
    "E34CheckpointError",
    "FEATURE_DIM",
    "Head",
    "LoRALinear",
    "checkpoint_inference_config",
    "encode_pair",
    "inject_lora",
    "inter_feats",
    "load_e34_payload",
    "restore_lora_parameters",
    "score_parts",
    "to_moescore_scale",
    "to_score",
    "wrap_linear",
]
