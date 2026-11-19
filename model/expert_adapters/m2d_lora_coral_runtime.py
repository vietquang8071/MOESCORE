"""Checkpoint-specific runtime for the E31 M2D-CLAP audio/text LoRA CORAL model.

The source notebook serializes ``[p for p in model.parameters() if p.requires_grad]``.
The ordered-list restore below follows that exact traversal after applying its
audio-then-text injection procedure. The E34 named checkpoint loader is not used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from model.expert_adapters.m2d_lora_runtime import (
    EMBEDDING_DIM,
    FEATURE_DIM,
    encode_pair,
    inject_lora,
    inter_feats,
)


N_LEVELS = 10
CORAL_FEATURE_DIM = 4 * EMBEDDING_DIM + 1
CHECKPOINT_KEYS = frozenset({"head", "lora", "cfg"})
EXPECTED_CFG = {
    "variant": "audio_text",
    "seed": 7,
    "epochs": 5,
    "patience": 2,
    "batch_size": 16,
    "eval_batch": 64,
    "lr_head": 0.001,
    "lr_lora": 0.0005,
    "weight_decay": 0.01,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_blocks": 4,
    "lora_text_layers": 2,
    "workers": 2,
    "run_logo": False,
    "smoke": False,
}
EXPECTED_HEAD_SHAPES = {
    "net.0.weight": (1024, CORAL_FEATURE_DIM),
    "net.0.bias": (1024,),
    "net.3.weight": (256, 1024),
    "net.3.bias": (256,),
    "net.5.weight": (N_LEVELS, 256),
    "net.5.bias": (N_LEVELS,),
}
EXPECTED_LORA_SHAPES = tuple(
    shape
    for _ in range(4)
    for shape in ((8, 768), (2304, 8), (8, 768), (768, 8))
) + tuple(shape for _ in range(4) for shape in ((8, 768), (768, 8)))


class M2DLoRACoralCheckpointError(ValueError):
    """Raised when an E31 list checkpoint cannot restore the expected model."""


class CoralHead(nn.Module):
    """E31's 3073-to-10 CORAL MLP, with its original module/key layout."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CORAL_FEATURE_DIM, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, N_LEVELS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _validate_cfg(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise M2DLoRACoralCheckpointError("checkpoint cfg must be a mapping")
    missing = set(EXPECTED_CFG) - set(config)
    unexpected = set(config) - set(EXPECTED_CFG)
    if missing or unexpected:
        raise M2DLoRACoralCheckpointError(
            f"checkpoint cfg keys are incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, expected in EXPECTED_CFG.items():
        actual = config[key]
        if type(actual) is not type(expected) or actual != expected:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint cfg {key} must be {expected!r}; got {actual!r}"
            )
    return config


def _validate_head_state(head_state: Any) -> None:
    if not isinstance(head_state, Mapping):
        raise M2DLoRACoralCheckpointError("checkpoint head must be a state-dictionary mapping")
    missing = set(EXPECTED_HEAD_SHAPES) - set(head_state)
    unexpected = set(head_state) - set(EXPECTED_HEAD_SHAPES)
    if missing or unexpected:
        raise M2DLoRACoralCheckpointError(
            f"checkpoint head keys are incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, shape in EXPECTED_HEAD_SHAPES.items():
        value = head_state[name]
        if not isinstance(value, torch.Tensor):
            raise M2DLoRACoralCheckpointError(f"checkpoint head tensor {name} is not a tensor")
        if tuple(value.shape) != shape:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint head tensor {name} has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.dtype != torch.float32:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint head tensor {name} dtype must be torch.float32, got {value.dtype}"
            )
        if not torch.isfinite(value).all().item():
            raise M2DLoRACoralCheckpointError(f"checkpoint head tensor {name} must be finite")


def _validate_lora_list(values: Any) -> None:
    if not isinstance(values, list) or len(values) != len(EXPECTED_LORA_SHAPES):
        count = len(values) if isinstance(values, list) else type(values).__name__
        raise M2DLoRACoralCheckpointError(
            f"checkpoint lora must be an ordered list of 24 tensors; got {count}"
        )
    for index, (value, shape) in enumerate(zip(values, EXPECTED_LORA_SHAPES)):
        if not isinstance(value, torch.Tensor):
            raise M2DLoRACoralCheckpointError(f"checkpoint lora[{index}] is not a tensor")
        if tuple(value.shape) != shape:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint lora[{index}] has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.dtype != torch.float32:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint lora[{index}] dtype must be torch.float32, got {value.dtype}"
            )
        if not torch.isfinite(value).all().item():
            raise M2DLoRACoralCheckpointError(f"checkpoint lora[{index}] must be finite")


def load_coral_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load and validate the E31 artifact on CPU using safe weights-only loading."""
    source = Path(path).expanduser()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise M2DLoRACoralCheckpointError(
            f"could not load E31 CORAL checkpoint {source} with weights_only=True: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise M2DLoRACoralCheckpointError("E31 CORAL checkpoint must contain a mapping")
    missing = CHECKPOINT_KEYS - set(payload)
    unexpected = set(payload) - CHECKPOINT_KEYS
    if missing or unexpected:
        raise M2DLoRACoralCheckpointError(
            f"E31 CORAL checkpoint keys are incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    _validate_cfg(payload["cfg"])
    _validate_head_state(payload["head"])
    _validate_lora_list(payload["lora"])
    return payload


def _expected_target_paths(model: nn.Module, config: Mapping[str, Any]) -> list[str]:
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    text_root = getattr(model, "text_encoder", None)
    bert = getattr(text_root, "text_encoder", None)
    layers = getattr(getattr(getattr(bert, "encoder", None), "layer", None), "__len__", None)
    if blocks is None or not callable(layers):
        raise M2DLoRACoralCheckpointError("M2D model lacks the E31 backbone/text layer layout")
    num_text_layers = len(bert.encoder.layer)
    audio_count = config["lora_blocks"]
    text_count = config["lora_text_layers"]
    if len(blocks) != 12 or num_text_layers != 12:
        raise M2DLoRACoralCheckpointError(
            f"E31 requires 12 audio blocks and 12 BERT layers; got {len(blocks)} and {num_text_layers}"
        )
    paths = []
    for index in range(len(blocks) - audio_count, len(blocks)):
        for target in ("attn.qkv", "attn.proj"):
            paths.append(f"audio.blocks.{index}.{target}")
    for index in range(num_text_layers - text_count, num_text_layers):
        for target in ("attention.self.query", "attention.self.value"):
            paths.append(f"text.encoder.layer.{index}.{target}")
    return paths


def inject_coral_lora(model: nn.Module, config: Mapping[str, Any]) -> list[str]:
    """Inject the E31 targets and confirm their module order before restoration."""
    _validate_cfg(config)
    expected_targets = _expected_target_paths(model, config)
    e34_config = {
        "lora_rank": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "audio_lora_blocks": config["lora_blocks"],
        "text_lora_layers": config["lora_text_layers"],
    }
    targets = inject_lora(model, e34_config)
    if targets != expected_targets:
        raise M2DLoRACoralCheckpointError(
            f"injected LoRA targets differ from E31 order: {targets!r} != {expected_targets!r}"
        )
    return targets


def _expected_serialized_parameter_names(targets: Sequence[str]) -> list[str]:
    names = []
    for target in targets:
        if target.startswith("audio."):
            module_path = "backbone." + target.removeprefix("audio.")
        elif target.startswith("text.encoder.layer."):
            layer, suffix = target.removeprefix("text.encoder.layer.").split(".", 1)
            module_path = f"text_encoder.text_encoder.encoder.layer.{layer}.{suffix}"
        else:
            raise M2DLoRACoralCheckpointError(f"unknown E31 LoRA target {target!r}")
        names.extend((module_path + ".A", module_path + ".B"))
    return names


def restore_ordered_lora(model: nn.Module, values: Sequence[torch.Tensor], *, targets=None) -> list[str]:
    """Restore the list serialized from E31's requires-grad parameter traversal."""
    if not isinstance(values, (list, tuple)) or len(values) != len(EXPECTED_LORA_SHAPES):
        raise M2DLoRACoralCheckpointError(
            f"ordered E31 LoRA restore requires 24 tensors; got {len(values) if isinstance(values, (list, tuple)) else type(values).__name__}"
        )
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                        if parameter.requires_grad]
    actual_names = [name for name, _ in named_parameters]
    if targets is None:
        cfg = dict(EXPECTED_CFG)
        targets = _expected_target_paths(model, cfg)
    expected_names = _expected_serialized_parameter_names(targets)
    if actual_names != expected_names:
        raise M2DLoRACoralCheckpointError(
            "M2D requires-grad parameter traversal differs from E31 serialized order: "
            f"{actual_names!r} != {expected_names!r}"
        )
    if len(named_parameters) != len(EXPECTED_LORA_SHAPES):
        raise M2DLoRACoralCheckpointError(
            f"M2D has {len(named_parameters)} trainable parameters; E31 requires 24"
        )
    for index, ((name, parameter), value, expected_shape) in enumerate(
        zip(named_parameters, values, EXPECTED_LORA_SHAPES)
    ):
        if not isinstance(value, torch.Tensor):
            raise M2DLoRACoralCheckpointError(f"checkpoint lora[{index}] for {name} is not a tensor")
        if tuple(parameter.shape) != expected_shape or tuple(value.shape) != tuple(parameter.shape):
            raise M2DLoRACoralCheckpointError(
                f"checkpoint lora[{index}] for {name} shape mismatch: "
                f"saved={tuple(value.shape)}, expected={tuple(parameter.shape)}"
            )
        if value.dtype != torch.float32 or parameter.dtype != torch.float32:
            raise M2DLoRACoralCheckpointError(
                f"checkpoint lora[{index}] for {name} dtype must use torch.float32"
            )
        if not torch.isfinite(value).all().item():
            raise M2DLoRACoralCheckpointError(f"checkpoint lora[{index}] for {name} must be finite")
    with torch.no_grad():
        for (_, parameter), value in zip(named_parameters, values):
            parameter.copy_(value.to(device=parameter.device))
    return actual_names


def load_coral_head(head: CoralHead, state: Mapping[str, Any]) -> None:
    """Validate then strictly restore the standalone ten-logit E31 head."""
    _validate_head_state(state)
    try:
        head.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise M2DLoRACoralCheckpointError(f"E31 CORAL head is incompatible: {exc}") from exc


__all__ = [
    "CORAL_FEATURE_DIM", "EXPECTED_CFG", "EXPECTED_HEAD_SHAPES", "EXPECTED_LORA_SHAPES",
    "N_LEVELS", "CoralHead", "M2DLoRACoralCheckpointError", "encode_pair",
    "inject_coral_lora", "inter_feats", "load_coral_checkpoint", "load_coral_head",
    "restore_ordered_lora",
]
