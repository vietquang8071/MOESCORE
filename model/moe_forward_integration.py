"""Lightweight frozen-expert to trainable-gate forward integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import (
    EXPECTED_FEATURE_WIDTH,
    EXPERT_FEATURE_DIMS,
    EXPERT_ORDER,
    validate_and_assemble_expert_outputs,
)
from model.moe_gating_core import GatingNetwork


class MoEForwardIntegration(nn.Module):
    """Run existing adapters frozen, then train the source-compatible gate.

    This class accepts already constructed lightweight adapters. It never
    resolves factories, loads checkpoints, or constructs pretrained models.
    """

    def __init__(
        self,
        experts: Mapping[str, ExpertAdapter],
        gating_network: Optional[GatingNetwork] = None,
    ) -> None:
        super().__init__()
        if not isinstance(experts, Mapping):
            raise TypeError("experts must be a mapping keyed by expert name")
        actual_names = set(experts)
        expected_names = set(EXPERT_ORDER)
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        if missing or unexpected:
            raise ValueError(
                f"Expert adapter registry mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        ordered_experts: dict[str, ExpertAdapter] = {}
        object_ids: set[int] = set()
        for name in EXPERT_ORDER:
            adapter = experts[name]
            if not isinstance(adapter, ExpertAdapter):
                raise TypeError(f"{name} must be an ExpertAdapter instance")
            expected_dim = EXPERT_FEATURE_DIMS[name]
            if adapter.feature_dim != expected_dim:
                raise ValueError(
                    f"{name}.feature_dim must be {expected_dim}; got {adapter.feature_dim}"
                )
            if id(adapter) in object_ids:
                raise ValueError("Each expert name must reference a distinct adapter instance")
            object_ids.add(id(adapter))
            ordered_experts[name] = adapter

        if gating_network is None:
            gating_network = GatingNetwork()
        if not isinstance(gating_network, GatingNetwork):
            raise TypeError("gating_network must be a GatingNetwork instance")
        if gating_network.input_feature_dim != EXPECTED_FEATURE_WIDTH:
            raise ValueError(
                f"gating network must accept {EXPECTED_FEATURE_WIDTH} features; "
                f"got {gating_network.input_feature_dim}"
            )
        if gating_network.num_experts != len(EXPERT_ORDER):
            raise ValueError("gating network expert count must match EXPERT_ORDER")

        self.experts = nn.ModuleDict(ordered_experts)
        self.gating_network = gating_network
        self._freeze_experts()

    def _freeze_experts(self) -> None:
        for adapter in self.experts.values():
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "MoEForwardIntegration":
        if not isinstance(mode, bool):
            raise ValueError(f"training mode must be bool; got {mode!r}")
        super().train(mode)
        # The gate follows the requested mode; wrapped expert trees never do.
        self._freeze_experts()
        return self

    def forward(
        self, batch: Mapping[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping accepted by the expert adapters")

        with torch.no_grad():
            expert_outputs = {
                name: self.experts[name](batch)
                for name in EXPERT_ORDER
            }

        return self.forward_from_expert_outputs(expert_outputs)

    def forward_from_expert_outputs(
        self,
        expert_outputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the existing assembler and gate on already computed outputs.

        This boundary also supports verification artifacts produced by
        sequential expert processes. It does not detach supplied tensors or
        construct any model; callers control whether their expert outputs are
        detached. The normal adapter ``forward`` path runs experts under
        ``no_grad`` before delegating here.
        """
        if not isinstance(expert_outputs, Mapping):
            raise TypeError("expert_outputs must be a mapping keyed by expert name")

        scores, combined_features = validate_and_assemble_expert_outputs(
            expert_outputs
        )
        first_gate_parameter = next(self.gating_network.parameters())
        if combined_features.device != first_gate_parameter.device:
            raise ValueError(
                f"expert features are on {combined_features.device}; "
                f"gating network is on {first_gate_parameter.device}"
            )
        if combined_features.dtype != first_gate_parameter.dtype:
            raise TypeError(
                f"expert features use dtype {combined_features.dtype}; "
                f"gating network uses {first_gate_parameter.dtype}"
            )

        weights = self.gating_network(combined_features)
        expected_weight_shape = (scores.shape[0], len(EXPERT_ORDER))
        if tuple(weights.shape) != expected_weight_shape:
            raise ValueError(
                f"gating weights must have shape {expected_weight_shape}; "
                f"got {tuple(weights.shape)}"
            )
        if weights.device != scores.device or weights.dtype != scores.dtype:
            raise ValueError("gating weights must match expert score device and dtype")
        if not torch.isfinite(weights).all().item():
            raise ValueError("gating weights must contain only finite values")
        if (weights < 0).any().item() or not torch.allclose(
            weights.sum(dim=1),
            torch.ones(scores.shape[0], device=weights.device, dtype=weights.dtype),
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError("gating weights must be nonnegative and sum to one per sample")

        # Preserve the original MOESCORE four-term sum and score ordering.
        final_score = (
            scores[:, 0:1] * weights[:, 0:1]
            + scores[:, 1:2] * weights[:, 1:2]
            + scores[:, 2:3] * weights[:, 2:3]
            + scores[:, 3:4] * weights[:, 3:4]
        )
        return final_score, weights


__all__ = ["MoEForwardIntegration"]
