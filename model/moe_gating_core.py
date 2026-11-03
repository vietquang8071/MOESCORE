"""Lightweight original MOESCORE gating network for verified adapter features."""

from __future__ import annotations

import torch
from torch import nn

from model.expert_adapters.registry import (
    EXPECTED_FEATURE_WIDTH,
    EXPERT_ORDER,
)


GATING_HIDDEN_DIM = 512
GATING_DROPOUT = 0.1


class GatingNetwork(nn.Module):
    """Original Linear/LN/ReLU/Dropout/Linear/Softmax gate.

    The input width is derived from the verified LOCAL-08 expert registry.
    Scores are deliberately not consumed here; this module only maps fused
    expert features to sample-specific mixture weights.
    """

    def __init__(
        self,
        input_feature_dim: int = EXPECTED_FEATURE_WIDTH,
        num_experts: int = len(EXPERT_ORDER),
    ) -> None:
        super().__init__()
        if type(input_feature_dim) is not int or input_feature_dim != EXPECTED_FEATURE_WIDTH:
            raise ValueError(
                f"input_feature_dim must match the registered feature width "
                f"{EXPECTED_FEATURE_WIDTH}; got {input_feature_dim!r}"
            )
        if type(num_experts) is not int or num_experts != len(EXPERT_ORDER):
            raise ValueError(
                f"num_experts must be four to match EXPERT_ORDER; got {num_experts!r}"
            )

        self.input_feature_dim = EXPECTED_FEATURE_WIDTH
        self.num_experts = len(EXPERT_ORDER)
        self.net = nn.Sequential(
            nn.Linear(self.input_feature_dim, GATING_HIDDEN_DIM),
            nn.LayerNorm(GATING_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(GATING_DROPOUT),
            nn.Linear(GATING_HIDDEN_DIM, self.num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(combined_features, torch.Tensor):
            raise TypeError("combined_features must be a torch.Tensor")
        if combined_features.ndim != 2:
            raise ValueError(
                "combined_features must have shape [B,7936]; "
                f"got {tuple(combined_features.shape)}"
            )
        if combined_features.shape[0] <= 0:
            raise ValueError("combined_features batch dimension must be positive")
        if combined_features.shape[1] != self.input_feature_dim:
            raise ValueError(
                f"combined_features width must be {self.input_feature_dim}; "
                f"got {combined_features.shape[1]}"
            )
        if not torch.is_floating_point(combined_features):
            raise TypeError("combined_features must use a floating-point dtype")
        first_layer = self.net[0]
        if combined_features.device != first_layer.weight.device:
            raise ValueError(
                f"combined_features is on {combined_features.device}; "
                f"gating network is on {first_layer.weight.device}"
            )
        if combined_features.dtype != first_layer.weight.dtype:
            raise TypeError(
                f"combined_features uses dtype {combined_features.dtype}; "
                f"gating network uses {first_layer.weight.dtype}"
            )
        if not torch.isfinite(combined_features).all().item():
            raise ValueError("combined_features must contain only finite values")
        return self.net(combined_features)


__all__ = [
    "GATING_DROPOUT",
    "GATING_HIDDEN_DIM",
    "GatingNetwork",
]
