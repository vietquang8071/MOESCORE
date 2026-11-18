"""Set A's affine cosine baseline and learned residual regressor."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


SET_A_COMPONENT_ORDER = ("top2_m2d", "top2_msclap", "top2_mga")
SET_A_COMPONENT_DIMS = {
    "top2_m2d": 768,
    "top2_msclap": 1024,
    "top2_mga": 1024,
}
SET_A_COMPONENT_FEATURE_DIMS = {
    name: 4 * embedding_dim + 1
    for name, embedding_dim in SET_A_COMPONENT_DIMS.items()
}
SET_A_FEATURE_DIM = sum(SET_A_COMPONENT_FEATURE_DIMS.values())
SET_A_FEATURE_REPRESENTATION = "paired_cross_4d_plus_1_v2"
SET_A_FEATURE_BLOCKS = (
    "audio_embedding",
    "text_embedding",
    "absolute_difference_l2",
    "elementwise_product_l2",
    "cosine_l2_scalar",
)


def feature_order_contract() -> dict[str, object]:
    """Return a JSON-compatible, ordered contract for checkpoints and reports."""
    return {
        "component_order": list(SET_A_COMPONENT_ORDER),
        "component_embedding_dims": dict(SET_A_COMPONENT_DIMS),
        "component_feature_dims": dict(SET_A_COMPONENT_FEATURE_DIMS),
        "feature_width": SET_A_FEATURE_DIM,
        "feature_representation": SET_A_FEATURE_REPRESENTATION,
        "component_feature_blocks": list(SET_A_FEATURE_BLOCKS),
    }


class SetAResidualHead(nn.Module):
    """Predict a normalized score from arithmetic-mean cosine plus residual.

    ``component_scores`` is ordered ``[M2D, MS-CLAP, MGA]`` with shape
    ``[B, 3]``. ``features5`` is the uncompressed concatenation of the three
    paired ``4D+1`` vectors with shape ``[B, 11267]``.
    """

    def __init__(
        self,
        *,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
            raise TypeError("hidden_dims must be a sequence of two positive integers")
        if len(hidden_dims) != 2 or any(type(value) is not int or value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain exactly two positive integers")
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a number in [0, 1)")

        first, second = hidden_dims
        self.hidden_dims = (first, second)
        self.dropout_probability = float(dropout)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.residual_mlp = nn.Sequential(
            nn.Linear(SET_A_FEATURE_DIM, first),
            nn.LayerNorm(first),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, 1),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    @property
    def architecture_config(self) -> dict[str, object]:
        return {
            "feature_dim": SET_A_FEATURE_DIM,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_probability,
        }

    def forward(self, component_scores: torch.Tensor, features5: torch.Tensor) -> torch.Tensor:
        if not isinstance(component_scores, torch.Tensor) or not isinstance(features5, torch.Tensor):
            raise TypeError("component_scores and features5 must be torch.Tensor values")
        if component_scores.ndim != 2 or component_scores.shape[1] != len(SET_A_COMPONENT_ORDER):
            raise ValueError(
                "component_scores must have shape [B, 3] in M2D, MS-CLAP, MGA order; "
                f"got {tuple(component_scores.shape)}"
            )
        if features5.ndim != 2 or features5.shape[1] != SET_A_FEATURE_DIM:
            raise ValueError(
                f"features5 feature width must be {SET_A_FEATURE_DIM}; got {tuple(features5.shape)}"
            )
        if component_scores.shape[0] != features5.shape[0]:
            raise ValueError(
                "component_scores and features5 batch dimensions differ: "
                f"{component_scores.shape[0]} != {features5.shape[0]}"
            )
        if component_scores.shape[0] <= 0:
            raise ValueError("component_scores and features5 batch size must be positive")
        if component_scores.dtype != torch.float32 or features5.dtype != torch.float32:
            raise TypeError("component_scores and features5 must use float32")
        if component_scores.device != features5.device:
            raise ValueError(
                "component_scores and features5 must share a device; "
                f"got {component_scores.device} and {features5.device}"
            )
        parameter = next(self.parameters())
        if parameter.device != features5.device:
            raise ValueError(
                f"SetAResidualHead is on {parameter.device}, but inputs are on {features5.device}"
            )
        if not torch.isfinite(component_scores).all().item() or not torch.isfinite(features5).all().item():
            raise ValueError("component_scores and features5 must contain only finite values")

        mean_cosine = (
            component_scores[:, 0:1]
            + component_scores[:, 1:2]
            + component_scores[:, 2:3]
        ) / 3.0
        return self.scale * mean_cosine + self.bias + self.residual_mlp(features5)


__all__ = [
    "SET_A_COMPONENT_DIMS",
    "SET_A_COMPONENT_FEATURE_DIMS",
    "SET_A_COMPONENT_ORDER",
    "SET_A_FEATURE_BLOCKS",
    "SET_A_FEATURE_DIM",
    "SET_A_FEATURE_REPRESENTATION",
    "SetAResidualHead",
    "feature_order_contract",
]
