"""Frozen LAION/Whisper feature regression head for E6."""

from collections.abc import Sequence

import torch
from torch import nn


SET_B_FEATURE_DIM = 4609


def feature_order_contract() -> dict:
    return {"feature_dim": 4609, "order": ["top2_laion", "whisper_audio", "whisper_text"],
            "dimensions": [2049, 1280, 1280], "laion_representation": "paired_cross_4d_plus_1_v2"}


class SetBResidualHead(nn.Module):
    def __init__(self, *, hidden_dims: Sequence[int] = (256, 64), dropout: float = 0.2):
        super().__init__()
        if len(hidden_dims) != 2 or any(type(x) is not int or x <= 0 for x in hidden_dims):
            raise ValueError("hidden_dims must contain two positive integers")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.residual_mlp = nn.Sequential(
            nn.Linear(4609, hidden_dims[0]), nn.LayerNorm(hidden_dims[0]), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dims[0], hidden_dims[1]), nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    @property
    def architecture_config(self) -> dict:
        return {"feature_dim": 4609, "hidden_dims": list(self.hidden_dims), "dropout": self.dropout}

    def forward(self, laion_score: torch.Tensor, features6: torch.Tensor) -> torch.Tensor:
        if laion_score.shape != (features6.shape[0], 1) or features6.ndim != 2 or features6.shape[1] != 4609:
            raise ValueError("E6 requires score [B,1] and features [B,4609]")
        if laion_score.dtype != torch.float32 or features6.dtype != torch.float32:
            raise TypeError("E6 inputs must be float32")
        if laion_score.device != features6.device or self.scale.device != features6.device:
            raise ValueError("E6 device mismatch")
        if not torch.isfinite(laion_score).all() or not torch.isfinite(features6).all():
            raise ValueError("E6 inputs must be finite")
        return self.scale * laion_score + self.bias + self.residual_mlp(features6)
