"""Common, checkpoint-independent contract for MOESCORE expert adapters."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, final

import torch
from torch import nn


def _validate_feature_dim(feature_dim: int, *, expert_name: str = "ExpertAdapter") -> int:
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError(
            f"{expert_name} feature_dim must be a positive integer; got {feature_dim!r}"
        )
    return feature_dim


def validate_expert_output(
    score: Any,
    features: Any,
    *,
    feature_dim: int,
    expert_name: str = "ExpertAdapter",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate the future adapter output without reshaping or transforming it.

    The accepted shapes are exactly ``score=[B, 1]`` and
    ``features=[B, feature_dim]``. Both tensors must be finite floating-point
    tensors on the same device. The original tensor objects are returned.
    """
    feature_dim = _validate_feature_dim(feature_dim, expert_name=expert_name)
    if not isinstance(score, torch.Tensor):
        raise TypeError(f"{expert_name} score must be a torch.Tensor; got {type(score).__name__}")
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"{expert_name} features must be a torch.Tensor; got {type(features).__name__}"
        )
    if score.ndim != 2 or score.shape[1] != 1:
        raise ValueError(
            f"{expert_name} score must have shape [B, 1]; got {tuple(score.shape)}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"{expert_name} features must have shape [B, D]; got {tuple(features.shape)}"
        )
    if features.shape[1] != feature_dim:
        raise ValueError(
            f"{expert_name} features width mismatch: declared feature_dim={feature_dim}, "
            f"got {features.shape[1]}"
        )
    if score.shape[0] != features.shape[0]:
        raise ValueError(
            f"{expert_name} score/features batch dimension mismatch: "
            f"{score.shape[0]} != {features.shape[0]}"
        )
    if score.shape[0] <= 0:
        raise ValueError(f"{expert_name} output batch dimension must be positive; got 0")
    if not torch.is_floating_point(score):
        raise TypeError(f"{expert_name} score must use a floating-point dtype; got {score.dtype}")
    if not torch.is_floating_point(features):
        raise TypeError(
            f"{expert_name} features must use a floating-point dtype; got {features.dtype}"
        )
    if score.device != features.device:
        raise ValueError(
            f"{expert_name} score and features must be on the same device; "
            f"got {score.device} and {features.device}"
        )
    if not torch.isfinite(score).all().item():
        raise ValueError(f"{expert_name} score must contain only finite values")
    if not torch.isfinite(features).all().item():
        raise ValueError(f"{expert_name} features must contain only finite values")
    return score, features


class ExpertAdapter(nn.Module, ABC):
    """Small base class defining a feature width and adapter output contract.

    Concrete adapters implement ``_forward(batch)`` and return a score tensor
    shaped ``[B, 1]`` plus a feature tensor shaped ``[B, feature_dim]``. The
    final public ``forward`` validates the pair automatically. The base class
    adds no parameters or checkpoint behavior.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self._feature_dim = _validate_feature_dim(feature_dim, expert_name=type(self).__name__)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @final
    def forward(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the adapter implementation and enforce the shared output contract."""
        output = self._forward(batch)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                f"{type(self).__name__} _forward must return a (score, features) tuple"
            )
        return self.validate_output(*output)

    @abstractmethod
    def _forward(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the original expert score and feature pair for ``batch``."""
        raise NotImplementedError

    def validate_output(
        self, score: Any, features: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the common shape/value checks using this adapter's width."""
        return validate_expert_output(
            score,
            features,
            feature_dim=self.feature_dim,
            expert_name=type(self).__name__,
        )
