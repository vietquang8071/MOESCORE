"""ExpertAdapter facade over verified, precomputed Set A component outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

import torch
from torch import nn

from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_DIMS,
    SET_A_COMPONENT_FEATURE_DIMS,
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_DIM,
    SetAResidualHead,
)
from model.expert_adapters.base import ExpertAdapter, validate_expert_output
from model.expert_adapters.paired_cross_features import build_paired_cross_features


class SetAResidualAdapter(ExpertAdapter):
    """Consume ``batch['component_outputs']`` and return score plus 11,267 features.

    The nested mapping has keys ``top2_m2d``, ``top2_msclap``, and ``top2_mga``.
    Each value is a mapping with ``score: [B,1]`` and ``features: [B,4D+1]``.
    Component outputs are detached before regression because all three source
    backbones are frozen and may be absent during cached-feature training.
    """

    def __init__(
        self,
        head: Optional[SetAResidualHead] = None,
        *,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__(feature_dim=SET_A_FEATURE_DIM)
        if head is not None and not isinstance(head, SetAResidualHead):
            raise TypeError("head must be a SetAResidualHead or None")
        self.head = head or SetAResidualHead(hidden_dims=hidden_dims, dropout=dropout)

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("SetAResidualAdapter batch must be a mapping")
        components = batch.get("component_outputs")
        if not isinstance(components, Mapping):
            raise TypeError("batch['component_outputs'] must be a mapping")
        expected_names = set(SET_A_COMPONENT_ORDER)
        if set(components) != expected_names:
            raise ValueError(
                "component names must be exactly "
                f"{SET_A_COMPONENT_ORDER}; got {tuple(components)}"
            )

        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        expected_device: Optional[torch.device] = None
        batch_size: Optional[int] = None
        for name in SET_A_COMPONENT_ORDER:
            value = components[name]
            if not isinstance(value, Mapping) or set(value) != {"score", "features"}:
                raise TypeError(
                    f"component_outputs[{name!r}] must contain exactly 'score' and 'features'"
                )
            score, features = value["score"], value["features"]
            if not isinstance(score, torch.Tensor) or not isinstance(features, torch.Tensor):
                raise TypeError(f"{name} score and features must be torch.Tensor values")
            expected_width = SET_A_COMPONENT_FEATURE_DIMS[name]
            if score.ndim != 2 or score.shape[1] != 1:
                raise ValueError(f"{name} score must have shape [B, 1]; got {tuple(score.shape)}")
            if features.ndim != 2 or features.shape[1] != expected_width:
                raise ValueError(
                    f"{name} features width mismatch: expected {expected_width}, got {tuple(features.shape)}"
                )
            if score.shape[0] <= 0 or score.shape[0] != features.shape[0]:
                raise ValueError(f"{name} score/features batch dimension mismatch")
            if score.dtype != torch.float32 or features.dtype != torch.float32:
                raise TypeError(f"{name} score and features must use float32")
            if score.dtype != features.dtype:
                raise TypeError(f"{name} score/features dtype mismatch: {score.dtype} != {features.dtype}")
            if score.device != features.device:
                raise ValueError(
                    f"{name} score and features must share a device; "
                    f"got {score.device} and {features.device}"
                )
            if batch_size is None:
                batch_size = int(score.shape[0])
                expected_device = score.device
            elif score.shape[0] != batch_size:
                raise ValueError(f"{name} batch size {score.shape[0]} differs from {batch_size}")
            if score.device != expected_device or features.device != expected_device:
                raise ValueError(
                    f"{name} device mismatch: outputs are on {score.device}/{features.device}; "
                    f"expected {expected_device}"
                )
            pairs[name] = (score, features)

        # The shared output validator checks finite values with tensor scalar
        # reads, so first reject all cross-component device mismatches above.
        for name in SET_A_COMPONENT_ORDER:
            validate_expert_output(
                pairs[name][0],
                pairs[name][1],
                feature_dim=SET_A_COMPONENT_FEATURE_DIMS[name],
                expert_name=name,
            )

        # Rebuild each component from its first two blocks. Equality against the
        # existing shared helper proves block ordering and all three derived
        # blocks without introducing another feature implementation.
        for name in SET_A_COMPONENT_ORDER:
            score, features = pairs[name]
            embedding_dim = SET_A_COMPONENT_DIMS[name]
            expected = build_paired_cross_features(
                features[:, :embedding_dim], features[:, embedding_dim : 2 * embedding_dim]
            )
            if not torch.allclose(features, expected, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"{name} feature representation/order does not match the five-block 4D+1 contract"
                )
            if not torch.allclose(score, expected[:, -1:], rtol=1e-5, atol=1e-6):
                raise ValueError(f"{name} score is inconsistent with its cosine feature scalar")

        component_scores = torch.cat([pairs[name][0] for name in SET_A_COMPONENT_ORDER], dim=1).detach()
        features5 = torch.cat([pairs[name][1] for name in SET_A_COMPONENT_ORDER], dim=1).detach()
        if features5.shape[1] != SET_A_FEATURE_DIM:
            raise ValueError(
                f"Set A concatenated feature width must be {SET_A_FEATURE_DIM}; got {features5.shape[1]}"
            )
        return self.head(component_scores, features5), features5


__all__ = ["SetAResidualAdapter"]
