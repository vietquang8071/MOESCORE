"""Explicit conversion between XACLE raw labels and MOESCORE targets."""

from __future__ import annotations

from typing import Any

import torch


def _floating_tensor(value: Any, field: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{field} must contain only finite values")
    return tensor


def normalize_targets(raw_scores: Any) -> torch.Tensor:
    """Map original challenge scores in [0,10] to the source loss interval."""
    raw = _floating_tensor(raw_scores, "raw_scores")
    if torch.any(raw < 0.0).item() or torch.any(raw > 10.0).item():
        raise ValueError("raw_scores must remain on the original [0,10] scale")
    return (raw - 5.0) / 5.0


def denormalize_predictions(normalized_scores: Any) -> torch.Tensor:
    """Map normalized gate predictions to the challenge's original score scale."""
    normalized = _floating_tensor(normalized_scores, "normalized_scores")
    return normalized * 5.0 + 5.0


def score_interval_feasibility(scores: torch.Tensor, targets_normalized: torch.Tensor) -> dict[str, Any]:
    """Describe the per-row error lower bound imposed by convex score fusion."""
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have shape [B,4]")
    if not isinstance(targets_normalized, torch.Tensor) or targets_normalized.ndim != 1:
        raise ValueError("targets_normalized must have shape [B]")
    if scores.shape[0] <= 0 or scores.shape[0] != targets_normalized.shape[0]:
        raise ValueError("scores and targets_normalized must have the same positive batch size")
    if not torch.is_floating_point(scores) or not torch.is_floating_point(targets_normalized):
        raise TypeError("scores and targets_normalized must be floating-point tensors")
    if scores.device != targets_normalized.device or scores.dtype != targets_normalized.dtype:
        raise ValueError("scores and targets_normalized must share device and dtype")
    if not torch.isfinite(scores).all().item() or not torch.isfinite(targets_normalized).all().item():
        raise ValueError("scores and targets_normalized must be finite")
    lower = scores.min(dim=1).values
    upper = scores.max(dim=1).values
    distance = torch.maximum(lower - targets_normalized, targets_normalized - upper).clamp_min(0)
    below = targets_normalized < lower
    above = targets_normalized > upper
    count = int(scores.shape[0])
    return {
        "sample_count": count,
        "outside_interval_count": int((below | above).sum().item()),
        "outside_interval_fraction": float((below | above).float().mean().item()),
        "below_interval_count": int(below.sum().item()),
        "above_interval_count": int(above.sum().item()),
        "minimum_absolute_error": float(distance.min().item()),
        "mean_feasibility_distance": float(distance.mean().item()),
        "max_feasibility_distance": float(distance.max().item()),
        "mean_squared_error_lower_bound": float(distance.square().mean().item()),
    }


__all__ = ["denormalize_predictions", "normalize_targets", "score_interval_feasibility"]
