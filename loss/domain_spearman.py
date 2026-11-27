"""Differentiable L2 Soft Spearman losses for the DSS gate only."""

from __future__ import annotations

import torch
from torch import nn

from loss.l2_soft_rank import l2_soft_rank


def _average_ranks(target: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(target, stable=True)
    sorted_target = target[order]
    _, counts = torch.unique_consecutive(sorted_target, return_counts=True)
    end = counts.cumsum(0)
    rank_by_group = end - (counts - 1).to(end.dtype) / 2
    ranks = torch.empty_like(target)
    ranks[order] = torch.repeat_interleave(rank_by_group, counts).to(target.dtype)
    return ranks


class SoftSpearmanLoss(nn.Module):
    def __init__(self, regularization_strength: float = 0.1, epsilon: float = 1e-8):
        super().__init__()
        if regularization_strength <= 0 or epsilon <= 0:
            raise ValueError("regularization_strength and epsilon must be positive")
        self.regularization_strength = regularization_strength
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.ndim != 1 or targets.shape != predictions.shape or predictions.numel() < 2:
            raise ValueError("predictions and targets must be matching rank-one vectors of length >= 2")
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("ranking inputs must be finite")
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            prediction = predictions.float() if predictions.dtype not in (torch.float32, torch.float64) else predictions
            target = targets.to(device=prediction.device, dtype=prediction.dtype)
            hard = _average_ranks(target)
            centered_hard = hard - hard.mean()
            if torch.count_nonzero(centered_hard) == 0:
                raise ValueError("constant target ranks make Spearman correlation undefined")
            soft = l2_soft_rank(prediction, self.regularization_strength)
            centered_soft = soft - soft.mean()
            numerator = torch.dot(centered_soft, centered_hard)
            denominator = centered_soft.norm().clamp_min(self.epsilon) * centered_hard.norm().clamp_min(self.epsilon)
            loss = 1 - numerator / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Soft Spearman loss is nonfinite")
        return loss


class DomainWiseSoftSpearmanLoss(nn.Module):
    def __init__(self, min_group_size: int = 8, regularization_strength: float = 0.1):
        super().__init__()
        if min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        self.min_group_size = min_group_size
        self.single = SoftSpearmanLoss(regularization_strength)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, model_ids: torch.Tensor):
        if model_ids.shape != predictions.shape or targets.shape != predictions.shape:
            raise ValueError("prediction, target and domain vectors must align")
        losses, valid, invalid = [], [], []
        per_domain = {}
        for domain in torch.unique(model_ids, sorted=True).tolist():
            mask = model_ids == domain
            if int(mask.sum()) < self.min_group_size or torch.unique(targets[mask]).numel() < 2:
                invalid.append(int(domain))
                continue
            loss = self.single(predictions[mask], targets[mask])
            losses.append(loss)
            per_domain[int(domain)] = float(loss.detach())
            valid.append(int(domain))
        if not losses:
            raise ValueError("all domain groups are invalid for Soft Spearman")
        self.last_per_domain = per_domain
        return torch.stack(losses).mean(), valid, invalid


class HybridSoftSpearmanLoss(nn.Module):
    def __init__(self, regularization_strength: float = 0.1, min_group_size: int = 8):
        super().__init__()
        self.global_loss = SoftSpearmanLoss(regularization_strength)
        self.domain_loss = DomainWiseSoftSpearmanLoss(min_group_size, regularization_strength)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, model_ids: torch.Tensor):
        global_loss = self.global_loss(predictions, targets)
        domain_loss, valid, invalid = self.domain_loss(predictions, targets, model_ids)
        total = 0.5 * global_loss + 0.5 * domain_loss
        if not torch.isfinite(total):
            raise FloatingPointError("hybrid Soft Spearman loss is nonfinite")
        return global_loss, domain_loss, total, {"valid_domains": valid, "invalid_domains": invalid,
                                                "per_domain_loss": self.domain_loss.last_per_domain}
