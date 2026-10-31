"""The released MOESCORE losses with class balancing disabled."""

import torch
from torch import Tensor


def expert4_loss(prediction: Tensor, target: Tensor, tau: float = 0.25,
                 margin: float = 0.1, beta: float = 1.0,
                 gamma: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    if prediction.ndim != 1 or target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape [B]")
    if not prediction.numel() or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("scores must be nonempty and finite")
    error = prediction - target
    clipped_mse = ((error.abs() > tau) * error.square()).mean()
    pred_diff = prediction[:, None] - prediction[None, :]
    target_diff = target[:, None] - target[None, :]
    contrastive = torch.clamp((pred_diff - target_diff).abs() - margin, min=0).mean() / 2
    return beta * clipped_mse + gamma * contrastive, clipped_mse, contrastive
