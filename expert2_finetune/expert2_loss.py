"""The MOESCORE clipped MSE plus pairwise score-difference objective."""

import torch


def hybrid_loss(pred, target, tau=0.25, margin=0.1):
    if pred.ndim != 1 or target.ndim != 1:
        raise ValueError("pred and target must be one-dimensional [B] tensors")
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes differ")
    error = pred - target
    mse = ((error.abs() > tau) * error.square()).mean()
    pred_diff = pred[:, None] - pred[None, :]
    target_diff = target[:, None] - target[None, :]
    pair = (pred_diff - target_diff).abs().sub(margin).clamp_min(0).mean() / 2
    return mse + 0.5 * pair, mse, pair
