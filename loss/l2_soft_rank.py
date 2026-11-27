"""Exact L2 soft rank via pool-adjacent-violators isotonic projection.

This is the L2 permutahedron projection of Blondel et al. (ICML 2020),
equivalent to torchsort.soft_rank(..., regularization='l2'). The small DSS
batch sizes make a pure PyTorch implementation practical when torchsort's
CUDA extension cannot be built against the installed CUDA toolkit.
"""

from __future__ import annotations

import torch


def l2_soft_rank(values: torch.Tensor, regularization_strength: float = 0.1) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("L2 soft rank requires a rank-one vector of length >= 2")
    if regularization_strength <= 0 or not torch.isfinite(values).all():
        raise ValueError("L2 soft rank requires finite values and positive regularization")
    scaled = values / regularization_strength
    sorted_values, order = torch.sort(scaled, descending=True, stable=True)
    descending_ranks = torch.arange(values.numel(), 0, -1, device=values.device, dtype=values.dtype)
    differences = sorted_values - descending_ranks
    blocks: list[tuple[torch.Tensor, int]] = []
    for item in differences.unbind():
        blocks.append((item, 1))
        while len(blocks) >= 2 and bool(blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]):
            right_sum, right_count = blocks.pop()
            left_sum, left_count = blocks.pop()
            blocks.append((left_sum + right_sum, left_count + right_count))
    isotonic = torch.cat([(total / count).expand(count) for total, count in blocks])
    inverse_order = torch.argsort(order)
    return (sorted_values - isotonic)[inverse_order]
