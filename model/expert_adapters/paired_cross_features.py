"""Stateless construction of paired audio/text interaction features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_paired_cross_features(
    audio_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
) -> torch.Tensor:
    """Return ``[audio, text, |a_hat-t_hat|, a_hat*t_hat, cosine*1_D]``.

    The first two blocks preserve the input embedding values after conversion
    to float32. L2-normalized copies are used only for the three interaction
    blocks. This utility has no model state and does not produce an expert
    score.
    """
    if not isinstance(audio_embedding, torch.Tensor) or not isinstance(text_embedding, torch.Tensor):
        raise TypeError("audio_embedding and text_embedding must be torch.Tensor values")
    if audio_embedding.ndim != 2 or text_embedding.ndim != 2:
        raise ValueError("audio_embedding and text_embedding must have shape [B,D]")
    if audio_embedding.shape != text_embedding.shape:
        raise ValueError(
            "audio_embedding and text_embedding shapes must match exactly; "
            f"got {tuple(audio_embedding.shape)} and {tuple(text_embedding.shape)}"
        )
    batch_size, embedding_dim = audio_embedding.shape
    if batch_size <= 0 or embedding_dim <= 0:
        raise ValueError("audio/text batch size and embedding dimension must be positive")
    if not torch.is_floating_point(audio_embedding) or not torch.is_floating_point(text_embedding):
        raise TypeError("audio_embedding and text_embedding must use floating-point dtypes")
    if audio_embedding.device != text_embedding.device:
        raise ValueError(
            "audio_embedding and text_embedding must be on the same device; "
            f"got {audio_embedding.device} and {text_embedding.device}"
        )
    if not torch.isfinite(audio_embedding).all().item() or not torch.isfinite(text_embedding).all().item():
        raise ValueError("audio_embedding and text_embedding must contain only finite values")

    audio = audio_embedding.float()
    text = text_embedding.float()
    if not torch.isfinite(audio).all().item() or not torch.isfinite(text).all().item():
        raise ValueError("audio/text embeddings must remain finite after float32 conversion")
    audio_normalized = F.normalize(audio, p=2, dim=-1)
    text_normalized = F.normalize(text, p=2, dim=-1)
    absolute_difference = torch.abs(audio_normalized - text_normalized)
    elementwise_product = audio_normalized * text_normalized
    cosine_similarity = torch.sum(
        audio_normalized * text_normalized,
        dim=-1,
        keepdim=True,
    )
    repeated_cosine = cosine_similarity.expand(-1, embedding_dim)
    return torch.cat(
        [audio, text, absolute_difference, elementwise_product, repeated_cosine],
        dim=-1,
    )


__all__ = ["build_paired_cross_features"]
