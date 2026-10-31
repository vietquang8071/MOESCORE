"""Expert 4 fusion and regression, following Basic_comp.py's active path."""
import torch
from torch import Tensor, nn


class SeqCoAttn(nn.Module):
    def __init__(self, dim: int = 512, nhead: int = 8):
        super().__init__()
        self.dim = dim
        self.audio_to_text_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, audio: Tensor, text: Tensor, audio_mask: Tensor, text_mask: Tensor) -> Tensor:
        if audio.ndim != 3 or text.ndim != 3 or audio.shape[0] != text.shape[0] or audio.shape[-1] != self.dim or text.shape[-1] != self.dim:
            raise ValueError("audio and text must be [B, T, 512]")
        for mask, seq in ((audio_mask, audio), (text_mask, text)):
            if mask.dtype != torch.bool or tuple(mask.shape) != tuple(seq.shape[:2]) or mask.all(dim=1).any():
                raise ValueError("padding mask must be bool [B, T] with at least one valid token")
        a, _ = self.audio_to_text_attn(audio, text, text, key_padding_mask=text_mask, need_weights=False)
        t, _ = self.text_to_audio_attn(text, audio, audio, key_padding_mask=audio_mask, need_weights=False)
        return torch.cat((self.pool(a.transpose(1, 2)).squeeze(-1), self.pool(t.transpose(1, 2)).squeeze(-1)), dim=-1)


class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.3), nn.Linear(2048, 1), nn.Tanh())

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)
