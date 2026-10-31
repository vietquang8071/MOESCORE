"""Standalone Expert 4; no other expert or gating network is imported."""
from torch import nn
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .seq_coattn import RegressionHead, SeqCoAttn


class Expert4(nn.Module):
    def __init__(self, audio_encoder, text_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.seq_coattn = SeqCoAttn()
        self.head = RegressionHead()

    @classmethod
    def from_pretrained(cls, beats_checkpoint, clap_checkpoint, unilm_source_dir):
        return cls(AudioEncoder(beats_checkpoint, unilm_source_dir), TextEncoder.load_pretrained(clap_checkpoint))

    def forward(self, batch):
        audio, audio_mask = self.audio_encoder(batch['wavs'], batch['wav_lens'])
        text, text_mask = self.text_encoder(batch['captions'])
        fused = self.seq_coattn(audio, text, audio_mask, text_mask)
        return self.head(fused), fused
