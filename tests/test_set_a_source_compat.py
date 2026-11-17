from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from model.expert_adapters.set_a_mga_source_adapter import SetAMGASourceAdapter


class FakeASE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.codebook = nn.Parameter(torch.ones(2, 1024))
        self.audio_input_shape = None

    def encode_audio(self, audio):
        self.audio_input_shape = tuple(audio.shape)
        mean = audio.mean(dim=1, keepdim=True) * self.scale
        return None, mean.unsqueeze(1).expand(-1, 2, 1024)

    def encode_text(self, captions):
        batch = len(captions)
        words = torch.ones(batch, 2, 1024, device=self.scale.device) * self.scale
        mask = torch.ones(batch, 2, device=self.scale.device)
        return None, words, mask

    def msc(self, values, codebook, mask=None):
        if mask is not None:
            values = values * mask.unsqueeze(-1)
        return values.mean(dim=1)


class SetAMGASourceCompatibilityTests(unittest.TestCase):
    def test_matches_top2_2026_16khz_waveform_contract_and_returns_4d_plus_1(self):
        model = FakeASE()
        adapter = SetAMGASourceAdapter(model)
        batch = {
            "expert3_wavs": torch.ones(2, 1, 160_000, dtype=torch.float32),
            "captions_raw": ["first caption", "second caption"],
        }
        score, features = adapter(batch)
        self.assertEqual(model.audio_input_shape, (2, 160_000))
        self.assertFalse(model.scale.requires_grad)
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertEqual(tuple(features.shape), (2, 4097))
        torch.testing.assert_close(score, features[:, -1:])
        self.assertFalse(adapter.training)

    def test_rejects_wrong_source_waveform_length_and_bad_caption_batch(self):
        adapter = SetAMGASourceAdapter(FakeASE())
        with self.assertRaisesRegex(ValueError, "160000"):
            adapter({
                "expert3_wavs": torch.zeros(1, 1, 320_000),
                "captions_raw": ["caption"],
            })
        with self.assertRaisesRegex(ValueError, "captions"):
            adapter({
                "expert3_wavs": torch.zeros(2, 1, 160_000),
                "captions_raw": ["only one"],
            })


if __name__ == "__main__":
    unittest.main()
