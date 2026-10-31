import unittest
import torch
from torch import nn
from expert4_finetune.model import Expert4


class DummyAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)

    def forward(self, wavs, lengths):
        out = self.projection(wavs[:, :, :4].transpose(1, 2))
        return out, torch.zeros(out.shape[:2], dtype=torch.bool)


class DummyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)

    def forward(self, captions):
        x = self.projection(torch.ones(len(captions), 3, 1))
        return x, torch.zeros(x.shape[:2], dtype=torch.bool)


class ModelTests(unittest.TestCase):
    def test_forward_backward_optimizer_step(self):
        model = Expert4(DummyAudio(), DummyText())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        batch = {'wavs': torch.randn(2, 1, 8), 'wav_lens': torch.tensor([8, 8]), 'captions': ['a', 'b']}
        scores, fused = model(batch)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertEqual(tuple(fused.shape), (2, 1024))
        scores.sum().backward()
        self.assertIsNotNone(model.audio_encoder.projection.weight.grad)
        self.assertIsNotNone(model.text_encoder.projection.weight.grad)
        optimizer.step()


if __name__ == '__main__':
    unittest.main()
