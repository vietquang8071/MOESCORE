import unittest

import torch
from torch import nn


class FakeASE(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Module()
        self.audio_encoder.audio_enc = nn.Module()
        self.audio_encoder.audio_enc.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        self.audio_encoder.audio_enc.layers[3].blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.audio_encoder.audio_enc.norm = nn.LayerNorm(2)
        self.audio_encoder.audio_enc.bn0 = nn.BatchNorm1d(2)
        self.text_encoder = nn.Module()
        self.text_encoder.text_encoder = nn.Module()
        self.text_encoder.text_encoder.encoder = nn.Module()
        self.text_encoder.text_encoder.encoder.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.frame_proj = nn.Linear(2, 2)
        self.word_proj = nn.Linear(2, 2)
        self.codebook = nn.Parameter(torch.ones(2, 2))
        self.temp = nn.Parameter(torch.ones(()))

    def encode_audio(self, audio):
        return None, self.frame_proj(audio[:, :2]).unsqueeze(1)

    def encode_text(self, text):
        x = torch.tensor([[1.0, 0.0] if t == "x" else [0.0, 1.0] for t in text])
        return None, self.word_proj(x).unsqueeze(1), torch.ones(len(text), 1)

    def msc(self, x, codebook, mask=None):
        return x.mean(1)


class Expert2Test(unittest.TestCase):
    def test_strict_load_ignores_only_legacy_bert_position_ids(self):
        from MOESCORE.expert2_finetune.expert2_model import load_mga_state

        model = nn.Linear(2, 2)
        state = model.state_dict()
        state["text_encoder.text_encoder.embeddings.position_ids"] = torch.arange(4).unsqueeze(0)
        load_mga_state(model, state)
        state["unexpected.weight"] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            load_mga_state(model, state)

    def test_cosine_and_trainable_policy(self):
        from MOESCORE.expert2_finetune.expert2_model import Expert2

        model = Expert2(FakeASE(), freeze_policy="partial")
        score, features = model({"mga_wavs": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), "caption_tokens": ["x", "y"]})
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertEqual(tuple(features.shape), (2, 4))
        self.assertTrue(torch.allclose(score[:, 0], (features[:, :2] * features[:, 2:]).sum(-1)))
        self.assertFalse(model.model.model.temp.requires_grad)
        self.assertTrue(model.model.model.codebook.requires_grad)
        self.assertTrue(model.model.model.text_encoder.text_encoder.encoder.layer[11].weight.requires_grad)
        self.assertFalse(model.model.model.text_encoder.text_encoder.encoder.layer[0].weight.requires_grad)
        model.train()
        self.assertFalse(model.model.model.text_encoder.text_encoder.training)
        self.assertTrue(model.model.model.text_encoder.text_encoder.encoder.layer[11].training)

    def test_full_policy_keeps_audio_frontend_statistics_fixed(self):
        from MOESCORE.expert2_finetune.expert2_model import Expert2

        model = Expert2(FakeASE(), freeze_policy="full").train()
        self.assertTrue(model.model.model.audio_encoder.audio_enc.layers[3].training)
        self.assertFalse(model.model.model.audio_encoder.audio_enc.bn0.training)


if __name__ == "__main__":
    unittest.main()
