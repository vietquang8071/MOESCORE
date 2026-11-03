"""Prove unified batches meet the four existing adapter boundaries using stubs."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from model.expert_adapters.expert1_adapter import Expert1Adapter
from model.expert_adapters.expert2_adapter import Expert2Adapter
from model.expert_adapters.expert3_adapter import Expert3Adapter
from model.expert_adapters.expert4_adapter import Expert4Adapter
from model.expert_adapters.registry import validate_and_assemble_expert_outputs

from data.unified_collate import collate_unified


def _sample(index: int, *, length: int = 160_000):
    return {
        "sample_id": f"validation:{index:05d}.wav",
        "wave_file_name": f"{index:05d}.wav",
        "split": "validation",
        "captions_raw": f" caption {index} ",
        "expert1_wavs": torch.full((1, 480_000), float(index)),
        "expert2_wavs": torch.full((1, 320_000), float(index + 10)),
        "expert3_wavs": torch.full((1, 160_000), float(index + 20)),
        "expert4_wavs": torch.cat(
            (torch.full((1, length), float(index + 30)), torch.zeros((1, 160_000 - length))),
            dim=1,
        ),
        "expert4_wav_lens": length,
        "label": torch.tensor(float(index), dtype=torch.float32),
    }


class _E1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, audio, captions):
        self.calls.append((audio, list(captions)))
        audio_value = audio.mean(dim=(1, 2)).unsqueeze(1) * self.weight
        text_value = torch.tensor([len(caption) for caption in captions], dtype=torch.float32).unsqueeze(1)
        return audio_value.expand(-1, 512), text_value.expand(-1, 512)


class _E1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _E1Encoder()


class _MappingModel(nn.Module):
    def __init__(self, key: str, width: int, *, score_rank=2):
        super().__init__()
        self.key = key
        self.width = width
        self.score_rank = score_rank
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, batch):
        self.calls.append(batch)
        audio = batch[self.key]
        value = audio.mean(dim=tuple(range(1, audio.ndim))) * self.scale
        features = value.unsqueeze(1).expand(-1, self.width).contiguous()
        score = value if self.score_rank == 1 else value.unsqueeze(1)
        return score, features


class _E3Model(_MappingModel):
    embedding_dim = 768
    cross_feature_dim = 3840
    concat_assignment = "released"

    def __init__(self):
        super().__init__("wavs", 3840)


class _E4Seq(nn.Module):
    dim = 512


class _E4Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 2), nn.ReLU())


class UnifiedAdapterIntegrationTests(unittest.TestCase):
    def test_one_batch_reaches_all_existing_adapter_contracts_in_order(self):
        samples = [_sample(0, length=80_000), _sample(1, length=160_000)]
        batch = collate_unified(samples)

        e1_model = _E1Model()
        e2_model = _MappingModel("mga_wavs", 2048)
        e3_model = _E3Model()
        e4_model = _MappingModel("wavs", 1024, score_rank=1)
        e4_model.seq_coattn = _E4Seq()
        e4_model.head = _E4Head()

        adapters = {
            "expert1": Expert1Adapter(e1_model, embedding_dim=512),
            "expert2": Expert2Adapter(e2_model, feature_dim=2048),
            "expert3": Expert3Adapter(e3_model),
            "expert4": Expert4Adapter(e4_model),
        }
        outputs = {name: adapter(batch) for name, adapter in adapters.items()}

        self.assertEqual(len(e1_model.model.calls), 1)
        self.assertTrue(torch.equal(e1_model.model.calls[0][0], batch["expert1_wavs"]))
        self.assertEqual(e1_model.model.calls[0][1], batch["captions_raw"])
        self.assertEqual(len(e2_model.calls), 1)
        self.assertTrue(torch.equal(e2_model.calls[0]["mga_wavs"], batch["expert2_wavs"]))
        self.assertEqual(e2_model.calls[0]["caption_tokens"], ["caption 0", "caption 1"])
        self.assertEqual(len(e3_model.calls), 1)
        self.assertTrue(torch.equal(e3_model.calls[0]["wavs"], batch["expert3_wavs"]))
        self.assertEqual(e3_model.calls[0]["captions"], ["caption 0", "caption 1"])
        self.assertEqual(len(e4_model.calls), 1)
        self.assertTrue(torch.equal(e4_model.calls[0]["wavs"], batch["expert4_wavs"]))
        self.assertTrue(torch.equal(e4_model.calls[0]["wav_lens"], torch.tensor([80_000, 160_000])))
        self.assertEqual(e4_model.calls[0]["captions"], ["caption 0", "caption 1"])

        scores, features = validate_and_assemble_expert_outputs(outputs, batch_size=2)
        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertEqual(tuple(features.shape), (2, 7936))
        self.assertEqual([tuple(outputs[name][1].shape[1:]) for name in adapters], [(1024,), (2048,), (3840,), (1024,)])

        for adapter in adapters.values():
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
            adapter.train()
            self.assertFalse(adapter.model.training)
            self.assertTrue(all(not module.training for module in adapter.model.modules()))

    def test_stub_adapter_integration_handles_batch_sizes_one_and_four(self):
        for batch_size in (1, 4):
            batch = collate_unified([_sample(i, length=80_000 + i * 20_000) for i in range(batch_size)])
            adapters = {
                "expert1": Expert1Adapter(_E1Model(), embedding_dim=512),
                "expert2": Expert2Adapter(_MappingModel("mga_wavs", 2048), feature_dim=2048),
                "expert3": Expert3Adapter(_E3Model()),
            }
            model4 = _MappingModel("wavs", 1024, score_rank=1)
            model4.seq_coattn = _E4Seq()
            model4.head = _E4Head()
            adapters["expert4"] = Expert4Adapter(model4)
            outputs = {name: adapter(batch) for name, adapter in adapters.items()}
            scores, features = validate_and_assemble_expert_outputs(outputs, batch_size=batch_size)
            self.assertEqual(tuple(scores.shape), (batch_size, 4))
            self.assertEqual(tuple(features.shape), (batch_size, 7936))


if __name__ == "__main__":
    unittest.main()
