"""Tests for the E31 list-checkpoint M2D-LoRA CORAL adapter."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model.expert_adapters.m2d_lora_coral_runtime import (
    EXPECTED_LORA_SHAPES,
    M2DLoRACoralCheckpointError,
    CoralHead,
    inject_coral_lora,
    load_coral_checkpoint,
    restore_ordered_lora,
)
from model.expert_adapters.m2d_lora_coral_adapter import M2DLoRACoralAdapter
from model.expert_adapters.m2d_lora_runtime import inter_feats


def _cfg():
    return {
        "variant": "audio_text", "seed": 7, "epochs": 5, "patience": 2,
        "batch_size": 16, "eval_batch": 64, "lr_head": 0.001,
        "lr_lora": 0.0005, "weight_decay": 0.01, "lora_r": 8,
        "lora_alpha": 16, "lora_blocks": 4, "lora_text_layers": 2,
        "workers": 2, "run_logo": False, "smoke": False,
    }


def _payload():
    return {
        "head": CoralHead().state_dict(),
        "lora": [torch.full(shape, index / 100.0, dtype=torch.float32)
                 for index, shape in enumerate(EXPECTED_LORA_SHAPES)],
        "cfg": _cfg(),
    }


class _AudioAttention(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)


class _AudioBlock(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.attn = _AudioAttention(width)


class _TextLayer(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(width, width)
        self.attention.self.value = nn.Linear(width, width)


class _TinyE31Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_AudioBlock() for _ in range(12)])
        self.text_encoder = nn.Module()
        self.text_encoder.text_encoder = nn.Module()
        self.text_encoder.text_encoder.encoder = nn.Module()
        self.text_encoder.text_encoder.encoder.layer = nn.ModuleList([_TextLayer() for _ in range(12)])


class _AudioProjection(nn.Module):
    def forward(self, values):
        return values


class _FakeM2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.backbone = nn.Module()
        self.backbone.audio_proj = _AudioProjection()

    def to_normalized_feature(self, wavs):
        return wavs

    def encode_lms(self, features):
        mean = features.mean(dim=1, keepdim=False).unsqueeze(-1)
        scale = torch.arange(768, dtype=features.dtype, device=features.device).unsqueeze(0) + 1.0
        return mean * scale

    def encode_clap_text(self, captions, truncate=True):
        return torch.stack([
            torch.full((768,), float(index + len(text)), device=self.anchor.device)
            for index, text in enumerate(captions)
        ])


class _FixedHead(nn.Module):
    def forward(self, features):
        logits = torch.zeros((features.shape[0], 10), dtype=torch.float32, device=features.device)
        logits[:, 0] = 2.0
        logits[:, 1] = -2.0
        return logits


class M2DLoRACoralAdapterTests(unittest.TestCase):
    def test_checkpoint_loader_accepts_verified_structure_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coral.pt"
            torch.save(_payload(), path)
            loaded = load_coral_checkpoint(path)
            self.assertEqual(tuple(loaded), ("head", "lora", "cfg"))
            self.assertEqual(len(loaded["lora"]), 24)

            for mutate, message in (
                (lambda value: value.pop("lora"), "keys"),
                (lambda value: value["cfg"].update(variant="audio"), "variant"),
                (lambda value: value["head"].pop("net.5.bias"), "keys"),
                (lambda value: value["head"].__setitem__("net.5.weight", torch.zeros(9, 256)), "shape"),
                (lambda value: value["head"].__setitem__("net.5.weight", torch.full((10, 256), float("nan"))), "finite"),
                (lambda value: value["head"].__setitem__("net.5.weight", torch.zeros(10, 256, dtype=torch.float64)), "dtype"),
                (lambda value: value["lora"].__setitem__(0, torch.zeros(7, 768)), "shape"),
                (lambda value: value["lora"].__setitem__(0, torch.full((8, 768), float("nan"))), "finite"),
                (lambda value: value["lora"].__setitem__(0, torch.zeros(8, 768, dtype=torch.float64)), "dtype"),
            ):
                corrupted = _payload()
                mutate(corrupted)
                torch.save(corrupted, path)
                with self.subTest(message=message), self.assertRaisesRegex(M2DLoRACoralCheckpointError, message):
                    load_coral_checkpoint(path)

    def test_injected_targets_and_parameter_traversal_match_e31_serialization(self):
        model = _TinyE31Model()
        model.requires_grad_(False)
        targets = inject_coral_lora(model, _cfg())
        expected_targets = []
        for index in range(8, 12):
            expected_targets.extend((f"audio.blocks.{index}.attn.qkv", f"audio.blocks.{index}.attn.proj"))
        for index in range(10, 12):
            expected_targets.extend((f"text.encoder.layer.{index}.attention.self.query",
                                     f"text.encoder.layer.{index}.attention.self.value"))
        self.assertEqual(targets, expected_targets)

        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(named), 24)
        values = [torch.full_like(parameter, (index + 1) / 100.0) for index, (_, parameter) in enumerate(named)]
        restored_names = restore_ordered_lora(model, values)
        self.assertEqual(restored_names, [name for name, _ in named])
        for index, (name, parameter) in enumerate(named):
            torch.testing.assert_close(parameter, torch.full_like(parameter, (index + 1) / 100.0), msg=name)
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters()
                            if not name.endswith((".A", ".B"))))

    def test_restorer_rejects_wrong_count_shape_dtype_and_nonfinite_values(self):
        model = _TinyE31Model()
        model.requires_grad_(False)
        inject_coral_lora(model, _cfg())
        valid = [torch.zeros(shape, dtype=torch.float32) for shape in EXPECTED_LORA_SHAPES]
        with self.assertRaisesRegex(M2DLoRACoralCheckpointError, "24"):
            restore_ordered_lora(model, valid[:-1])
        malformed = list(valid)
        malformed[0] = torch.zeros(7, 768)
        with self.assertRaisesRegex(M2DLoRACoralCheckpointError, "shape"):
            restore_ordered_lora(model, malformed)
        malformed = list(valid)
        malformed[0] = torch.zeros(8, 768, dtype=torch.float64)
        with self.assertRaisesRegex(M2DLoRACoralCheckpointError, "dtype"):
            restore_ordered_lora(model, malformed)
        malformed = list(valid)
        malformed[0] = torch.full((8, 768), float("inf"))
        with self.assertRaisesRegex(M2DLoRACoralCheckpointError, "finite"):
            restore_ordered_lora(model, malformed)

    def test_coral_head_is_10_logits_and_strict_state_dict(self):
        head = CoralHead()
        result = head(torch.zeros((3, 3073)))
        self.assertEqual(tuple(result.shape), (3, 10))
        self.assertEqual(tuple(head.net[0].weight.shape), (1024, 3073))
        self.assertEqual(tuple(head.net[3].weight.shape), (256, 1024))
        self.assertEqual(tuple(head.net[5].weight.shape), (10, 256))
        state = head.state_dict()
        with self.assertRaises(RuntimeError):
            head.load_state_dict({key: value for key, value in state.items() if key != "net.5.bias"}, strict=True)

    def test_e31_interaction_features_and_normalized_coral_score(self):
        audio = torch.linspace(-1.0, 2.0, 1536).reshape(2, 768)
        text = torch.linspace(3.0, -2.0, 1536).reshape(2, 768)
        audio_z = (audio - audio.mean(-1, keepdim=True)) / (audio.std(-1, keepdim=True) + 1e-6)
        text_z = (text - text.mean(-1, keepdim=True)) / (text.std(-1, keepdim=True) + 1e-6)
        expected = torch.cat([audio_z, text_z, audio_z * text_z,
                              (audio_z - text_z).abs(), (audio_z * text_z).sum(-1, keepdim=True)], dim=-1)
        torch.testing.assert_close(inter_feats(audio, text), expected)

        adapter = M2DLoRACoralAdapter(_FakeM2D(), _FixedHead())
        wavs = torch.ones((2, 1, 160_000), dtype=torch.float32)
        score, features = adapter({"expert3_wavs": wavs, "captions_raw": ["one", "two"]})
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertEqual(tuple(features.shape), (2, 3073))
        self.assertEqual(score.dtype, torch.float32)
        self.assertEqual(features.dtype, torch.float32)
        expected_raw = torch.sigmoid(torch.tensor(2.0)) + torch.sigmoid(torch.tensor(-2.0)) + 8 * 0.5
        torch.testing.assert_close(score[:, 0], torch.full((2,), (expected_raw.item() - 5.0) / 5.0))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(score.device, features.device)
        self.assertFalse(any(parameter.requires_grad for parameter in adapter.parameters()))
        adapter.train()
        self.assertFalse(adapter.training)

    def test_adapter_import_does_not_load_portable_m2d_or_transformers(self):
        repo = Path(__file__).resolve().parents[1]
        code = (
            "import sys; import model.expert_adapters.m2d_lora_coral_adapter; "
            "assert 'transformers' not in sys.modules; "
            "assert 'expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d' "
            "not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=repo,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
