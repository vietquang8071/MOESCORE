"""Focused tests for the inference-only E34 M2D-LoRA adapter."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

from model.expert_adapters.m2d_lora_runtime import (
    E34CheckpointError,
    Head,
    LoRALinear,
    inject_lora,
    inter_feats,
    restore_lora_parameters,
    score_parts,
    to_moescore_scale,
    to_score,
)
from model.expert_adapters.m2d_lora_adapter import M2DLoRAAdapter
from model.expert_adapters import m2d_lora_adapter as adapter_module


class _Attention(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)


class _AudioBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attn = _Attention(width)


class _TextLayer(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(width, width)
        self.attention.self.value = nn.Linear(width, width)


class _TinyE34Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_AudioBlock() for _ in range(4)])
        self.text_encoder = nn.Module()
        self.text_encoder.text_encoder = nn.Module()
        self.text_encoder.text_encoder.encoder = nn.Module()
        self.text_encoder.text_encoder.encoder.layer = nn.ModuleList(
            [_TextLayer() for _ in range(2)]
        )


class _FakeAudioProjection(nn.Module):
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_proj = _FakeAudioProjection()


class _FakeM2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.anchor = nn.Parameter(torch.ones(()))

    def to_normalized_feature(self, wavs: torch.Tensor) -> torch.Tensor:
        return wavs

    def encode_lms(self, features: torch.Tensor) -> torch.Tensor:
        values = features.mean(dim=1, keepdim=True)
        scale = torch.arange(768, dtype=features.dtype, device=features.device)
        return values * (scale.unsqueeze(0) + 1.0)

    def encode_clap_text(self, captions: list[str], truncate: bool = False) -> torch.Tensor:
        scale = torch.arange(768, dtype=torch.float32, device=self.anchor.device)
        rows = [scale * (len(text) % 5 + 1) + index for index, text in enumerate(captions)]
        return torch.stack(rows)


class M2DLoRAAdapterTests(unittest.TestCase):
  def test_importing_adapter_does_not_load_portable_m2d_or_transformers(self) -> None:
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "import model.expert_adapters.m2d_lora_adapter; "
        "assert 'transformers' not in sys.modules; "
        "assert 'expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d' "
        "not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=False
    )
    self.assertEqual(result.returncode, 0, result.stderr)


  def test_inter_feats_matches_e34_interaction_layout(self) -> None:
    audio = torch.linspace(-2.0, 3.0, 1536, dtype=torch.float32).reshape(2, 768)
    text = torch.linspace(4.0, -1.0, 1536, dtype=torch.float32).reshape(2, 768)
    audio_z = (audio - audio.mean(-1, keepdim=True)) / (audio.std(-1, keepdim=True) + 1e-6)
    text_z = (text - text.mean(-1, keepdim=True)) / (text.std(-1, keepdim=True) + 1e-6)
    expected = torch.cat(
        [audio_z, text_z, audio_z * text_z, (audio_z - text_z).abs(),
         (audio_z * text_z).sum(-1, keepdim=True)],
        dim=-1,
    )

    actual = inter_feats(audio, text)

    self.assertEqual(actual.shape, (2, 3073))
    self.assertEqual(actual.dtype, torch.float32)
    torch.testing.assert_close(actual, expected)


  def test_e34_coral_regression_and_mix_restore_raw_scores(self) -> None:
    output = torch.zeros((2, 11), dtype=torch.float32)
    output[1, :10] = 0.5
    output[:, 10] = torch.tensor([2.0, -1.0])
    stats = (4.0, 2.0)

    parts = score_parts(output, stats, mix=0.25)

    self.assertAlmostEqual(parts["coral"][0].item(), 0.5)
    self.assertAlmostEqual(
        parts["coral"][1].item(),
        (10 * torch.sigmoid(torch.tensor(0.5)).item() - 4) / 2,
        delta=1e-6,
    )
    self.assertEqual(parts["reg"].tolist(), [2.0, -1.0])
    expected_mix = [0.75 * 0.5 + 0.25 * 2.0,
                    0.75 * parts["coral"][1].item() + 0.25 * -1.0]
    self.assertAlmostEqual(parts["mix"][0].item(), expected_mix[0], delta=1e-6)
    self.assertAlmostEqual(parts["mix"][1].item(), expected_mix[1], delta=1e-6)
    coral = to_score(output, stats, select_on="coral", mix=0.25)
    self.assertAlmostEqual(coral[0].item(), 5.0, delta=1e-6)
    self.assertAlmostEqual(coral[1].item(), 4 + 2 * parts["coral"][1].item(), delta=1e-6)
    self.assertEqual(to_score(output, stats, select_on="reg", mix=0.25).tolist(), [8.0, 2.0])
    mixed_raw = to_score(output, stats, select_on="mix", mix=0.25)
    self.assertAlmostEqual(mixed_raw[0].item(), 5.75, delta=1e-6)
    self.assertAlmostEqual(mixed_raw[1].item(), 4 + 2 * expected_mix[1], delta=1e-6)
    normalized = to_moescore_scale(mixed_raw)
    self.assertAlmostEqual(normalized[0].item(), (5.75 - 5.0) / 5.0, delta=1e-6)
    self.assertAlmostEqual(normalized[1].item(), (mixed_raw[1].item() - 5.0) / 5.0, delta=1e-6)


  def test_lora_injection_uses_e34_target_names_and_freezes_base_layers(self) -> None:
    model = _TinyE34Model()
    names = inject_lora(
        model,
        {"lora_rank": 2, "lora_alpha": 4, "audio_lora_blocks": 4, "text_lora_layers": 2},
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    self.assertEqual(len(names), 12)
    self.assertEqual(len(trainable), 24)
    self.assertIn("backbone.blocks.3.attn.qkv.A", trainable)
    self.assertIn("text_encoder.text_encoder.encoder.layer.1.attention.self.value.B", trainable)
    self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".base." in name))
    layer = model.backbone.blocks[0].attn.qkv
    self.assertIsInstance(layer, LoRALinear)
    self.assertEqual(layer.scale, 2.0)


  def test_lora_restore_requires_complete_matching_checkpoint_parameters(self) -> None:
    model = _TinyE34Model()
    inject_lora(
        model,
        {"lora_rank": 2, "lora_alpha": 4, "audio_lora_blocks": 4, "text_lora_layers": 2},
    )
    saved = {
        name: torch.full_like(parameter, 0.125)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    restore_lora_parameters(model, saved)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)), name)

    with self.assertRaisesRegex(E34CheckpointError, "missing"):
        restore_lora_parameters(model, {name: value for name, value in saved.items() if name != next(iter(saved))})
    with self.assertRaisesRegex(E34CheckpointError, "unexpected"):
        restore_lora_parameters(model, {**saved, "wrong.module.A": torch.zeros(2, 8)})
    wrong_shape = dict(saved)
    first = next(iter(wrong_shape))
    wrong_shape[first] = torch.zeros(1)
    with self.assertRaisesRegex(E34CheckpointError, "shape"):
        restore_lora_parameters(model, wrong_shape)


  def test_adapter_runs_prediction_head_and_stays_frozen_in_eval_mode(self) -> None:
    model = _FakeM2D()
    head = Head()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[-1].bias[10] = 2.0
    adapter = M2DLoRAAdapter(model, head, label_stats=(7.0, 2.0), select_on="mix", mix=0.5)
    batch = {
        "expert3_wavs": torch.linspace(-1.0, 1.0, 2 * 160_000).reshape(2, 1, 160_000),
        "captions_raw": ["short sound", "a longer audio caption"],
    }

    score, features = adapter(batch)

    self.assertEqual(score.shape, (2, 1))
    self.assertEqual(features.shape, (2, 3073))
    self.assertEqual(score.dtype, torch.float32)
    self.assertEqual(features.dtype, torch.float32)
    self.assertEqual(score.device, torch.device("cpu"))
    self.assertEqual(features.device, score.device)
    torch.testing.assert_close(score[:, 0], torch.tensor([0.6, 0.6]), atol=1e-6, rtol=0)
    self.assertTrue(torch.isfinite(score).all() and torch.isfinite(features).all())
    self.assertFalse(adapter.training or model.training or head.training)
    self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    adapter.train()
    self.assertFalse(adapter.training or model.training or head.training)
    self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))


  def test_from_config_reports_missing_explicit_checkpoint(self) -> None:
    with self.assertRaisesRegex(ValueError, "finetuned_checkpoint"):
        M2DLoRAAdapter.from_config(
            {"base_checkpoint": "/definitely/missing/checkpoint-30.pth"},
            device="cpu",
        )

  def test_huggingface_home_resolves_to_the_standard_hub_cache(self) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        hub = root / "hub"
        hub.mkdir()

        self.assertEqual(adapter_module._text_encoder_cache_dir(root), hub)

        direct = root / "direct-cache"
        direct.mkdir()
        (direct / "models--google-bert--bert-base-uncased").mkdir()
        self.assertEqual(adapter_module._text_encoder_cache_dir(direct), direct)
