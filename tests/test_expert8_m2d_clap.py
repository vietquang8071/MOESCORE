from __future__ import annotations

import torch
from torch import nn
import numpy as np
import soundfile as sf

from model.expert_adapters.expert8_m2d_clap_runtime import (
    EXPECTED_CFG,
    EXPECTED_LORA_SHAPES,
    CoralHead,
    Expert8CheckpointError,
    expert8_interaction_features,
    inject_expert8_lora,
    load_expert8_checkpoint,
    restore_ordered_expert8_lora,
)
from model.expert_adapters.expert8_m2d_clap_adapter import Expert8M2DCLAPAdapter
from data.unified_dataset import UnifiedXACLEDataset


def _checkpoint_payload():
    return {
        "head": CoralHead().state_dict(),
        "lora": [torch.full(shape, (index + 1) / 100.0) for index, shape in enumerate(EXPECTED_LORA_SHAPES)],
        "cfg": dict(EXPECTED_CFG),
    }


class _AudioAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 2304)
        self.proj = nn.Linear(768, 768)


class _AudioBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AudioAttention()


class _TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(768, 768)
        self.attention.self.value = nn.Linear(768, 768)


class _LayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_AudioBlock() for _ in range(12)])
        self.text_encoder = nn.Module()
        self.text_encoder.text_encoder = nn.Module()
        self.text_encoder.text_encoder.encoder = nn.Module()
        self.text_encoder.text_encoder.encoder.layer = nn.ModuleList([_TextLayer() for _ in range(12)])


class _FakeEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def to_normalized_feature(self, wavs):
        return wavs

    def encode_lms(self, features):
        return features.mean(dim=-1, keepdim=True).expand(-1, 768)

    def backbone_audio_projection(self, frames):
        return frames

    def encode_clap_text(self, captions, truncate=True):
        values = torch.arange(1, 769, dtype=torch.float32, device=self.anchor.device)
        return torch.stack([values * (len(caption) + 1) for caption in captions])


class _FakeM2D(_FakeEmbeddingModel):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.audio_proj = nn.Identity()
        self.cfg = type("Cfg", (), {"sample_rate": 16000})()


def test_e8_checkpoint_validates_exact_keys_config_head_and_24_lora_tensors(tmp_path):
    path = tmp_path / "m2d-lora-Nam.pt"
    torch.save(_checkpoint_payload(), path)
    payload = load_expert8_checkpoint(path)
    assert set(payload) == {"head", "lora", "cfg"}
    assert len(payload["lora"]) == 24
    assert tuple(payload["head"]["net.5.weight"].shape) == (10, 256)


def test_e8_restores_list_in_exact_audio_then_text_target_order():
    model = _LayoutModel()
    model.requires_grad_(False)
    targets = inject_expert8_lora(model, EXPECTED_CFG)
    expected = []
    for index in range(8, 12):
        expected.extend((f"audio.blocks.{index}.attn.qkv", f"audio.blocks.{index}.attn.proj"))
    for index in range(10, 12):
        expected.extend((f"text.encoder.layer.{index}.attention.self.query",
                         f"text.encoder.layer.{index}.attention.self.value"))
    assert targets == expected
    restored = restore_ordered_expert8_lora(model, _checkpoint_payload()["lora"], targets=targets)
    assert len(restored) == 24
    assert [name for name, value in model.named_parameters() if value.requires_grad] == restored
    for index, (_, parameter) in enumerate((item for item in model.named_parameters() if item[1].requires_grad)):
        torch.testing.assert_close(parameter, torch.full_like(parameter, (index + 1) / 100.0))


def test_e8_features_match_authoritative_standardize_then_interactions():
    audio = torch.arange(768, dtype=torch.float32).reshape(1, 768)
    text = torch.arange(768, 0, -1, dtype=torch.float32).reshape(1, 768)
    actual = expert8_interaction_features(audio, text)
    a = (audio - audio.mean(-1, keepdim=True)) / (audio.std(-1, keepdim=True) + 1e-6)
    t = (text - text.mean(-1, keepdim=True)) / (text.std(-1, keepdim=True) + 1e-6)
    expected = torch.cat((a, t, a * t, (a - t).abs(), (a * t).sum(-1, keepdim=True)), dim=-1)
    assert actual.dtype == torch.float32
    assert tuple(actual.shape) == (1, 3073)
    torch.testing.assert_close(actual, expected)


def test_e8_frozen_adapter_emits_ten_logit_coral_score_on_moescore_scale():
    adapter = Expert8M2DCLAPAdapter(_FakeM2D(), CoralHead())
    assert not adapter.training
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    wavs = torch.zeros((2, 1, 160_000), dtype=torch.float32)
    score, features = adapter({"expert3_wavs": wavs, "captions_raw": ["one", "two words"]})
    assert tuple(score.shape) == (2, 1)
    assert tuple(features.shape) == (2, 3073)
    assert score.dtype == features.dtype == torch.float32
    assert torch.isfinite(score).all() and torch.isfinite(features).all()
    logits = adapter.head(features)
    expected = (torch.sigmoid(logits.float()).sum(-1, keepdim=True) - 5.0) / 5.0
    torch.testing.assert_close(score, expected)


def test_e8_rejects_wrong_sample_count_and_nonmatching_parameter_mapping():
    model = _LayoutModel()
    model.requires_grad_(False)
    inject_expert8_lora(model, EXPECTED_CFG)
    with torch.no_grad():
        model.backbone.blocks[8].attn.proj.A.requires_grad_(False)
    with torch.no_grad():
        try:
            restore_ordered_expert8_lora(model, _checkpoint_payload()["lora"])
        except Expert8CheckpointError as exc:
            assert "traversal" in str(exc) or "24" in str(exc)
        else:
            raise AssertionError("unexpected trainable-parameter traversal was accepted")


def test_e8_audio_preprocessing_requires_16khz_and_never_resamples(tmp_path):
    dataset = object.__new__(UnifiedXACLEDataset)
    bad_rate = tmp_path / "8khz.wav"
    sf.write(bad_rate, np.zeros(8000, dtype=np.float32), 8000)
    try:
        dataset._expert8_audio(bad_rate)
    except ValueError as exc:
        assert "16000" in str(exc) or "16000" in str(exc).replace(",", "")
    else:
        raise AssertionError("unsupported sample rate was silently reinterpreted")
    valid = tmp_path / "16khz.wav"
    sf.write(valid, np.ones(8000, dtype=np.float32), 16000)
    waveform = dataset._expert8_audio(valid)
    assert tuple(waveform.shape) == (1, 160000)
    assert waveform.dtype == torch.float32
    torch.testing.assert_close(waveform[0, :8000], torch.ones(8000), atol=1e-4, rtol=0)
    assert torch.count_nonzero(waveform[0, 8000:]) == 0
