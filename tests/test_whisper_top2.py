import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from model.feature_extractors.whisper_top2 import (
    FEATURE_EXTRACTION_VERSION,
    MODEL_ID,
    WhisperTop2FeatureExtractor,
    _checkpoint_identity,
    _local_model_files,
)


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, *, sampling_rate, return_tensors):
        self.calls.append((audio, sampling_rate, return_tensors))
        return SimpleNamespace(
            input_features=torch.zeros(len(audio), 80, 3000, dtype=torch.float32)
        )


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, captions, *, return_tensors, padding):
        self.calls.append((list(captions), return_tensors, padding))
        widths = [2 if caption.strip() == "rain" else 3 for caption in captions]
        width = max(widths)
        token_ids = torch.zeros(len(captions), width, dtype=torch.long)
        for row, length in enumerate(widths):
            token_ids[row, :length] = torch.arange(1, length + 1) + row * 3
        return SimpleNamespace(input_ids=token_ids)


class FakeWhisperModel(nn.Module):
    def __init__(self, *, bad_output=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(_name_or_path="openai/whisper-large-v2")
        self.bad_output = bad_output
        self.calls = []
        self.inference_modes = []

    def forward(self, input_features, *, decoder_input_ids):
        self.calls.append((input_features.detach().clone(), decoder_input_ids.detach().clone()))
        self.inference_modes.append(torch.is_inference_mode_enabled())
        batch_size, sequence_length = decoder_input_ids.shape
        audio_values = torch.arange(batch_size, dtype=torch.float32).view(-1, 1, 1) + 10
        audio_hidden = audio_values.expand(batch_size, 3, 1280).clone()
        decoder_hidden = decoder_input_ids.float().unsqueeze(-1).expand(-1, -1, 1280).clone()
        if self.bad_output:
            decoder_hidden[0, 0, 0] = float("nan")
        return SimpleNamespace(
            encoder_last_hidden_state=audio_hidden,
            last_hidden_state=decoder_hidden,
        )


def make_extractor(*, bad_output=False):
    feature_extractor = FakeFeatureExtractor()
    tokenizer = FakeTokenizer()
    model = FakeWhisperModel(bad_output=bad_output)
    extractor = WhisperTop2FeatureExtractor(
        feature_extractor,
        tokenizer,
        model,
        device="cpu",
        model_name_or_path="openai/whisper-large-v2",
        checkpoint_identity={"model_id": "openai/whisper-large-v2", "test": True},
    )
    return extractor, feature_extractor, tokenizer, model


def batch(size=1):
    return {
        "expert3_wavs": torch.zeros(size, 1, 160_000, dtype=torch.float32),
        "captions_raw": [" rain " if index == 0 else "engine in distance" for index in range(size)],
    }


def write_local_snapshot(directory, *, model_id=MODEL_ID):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"_name_or_path": model_id, "model_type": "whisper", "d_model": 1280}),
        encoding="utf-8",
    )
    (directory / "preprocessor_config.json").write_text(
        json.dumps({"feature_size": 80, "sampling_rate": 16_000}), encoding="utf-8"
    )
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "vocab.json").write_text("{}", encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"synthetic checkpoint marker")


def test_outputs_match_original_unmasked_sequence_means_and_exact_calls():
    extractor, frontend, tokenizer, model = make_extractor()
    data = batch(2)

    audio, text = extractor(data)

    assert audio.shape == (2, 1280)
    assert text.shape == (2, 1280)
    assert audio.dtype == torch.float32 and text.dtype == torch.float32
    assert audio.device.type == "cpu" and text.device.type == "cpu"
    assert torch.equal(audio[:, 0], torch.tensor([10.0, 11.0]))
    # The shorter caption's padded decoder position is included: (1 + 2 + 0) / 3.
    assert torch.equal(text[:, 0], torch.tensor([1.0, 5.0]))
    assert len(frontend.calls) == 1
    audio_list, sample_rate, tensor_format = frontend.calls[0]
    assert [item.shape for item in audio_list] == [(160_000,), (160_000,)]
    assert sample_rate == 16_000 and tensor_format == "pt"
    assert tokenizer.calls == [([" rain ", "engine in distance"], "pt", True)]
    model_inputs, decoder_ids = model.calls[0]
    assert model_inputs.shape == (2, 80, 3000)
    assert decoder_ids.tolist() == [[1, 2, 0], [4, 5, 6]]
    assert len(model.calls[0]) == 2
    assert model.inference_modes == [True]


@pytest.mark.parametrize("size", [1, 3])
def test_supports_single_and_multiple_items_with_deterministic_float32_outputs(size):
    extractor, _, _, _ = make_extractor()
    first = extractor(batch(size))
    second = extractor(batch(size))
    assert first[0].shape == (size, 1280)
    assert first[1].shape == (size, 1280)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(output.dtype == torch.float32 for output in first)
    assert all(torch.isfinite(output).all() for output in first)


def test_model_parameters_are_frozen_and_model_is_in_evaluation_mode():
    extractor, _, _, model = make_extractor()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert extractor.training is False


def test_rejects_missing_waveform_and_invalid_waveform_contract():
    extractor, _, _, _ = make_extractor()
    with pytest.raises(KeyError, match="expert3_wavs"):
        extractor({"captions_raw": ["rain"]})
    malformed = batch()
    malformed["expert3_wavs"] = torch.zeros(1, 1, 159_999)
    with pytest.raises(ValueError, match="160000"):
        extractor(malformed)
    malformed = batch()
    malformed["expert3_wavs"] = torch.zeros(1, 160_000)
    with pytest.raises(ValueError, match="shape"):
        extractor(malformed)


def test_rejects_wrong_waveform_dtype_and_nonfinite_waveform():
    extractor, _, _, _ = make_extractor()
    malformed = batch()
    malformed["expert3_wavs"] = malformed["expert3_wavs"].double()
    with pytest.raises(TypeError, match="float32"):
        extractor(malformed)
    malformed = batch()
    malformed["expert3_wavs"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        extractor(malformed)


def test_rejects_invalid_sample_rate_caption_alignment_and_empty_captions():
    with pytest.raises(ValueError, match="16000"):
        WhisperTop2FeatureExtractor(
            FakeFeatureExtractor(), FakeTokenizer(), FakeWhisperModel(), sample_rate=8_000
        )
    extractor, _, _, _ = make_extractor()
    malformed = batch()
    malformed["sample_rate"] = 48_000
    with pytest.raises(ValueError, match="sample_rate"):
        extractor(malformed)
    with pytest.raises(ValueError, match="one caption"):
        extractor({**batch(2), "captions_raw": ["one"]})
    with pytest.raises(ValueError, match="non-empty"):
        extractor({**batch(), "captions_raw": ["  "]})


def test_rejects_nonfinite_model_outputs():
    extractor, _, _, _ = make_extractor(bad_output=True)
    with pytest.raises(ValueError, match="non-finite"):
        extractor(batch())


def test_offline_local_checkpoint_loading_uses_exact_huggingface_classes(tmp_path, monkeypatch):
    model_dir = tmp_path / "whisper-large-v2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "openai/whisper-large-v2", "model_type": "whisper", "d_model": 1280}),
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"feature_size": 80, "sampling_rate": 16_000}), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"synthetic checkpoint marker")

    calls = []

    class Loader:
        def __init__(self, result):
            self.result = result

        def from_pretrained(self, path, **kwargs):
            calls.append((type(self).__name__, path, kwargs))
            return self.result

    class AutoFeatureExtractorLoader(Loader):
        pass

    class WhisperTokenizerLoader(Loader):
        pass

    class WhisperModelLoader(Loader):
        pass

    model = FakeWhisperModel()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoFeatureExtractor=AutoFeatureExtractorLoader(FakeFeatureExtractor()),
            WhisperTokenizer=WhisperTokenizerLoader(FakeTokenizer()),
            WhisperModel=WhisperModelLoader(model),
        ),
    )

    extractor = WhisperTop2FeatureExtractor.from_config(
        {
            "runtime": {"device": "cpu", "offline": True, "cache_dir": str(tmp_path / "hf-cache")},
            "whisper_top2": {"model_name_or_path": str(model_dir)},
        }
    )

    assert [call[0] for call in calls] == [
        "AutoFeatureExtractorLoader",
        "WhisperTokenizerLoader",
        "WhisperModelLoader",
    ]
    assert all(call[1] == str(model_dir) for call in calls)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["cache_dir"] == str(tmp_path / "hf-cache") for call in calls)
    assert extractor.checkpoint_identity["model_id"] == "openai/whisper-large-v2"
    assert extractor.preprocessing_identity["sample_rate"] == 16_000
    assert extractor.feature_extraction_version == FEATURE_EXTRACTION_VERSION


def test_rejects_incomplete_local_checkpoint_before_loading(tmp_path, monkeypatch):
    model_dir = tmp_path / "incomplete-whisper-large-v2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "openai/whisper-large-v2", "model_type": "whisper", "d_model": 1280}),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        WhisperTop2FeatureExtractor.from_config(
            {
                "runtime": {"device": "cpu", "offline": True},
                "whisper_top2": {"model_name_or_path": str(model_dir)},
            }
        )


def test_local_large_v2_named_ancestor_cannot_override_explicit_large_v1_identity(tmp_path):
    wrong_variant = tmp_path / "whisper-large-v2" / "renamed-model"
    write_local_snapshot(wrong_variant, model_id="openai/whisper-large-v1")
    with pytest.raises(ValueError, match="different Whisper variant"):
        _local_model_files(wrong_variant)


def test_huggingface_snapshot_symlink_fingerprints_snapshot_not_blobs(tmp_path, monkeypatch):
    repo_root = tmp_path / "models--openai--whisper-large-v2"
    blob_root = repo_root / "blobs"
    snapshot = repo_root / "snapshots" / "commit-123"
    blob_root.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    source_files = {
        "config.json": json.dumps({"_name_or_path": MODEL_ID, "model_type": "whisper", "d_model": 1280}),
        "preprocessor_config.json": json.dumps({"feature_size": 80, "sampling_rate": 16_000}),
        "tokenizer_config.json": "{}",
        "vocab.json": "{}",
        "merges.txt": "#version: 0.2\n",
        "model.safetensors": "synthetic checkpoint marker",
    }
    for name, content in source_files.items():
        blob = blob_root / f"blob-{name}"
        blob.write_text(content, encoding="utf-8")
        (snapshot / name).symlink_to(blob)

    hub_calls = []

    def try_to_load_from_cache(repo_id, filename, *, cache_dir=None, revision=None):
        hub_calls.append((repo_id, filename, cache_dir, revision))
        return str(snapshot / filename) if filename == "config.json" else None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache),
    )

    identity = _checkpoint_identity(MODEL_ID, cache_dir=tmp_path / "hf-cache", revision="commit-123")

    assert identity["model_id"] == MODEL_ID
    assert "model.safetensors" in identity["files"]
    assert hub_calls == [(MODEL_ID, "config.json", str(tmp_path / "hf-cache"), "commit-123")]


def test_rejects_alternative_whisper_variant_before_loading():
    with pytest.raises(ValueError, match="large-v2"):
        WhisperTop2FeatureExtractor.from_config(
            {
                "runtime": {"device": "cpu", "offline": True},
                "whisper_top2": {"model_name_or_path": "openai/whisper-large-v3"},
            }
        )


def test_close_releases_model_and_is_idempotent():
    extractor, _, _, _ = make_extractor()
    extractor.close()
    extractor.close()
    with pytest.raises(RuntimeError, match="closed"):
        extractor(batch())
