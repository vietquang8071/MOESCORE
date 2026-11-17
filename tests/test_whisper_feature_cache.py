import json
from types import SimpleNamespace

import numpy as np
import pytest

from data.whisper_feature_cache import (
    WhisperFeatureCacheWriter,
    build_whisper_dataset_identity,
    validate_whisper_cache_resume,
)
from scripts.extract_whisper_features import _cache_preprocessing_identity


def features(values):
    values = np.asarray(values, dtype=np.float32)
    return np.repeat(values.reshape(-1, 1), 1280, axis=1)


def make_writer(
    path,
    *,
    split="train",
    checkpoint="checkpoint-a",
    sample_rate=16_000,
    batch_size=1,
    dataset_identity=None,
):
    return WhisperFeatureCacheWriter(
        path,
        split=split,
        source_checkpoint_identity={"model_id": "openai/whisper-large-v2", "digest": checkpoint},
        preprocessing_identity={
            "sample_rate": sample_rate,
            "pooling": "mean",
            "extraction_batch_size": batch_size,
        },
        dataset_identity=dataset_identity or {
            "split": split,
            "sample_count": 3,
            "ordered_samples_sha256": "a" * 64,
        },
    )


def test_atomic_shards_preserve_sample_alignment_and_float32_roundtrip(tmp_path):
    cache_path = tmp_path / "train"
    writer = make_writer(cache_path)
    writer.append(["train:first.wav", "train:second.wav"], features([1, 2]), features([11, 12]))
    writer.append(["train:third.wav"], features([3]), features([13]))

    restored = make_writer(cache_path).load()

    assert restored["sample_ids"] == ["train:first.wav", "train:second.wav", "train:third.wav"]
    assert restored["whisper_audio_features"].dtype == np.float32
    assert restored["whisper_text_features"].dtype == np.float32
    assert restored["whisper_audio_features"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert restored["whisper_text_features"][:, 0].tolist() == [11.0, 12.0, 13.0]
    assert all(path.suffix != ".tmp" for path in cache_path.iterdir())
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "train"
    assert manifest["feature_dimensions"] == {"audio": 1280, "text": 1280, "total": 2560}
    assert manifest["source_checkpoint_identity"]["digest"] == "checkpoint-a"
    assert manifest["preprocessing_identity"]["sample_rate"] == 16_000
    assert manifest["dataset_identity"]["ordered_samples_sha256"] == "a" * 64
    assert manifest["feature_extraction_version"] == "top2-whisper-large-v2-meanpool-v1"


def test_reopening_cache_resumes_after_existing_sample_ids(tmp_path):
    cache_path = tmp_path / "validation"
    first = make_writer(cache_path, split="validation")
    first.append(["validation:one.wav"], features([1]), features([2]))

    resumed = make_writer(cache_path, split="validation")
    assert resumed.sample_ids == ("validation:one.wav",)
    resumed.append(["validation:two.wav"], features([3]), features([4]))
    assert resumed.sample_ids == ("validation:one.wav", "validation:two.wav")


def test_rejects_duplicate_sample_ids_within_or_across_shards(tmp_path):
    writer = make_writer(tmp_path / "train")
    with pytest.raises(ValueError, match="split prefix"):
        writer.append(["validation:wrong-split.wav"], features([1]), features([2]))
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(["train:one.wav", "train:one.wav"], features([1, 2]), features([3, 4]))
    writer.append(["train:one.wav"], features([1]), features([3]))
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(["train:one.wav"], features([2]), features([4]))


def test_rejects_feature_rows_beyond_the_identified_dataset_size(tmp_path):
    writer = make_writer(tmp_path / "train", dataset_identity={
        "split": "train",
        "sample_count": 1,
        "ordered_samples_sha256": "a" * 64,
    })
    writer.append(["train:one.wav"], features([1]), features([2]))
    with pytest.raises(ValueError, match="exceeds the dataset identity sample count"):
        writer.append(["train:two.wav"], features([3]), features([4]))


def test_rejects_nonfinite_or_wrong_width_features(tmp_path):
    writer = make_writer(tmp_path / "train")
    invalid = features([1])
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        writer.append(["train:one.wav"], invalid, features([2]))
    with pytest.raises(ValueError, match="1280"):
        writer.append(["train:one.wav"], np.zeros((1, 1279), dtype=np.float32), features([2]))


def test_rejects_checkpoint_preprocessing_and_split_identity_mismatches(tmp_path):
    cache_path = tmp_path / "train"
    make_writer(cache_path).append(["train:one.wav"], features([1]), features([2]))
    with pytest.raises(ValueError, match="identity mismatch"):
        make_writer(cache_path, checkpoint="checkpoint-b")
    with pytest.raises(ValueError, match="preprocessing identity mismatch"):
        make_writer(cache_path, sample_rate=48_000)
    with pytest.raises(ValueError, match="preprocessing identity mismatch"):
        make_writer(cache_path, batch_size=2)
    with pytest.raises(ValueError, match="split mismatch"):
        make_writer(cache_path, split="test")
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        make_writer(cache_path, dataset_identity={
            "split": "train",
            "sample_count": 3,
            "ordered_samples_sha256": "b" * 64,
        })


def test_dataset_identity_changes_with_caption_audio_or_row_order(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first waveform")
    second_audio.write_bytes(b"second waveform")
    records = [
        {"sample_id": "train:first.wav", "split": "train", "captions_raw": "rain", "audio_path": first_audio},
        {"sample_id": "train:second.wav", "split": "train", "captions_raw": "engine", "audio_path": second_audio},
    ]

    original = build_whisper_dataset_identity(records, split="train")
    changed_caption = [dict(records[0], captions_raw="heavy rain"), records[1]]
    changed_audio = [records[0], dict(records[1], audio_path=first_audio)]

    assert build_whisper_dataset_identity(changed_caption, split="train") != original
    assert build_whisper_dataset_identity(changed_audio, split="train") != original
    assert build_whisper_dataset_identity(list(reversed(records)), split="train") != original


def test_resume_requires_an_ordered_prefix_and_original_batch_boundaries():
    ids = ["train:one.wav", "train:two.wav", "train:three.wav", "train:four.wav"]
    assert validate_whisper_cache_resume(ids, ids[:2], batch_size=2) == 2
    with pytest.raises(ValueError, match="ordered prefix"):
        validate_whisper_cache_resume(ids, [ids[0], ids[2]], batch_size=2)
    with pytest.raises(ValueError, match="batch boundary"):
        validate_whisper_cache_resume(ids, ids[:1], batch_size=2)
    # A short batch is valid only when it completes the dataset.
    assert validate_whisper_cache_resume(ids[:3], ids[:3], batch_size=2) == 3


def test_cli_preprocessing_identity_changes_with_batch_size_and_execution_device():
    class Model:
        def parameters(self):
            return iter([SimpleNamespace(dtype="torch.float32")])

    extractor = SimpleNamespace(
        preprocessing_identity={"sample_rate": 16_000},
        model=Model(),
        device="cpu",
    )
    single = _cache_preprocessing_identity(extractor, batch_size=1)
    pair = _cache_preprocessing_identity(extractor, batch_size=2)
    assert single["extraction_batch_size"] == 1
    assert pair["extraction_batch_size"] == 2
    assert single["inference_device"] == "cpu"
    assert single["model_parameter_dtype"] == "torch.float32"
    assert single != pair


def test_train_validation_and_test_caches_are_independent(tmp_path):
    for split, value in (("train", 1), ("validation", 2), ("test", 3)):
        writer = make_writer(tmp_path / split, split=split)
        writer.append([f"{split}:same.wav"], features([value]), features([value + 10]))
        assert writer.load()["sample_ids"] == [f"{split}:same.wav"]


def test_rejects_tampered_manifest_before_reuse(tmp_path):
    cache_path = tmp_path / "train"
    writer = make_writer(cache_path)
    writer.append(["train:one.wav"], features([1]), features([2]))
    manifest_path = cache_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        make_writer(cache_path)


def test_rejects_tampered_shard_before_reuse(tmp_path):
    cache_path = tmp_path / "train"
    writer = make_writer(cache_path)
    writer.append(["train:one.wav"], features([1]), features([2]))
    shard = next(cache_path.glob("shard-*.npz"))
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        make_writer(cache_path)
