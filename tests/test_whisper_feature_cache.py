import json

import numpy as np
import pytest

from data.whisper_feature_cache import WhisperFeatureCacheWriter


def features(values):
    values = np.asarray(values, dtype=np.float32)
    return np.repeat(values.reshape(-1, 1), 1280, axis=1)


def make_writer(path, *, split="train", checkpoint="checkpoint-a", sample_rate=16_000):
    return WhisperFeatureCacheWriter(
        path,
        split=split,
        source_checkpoint_identity={"model_id": "openai/whisper-large-v2", "digest": checkpoint},
        preprocessing_identity={"sample_rate": sample_rate, "pooling": "mean"},
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
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(["train:one.wav", "train:one.wav"], features([1, 2]), features([3, 4]))
    writer.append(["train:one.wav"], features([1]), features([3]))
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(["train:one.wav"], features([2]), features([4]))


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
    with pytest.raises(ValueError, match="split mismatch"):
        make_writer(cache_path, split="test")


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
