"""Small synthetic Full9 plus Whisper cache for E6 integration tests."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.moe_expert_cache import build_assembly_identity, build_expert_cache_identity
from data.moe_feature_store import MoeFeatureStoreWriter
from data.set_b_cached_dataset import SetBCachedDataset
from data.whisper_feature_cache import WhisperFeatureCacheWriter, build_whisper_dataset_identity
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.registry import TOP2_EXPERT_INPUT_CONTRACTS, profile_contract


def synthetic_caches(root: Path, *, split="train", count=2, labels=None,
                     whisper_order=None, whisper_caption_suffix=""):
    full9, whisper, audio_dir = root / "full9", root / "whisper", root / split / "wav"
    audio_dir.mkdir(parents=True, exist_ok=True)
    order, dimensions, width = profile_contract("full_9")
    records = []
    whisper_records = []
    for index in range(count):
        name = f"{index}.wav"
        path = audio_dir / name
        path.write_bytes(f"wave-{split}-{index}".encode())
        sample_id = f"{split}:{name}"
        caption = f"caption {index}"
        records.append({"sample_id": sample_id, "wave_file_name": name, "raw_caption": caption,
                        "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **({"label_raw": (labels or [5.1] * count)[index]} if split != "test" else {})})
        whisper_records.append({"sample_id": sample_id, "captions_raw": caption + whisper_caption_suffix, "audio_path": path, "split": split})
    source = {"schema_version": 1, "split": split, "records": records}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    cache_ids, models = {}, {}
    representation = {"feature_representation": "paired_cross_4d_plus_1_v2",
        "feature_blocks": ["audio_embedding", "text_embedding", "absolute_difference_l2", "elementwise_product_l2", "cosine_l2_scalar"],
        "feature_width_formula": "4D+1"}
    for index, name in enumerate(order):
        contract = {"score_shape": [None, 1], "feature_shape": [None, dimensions[name]],
                    "score_dtype": "float32", "feature_dtype": "float32"}
        if name == "top2_laion":
            contract.update(representation)
        input_contract = asdict(TOP2_EXPERT_INPUT_CONTRACTS[name]) if name in TOP2_EXPERT_INPUT_CONTRACTS else {"audio_key": "expert3_wavs", "sample_rate_hz": 16000}
        model_id = {"sha256": f"{index + 1:064x}"}
        cache_ids[name] = build_expert_cache_identity(expert_name=name, checkpoint_provenance=model_id,
            implementation_provenance={"source": "synthetic"}, preprocessing_provenance={"policy": "synthetic"},
            input_contract=input_contract, inference_policy={"module_mode": "eval"},
            score_contract=contract, relevant_dependencies={"torch": "test"})
        models[name] = model_id
    ids = [r["sample_id"] for r in records]
    scores = np.zeros((count, len(order)), np.float32)
    features = np.zeros((count, width), np.float32)
    g = torch.Generator().manual_seed(1)
    laion = build_paired_cross_features(torch.randn(count, 512, generator=g), torch.randn(count, 512, generator=g)).numpy()
    offset = sum(dimensions[n] for n in order[:order.index("top2_laion")])
    features[:, offset:offset + 2049] = laion
    scores[:, order.index("top2_laion")] = laion[:, -1]
    assembly = build_assembly_identity(expert_profile="full_9", expert_order=order,
        expert_cache_identities=cache_ids, feature_dimensions=dimensions, source_manifest_identity=source_hash)
    writer = MoeFeatureStoreWriter(full9, split=split, expected_sample_ids=ids, source_manifest_sha256=source_hash,
        checkpoint_provenance=models, preprocessing_identity={"policy": "synthetic", "source_manifest_sha256": source_hash},
        profile="full_9", per_expert_cache_identities=cache_ids, assembly_identity=assembly)
    writer.write_shard(0, sample_ids=ids, wave_file_names=[r["wave_file_name"] for r in records],
        scores=scores, features=features,
        labels_raw=np.asarray([r["label_raw"] for r in records], np.float32) if split != "test" else None)
    writer.finalize()
    (full9 / split / "source_manifest.json").write_text(json.dumps({**source, "sha256": source_hash}))
    project = Path(__file__).resolve().parents[1]
    preprocessing = {"extraction_batch_size": 2,
        "decoder_padding_scope": "per extraction batch; padded positions participate in mean pooling",
        "unified_waveform_preprocessing": {"source_sha256": hashlib.sha256((project / "data/unified_dataset.py").read_bytes()).hexdigest()},
        "collation_source_sha256": hashlib.sha256((project / "data/unified_collate.py").read_bytes()).hexdigest(),
        "inference_device": "cpu", "model_parameter_dtype": "torch.float32"}
    cache = WhisperFeatureCacheWriter(whisper / split, split=split,
        source_checkpoint_identity={"model_id": "openai/whisper-large-v2", "sha256": "synthetic"},
        preprocessing_identity=preprocessing, dataset_identity=build_whisper_dataset_identity(whisper_records, split=split))
    cache.append([ids[index] for index in (whisper_order or range(count))],
                 np.zeros((count, 1280), np.float32), np.ones((count, 1280), np.float32))
    return full9, whisper, audio_dir


def test_join_uses_verified_source_and_preserves_label_precision(tmp_path):
    full9, whisper, audio = synthetic_caches(tmp_path)
    dataset = SetBCachedDataset(full9, whisper, split="train", source_audio_dir=audio)
    assert len(dataset) == 2
    assert dataset[0]["component_outputs"]["top2_laion"]["features"].shape == (2049,)
    assert dataset[0]["target_normalized"].item() == pytest.approx((np.float32(5.1) - 5) / 5)


def test_changed_source_audio_is_rejected(tmp_path):
    full9, whisper, audio = synthetic_caches(tmp_path)
    (audio / "0.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source WAV changed"):
        SetBCachedDataset(full9, whisper, split="train", source_audio_dir=audio)


def test_test_cache_does_not_require_labels(tmp_path):
    full9, whisper, audio = synthetic_caches(tmp_path, split="test")
    row = SetBCachedDataset(full9, whisper, split="test", source_audio_dir=audio)[0]
    assert "target_normalized" not in row


def test_rejects_whisper_order_and_incomplete_cache(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path)
    manifest_path = whisper / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_identity"]["sample_count"] += 1
    from data.whisper_feature_cache import _identity_sha256
    manifest["metadata_sha256"] = _identity_sha256({k: v for k, v in manifest.items() if k != "metadata_sha256"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dataset identity"):
        SetBCachedDataset(full9, whisper, split="train")


def test_rejects_different_whisper_checkpoint(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path)
    manifest_path = whisper / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_checkpoint_identity"]["model_id"] = "openai/whisper-tiny"
    from data.whisper_feature_cache import _identity_sha256
    manifest["metadata_sha256"] = _identity_sha256({k: v for k, v in manifest.items() if k != "metadata_sha256"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Whisper cache checkpoint"):
        SetBCachedDataset(full9, whisper, split="train")


def test_rejects_changed_caption_and_reordered_whisper_ids(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path / "caption", whisper_caption_suffix=" changed")
    with pytest.raises(ValueError, match="dataset identity"):
        SetBCachedDataset(full9, whisper, split="train")
    full9, whisper, _ = synthetic_caches(tmp_path / "order", whisper_order=[1, 0])
    with pytest.raises(ValueError, match="sample order"):
        SetBCachedDataset(full9, whisper, split="train")
