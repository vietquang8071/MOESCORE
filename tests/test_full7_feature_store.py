import numpy as np
import pytest
from types import SimpleNamespace

from data.moe_feature_store import MoeFeatureStoreWriter, validate_feature_store
from model.expert_adapters.registry import profile_contract


def test_full7_store_schema_and_label_free_test(tmp_path):
    order, _, width = profile_contract("full_7")
    for split in ("train", "validation", "test"):
        ids = [f"{split}:a.wav"]
        writer = MoeFeatureStoreWriter(
            tmp_path, split=split, expected_sample_ids=ids,
            source_manifest_sha256="a" * 64,
            checkpoint_provenance={name: {"sha256": "b" * 64} for name in order},
            preprocessing_identity={"policy": "synthetic"}, profile="full_7",
            assembly_identity={"source_full9_store_identity": "c" * 64,
                               "source_whisper_manifest_sha256": "d" * 64,
                               "score_contracts": {name: {"source": "m2d_lora" if name == "expert7" else name, "semantics": "synthetic_normalized",
                                                         "target_scale": "normalized_minus1_to1",
                                                         "output_scale": "normalized_minus1_to1", "conversion": "identity",
                                                         "audit_status": "SYNTHETIC_VERIFIED", "identity": "b" * 64}
                                                   for name in order},
                               "shared_source_contract": {"synthetic": True},
                               "ordered_source_fingerprints": [{"sample_id": ids[0],
                                   "wav_sha256": "e" * 64, "caption_sha256": "f" * 64}],
                               "head_membership": {name: {"synthetic": True, "per_row": ["SYNTHETIC_UNTRAINED_HEAD"]}
                                                   for name in ("expert5", "expert6")},
                               "synthetic_provenance": True},
        )
        writer.write_shard(0, sample_ids=ids, wave_file_names=["a.wav"],
                           scores=np.zeros((1, 7), np.float32),
                           features=np.zeros((1, width), np.float32),
                           labels_raw=None if split == "test" else np.array([5.0], np.float32))
        writer.finalize()
        manifest = validate_feature_store(tmp_path, split=split)
        assert manifest["schema_version"] == 3
        assert manifest["score_width"] == 7
        assert manifest["labels_present"] is (split != "test")
        if split == "test":
            assert manifest.get("labels_scale") is None


def test_full7_requires_assembly_and_score_contracts(tmp_path):
    order, _, _ = profile_contract("full_7")
    with pytest.raises(ValueError):
        MoeFeatureStoreWriter(tmp_path, split="train", expected_sample_ids=["train:a"],
                              source_manifest_sha256="a" * 64,
                              checkpoint_provenance={name: {"id": name} for name in order},
                              preprocessing_identity={"id": "x"}, profile="full_7")


def test_partial_full7_shard_rejects_changed_source_identity(tmp_path):
    order, _, width = profile_contract("full_7")
    contracts = {name: {"source": "m2d_lora" if name == "expert7" else name, "semantics": "synthetic", "output_scale": "normalized_minus1_to1",
                        "target_scale": "normalized_minus1_to1",
                        "conversion": "identity", "audit_status": "SYNTHETIC_VERIFIED", "identity": "e" * 64}
                 for name in order}
    def writer(source_hash):
        return MoeFeatureStoreWriter(tmp_path, split="train", expected_sample_ids=["train:a.wav"],
            source_manifest_sha256=source_hash,
            checkpoint_provenance={name: {"sha256": "e" * 64} for name in order},
            preprocessing_identity={"synthetic": True}, profile="full_7",
            assembly_identity={"source_full9_store_identity": source_hash,
                               "source_whisper_manifest_sha256": "f" * 64,
                               "score_contracts": contracts, "shared_source_contract": {"synthetic": True},
                               "ordered_source_fingerprints": [{"sample_id": "train:a.wav",
                                   "wav_sha256": "e" * 64, "caption_sha256": "f" * 64}],
                               "head_membership": {name: {"synthetic": True, "per_row": ["SYNTHETIC_UNTRAINED_HEAD"]}
                                                   for name in ("expert5", "expert6")},
                               "synthetic_provenance": True})
    first = writer("a" * 64)
    kwargs = {"sample_ids": ["train:a.wav"], "wave_file_names": ["a.wav"],
              "scores": np.zeros((1, 7), np.float32), "features": np.zeros((1, width), np.float32),
              "labels_raw": np.array([5.0], np.float32)}
    first.write_shard(0, **kwargs)
    with pytest.raises(ValueError, match="differs"):
        writer("b" * 64).write_shard(0, **kwargs)


def test_training_preflight_preserves_each_split_wav_provenance_without_comparing_splits():
    from scripts.train_moe_gating import _store_provenance
    order, dims, _ = profile_contract("full_7")
    provenance = {name: {"sha256": "a" * 64} for name in order}
    common = {"expert_profile": "full_7", "expert_order": list(order),
              "feature_dimensions": dims, "checkpoint_provenance": provenance,
              "assembly_identity": {"shared_source_contract": {"id": "x"},
                                    "score_contracts": {name: {"identity": "a" * 64} for name in order}}}
    train_manifest = {**common, "assembly_identity": {**common["assembly_identity"],
        "ordered_source_fingerprints": [{"sample_id": "train:a.wav", "wav_sha256": "f" * 64}]}}
    val_manifest = {**common, "assembly_identity": {**common["assembly_identity"],
        "ordered_source_fingerprints": [{"sample_id": "validation:b.wav", "wav_sha256": "f" * 64}]}}
    _, identity = _store_provenance(
        SimpleNamespace(manifest=train_manifest, identity="a" * 64),
        SimpleNamespace(manifest=val_manifest, identity="b" * 64),
    )
    assert identity["source_wav_sha256_by_split"] == {
        "train": ["f" * 64], "validation": ["f" * 64],
    }
    assert identity["source_sample_ids_by_split"] == {
        "train": ["train:a.wav"], "validation": ["validation:b.wav"],
    }
    assert "audio_overlap_diagnostic" not in identity


def test_moe_config_rejects_removed_overlap_configuration(tmp_path):
    import yaml
    from training.config import load_moe_training_config

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs/moe_full7.example.yaml").read_text())
    config_path = tmp_path / "moe.yaml"
    raw["data"]["feature_store_root"] = str(tmp_path / "store")
    config_path.write_text(yaml.safe_dump(raw))
    assert "allow_train_validation_audio_overlap" not in load_moe_training_config(config_path)["data"]
    raw["data"]["allow_train_validation_audio_overlap"] = True
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown field"):
        load_moe_training_config(config_path)


@pytest.mark.parametrize("rows, message", [
    ([{"sample_id": "train:a.wav", "wav_sha256": "invalid"}], "SHA-256"),
    ([{"sample_id": "train:a.wav", "wav_sha256": "a" * 64},
      {"sample_id": "train:a.wav", "wav_sha256": "b" * 64}], "duplicate train source sample ID"),
])
def test_global_gate_still_rejects_invalid_per_split_source_identity(rows, message):
    from scripts.train_moe_gating import _store_provenance
    order, dims, _ = profile_contract("full_7")
    provenance = {name: {"sha256": "a" * 64} for name in order}
    common = {"expert_profile": "full_7", "expert_order": list(order),
              "feature_dimensions": dims, "checkpoint_provenance": provenance,
              "assembly_identity": {"shared_source_contract": {"id": "x"},
                                    "score_contracts": {name: {"identity": "a" * 64} for name in order}}}
    train = SimpleNamespace(manifest={**common, "assembly_identity": {**common["assembly_identity"],
        "ordered_source_fingerprints": rows}}, identity="a" * 64)
    validation = SimpleNamespace(manifest={**common, "assembly_identity": {**common["assembly_identity"],
        "ordered_source_fingerprints": [{"sample_id": "validation:b.wav", "wav_sha256": "c" * 64}]}},
        identity="b" * 64)
    with pytest.raises(ValueError, match=message):
        _store_provenance(train, validation)


def test_global_gate_loaders_keep_all_official_train_and_validation_rows():
    from scripts.train_moe_gating import _feature_loader
    from training.sampler import StatefulRandomSampler

    train_rows = list(range(7500))
    validation_rows = list(range(3000))
    train_loader = _feature_loader(
        train_rows, batch_size=8, sampler=StatefulRandomSampler(train_rows, seed=9),
        collate_fn=lambda batch: batch,
    )
    validation_loader = _feature_loader(
        validation_rows, batch_size=8, collate_fn=lambda batch: batch,
    )
    train_seen = [row for batch in train_loader for row in batch]
    validation_seen = [row for batch in validation_loader for row in batch]
    assert len(train_seen) == len(set(train_seen)) == 7500
    assert len(validation_seen) == len(set(validation_seen)) == 3000


def test_inference_no_longer_exposes_cross_split_wav_rejection_helper():
    import scripts.infer_moe_gating as inference
    assert not hasattr(inference, "_reject_test_wav_overlap")
