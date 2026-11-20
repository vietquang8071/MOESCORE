from types import SimpleNamespace


def test_full8_assembly_processes_train_validation_and_test_with_shared_wav(monkeypatch, tmp_path):
    import data.full8_assembler as assembler

    shared = "a" * 64
    manifests = {
        split: {"assembly_identity": {"ordered_source_fingerprints": [
            {"sample_id": f"{split}:sample.wav", "wav_sha256": shared},
        ]}}
        for split in ("train", "validation", "test")
    }
    monkeypatch.setattr(assembler, "validate_feature_store", lambda root, split, require_labels: manifests[split])
    built = []
    monkeypatch.setattr(
        assembler, "build_full8_split",
        lambda *args, split, **kwargs: built.append(split) or {"split": split, "sample_count": 1},
    )

    report = assembler.build_full8_stores("full7", "e8-cache", tmp_path / "full8")

    assert report["status"] == "PASS"
    assert built == ["train", "validation", "test"]
    assert "audio_overlap_diagnostic" not in report


def test_global_gate_provenance_keeps_split_hashes_without_comparing_them():
    from model.expert_adapters.registry import profile_contract
    from scripts.train_moe_gating import _store_provenance

    order, dimensions, _ = profile_contract("full_7")
    provenance = {name: {"sha256": "b" * 64} for name in order}
    common = {
        "expert_profile": "full_7",
        "expert_order": list(order),
        "feature_dimensions": dimensions,
        "checkpoint_provenance": provenance,
        "assembly_identity": {
            "shared_source_contract": {"id": "source"},
            "score_contracts": {name: {"identity": "a" * 64} for name in order},
        },
    }
    train = SimpleNamespace(
        manifest={**common, "assembly_identity": {**common["assembly_identity"],
            "ordered_source_fingerprints": [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64}]}},
        identity="c" * 64,
    )
    validation = SimpleNamespace(
        manifest={**common, "assembly_identity": {**common["assembly_identity"],
            "ordered_source_fingerprints": [{"sample_id": "validation:b.wav", "wav_sha256": "a" * 64}]}},
        identity="d" * 64,
    )

    checkpoint_provenance, identity = _store_provenance(train, validation)

    assert checkpoint_provenance == provenance
    assert identity["source_wav_sha256_by_split"] == {
        "train": ["a" * 64], "validation": ["a" * 64],
    }
    assert "audio_overlap_diagnostic" not in identity


def test_gate_inference_does_not_compare_test_wav_to_training_splits(monkeypatch, tmp_path):
    """Shared WAV content does not interfere with label-free inference alignment."""
    import scripts.infer_moe_gating as inference

    test_manifest = {
        "expert_profile": "full_7",
        "checkpoint_provenance": {f"expert{i}": {"sha256": f"{i:064x}"} for i in range(1, 8)},
        "assembly_identity": {"ordered_source_fingerprints": [
            {"sample_id": "test:a.wav", "wav_sha256": "a" * 64},
            {"sample_id": "test:b.wav", "wav_sha256": "b" * 64},
        ]},
    }

    class Dataset:
        manifest = test_manifest
        identity = "f" * 64

        def __init__(self, *args, **kwargs):
            assert kwargs == {"split": "test", "require_labels": False}

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    class Gate:
        profile = "full_7"

        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

    class Manager:
        def load_for_inference(self, *args, **kwargs):
            return {
                "feature_store_identity": {
                    "preprocessing_contract": {"contract": "same"},
                    "source_wav_sha256_by_split": {
                        "train": ["a" * 64], "validation": ["b" * 64],
                    },
                },
                "epoch": 1,
            }

    class Evaluator:
        def __init__(self, gate):
            pass

        def predict(self, loader):
            assert len(list(loader)) == 2
            return [{"wave_file_name": "a.wav", "predicted_score": 5.0},
                    {"wave_file_name": "b.wav", "predicted_score": 5.0}]

    monkeypatch.setattr(inference, "MoeTrainingDataset", Dataset)
    monkeypatch.setattr(inference, "GatingNetwork", Gate)
    monkeypatch.setattr(inference, "MoeCheckpointManager", Manager)
    monkeypatch.setattr(inference, "MoeGateEvaluator", Evaluator)
    monkeypatch.setattr(inference, "collate_moe_features", lambda batch: batch)
    monkeypatch.setattr(inference, "preprocessing_contract", lambda manifest: {"contract": "same"})
    monkeypatch.setattr(inference, "write_submission_csv", lambda rows, output, filename_column: tmp_path / "submission.csv")
    monkeypatch.setattr(inference, "profile_contract", lambda profile: (tuple(f"expert{i}" for i in range(1, 8)), (), 29959))

    report = inference.infer_checkpoint(
        {"runtime": {"device": "cpu"}, "data": {"feature_store_root": str(tmp_path), "profile": "full_7"},
         "inference": {"batch_size": 1}},
        "gate.pt", str(tmp_path / "submission.csv"), device_override="cpu",
    )

    assert report["status"] == "PASS"
    assert report["sample_count"] == 2


def test_independent_source_identity_validation_keeps_hash_and_sample_id_checks():
    import pytest
    from data.source_identity import validate_source_fingerprints

    with pytest.raises(ValueError, match="SHA-256"):
        validate_source_fingerprints(
            [{"sample_id": "train:a.wav", "wav_sha256": "not-a-hash"}], split="train",
        )
    with pytest.raises(ValueError, match="duplicate train source sample ID"):
        validate_source_fingerprints(
            [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64},
             {"sample_id": "train:a.wav", "wav_sha256": "b" * 64}],
            split="train",
        )
