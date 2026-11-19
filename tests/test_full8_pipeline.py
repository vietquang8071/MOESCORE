from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import pytest

from model.expert_adapters.registry import (
    EXPERT_PROFILES,
    FULL7_EXPERT_ORDER,
    FULL8_EXPERT_ORDER,
    FULL8_FEATURE_DIMS,
    profile_contract,
    validate_and_assemble_expert_outputs,
)
from model.moe_gating_core import GatingNetwork
from data.full8_contract import build_expert8_score_contract, validate_full8_score_contracts
from data.moe_expert_cache import build_expert_cache_identity
from data.moe_feature_store import MoeFeatureStoreDataset, MoeFeatureStoreWriter, FULL8_FEATURE_STORE_SCHEMA_VERSION


def test_full8_is_full7_order_plus_independent_e8_with_fixed_dimensions():
    order, dimensions, width = profile_contract("full_8")
    assert order == (*FULL7_EXPERT_ORDER, "expert8")
    assert order == FULL8_EXPERT_ORDER
    assert dimensions == dict(FULL8_FEATURE_DIMS)
    assert dimensions["expert8"] == 3073
    assert width == 33032
    assert EXPERT_PROFILES["full_7"] == FULL7_EXPERT_ORDER
    assert profile_contract("full_7")[2] == 29959


def test_full8_score_feature_assembly_and_gate_shapes_are_eight_wide():
    order, dimensions, width = profile_contract("full_8")
    outputs = {
        name: (
            torch.full((3, 1), index / 10, dtype=torch.float32),
            torch.full((3, dimensions[name]), float(index), dtype=torch.float32),
        )
        for index, name in enumerate(order)
    }
    scores, features = validate_and_assemble_expert_outputs(outputs, profile="full_8")
    assert tuple(scores.shape) == (3, 8)
    assert tuple(features.shape) == (3, 33032)
    assert scores.dtype == features.dtype == torch.float32
    assert width == features.shape[1]
    gate = GatingNetwork(input_feature_dim=width, num_experts=8, profile="full_8")
    weights = gate(features)
    assert tuple(weights.shape) == (3, 8)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_full8_score_contract_preserves_full7_contracts_and_binds_e8_checkpoint():
    from model.expert_adapters.registry import FULL8_EXPERT_ORDER

    contracts = {}
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        contracts[name] = {
            "source": ("expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "m2d_lora")[index],
            "semantics": "synthetic_score_fixture",
            "target_scale": "normalized_minus1_to1",
            "output_scale": "normalized_minus1_to1",
            "conversion": "identity",
            "identity": f"{index + 1:064x}",
            "audit_status": "SYNTHETIC_VERIFIED",
        }
    checkpoint_sha = "a" * 64
    contracts["expert8"] = build_expert8_score_contract(checkpoint_sha, synthetic=True)
    assert set(contracts) == set(FULL8_EXPERT_ORDER)
    validate_full8_score_contracts(contracts, synthetic=True, expert8_checkpoint_sha256=checkpoint_sha)
    corrupt = dict(contracts)
    corrupt["expert8"] = {**contracts["expert8"], "identity": "b" * 64}
    try:
        validate_full8_score_contracts(corrupt, synthetic=True, expert8_checkpoint_sha256=checkpoint_sha)
    except ValueError as exc:
        assert "expert8" in str(exc).lower() or "checkpoint" in str(exc).lower()
    else:
        raise AssertionError("E8 score contract not bound to checkpoint identity")


def _synthetic_full8_assembly(split, sample_ids, source_hash):
    from model.expert_adapters.expert8_m2d_clap_runtime import EXPECTED_BASE_M2D_SHA256

    lora_sha = "a" * 64
    cache_identity = build_expert_cache_identity(
        expert_name="expert8",
        checkpoint_provenance={
            "base_checkpoint": {"sha256": EXPECTED_BASE_M2D_SHA256},
            "finetuned_checkpoint": {"sha256": lora_sha},
        },
        implementation_provenance={"adapter_sha256": "b" * 64},
        preprocessing_provenance={"audio": "16khz_mono_pad_truncate_10s"},
        input_contract={"audio_shape": [1, 160000], "caption": "raw"},
        inference_policy={"device": "cpu", "dtype": "float32"},
        score_contract={"policy": "coral10_then_normalize"},
        relevant_dependencies={"torch": "synthetic"},
    )
    contracts = {}
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        contracts[name] = {
            "source": ("expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "m2d_lora")[index],
            "semantics": "synthetic_score_fixture", "target_scale": "normalized_minus1_to1",
            "output_scale": "normalized_minus1_to1", "conversion": "identity",
            "identity": f"{index + 1:064x}", "audit_status": "SYNTHETIC_VERIFIED",
        }
    contracts["expert8"] = build_expert8_score_contract(lora_sha, synthetic=True)
    return {
        "schema_version": 1, "expert_profile": "full_8", "split": split,
        "source_manifest_sha256": source_hash,
        "full7_store_identity_sha256": "c" * 64,
        "full7_assembly_contract_sha256": "d" * 64,
        "expert8_cache_identity": cache_identity,
        "expert8_cache_id": cache_identity["cache_id"],
        "expert8_cache_manifest_sha256": "e" * 64,
        "ordered_source_fingerprints": [
            {"sample_id": sample_id, "wav_sha256": "f" * 64, "caption_sha256": "1" * 64}
            for sample_id in sample_ids
        ],
        "score_contracts": contracts,
        "shared_source_contract": {"source": "synthetic"},
        "head_membership": {
            name: {"per_row": ["SYNTHETIC_UNTRAINED_HEAD"] * len(sample_ids)}
            for name in ("expert5", "expert6")
        },
        "synthetic_provenance": True,
        "core_assembly_contract": "full8_v1",
    }, cache_identity


def test_full8_feature_store_has_new_schema_and_test_rows_are_label_free(tmp_path):
    source_hash = "2" * 64
    train_ids = ["train:a.wav", "train:b.wav"]
    test_ids = ["test:c.wav"]
    cache_identity = None
    for split, sample_ids in (("train", train_ids), ("test", test_ids)):
        assembly, cache_identity = _synthetic_full8_assembly(split, sample_ids, source_hash)
        provenance = {name: {"checkpoint_sha256": f"{i + 1:064x}"}
                      for i, name in enumerate(FULL7_EXPERT_ORDER)}
        provenance["expert8"] = cache_identity["checkpoint_provenance"]
        writer = MoeFeatureStoreWriter(
            tmp_path / "full8", split=split, expected_sample_ids=sample_ids,
            source_manifest_sha256=source_hash, checkpoint_provenance=provenance,
            preprocessing_identity={"full8": "fixture"}, profile="full_8",
            assembly_identity=assembly,
        )
        writer.write_shard(
            0, sample_ids=sample_ids, wave_file_names=[sample_id.split(":", 1)[1] for sample_id in sample_ids],
            scores=np.zeros((len(sample_ids), 8), dtype=np.float32),
            features=np.zeros((len(sample_ids), 33032), dtype=np.float32),
            labels_raw=np.full(len(sample_ids), 5.0, dtype=np.float32) if split == "train" else None,
        )
        writer.finalize()
        manifest = __import__("data.moe_feature_store", fromlist=["validate_feature_store"]).validate_feature_store(
            tmp_path / "full8", split=split, require_labels=split == "train"
        )
        assert manifest["schema_version"] == FULL8_FEATURE_STORE_SCHEMA_VERSION
        assert manifest["expert_profile"] == "full_8"
        assert manifest["feature_width"] == 33032 and manifest["score_width"] == 8
        assert manifest["labels_present"] is (split == "train")
    test_ds = MoeFeatureStoreDataset(tmp_path / "full8", split="test")
    assert "label_raw" not in test_ds[0]


def test_e8_cache_manifest_binds_exact_order_and_checkpoint_identity():
    from scripts.extract_expert8_features import make_expert8_cache_manifest, validate_expert8_cache_manifest
    from data.moe_expert_cache import build_expert_cache_identity
    from model.expert_adapters.expert8_m2d_clap_runtime import EXPECTED_BASE_M2D_SHA256

    identity = build_expert_cache_identity(
        expert_name="expert8",
        checkpoint_provenance={"base_checkpoint": {"sha256": EXPECTED_BASE_M2D_SHA256},
                               "finetuned_checkpoint": {"sha256": "a" * 64}},
        implementation_provenance={"adapter": "b" * 64},
        preprocessing_provenance={"sample_rate_hz": 16000}, input_contract={"samples": 160000},
        inference_policy={"device": "cpu"}, score_contract={"score": "normalized"},
        relevant_dependencies={"torch": "synthetic"},
    )
    records = [
        {"sample_id": f"validation:{name}", "wave_file_name": name,
         "raw_caption": f"caption {index}", "wav_sha256": f"{index + 1:064x}"}
        for index, name in enumerate(("a.wav", "b.wav"))
    ]
    manifest = make_expert8_cache_manifest(
        split="validation", records=records, cache_identity=identity, shards=[]
    )
    validate_expert8_cache_manifest(
        manifest, split="validation", expected_records=records, cache_identity=identity
    )
    changed = [records[1], records[0]]
    try:
        validate_expert8_cache_manifest(
            manifest, split="validation", expected_records=changed, cache_identity=identity
        )
    except ValueError as exc:
        assert "source" in str(exc).lower() or "order" in str(exc).lower()
    else:
        raise AssertionError("reordered E8 cache rows were accepted")


def test_full8_assembly_appends_aligned_e8_cache_without_changing_full7_columns(tmp_path):
    import hashlib
    import json

    from data.full8_assembler import build_full8_split
    from data.moe_expert_cache import build_sample_identity, save_expert_shard, sha256_file
    from data.moe_feature_store import _canonical_json
    from scripts.extract_expert8_features import make_expert8_cache_manifest

    sample_ids = ["train:a.wav", "train:b.wav"]
    filenames = ["a.wav", "b.wav"]
    captions = ["first caption", "second caption"]
    wav_hashes = ["a" * 64, "b" * 64]
    contracts = {}
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        contracts[name] = {
            "source": ("expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "m2d_lora")[index],
            "semantics": "synthetic_score_fixture", "target_scale": "normalized_minus1_to1",
            "output_scale": "normalized_minus1_to1", "conversion": "identity",
            "identity": f"{index + 1:064x}", "audit_status": "SYNTHETIC_VERIFIED",
        }
    source_hash = "2" * 64
    full7_provenance = {name: {"checkpoint_sha256": f"{i + 1:064x}"}
                        for i, name in enumerate(FULL7_EXPERT_ORDER)}
    full7_assembly = {
        "source_full9_store_identity": "3" * 64,
        "source_whisper_manifest_sha256": "4" * 64,
        "score_contracts": contracts,
        "shared_source_contract": {"synthetic": True},
        "ordered_source_fingerprints": [
            {"sample_id": sid, "wav_sha256": wav,
             "caption_sha256": hashlib.sha256(caption.encode()).hexdigest()}
            for sid, wav, caption in zip(sample_ids, wav_hashes, captions)
        ],
        "head_membership": {
            name: {"per_row": ["SYNTHETIC_UNTRAINED_HEAD"] * 2}
            for name in ("expert5", "expert6")
        },
        "synthetic_provenance": True,
    }
    full7_root = tmp_path / "full7"
    full7_writer = MoeFeatureStoreWriter(
        full7_root, split="train", expected_sample_ids=sample_ids,
        source_manifest_sha256=source_hash, checkpoint_provenance=full7_provenance,
        preprocessing_identity={"fixture": "full7"}, profile="full_7",
        assembly_identity=full7_assembly,
    )
    scores7 = np.tile(np.arange(7, dtype=np.float32), (2, 1))
    features7 = np.vstack((np.ones(29959, np.float32), np.full(29959, 2, np.float32)))
    full7_writer.write_shard(
        0, sample_ids=sample_ids, wave_file_names=filenames,
        scores=scores7, features=features7, labels_raw=np.array([4, 8], np.float32),
    )
    full7_writer.finalize()

    from model.expert_adapters.expert8_m2d_clap_runtime import EXPECTED_BASE_M2D_SHA256
    lora_sha = "5" * 64
    e8_identity = build_expert_cache_identity(
        expert_name="expert8",
        checkpoint_provenance={
            "base_checkpoint": {"sha256": EXPECTED_BASE_M2D_SHA256},
            "finetuned_checkpoint": {"sha256": lora_sha},
        },
        implementation_provenance={"adapter": "6" * 64},
        preprocessing_provenance={"sample_rate_hz": 16000},
        input_contract={"samples": 160000}, inference_policy={"frozen": True},
        score_contract={"semantics": "coral"}, relevant_dependencies={"torch": "fixture"},
    )
    e8_records = [
        {"sample_id": sid, "wave_file_name": name, "raw_caption": caption, "wav_sha256": wav}
        for sid, name, caption, wav in zip(sample_ids, filenames, captions, wav_hashes)
    ]
    identity_rows = [{**row} for row in e8_records]
    preprocessing = {"expert_cache_id": e8_identity["cache_id"],
                     "input_contract": e8_identity["input_contract"],
                     "preprocessing_provenance": e8_identity["preprocessing_provenance"]}
    shard_identity = build_sample_identity("train", identity_rows, preprocessing, 3073)
    cache_dir = tmp_path / "e8cache" / "expert8" / e8_identity["cache_id"] / "train"
    shard_path = cache_dir / "shards" / f"shard_{shard_identity['sample_identity_sha256']}.npz"
    save_expert_shard(
        shard_path, expert_identity=e8_identity, sample_identity=shard_identity,
        scores=np.array([[-0.5], [0.25]], np.float32),
        features=np.vstack((np.full(3073, 3, np.float32), np.full(3073, 4, np.float32))),
        inference_metadata={"factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY", "frozen_eval": True},
    )
    shard_entry = {
        "index": 0, "start": 0, "count": 2,
        "sample_identity_sha256": shard_identity["sample_identity_sha256"],
        "relative_path": str(shard_path.relative_to(cache_dir)),
        "file_sha256": sha256_file(shard_path),
    }
    cache_manifest = make_expert8_cache_manifest(
        split="train", records=e8_records, cache_identity=e8_identity, shards=[shard_entry]
    )
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest_path.write_text(json.dumps(cache_manifest))

    report = build_full8_split(full7_root, tmp_path / "e8cache", tmp_path / "full8", split="train")
    assert report["feature_width"] == 33032 and report["score_width"] == 8
    full8 = MoeFeatureStoreDataset(tmp_path / "full8", split="train", require_labels=True)
    torch.testing.assert_close(full8[0]["scores"][:7], torch.from_numpy(scores7[0]))
    torch.testing.assert_close(full8[0]["scores"][7], torch.tensor(-0.5))
    torch.testing.assert_close(full8[1]["features"][:29959], torch.from_numpy(features7[1]))
    torch.testing.assert_close(full8[1]["features"][29959:], torch.full((3073,), 4.0))


def test_full8_train_score_audit_reports_all_eight_experts():
    from data.full8_contract import audit_train_scores_full8

    scores = np.linspace(-0.8, 0.8, 24, dtype=np.float32).reshape(3, 8)
    labels = np.array([2.0, 5.0, 8.0], dtype=np.float32)
    report = audit_train_scores_full8(scores, labels)
    assert report["sample_count"] == 3
    assert list(report["experts"]) == list(FULL8_EXPERT_ORDER)
    assert report["experts"]["expert8"]["non_finite_count"] == 0


def _full8_checkpoint_fixture():
    from data.audio_overlap import build_train_validation_audio_overlap_diagnostic

    train_records = [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64}]
    validation_records = [{"sample_id": "validation:b.wav", "wav_sha256": "b" * 64}]
    diagnostic = build_train_validation_audio_overlap_diagnostic(
        train_records, validation_records,
        allow_train_validation_audio_overlap=False,
    )
    identity = {
        "train": "c" * 64, "validation": "d" * 64,
        "preprocessing_contract": {"full8": "synthetic"},
        "source_wav_sha256_by_split": {
            "train": ["a" * 64], "validation": ["b" * 64],
        },
        "source_sample_ids_by_split": {
            "train": ["train:a.wav"], "validation": ["validation:b.wav"],
        },
        "audio_overlap_diagnostic": diagnostic,
        "score_review_policy": "EXPERIMENTAL_SKIPPED",
    }
    config = {"runtime": {"seed": 2027}, "data": {
        "profile": "full_8", "allow_train_validation_audio_overlap": False,
        "score_review_policy": "EXPERIMENTAL_SKIPPED",
    }}
    provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"}
                  for index, name in enumerate(FULL8_EXPERT_ORDER)}
    from model.expert_adapters.expert8_m2d_clap_runtime import EXPECTED_BASE_M2D_SHA256
    provenance["expert8"] = {
        "base_checkpoint": {"sha256": EXPECTED_BASE_M2D_SHA256},
        "finetuned_checkpoint": {"sha256": "e" * 64},
    }
    return config, identity, provenance


def test_full8_checkpoint_format5_provenance_and_resume_policy(tmp_path):
    from training.checkpoint_manager import (
        FULL8_CHECKPOINT_FORMAT_VERSION, MoeCheckpointManager, _validate_gate_payload,
    )

    config, identity, provenance = _full8_checkpoint_fixture()
    gate = GatingNetwork(33032, 8, profile="full_8")
    optimizer = torch.optim.AdamW(gate.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    manager = MoeCheckpointManager()
    checkpoint = tmp_path / "full8_latest.pt"
    manager.save(
        checkpoint, gating_network=gate, optimizer=optimizer, scheduler=scheduler,
        epoch=1, global_step=9, config=config,
        feature_store_identity=identity, expert_checkpoint_provenance=provenance,
        best_srcc=0.42,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format_version"] == FULL8_CHECKPOINT_FORMAT_VERSION == 5
    assert payload["expert_profile"] == "full_8"
    assert payload["expert_order"][-1] == "expert8"
    assert payload["gate_architecture"]["input_width"] == 33032
    assert payload["feature_store_identity"]["audio_overlap_diagnostic"] == identity["audio_overlap_diagnostic"]
    _validate_gate_payload(payload, gate)
    corrupt_source = dict(payload)
    corrupt_source["source_code_sha256"] = {**payload["source_code_sha256"], "data/full8_assembler.py": "0" * 64}
    with pytest.raises(ValueError, match="source code"):
        _validate_gate_payload(corrupt_source, gate)
    restored_optimizer = torch.optim.AdamW(gate.parameters())
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
    loaded = manager.load(
        checkpoint, gating_network=gate, optimizer=restored_optimizer,
        scheduler=restored_scheduler, expected_config=config,
        expected_feature_store_identity=identity,
        expected_expert_checkpoint_provenance=provenance,
    )
    assert loaded["epoch"] == 1 and loaded["best_srcc"] == 0.42

    changed_config = {**config, "data": {**config["data"], "score_review_policy": "HUMAN_REVIEWED"}}
    changed_identity = {**identity, "score_review_policy": "HUMAN_REVIEWED",
                       "score_review_sha256": "f" * 64}
    with pytest.raises(ValueError, match="score review policy mismatch|feature store identity mismatch"):
        manager.load(
            checkpoint, gating_network=gate, optimizer=restored_optimizer,
            scheduler=restored_scheduler, expected_config=changed_config,
            expected_feature_store_identity=changed_identity,
            expected_expert_checkpoint_provenance=provenance,
        )
    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        manager.load_for_inference(
            checkpoint, gating_network=gate,
            expected_expert_checkpoint_provenance={**provenance, "expert8": {"finetuned_checkpoint": {"sha256": "0" * 64}}},
        )
    from model.moe_gating_core import GatingNetwork as Gate
    full7 = Gate(29959, 7, profile="full_7")
    with pytest.raises(ValueError, match="expert profile|full_8 architecture"):
        _validate_gate_payload(payload, full7)


def test_full8_config_is_independent_and_profile_checked(tmp_path):
    import yaml
    from training.config import load_moe_training_config

    raw = yaml.safe_load(Path("configs/moe_full8.example.yaml").read_text())
    raw["data"]["feature_store_root"] = str(tmp_path / "store")
    raw["output_dir"] = str(tmp_path / "gate")
    config_path = tmp_path / "full8.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    loaded = load_moe_training_config(config_path)
    assert loaded["data"]["profile"] == "full_8"
    assert loaded["data"]["allow_train_validation_audio_overlap"] is False


def test_full8_experimental_review_bypass_runs_train_only_audit_without_review_file(monkeypatch, tmp_path):
    import scripts.train_moe_gating as training

    calls = []
    audit = {"split": "train", "diagnostics": {"sample_count": 7}}
    monkeypatch.setattr(training, "audit_store", lambda root: calls.append(root) or audit)
    manifest = {
        "expert_order": list(FULL8_EXPERT_ORDER),
        "score_contracts": {name: {} for name in FULL8_EXPERT_ORDER},
        "assembly_identity": {"synthetic_provenance": False},
    }
    identity = {}
    returned = training._resolve_full7_score_review(
        tmp_path, manifest, identity, tmp_path / "does_not_exist.json", skip_review=True,
    )
    assert calls == [tmp_path]
    assert returned[0] is audit
    assert returned[1:] == (None, "EXPERIMENTAL_SKIPPED", False)
    assert identity["score_review_policy"] == "EXPERIMENTAL_SKIPPED"
    assert "score_review_sha256" not in identity
    with pytest.raises(ValueError, match="requires a score review file"):
        training._resolve_full7_score_review(
            tmp_path, manifest, {}, None, skip_review=False,
        )


def test_full8_builder_allows_only_configured_train_validation_overlap(monkeypatch, tmp_path):
    import data.full8_assembler as assembler

    source_rows = {
        "train": [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64}],
        "validation": [{"sample_id": "validation:b.wav", "wav_sha256": "a" * 64}],
        "test": [{"sample_id": "test:c.wav", "wav_sha256": "c" * 64}],
    }
    monkeypatch.setattr(
        assembler, "validate_feature_store",
        lambda root, split, require_labels: {
            "assembly_identity": {"ordered_source_fingerprints": source_rows[split]}
        },
    )
    calls = []
    monkeypatch.setattr(
        assembler, "build_full8_split",
        lambda *args, split, **kwargs: calls.append(split) or {"split": split},
    )
    with pytest.raises(ValueError, match="duplicate WAV|overlap"):
        assembler.build_full8_stores("base", "cache", tmp_path / "strict")
    accepted = assembler.build_full8_stores(
        "base", "cache", tmp_path / "experimental",
        allow_train_validation_audio_overlap=True,
    )
    assert calls == ["train", "validation", "test"]
    assert accepted["audio_overlap_diagnostic"]["overlapping_sha256_count"] == 1

    source_rows["test"] = [{"sample_id": "test:c.wav", "wav_sha256": "a" * 64}]
    with pytest.raises(ValueError, match="involving official test"):
        assembler.build_full8_stores(
            "base", "cache", tmp_path / "bad_test",
            allow_train_validation_audio_overlap=True,
        )


def test_full8_global_gate_trains_on_full_train_store_and_saves_loadable_checkpoint(tmp_path):
    import hashlib
    import json

    from data.moe_feature_store import MoeFeatureStoreWriter
    from scripts.train_moe_gating import run_training
    from training.checkpoint_manager import MoeCheckpointManager

    store_root = tmp_path / "full8_store"
    provenance = None
    rng = np.random.default_rng(2027)
    for split, labels in (("train", [1.0, 4.0, 7.0, 9.0]),
                          ("validation", [2.0, 3.5, 6.5, 8.5]),
                          ("test", [None, None])):
        sample_ids = [f"{split}:{index}.wav" for index in range(len(labels))]
        assembly, cache_identity = _synthetic_full8_assembly(
            split, sample_ids, "2" * 64,
        )
        for index, row in enumerate(assembly["ordered_source_fingerprints"]):
            row["wav_sha256"] = hashlib.sha256(f"{split}:{index}".encode()).hexdigest()
            row["caption_sha256"] = hashlib.sha256(f"caption:{split}:{index}".encode()).hexdigest()
        provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"}
                      for index, name in enumerate(FULL7_EXPERT_ORDER)}
        provenance["expert8"] = cache_identity["checkpoint_provenance"]
        writer = MoeFeatureStoreWriter(
            store_root, split=split, expected_sample_ids=sample_ids,
            source_manifest_sha256="2" * 64, checkpoint_provenance=provenance,
            preprocessing_identity={"full8": "synthetic-gate-test"},
            profile="full_8", assembly_identity=assembly,
        )
        scores = rng.uniform(-0.9, 0.9, size=(len(labels), 8)).astype(np.float32)
        features = rng.normal(size=(len(labels), 33032)).astype(np.float32)
        writer.write_shard(
            0, sample_ids=sample_ids,
            wave_file_names=[sample_id.split(":", 1)[1] for sample_id in sample_ids],
            scores=scores, features=features,
            labels_raw=None if split == "test" else np.asarray(labels, dtype=np.float32),
        )
        writer.finalize()

    config = {
        "runtime": {"device": "cpu", "seed": 2027},
        "data": {"feature_store_root": str(store_root), "profile": "full_8",
                 "allow_train_validation_audio_overlap": False},
        "output_dir": str(tmp_path / "gate"),
        "training": {
            "batch_size": 4, "max_epochs": 1, "learning_rate": 1e-5,
            "weight_decay": 0.0001, "diversity_weight": 0.2,
            "early_stop_patience": 0, "checkpoint_interval_steps": 1000,
            "loss": {"loss_weights": [1.0, 0.5],
                     "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
                     "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False}},
        },
        "validation": {"batch_size": 4}, "inference": {"batch_size": 4},
    }
    report = run_training(config, device_override="cpu")
    assert report["status"] == "PASS"
    assert report["history"][0]["train_rows_used"] == 4
    assert report["history"][0]["train_rows_available"] == 4
    assert list(report["score_audit"]["diagnostics"]["experts"]) == list(FULL8_EXPERT_ORDER)
    assert report["score_review_policy"] == "SYNTHETIC_NOT_APPLICABLE"
    checkpoint = torch.load(report["best_checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["expert_profile"] == "full_8"
    assert checkpoint["expert_checkpoint_provenance"] == provenance
    assert checkpoint["feature_store_identity"]["train"]
    assert checkpoint["feature_store_identity"]["validation"]
    assert json.loads(Path(report["run_manifest"]).read_text())["expert_profile"] == "full_8"
    gate = GatingNetwork(33032, 8, profile="full_8")
    loaded = MoeCheckpointManager().load_for_inference(
        report["best_checkpoint"], gating_network=gate,
        expected_expert_checkpoint_provenance=provenance,
    )
    assert loaded["format_version"] == 5
    from scripts.infer_moe_gating import infer_checkpoint
    inference = infer_checkpoint(
        config, report["best_checkpoint"], str(tmp_path / "full8_submission.csv"),
        device_override="cpu",
    )
    assert inference["status"] == "PASS" and inference["labels_required"] is False
    import csv
    with Path(inference["output_csv"]).open(newline="", encoding="utf-8") as stream:
        assert next(csv.reader(stream)) == ["wave_file_name", "pred_score"]
        assert sum(1 for _ in csv.reader(stream)) == 2
