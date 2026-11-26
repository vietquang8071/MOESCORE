"""Small CPU-only Full8 cache fixture; never used for real XACLE results."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from data.moe_feature_store import MoeFeatureStoreWriter
from model.expert_adapters.registry import FULL7_EXPERT_ORDER
from test_full8_pipeline import _synthetic_full8_assembly
from training.logo_runner import SOURCE_PATHS, digest, run_experiment


def test_cpu_fold_outputs_resume_and_three_heldout_splits(tmp_path):
    data = tmp_path / "data" / "meta_data"
    data.mkdir(parents=True)
    root = tmp_path / "artifacts"
    store = root / "full8_store"
    identities = {}
    layouts = {"train": [("a", 1), ("a", 2), ("b", 3), ("b", 4)],
               "validation": [("a", 1), ("a", 2)], "test": [("a", 1), ("a", 2)]}
    for split, layout in layouts.items():
        names = [f"{split}_{i}.wav" for i in range(len(layout))]
        ids = [f"{split}:{name}" for name in names]
        source_hash = hashlib.sha256(split.encode()).hexdigest()
        assembly, cache = _synthetic_full8_assembly(split, ids, source_hash)
        assembly["ordered_source_fingerprints"] = [
            {"sample_id": sample_id, "wav_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
             "caption_sha256": hashlib.sha256(b"caption").hexdigest()} for sample_id in ids]
        provenance = {name: {"checkpoint_sha256": f"{index+1:064x}"}
                      for index, name in enumerate(FULL7_EXPERT_ORDER)}
        provenance["expert8"] = cache["checkpoint_provenance"]
        writer = MoeFeatureStoreWriter(store, split=split, expected_sample_ids=ids,
            source_manifest_sha256=source_hash, checkpoint_provenance=provenance,
            preprocessing_identity={"source": "fixture"}, profile="full_8", assembly_identity=assembly)
        features = np.zeros((len(ids), 33032), dtype=np.float32)
        features[:, 0] = np.arange(len(ids), dtype=np.float32)
        scores = np.tile(np.linspace(-.8, .8, 8, dtype=np.float32), (len(ids), 1))
        scores[:, 0] += np.arange(len(ids), dtype=np.float32) * .1
        writer.write_shard(0, sample_ids=ids, wave_file_names=names, scores=scores, features=features,
            labels_raw=np.asarray([score for _, score in layout], dtype=np.float32) if split != "test" else None)
        writer.finalize()
        identities[split] = json.loads((store / split / "manifest.json").read_text())["store_identity_sha256"]
        with (data / f"{split}_average.csv").open("w", newline="") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=["wave_file_name", "text", "average_score", "model_ID", "model_name"])
            writer_csv.writeheader()
            writer_csv.writerows([{"wave_file_name": name, "text": "caption", "average_score": score,
                                  "model_ID": model_id, "model_name": model_id}
                                 for name, (model_id, score) in zip(names, layout)])
    gate_dir = root / "moe8_gate"
    gate_dir.mkdir()
    baseline = {"runtime": {"seed": 2027}, "data": {"profile": "full_8"},
                "training": {"batch_size": 2, "max_epochs": 1, "learning_rate": 1e-5,
                             "weight_decay": 1e-4, "diversity_weight": .2, "early_stop_patience": 50,
                             "loss": {"loss_weights": [1.0, .5],
                                      "clipped_mse": {"tau": .25, "mode": "utt", "beta": .99, "cbl": False},
                                      "contrastive": {"margin": .1, "beta": .99, "cbl": False}}},
                "validation": {"batch_size": 2}, "inference": {"batch_size": 2}}
    (gate_dir / "moe_full8.yaml").write_text(yaml.safe_dump(baseline))
    project = Path(__file__).resolve().parents[1]
    effective = json.loads(json.dumps(baseline))
    effective["data"]["score_review_policy"] = "SYNTHETIC_NOT_APPLICABLE"
    (gate_dir / "best_gate.pt").write_bytes(b"synthetic baseline gate; never used for initialization")
    (gate_dir / "run_manifest.json").write_text(json.dumps({
        "expert_profile": "full_8", "git_commit": "synthetic", "seed": 2027,
        "config_sha256": digest(effective),
        "gate_checkpoint_sha256": hashlib.sha256((gate_dir / "best_gate.pt").read_bytes()).hexdigest(),
        "score_review_policy": "SYNTHETIC_NOT_APPLICABLE",
        "source_code_sha256": {name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
        "train_store_identity_sha256": identities["train"],
        "validation_store_identity_sha256": identities["validation"]}))
    config = {"output_dir": str(tmp_path / "out"), "device": "cpu", "control_model_names": [],
              "allow_train_validation_audio_overlap": False, "allow_public_split_audio_overlap": False}
    result = run_experiment(config, dataset_root=str(data), artifacts_root=str(root),
                            generator_ids=["a"], device="cpu")
    assert result["status"] == "PARTIAL"
    fold = tmp_path / "out" / "fold_a"
    assert (fold / "checkpoints" / "best.pt").is_file()
    for split in ("train", "validation", "public_test"):
        with (fold / "predictions" / f"{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        assert {row["model_id"] for row in rows} == {"a"}
        assert {row["source_split"] for row in rows} == {"test" if split == "public_test" else split}
    resumed = run_experiment(config, dataset_root=str(data), artifacts_root=str(root),
                             generator_ids=["a"], device="cpu", resume=True)
    assert resumed["folds"][0]["status"] == "SKIPPED_VALIDATED"
    with (fold / "predictions" / "train.csv").open("a") as stream:
        stream.write("\n")
    import pytest
    with pytest.raises(ValueError, match="incompatible or incomplete"):
        run_experiment(config, dataset_root=str(data), artifacts_root=str(root),
                       generator_ids=["a"], device="cpu", resume=True)
