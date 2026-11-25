import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model.expert_adapters.registry import profile_contract
from research.xacle27.analysis.d02_gate_diagnostics import (
    resolve_precomputed_full8_artifacts,
    validate_checkpoint_store_compatibility,
    validate_full8_manifest_contracts,
)


SPLITS = {"train": 7500, "validation": 3000, "test": 3000}


def _manifest(split, count, *, provenance=None, identity=None):
    order, dimensions, width = profile_contract("full_8")
    sample_ids = [f"{split}:{index:05d}.wav" for index in range(count)]
    provenance = provenance or {name: {"sha256": name} for name in order}
    fingerprints = [{"sample_id": sample_id, "wav_sha256": "a" * 64,
                     "caption_sha256": "b" * 64} for sample_id in sample_ids]
    return {
        "schema_version": 4,
        "split": split,
        "expert_profile": "full_8",
        "expert_order": list(order),
        "feature_dimensions": dict(dimensions),
        "feature_width": width,
        "score_width": 8,
        "labels_present": split != "test",
        "labels_scale": "raw_0_to_10" if split != "test" else None,
        "store_identity_sha256": identity or (split[0] * 64),
        "checkpoint_provenance": provenance,
        "assembly_identity": {
            "shared_source_contract": {"fixture": "same"},
            "score_contracts": {name: {"fixture": name} for name in order},
            "expert8_cache_identity": {"cache_id": "fixture"},
            "ordered_source_fingerprints": fingerprints,
        },
        "shards": [{"index": 0, "sample_ids": sample_ids,
                    "wave_file_names": [Path(value).name for value in sample_ids],
                    "count": count}],
    }


class D02PrecomputedArtifactTests(unittest.TestCase):
    def _write_artifact_root(self, root):
        (root / "moe8_gate/moe8_gate").mkdir(parents=True)
        for name in ("best_gate.pt", "moe_full8.yaml", "run_manifest.json"):
            (root / "moe8_gate/moe8_gate" / name).write_text("artifact", encoding="utf-8")
        store = root / "moescore_full8_store/full8_store"
        for split, count in SPLITS.items():
            directory = store / split
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(
                json.dumps(_manifest(split, count)), encoding="utf-8"
            )
        return store

    def test_resolver_finds_unique_gate_config_run_manifest_and_full8_store(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_store = self._write_artifact_root(root)
            found = resolve_precomputed_full8_artifacts(root)

        self.assertEqual(found["gate_checkpoint"].name, "best_gate.pt")
        self.assertEqual(found["gate_config"].name, "moe_full8.yaml")
        self.assertEqual(found["run_manifest"].name, "run_manifest.json")
        self.assertEqual(found["feature_store_root"], expected_store.resolve())

    def test_resolver_rejects_ambiguous_checkpoints_and_feature_stores(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_artifact_root(root)
            duplicate = root / "backup/best_gate.pt"
            duplicate.parent.mkdir()
            duplicate.write_text("ambiguous", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous best_gate.pt"):
                resolve_precomputed_full8_artifacts(root)

    def test_resolver_rejects_multiple_complete_full8_stores(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_artifact_root(root)
            duplicate_store = root / "backup/full8_store"
            for split, count in SPLITS.items():
                directory = duplicate_store / split
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(
                    json.dumps(_manifest(split, count)), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "ambiguous Full8 feature stores"):
                resolve_precomputed_full8_artifacts(root)

    def test_resolver_reports_missing_public_test_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_artifact_root(root)
            (store / "test/manifest.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "test"):
                resolve_precomputed_full8_artifacts(root)

    def test_all_full8_manifests_match_split_counts_and_registry_contract(self):
        manifests = {split: _manifest(split, count) for split, count in SPLITS.items()}
        validated = validate_full8_manifest_contracts(manifests, expected_counts=SPLITS)
        self.assertEqual(set(validated), set(SPLITS))

        invalid = dict(manifests)
        invalid["test"] = {**manifests["test"], "feature_width": 29959}
        with self.assertRaisesRegex(ValueError, "feature width"):
            validate_full8_manifest_contracts(invalid, expected_counts=SPLITS)

    def test_checkpoint_and_store_provenance_must_match(self):
        order, dimensions, width = profile_contract("full_8")
        provenance = {name: {"sha256": name} for name in order}
        manifests = {split: _manifest(split, count, provenance=provenance)
                     for split, count in SPLITS.items()}
        identity = {
            "train": manifests["train"]["store_identity_sha256"],
            "validation": manifests["validation"]["store_identity_sha256"],
            "preprocessing_contract": {
                "shared_source_contract": {"fixture": "same"},
                "score_contracts": {name: {"fixture": name} for name in order},
                "checkpoint_provenance": provenance,
                "expert8_cache_identity": {"cache_id": "fixture"},
                "core_assembly_contract": "full8_v1",
            },
            "source_sample_ids_by_split": {
                split: [sample_id for shard in manifests[split]["shards"]
                        for sample_id in shard["sample_ids"]]
                for split in ("train", "validation")
            },
            "source_wav_sha256_by_split": {
                split: [row["wav_sha256"] for row in manifests[split]["assembly_identity"]["ordered_source_fingerprints"]]
                for split in ("train", "validation")
            },
        }
        payload = {
            "expert_profile": "full_8",
            "expert_order": list(order),
            "expert_feature_dimensions": dict(dimensions),
            "gate_input_dim": width,
            "gate_expert_count": 8,
            "expert_checkpoint_provenance": provenance,
            "feature_store_identity": identity,
        }
        validate_checkpoint_store_compatibility(payload, manifests)

        mismatched = {**payload, "expert_checkpoint_provenance": {**provenance, "expert8": {"sha256": "wrong"}}}
        with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
            validate_checkpoint_store_compatibility(mismatched, manifests)


if __name__ == "__main__":
    unittest.main()
