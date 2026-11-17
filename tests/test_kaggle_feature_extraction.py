import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch

from expert4_finetune.source_compatibility import (
    ORIGINAL_COMMIT, ORIGINAL_SHA256, verify_bundled_import_equivalence,
)
from model.expert_adapters.registry import EXPERT_ORDER, EXPERT_FEATURE_DIMS
from data.moe_feature_store import validate_feature_store
from data.moe_expert_cache import build_sample_identity
from scripts.extract_moe_feature_store import (
    prepare_job, extract_worker, assemble_stores, load_chunk, chunk_path,
    job_identity, assembly_store_root, completed_stores_match, projected_dataset, validate_manifests,
)
from scripts.kaggle_moe_pipeline import generate_configs, discover_dataset

ROOT = Path(__file__).resolve().parents[1]


def audio_fixture(root, count=2):
    (root / "meta_data").mkdir(parents=True)
    for split in ("train", "validation", "test"):
        directory = root / "wav" / split
        directory.mkdir(parents=True)
        with (root / "meta_data" / f"{split}_average.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["wav_file_name", "text", "average_score"])
            for index in range(count):
                name = f"{index}.wav"
                writer.writerow([name, f" Caption {index} ", "not-read-during-extraction" if split == "test" else 4 + index])
                with wave.open(str(directory / name), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(np.full(1600, 100 + index, dtype=np.int16).tobytes())


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_discovery_requires_one_explicitly_identifiable_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_fixture(root / "slug" / "XACLE_dataset")
            expected = root / "slug" / "XACLE_dataset"
            self.assertEqual(discover_dataset(root), expected)
            audio_fixture(root / "other")
            with self.assertRaisesRegex(ValueError, "explicitly"):
                discover_dataset(root)
            self.assertEqual(discover_dataset(root, expected), expected)

    def test_bundled_source_only_allows_the_reviewed_import_differences(self):
        expected = {"git_commit": ORIGINAL_COMMIT, "sha256": ORIGINAL_SHA256}
        result = verify_bundled_import_equivalence(ROOT / "expert4_finetune/unilm", expected)
        self.assertEqual(result["normalized_sha256"], ORIGINAL_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("BEATs.py", "backbone.py", "modules.py"):
                shutil.copy2(ROOT / "expert4_finetune/unilm/beats" / name, root / name)
            with (root / "BEATs.py").open("a") as stream:
                stream.write("\n# unreviewed source edit\n")
            with self.assertRaisesRegex(ValueError, "differs beyond"):
                verify_bundled_import_equivalence(root, expected)
        with self.assertRaisesRegex(ValueError, "reviewed original"):
            verify_bundled_import_equivalence(ROOT / "expert4_finetune/unilm", {**expected, "sha256": "0" * 64})

    def test_user_checkpoint_mapping_is_used_in_both_yaml_and_resolved_config(self):
        from scripts.extract_moe_feature_store import CHECKPOINT_FIELDS
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {expert: {field: root / "custom" / expert / f"{field}.pt" for field in fields}
                       for expert, fields in CHECKPOINT_FIELDS.items()}
            config_path, _, resolved = generate_configs(ROOT, root / "data", root / "weights", root / "cache",
                root / "stores", root / "output", root, expert_checkpoints=mapping)
            saved = yaml.safe_load(config_path.read_text())
            for expert, fields in mapping.items():
                for field, path in fields.items():
                    self.assertEqual(saved["experts"][expert][field], str(path))
                    self.assertEqual(resolved["experts"][expert][field], path)
            with self.assertRaisesRegex(ValueError, "exactly the four"):
                generate_configs(ROOT, root / "data", root / "weights", root / "cache", root / "stores",
                                 root / "output", root, expert_checkpoints={"expert1": mapping["expert1"]})

    def test_label_free_test_projection_and_sequential_shard_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root, weights, output = root / "data", root / "weights", root / "store"
            audio_fixture(dataset_root)
            legacy_shard = output / "_extraction" / "expert1" / "train" / "shard_000000.npz"
            legacy_shard.parent.mkdir(parents=True)
            legacy_shard.write_bytes(b"legacy global-identity shard")
            legacy_store_marker = output / "train" / "legacy_store.marker"
            legacy_store_marker.parent.mkdir(parents=True)
            legacy_store_marker.write_bytes(b"legacy four-expert store")
            for name in ("laion-clap.pt", "mga-clap.pt", "checkpoint-30.pth", "BEATs_iter3_plus_AS2M.pt"):
                path = weights / "pretrained" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())  # Test serialization, never deserialize these fixtures.
            for expert in EXPERT_ORDER:
                path = weights / "finetuned_weights" / f"{expert}_best.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expert.encode())
            config_path, train_path, _ = generate_configs(ROOT, dataset_root, weights, root / "cache", output, root / "run", root)
            config, job, manifests = prepare_job(
                config_path, dataset_root, output, shard_size=1, limit=2, device="cpu"
            )
            training = __import__("yaml").safe_load(train_path.read_text())
            self.assertEqual(training["training"]["loss"]["loss_weights"], [1.0, 0.5])
            changed = json.loads(json.dumps(manifests))
            changed["train"]["records"][0]["label_raw"] = 9.0
            with self.assertRaisesRegex(ValueError, "manifest identity changed"):
                validate_manifests(job, changed)
            test_dataset, _ = projected_dataset(dataset_root, "test", root / "projection")
            self.assertTrue(all(row["label_value"] is None for row in test_dataset.rows))
            self.assertNotIn("average_score", test_dataset.headers)
            self.assertTrue(all("label_raw" not in row for row in manifests["test"]["records"]))

            class Stub(torch.nn.Module):
                def __init__(self, name):
                    super().__init__()
                    self.name = name
                    self.eval()

                def forward(self, batch):
                    count = len(batch["captions_raw"])
                    self.assert_no_labels(batch)
                    value = float(EXPERT_ORDER.index(self.name))
                    return torch.full((count, 1), value), torch.full((count, EXPERT_FEATURE_DIMS[self.name]), value)

                @staticmethod
                def assert_no_labels(batch):
                    if any(key in batch for key in ("labels", "labels_raw", "label", "model_id")):
                        raise AssertionError("labels/metadata entered expert forward")

            class InterruptedStub(Stub):
                calls = 0

                def forward(self, batch):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("simulated session interruption")
                    return super().forward(batch)

            def shard_identity(selected_expert, split, selected_records):
                source = job["expert_cache_identities"][selected_expert]
                return build_sample_identity(
                    split,
                    selected_records,
                    {
                        "expert_cache_id": source["cache_id"],
                        "input_contract": source["input_contract"],
                        "preprocessing_provenance": source["preprocessing_provenance"],
                    },
                    job["feature_dimensions"][selected_expert],
                )

            def path_for(selected_expert, split, index, start, selected_records):
                sample = shard_identity(selected_expert, split, selected_records)
                return chunk_path(
                    output, selected_expert, split, index,
                    cache_id=job["expert_cache_identities"][selected_expert]["cache_id"],
                    sample_identity_sha256=sample["sample_identity_sha256"],
                ), sample

            with patch("scripts.extract_moe_feature_store.create_expert_adapter", side_effect=[InterruptedStub("expert1"), *[Stub(e) for e in EXPERT_ORDER]]) as factory, \
                 patch("scripts.verify_real_expert_to_gate._validate_real_module_origins", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "session interruption"):
                    extract_worker(config, job, manifests, dataset_root, output, "expert1", "cpu")
                first_path, first_sample = path_for("expert1", "train", 0, 0, manifests["train"]["records"][:1])
                second_path, _ = path_for("expert1", "train", 1, 1, manifests["train"]["records"][1:2])
                self.assertTrue(first_path.exists())
                self.assertFalse(second_path.exists())
                for expert in EXPERT_ORDER:
                    extract_worker(config, job, manifests, dataset_root, output, expert, "cpu")
                self.assertEqual(factory.call_count, 5)
                extract_worker(config, job, manifests, dataset_root, output, "expert1", "cpu")
                self.assertEqual(factory.call_count, 5)  # Valid shards skip model construction.
            assemble_stores(job, manifests, output)
            self.assertTrue(completed_stores_match(job, manifests, output))
            self.assertEqual(legacy_shard.read_bytes(), b"legacy global-identity shard")
            self.assertEqual(legacy_store_marker.read_bytes(), b"legacy four-expert store")
            final_store = assembly_store_root(output, job)
            self.assertFalse(validate_feature_store(final_store, split="test")["labels_present"])
            self.assertTrue(validate_feature_store(final_store, split="train")["labels_present"])
            _, resized_job, _ = prepare_job(
                config_path, dataset_root, output, shard_size=1, limit=2,
                batch_sizes=dict.fromkeys(EXPERT_ORDER, 2), device="cpu",
            )
            self.assertEqual(job_identity(job), job_identity(resized_job))
            path, sample = path_for("expert1", "train", 0, 0, manifests["train"]["records"][:1])
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash|corrupt"):
                load_chunk(path, expert="expert1", records=manifests["train"]["records"][:1],
                           identity=job["expert_cache_identities"]["expert1"], sample_identity=sample)
            (weights / "pretrained/laion-clap.pt").write_bytes(b"different checkpoint")
            _, changed_job, _ = prepare_job(
                config_path, dataset_root, output, shard_size=1, limit=2, device="cpu"
            )
            self.assertNotEqual(
                job["expert_cache_identities"]["expert1"]["cache_id"],
                changed_job["expert_cache_identities"]["expert1"]["cache_id"],
            )
            self.assertEqual(
                job["expert_cache_identities"]["expert2"]["cache_id"],
                changed_job["expert_cache_identities"]["expert2"]["cache_id"],
            )


if __name__ == "__main__":
    unittest.main()
