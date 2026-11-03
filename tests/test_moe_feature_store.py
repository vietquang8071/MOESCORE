import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.moe_feature_store import MoeFeatureStoreDataset, MoeFeatureStoreWriter
from data.moe_training_dataset import collate_moe_features
from model.expert_adapters.registry import EXPERT_ORDER
from scripts.build_moe_feature_store import _extract_raw_labels


def provenance():
    return {name: {"checkpoint_sha256": str(index + 1) * 64} for index, name in enumerate(EXPERT_ORDER)}


class MoeFeatureStoreTests(unittest.TestCase):
    def test_source_labels_must_be_complete_per_split(self):
        present, labels = _extract_raw_labels(
            [{"sample_id": "validation:a", "label_raw_0_to_10": 2.5},
             {"sample_id": "validation:b", "label_raw_0_to_10": 7.5}],
            "validation",
        )
        self.assertTrue(present)
        self.assertEqual(labels, [2.5, 7.5])
        present, labels = _extract_raw_labels(
            [{"sample_id": "test:a"}, {"sample_id": "test:b"}], "test"
        )
        self.assertFalse(present)
        self.assertEqual(labels, [None, None])
        with self.assertRaisesRegex(ValueError, "only some"):
            _extract_raw_labels(
                [{"sample_id": "validation:a", "label_raw": 5.0}, {"sample_id": "validation:b"}],
                "validation",
            )
        with self.assertRaisesRegex(ValueError, "training.*missing"):
            _extract_raw_labels([{"sample_id": "train:a"}], "train")

    def write_store(self, root, *, split="train", ids=None, labels=True):
        ids = ids or [f"{split}:{i}.wav" for i in range(3)]
        writer = MoeFeatureStoreWriter(
            root,
            split=split,
            expected_sample_ids=ids,
            source_manifest_sha256="a" * 64,
            checkpoint_provenance=provenance(),
            preprocessing_identity="unified-dataset-local09.1-v1",
        )
        for shard_index, start in enumerate(range(0, len(ids), 2)):
            selected = ids[start : start + 2]
            scores = np.full((len(selected), 4), shard_index + 0.25, dtype=np.float32)
            features = np.full((len(selected), 7936), shard_index + 1.5, dtype=np.float32)
            kwargs = {}
            if labels:
                kwargs["labels_raw"] = np.arange(start, start + len(selected), dtype=np.float32) + 4.0
            writer.write_shard(
                shard_index,
                sample_ids=selected,
                wave_file_names=[sample_id.split(":", 1)[1] for sample_id in selected],
                scores=scores,
                features=features,
                **kwargs,
            )
        return writer.finalize()

    def test_sharded_store_preserves_values_identity_and_collate_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_store(temporary)
            dataset = MoeFeatureStoreDataset(temporary, split="train")
            self.assertEqual(len(dataset), 3)
            first, second = dataset[0], dataset[1]
            self.assertEqual(first["sample_id"], "train:0.wav")
            self.assertEqual(first["wave_file_name"], "0.wav")
            self.assertEqual(first["label_raw"].dtype, torch.float32)
            self.assertEqual(first["scores"].dtype, torch.float32)
            self.assertEqual(first["features"].shape, (7936,))
            batch = collate_moe_features([second, first])
            self.assertEqual(batch["sample_ids"], ["train:1.wav", "train:0.wav"])
            self.assertEqual(batch["scores"].shape, (2, 4))
            self.assertEqual(batch["features"].shape, (2, 7936))
            self.assertEqual(batch["labels_raw"].shape, (2,))
            self.assertEqual(batch["labels_scale"], "raw_0_to_10")
            self.assertEqual(batch["labels_raw"].tolist(), [5.0, 4.0])
            self.assertTrue(torch.equal(batch["features"][0], torch.full((7936,), 1.5)))

    def test_feature_collate_b1_b2_b4_preserves_values_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            ids = [f"train:{index}.wav" for index in range(4)]
            writer = MoeFeatureStoreWriter(
                temporary,
                split="train",
                expected_sample_ids=ids,
                source_manifest_sha256="e" * 64,
                checkpoint_provenance=provenance(),
                preprocessing_identity={"contract": "synthetic-test-v1"},
            )
            scores = np.arange(16, dtype=np.float32).reshape(4, 4)
            features = np.repeat(np.arange(4, dtype=np.float32)[:, None], 7936, axis=1)
            writer.write_shard(
                0,
                sample_ids=ids,
                wave_file_names=[f"{index}.wav" for index in range(4)],
                scores=scores,
                features=features,
                labels_raw=np.asarray([0.0, 2.5, 7.5, 10.0], dtype=np.float32),
            )
            writer.finalize()
            dataset = MoeFeatureStoreDataset(temporary, split="train")
            for batch_size in (1, 2, 4):
                samples = [dataset[index] for index in range(batch_size)]
                batch = collate_moe_features(samples)
                self.assertEqual(batch["scores"].shape, (batch_size, 4))
                self.assertEqual(batch["features"].shape, (batch_size, 7936))
                self.assertEqual(batch["labels_raw"].shape, (batch_size,))
                self.assertEqual(batch["sample_ids"], ids[:batch_size])
                torch.testing.assert_close(batch["scores"], torch.from_numpy(scores[:batch_size]))
                torch.testing.assert_close(batch["features"][:, 0], torch.arange(batch_size, dtype=torch.float32))
                self.assertEqual(batch["scores"].device.type, "cpu")

    def test_train_requires_raw_labels_and_test_can_be_label_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = MoeFeatureStoreWriter(
                temporary,
                split="train",
                expected_sample_ids=["train:1.wav"],
                source_manifest_sha256="b" * 64,
                checkpoint_provenance=provenance(),
                preprocessing_identity="verified-v1",
            )
            with self.assertRaisesRegex(ValueError, "label"):
                writer.write_shard(
                    0,
                    sample_ids=["train:1.wav"],
                    wave_file_names=["1.wav"],
                    scores=np.zeros((1, 4), dtype=np.float32),
                    features=np.zeros((1, 7936), dtype=np.float32),
                )

        with tempfile.TemporaryDirectory() as temporary:
            self.write_store(temporary, split="test", labels=False)
            dataset = MoeFeatureStoreDataset(temporary, split="test", require_labels=False)
            self.assertNotIn("label_raw", dataset[0])

    def test_wrong_split_feature_width_nonfinite_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write_store(temporary)
            with self.assertRaisesRegex(ValueError, "split"):
                MoeFeatureStoreDataset(temporary, split="validation")

        cases = [
            {"features": np.zeros((1, 7935), dtype=np.float32), "sample_ids": ["train:a.wav"]},
            {"features": np.full((1, 7936), np.nan, dtype=np.float32), "sample_ids": ["train:a.wav"]},
            {"features": np.zeros((2, 7936), dtype=np.float32), "sample_ids": ["train:a.wav", "train:a.wav"]},
        ]
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises((ValueError, TypeError)):
                        writer = MoeFeatureStoreWriter(
                            temporary,
                            split="train",
                            expected_sample_ids=case["sample_ids"],
                            source_manifest_sha256="c" * 64,
                            checkpoint_provenance=provenance(),
                            preprocessing_identity="verified-v1",
                        )
                        count = len(case["sample_ids"])
                        writer.write_shard(
                            0,
                            sample_ids=case["sample_ids"],
                            wave_file_names=["a.wav"] * count,
                            scores=np.zeros((count, 4), dtype=np.float32),
                            features=case["features"],
                            labels_raw=np.ones(count, dtype=np.float32),
                        )

    def test_shard_write_is_idempotent_and_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = MoeFeatureStoreWriter(
                root,
                split="train",
                expected_sample_ids=["train:a.wav"],
                source_manifest_sha256="d" * 64,
                checkpoint_provenance=provenance(),
                preprocessing_identity="verified-v1",
            )
            kwargs = {
                "sample_ids": ["train:a.wav"],
                "wave_file_names": ["a.wav"],
                "scores": np.zeros((1, 4), dtype=np.float32),
                "features": np.ones((1, 7936), dtype=np.float32),
                "labels_raw": np.asarray([5.0], dtype=np.float32),
            }
            writer.write_shard(0, **kwargs)
            writer.write_shard(0, **kwargs)
            writer.finalize()
            feature_file = next(root.rglob("features.npy"))
            with feature_file.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "hash|corrupt|integrity"):
                MoeFeatureStoreDataset(root, split="train")


if __name__ == "__main__":
    unittest.main()
