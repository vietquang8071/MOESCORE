import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.moe_feature_store import MoeFeatureStoreWriter
from model.expert_adapters.registry import EXPERT_ORDER
from scripts.evaluate_moe_gating import evaluate_checkpoint
from scripts.infer_moe_gating import infer_checkpoint
from scripts.train_moe_gating import run_training


def _write_split(root: Path, split: str, *, labeled: bool) -> None:
    count = 4
    ids = [f"{split}:{index}.wav" for index in range(count)]
    labels = np.asarray([2.0, 4.0, 7.0, 9.0], dtype=np.float32)
    target = (labels - 5.0) / 5.0
    offsets = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    scores = target[:, None] + offsets[None, :]
    generator = torch.Generator().manual_seed(300 + len(split))
    features = torch.randn((count, 7936), generator=generator).numpy().astype(np.float32)
    writer = MoeFeatureStoreWriter(
        root,
        split=split,
        expected_sample_ids=ids,
        source_manifest_sha256=("a" if split == "train" else "b" if split == "validation" else "c") * 64,
        checkpoint_provenance={name: {"checkpoint_sha256": f"{index + 1}" * 64} for index, name in enumerate(EXPERT_ORDER)},
        preprocessing_identity={"contract": "synthetic-entrypoint-test-v1", "expert_order": list(EXPERT_ORDER)},
    )
    kwargs = {"labels_raw": labels} if labeled else {}
    writer.write_shard(
        0,
        sample_ids=ids,
        wave_file_names=[f"{index}.wav" for index in range(count)],
        scores=scores,
        features=features,
        **kwargs,
    )
    writer.finalize()


def _config(root: Path, output: Path) -> dict:
    return {
        "schema_version": 1,
        "runtime": {"device": "cpu", "seed": 2027},
        "data": {"feature_store_root": str(root)},
        "output_dir": str(output),
        "training": {
            "batch_size": 2,
            "max_epochs": 1,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "diversity_weight": 0.2,
            "early_stop_patience": 3,
            "checkpoint_interval_steps": 100,
            "loss": {
                "loss_weights": [1.0, -0.5],
                "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
                "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
            },
        },
        "validation": {"batch_size": 2},
        "inference": {"batch_size": 2},
    }


class MoeCloudEntrypointTests(unittest.TestCase):
    def test_train_validate_and_label_free_infer_commands_share_the_same_contract(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stores"
            output = Path(temporary) / "run"
            _write_split(root, "train", labeled=True)
            _write_split(root, "validation", labeled=True)
            _write_split(root, "test", labeled=False)
            config = _config(root, output)

            trained = run_training(config, device_override="cpu")
            self.assertEqual(trained["status"], "PASS")
            self.assertEqual(trained["global_step"], 2)
            self.assertTrue(Path(trained["latest_checkpoint"]).is_file())
            for name in ("srcc", "lcc", "ktau", "mse", "mae"):
                self.assertIn(name, trained["history"][0]["validation_metrics_raw"])

            evaluated = evaluate_checkpoint(config, trained["latest_checkpoint"], device_override="cpu")
            self.assertEqual(evaluated["status"], "PASS")
            self.assertEqual(evaluated["sample_count"], 4)
            self.assertEqual(evaluated["optimizer_steps"], 0)
            for name in ("srcc", "lcc", "ktau", "mse", "mae"):
                self.assertIn(name, evaluated["metrics_raw_scale"])

            csv_path = Path(temporary) / "submission.csv"
            inferred = infer_checkpoint(
                config,
                trained["latest_checkpoint"],
                str(csv_path),
                device_override="cpu",
            )
            self.assertEqual(inferred["status"], "PASS")
            self.assertFalse(inferred["labels_required"])
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["wav_file_name", "pred_score"])
                rows = list(reader)
            self.assertEqual([row["wav_file_name"] for row in rows], [f"{index}.wav" for index in range(4)])
            self.assertEqual(len({row["wav_file_name"] for row in rows}), 4)


if __name__ == "__main__":
    unittest.main()
