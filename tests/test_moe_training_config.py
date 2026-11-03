import tempfile
import unittest
from pathlib import Path

import yaml

from training.config import load_moe_training_config


def config_data():
    return {
        "schema_version": 1,
        "runtime": {"device": "auto", "seed": 2027},
        "data": {"feature_store_root": "../stores"},
        "output_dir": "../runs/moe",
        "training": {
            "batch_size": 2,
            "max_epochs": 5,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "diversity_weight": 0.2,
            "early_stop_patience": 4,
            "checkpoint_interval_steps": 10,
            "loss": {
                "loss_weights": [1.0, -0.5],
                "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
                "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
            },
        },
        "validation": {"batch_size": 2},
        "inference": {"batch_size": 2},
    }


class MoeTrainingConfigTests(unittest.TestCase):
    def test_loads_relative_paths_and_preserves_source_loss_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "moe.yaml"
            config_path.write_text(yaml.safe_dump(config_data()), encoding="utf-8")
            config = load_moe_training_config(config_path)
            self.assertEqual(config["runtime"]["device"], "auto")
            self.assertEqual(config["runtime"]["seed"], 2027)
            self.assertEqual(config["training"]["loss"]["loss_weights"], [1.0, -0.5])
            self.assertEqual(config["training"]["diversity_weight"], 0.2)
            self.assertEqual(Path(config["data"]["feature_store_root"]), Path(temporary).parent / "stores")
            self.assertEqual(Path(config["output_dir"]), Path(temporary).parent / "runs/moe")

    def test_unknown_fields_and_invalid_batch_settings_fail(self):
        for mutate, message in (
            (lambda data: data.update(extra=True), "unknown"),
            (lambda data: data["training"].update(batch_size=0), "batch_size"),
            (lambda data: data["training"]["loss"].update(loss_weights=[1.0]), "loss_weights"),
            (lambda data: data["runtime"].update(device="cuda:-1"), "device"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                data = config_data()
                mutate(data)
                config_path = Path(temporary) / "bad.yaml"
                config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_moe_training_config(config_path)


if __name__ == "__main__":
    unittest.main()
