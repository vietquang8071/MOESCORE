from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.set_a_cached_dataset import SetACachedDataset, write_set_a_cache
from model.composite_experts.set_a_residual import SetAResidualHead
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from test_set_a_residual import make_component_outputs, make_full9_store
from training.set_a_config import load_set_a_config
from training.set_a_trainer import (
    SetAResidualTrainer,
    load_set_a_checkpoint,
    set_deterministic_seed,
)


def inference_batch(dataset: SetACachedDataset):
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    return {"component_outputs": batch["component_outputs"]}


class SetATrainingTests(unittest.TestCase):
    def test_config_resolves_environment_paths_and_checks_required_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "set_a.yaml"
            config_path.write_text(
                "feature_store_root: ${SET_A_TEST_STORE}\n"
                "output_dir: ${SET_A_TEST_OUTPUT}\n"
                "runtime:\n  device: cpu\n  num_workers: 0\n"
                "model:\n  hidden_dims: [32, 8]\n  dropout: 0.1\n"
                "training:\n  epochs: 3\n  batch_size: 4\n  learning_rate: 0.002\n"
                "  weight_decay: 0.0001\n  gradient_clip: 1.5\n  seed: 19\n",
                encoding="utf-8",
            )
            previous_store = os.environ.get("SET_A_TEST_STORE")
            previous_output = os.environ.get("SET_A_TEST_OUTPUT")
            try:
                os.environ["SET_A_TEST_STORE"] = str(root / "features")
                os.environ["SET_A_TEST_OUTPUT"] = str(root / "runs")
                config = load_set_a_config(config_path)
            finally:
                if previous_store is None:
                    os.environ.pop("SET_A_TEST_STORE", None)
                else:
                    os.environ["SET_A_TEST_STORE"] = previous_store
                if previous_output is None:
                    os.environ.pop("SET_A_TEST_OUTPUT", None)
                else:
                    os.environ["SET_A_TEST_OUTPUT"] = previous_output
            self.assertEqual(config["feature_store_root"], (root / "features").resolve())
            self.assertEqual(config["output_dir"], (root / "runs").resolve())
            self.assertEqual(config["model"]["hidden_dims"], [32, 8])
            self.assertEqual(config["training"]["seed"], 19)

    def test_one_cpu_epoch_saves_final_and_latest_without_backbone_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "cache"
            make_full9_store(store, split="train")
            dataset = SetACachedDataset(store, split="train")
            model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(16, 8), dropout=0.0))
            trainer = SetAResidualTrainer(
                model,
                learning_rate=1e-3,
                weight_decay=1e-4,
                gradient_clip=1.0,
                seed=37,
                device="cpu",
            )
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            history = trainer.fit(dataset, epochs=1, batch_size=2, output_dir=root / "out")
            self.assertEqual(len(history), 1)
            self.assertTrue(torch.isfinite(torch.tensor(history[0]["mse"])))
            self.assertEqual(trainer.global_step, 1)
            for filename in ("expert5_latest.pt", "expert5_final.pt"):
                self.assertTrue((root / "out" / filename).is_file())
            checkpoint = torch.load(root / "out" / "expert5_final.pt", map_location="cpu", weights_only=False)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["feature_contract"]["feature_width"], 11267)
            self.assertEqual(checkpoint["feature_contract"]["component_order"], ["top2_m2d", "top2_msclap", "top2_mga"])
            self.assertEqual(checkpoint["score_normalization"]["formula"], "(target_raw - 5.0) / 5.0")
            self.assertTrue(all(key.startswith("head.") or key in {"scale", "bias"} for key in checkpoint["model_state_dict"]))
            self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))

    def test_trains_from_standalone_e5_cache_without_full9_backbones(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full9 = root / "full9"
            cache_ids, model_ids = make_full9_store(full9, split="train")
            set_a_cache_ids = {name: cache_ids[name] for name in ("top2_m2d", "top2_msclap", "top2_mga")}
            set_a_model_ids = {name: model_ids[name] for name in ("top2_m2d", "top2_msclap", "top2_mga")}
            sample_ids = ["train:0.wav", "train:1.wav"]
            write_set_a_cache(
                root / "e5_cache",
                split="train",
                source_manifest_path=full9 / "train" / "source_manifest.json",
                component_outputs=make_component_outputs(batch_size=2),
                component_sample_ids={name: sample_ids for name in set_a_cache_ids},
                component_cache_identities=set_a_cache_ids,
                component_model_identities=set_a_model_ids,
            )
            dataset = SetACachedDataset(root / "e5_cache", split="train")
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=3,
                device="cpu",
            )
            history = trainer.fit(dataset, epochs=1, batch_size=2, output_dir=root / "run")
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(history[0]["epoch"], 1.0)

    def test_checkpoint_load_preserves_inference_and_checks_component_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_root = root / "train_cache"
            test_root = root / "test_cache"
            make_full9_store(train_root, split="train")
            make_full9_store(test_root, split="test", labels=False)
            train = SetACachedDataset(train_root, split="train")
            test = SetACachedDataset(test_root, split="test")
            model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(12, 6), dropout=0.0))
            trainer = SetAResidualTrainer(model, seed=5, device="cpu")
            trainer.fit(train, epochs=1, batch_size=2, output_dir=root / "out")

            batch = inference_batch(test)
            model.eval()
            expected_score, expected_features = model(batch)
            restored = SetAResidualAdapter(SetAResidualHead(hidden_dims=(12, 6), dropout=0.0))
            load_set_a_checkpoint(
                root / "out" / "expert5_final.pt",
                model=restored,
                expected_dataset=test,
                map_location="cpu",
            )
            restored.eval()
            actual_score, actual_features = restored(batch)
            torch.testing.assert_close(actual_score, expected_score)
            torch.testing.assert_close(actual_features, expected_features)

            incompatible_arch = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_set_a_checkpoint(
                    root / "out" / "expert5_final.pt",
                    model=incompatible_arch,
                    expected_dataset=test,
                    map_location="cpu",
                )
            bad_width = torch.load(root / "out" / "expert5_final.pt", map_location="cpu", weights_only=False)
            bad_width["feature_contract"]["feature_width"] = 11266
            bad_width_path = root / "bad_width.pt"
            torch.save(bad_width, bad_width_path)
            with self.assertRaisesRegex(ValueError, "feature width"):
                load_set_a_checkpoint(
                    bad_width_path,
                    model=restored,
                    expected_dataset=test,
                    map_location="cpu",
                )

            test.component_cache_identities["top2_m2d"]["cache_id"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "component cache provenance"):
                load_set_a_checkpoint(
                    root / "out" / "expert5_final.pt",
                    model=restored,
                    expected_dataset=test,
                    map_location="cpu",
                )

    def test_resume_restores_optimizer_epoch_and_global_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "cache"
            make_full9_store(store, split="train")
            dataset = SetACachedDataset(store, split="train")
            first = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=11,
                device="cpu",
            )
            first.fit(dataset, epochs=1, batch_size=1, output_dir=root / "out")
            second = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=11,
                device="cpu",
            )
            second.fit(
                dataset,
                epochs=2,
                batch_size=1,
                output_dir=root / "out",
                resume_from=root / "out" / "expert5_latest.pt",
            )
            final = torch.load(root / "out" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(final["epoch"], 2)
            self.assertEqual(final["global_step"], 4)
            self.assertGreaterEqual(len(final["optimizer_state_dict"]["state"]), 1)

    def test_training_refuses_validation_and_test_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("validation", "test"):
                split_root = root / split
                make_full9_store(split_root, split=split, labels=(split == "validation"))
                dataset = SetACachedDataset(split_root, split=split)
                trainer = SetAResidualTrainer(
                    SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                    device="cpu",
                )
                with self.assertRaisesRegex(ValueError, "train split"):
                    trainer.fit(dataset, epochs=1, batch_size=1, output_dir=root / "out")

    def test_deterministic_seed_reproduces_cpu_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "cache"
            make_full9_store(store, split="train")
            dataset = SetACachedDataset(store, split="train")
            states = []
            for run in range(2):
                set_deterministic_seed(23)
                model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
                trainer = SetAResidualTrainer(model, seed=23, device="cpu")
                trainer.fit(dataset, epochs=1, batch_size=1, output_dir=root / f"out{run}")
                states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key])


if __name__ == "__main__":
    unittest.main()
