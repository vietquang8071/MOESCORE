from __future__ import annotations

import tempfile
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from data.set_a_cached_dataset import SetACachedDataset, write_set_a_cache
from model.composite_experts.set_a_residual import SetAResidualHead
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from test_set_a_residual import make_component_outputs, make_full9_store
from training.set_a_config import load_set_a_config
from training.set_a_trainer import (
    SetAResidualTrainer,
    _resolve_device,
    create_train_holdout_split,
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
            self.assertEqual(config["training"]["holdout_fraction"], 0.15)
            self.assertEqual(config["training"]["early_stopping_patience"], 5)
            self.assertEqual(config["training"]["min_delta"], 0.0)

    def test_device_resolution_handles_auto_cpu_cuda_unavailable_and_invalid_values(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(_resolve_device("auto"), torch.device("cpu"))
            self.assertEqual(_resolve_device("cpu"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                _resolve_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(_resolve_device("auto"), torch.device("cuda"))
            self.assertEqual(_resolve_device("cuda"), torch.device("cuda"))
        with self.assertRaisesRegex(ValueError, "device"):
            _resolve_device("tpu")

    def test_holdout_split_is_deterministic_disjoint_and_group_aware_when_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [f"train:{index}.wav" for index in range(8)]
            plain_root = root / "plain"
            make_full9_store(plain_root, ids=ids, label_values=[5.0] * len(ids))
            plain = SetACachedDataset(plain_root, split="train")
            first = create_train_holdout_split(plain, holdout_fraction=0.25, seed=17)
            again = create_train_holdout_split(plain, holdout_fraction=0.25, seed=17)
            self.assertEqual(first, again)
            self.assertEqual(first["strategy"], "sample_id_sha256_v1")
            self.assertFalse(set(first["fit_sample_ids"]) & set(first["holdout_sample_ids"]))
            self.assertEqual(
                set(first["fit_sample_ids"]) | set(first["holdout_sample_ids"]),
                set(ids),
            )

            grouped_root = root / "grouped"
            groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
            make_full9_store(
                grouped_root,
                ids=ids,
                label_values=[5.0] * len(ids),
                group_ids=groups,
            )
            grouped = SetACachedDataset(grouped_root, split="train")
            split = create_train_holdout_split(grouped, holdout_fraction=0.25, seed=17)
            self.assertEqual(split["strategy"], "group_id_sha256_v1")
            fit_groups = {groups[ids.index(sample_id)] for sample_id in split["fit_sample_ids"]}
            holdout_groups = {groups[ids.index(sample_id)] for sample_id in split["holdout_sample_ids"]}
            self.assertFalse(fit_groups & holdout_groups)
            self.assertFalse(set(split["fit_sample_ids"]) & set(split["holdout_sample_ids"]))

    def test_holdout_split_rejects_too_small_train_data_and_invalid_fraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_full9_store(root, ids=["train:only.wav"], label_values=[5.0])
            dataset = SetACachedDataset(root, split="train")
            with self.assertRaisesRegex(ValueError, "at least two"):
                create_train_holdout_split(dataset, holdout_fraction=0.15, seed=3)
            second_root = root / "two"
            make_full9_store(second_root, ids=["train:only.wav", "train:other.wav"], label_values=[5.0, 6.0])
            dataset = SetACachedDataset(second_root, split="train")
            with self.assertRaisesRegex(ValueError, "holdout_fraction"):
                create_train_holdout_split(dataset, holdout_fraction=1.0, seed=3)

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
            for metric in ("train_mse", "holdout_mse", "best_holdout_mse", "learning_rate"):
                self.assertTrue(torch.isfinite(torch.tensor(history[0][metric])))
            self.assertEqual(trainer.global_step, 1)
            for filename in ("expert5_latest.pt", "expert5_best.pt", "expert5_final.pt"):
                self.assertTrue((root / "out" / filename).is_file())
            checkpoint = torch.load(root / "out" / "expert5_final.pt", map_location="cpu", weights_only=False)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["feature_contract"]["feature_width"], 11267)
            self.assertEqual(checkpoint["feature_contract"]["component_order"], ["top2_m2d", "top2_msclap", "top2_mga"])
            self.assertEqual(checkpoint["score_normalization"]["formula"], "(target_raw - 5.0) / 5.0")
            self.assertEqual(checkpoint["checkpoint_role"], "final_best")
            self.assertEqual(checkpoint["training_state"]["best_epoch"], checkpoint["epoch"])
            self.assertEqual(checkpoint["training_state"]["holdout_split"]["strategy"], "sample_id_sha256_v1")
            self.assertTrue(all(key.startswith("head.") or key in {"scale", "bias"} for key in checkpoint["model_state_dict"]))
            self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
            latest = torch.load(root / "out" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(latest["checkpoint_role"], "latest")
            self.assertEqual(latest["training_state"]["epoch"], 1)
            combined = latest["training_state"]["fit_sample_ids"] + latest["training_state"]["holdout_sample_ids"]
            self.assertEqual(sorted(combined), sorted(dataset.sample_ids))

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
            self.assertIn("holdout_mse", history[0])

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
            self.assertEqual(final["global_step"], 2)
            self.assertGreaterEqual(len(final["optimizer_state_dict"]["state"]), 1)
            self.assertEqual(
                final["training_state"]["holdout_split"]["fingerprint"],
                torch.load(root / "out" / "expert5_best.pt", map_location="cpu", weights_only=False)
                ["training_state"]["holdout_split"]["fingerprint"],
            )

    def test_early_stopping_and_best_checkpoint_selection_use_holdout_mse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "cache"
            make_full9_store(store, split="train")
            dataset = SetACachedDataset(store, split="train")
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=13,
                device="cpu",
                early_stopping_patience=1,
            )
            with patch.object(trainer, "_evaluate_mse", side_effect=[0.4, 0.2, 0.3, 0.25]):
                history = trainer.fit(dataset, epochs=10, batch_size=1, output_dir=root / "out")
            self.assertEqual(trainer.completed_epochs, 2)
            self.assertEqual([row["holdout_mse"] for row in history], [0.2, 0.25])
            best = torch.load(root / "out" / "expert5_best.pt", map_location="cpu", weights_only=False)
            latest = torch.load(root / "out" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            final = torch.load(root / "out" / "expert5_final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["training_state"]["best_epoch"], 1)
            self.assertEqual(best["training_state"]["best_holdout_mse"], 0.2)
            self.assertEqual(latest["training_state"]["epochs_since_improvement"], 1)
            self.assertEqual(final["checkpoint_role"], "final_best")
            for key, value in best["model_state_dict"].items():
                torch.testing.assert_close(final["model_state_dict"][key], value)

    def test_resume_rejects_holdout_or_seed_mismatch_and_reproduces_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "cache"
            make_full9_store(store, split="train")
            dataset = SetACachedDataset(store, split="train")
            set_deterministic_seed(31)
            first_model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.1))
            first = SetAResidualTrainer(first_model, seed=31, device="cpu")
            first.fit(dataset, epochs=1, batch_size=1, output_dir=root / "run")

            bad_seed = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.1)),
                seed=32,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                bad_seed.fit(
                    dataset,
                    epochs=2,
                    batch_size=1,
                    output_dir=root / "run",
                    resume_from=root / "run" / "expert5_latest.pt",
                )

            bad_fraction = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.1)),
                seed=31,
                device="cpu",
                holdout_fraction=0.3,
            )
            with self.assertRaisesRegex(ValueError, "split configuration mismatch"):
                bad_fraction.fit(
                    dataset,
                    epochs=2,
                    batch_size=1,
                    output_dir=root / "run",
                    resume_from=root / "run" / "expert5_latest.pt",
                )

            set_deterministic_seed(31)
            uninterrupted_model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.1))
            uninterrupted = SetAResidualTrainer(uninterrupted_model, seed=31, device="cpu")
            uninterrupted.fit(dataset, epochs=2, batch_size=1, output_dir=root / "uninterrupted")
            resumed_model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.1))
            resumed = SetAResidualTrainer(resumed_model, seed=31, device="cpu")
            resumed.fit(
                dataset,
                epochs=2,
                batch_size=1,
                output_dir=root / "run",
                resume_from=root / "run" / "expert5_latest.pt",
            )
            uninterrupted_best = torch.load(root / "uninterrupted" / "expert5_best.pt", map_location="cpu", weights_only=False)
            resumed_best = torch.load(root / "run" / "expert5_best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(
                resumed_best["training_state"]["holdout_split"]["fingerprint"],
                uninterrupted_best["training_state"]["holdout_split"]["fingerprint"],
            )
            for key, value in uninterrupted_best["model_state_dict"].items():
                torch.testing.assert_close(resumed_best["model_state_dict"][key], value)

    def test_inference_cli_runs_with_auto_device_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_root = root / "train"
            test_root = root / "test"
            make_full9_store(train_root, split="train")
            make_full9_store(test_root, split="test", labels=False)
            dataset = SetACachedDataset(train_root, split="train")
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=7,
                device="cpu",
            )
            trainer.fit(dataset, epochs=1, batch_size=1, output_dir=root / "run")
            config_path = root / "config.yaml"
            config_path.write_text(
                f"feature_store_root: {test_root}\n"
                f"output_dir: {root / 'run'}\n"
                "runtime:\n  device: auto\n"
                "model:\n  hidden_dims: [8, 4]\n  dropout: 0.0\n"
                "training:\n  epochs: 1\n  batch_size: 2\n",
                encoding="utf-8",
            )
            output = root / "scores.csv"
            argv = [
                "infer_set_a_residual.py",
                "--config", str(config_path),
                "--checkpoint", str(root / "run" / "expert5_best.pt"),
                "--split", "test",
                "--output", str(output),
            ]
            from scripts import infer_set_a_residual
            with patch.object(sys, "argv", argv), patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(infer_set_a_residual.main(), 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

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
