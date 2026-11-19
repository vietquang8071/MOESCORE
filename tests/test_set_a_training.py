from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import SetAResidualHead
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from test_set_a_residual import make_full9_store
from training.set_a_config import load_set_a_config
from training.set_a_trainer import (
    CHECKPOINT_FORMAT_VERSION,
    SetAResidualTrainer,
    _resolve_device,
    load_set_a_checkpoint,
    set_deterministic_seed,
)


def inference_batch(dataset: SetACachedDataset):
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    return {"component_outputs": batch["component_outputs"]}


def make_train_validation(
    root: Path,
    *,
    train_ids: list[str] | None = None,
    validation_ids: list[str] | None = None,
    validation_labels: bool = True,
    validation_label_values: list[float] | None = None,
):
    train_ids = train_ids or [f"train:{index}.wav" for index in range(2)]
    validation_ids = validation_ids or [f"validation:{index}.wav" for index in range(2)]
    make_full9_store(root, split="train", ids=train_ids)
    make_full9_store(
        root,
        split="validation",
        ids=validation_ids,
        labels=validation_labels,
        label_values=validation_label_values,
    )
    return (
        SetACachedDataset(root, split="train", require_labels=True),
        SetACachedDataset(root, split="validation", require_labels=True)
        if validation_labels
        else SetACachedDataset(root, split="validation", require_labels=False),
    )


class SetATrainingTests(unittest.TestCase):
    def test_config_requires_official_validation_selection_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "set_a.yaml"
            config_path.write_text(
                "feature_store_root: ${SET_A_TEST_STORE}\n"
                "output_dir: ${SET_A_TEST_OUTPUT}\n"
                "runtime:\n  device: cpu\n"
                "model:\n  hidden_dims: [32, 8]\n  dropout: 0.1\n"
                "training:\n  epochs: 3\n  batch_size: 4\n"
                "  selection_metric: official_validation_mse\n",
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
            self.assertEqual(config["training"]["selection_metric"], "official_validation_mse")
            self.assertNotIn("holdout_fraction", config["training"])
            self.assertEqual(config["training"]["early_stopping_patience"], 5)

            config_path.write_text(
                "feature_store_root: x\noutput_dir: y\nruntime: {}\nmodel: {}\n"
                "training:\n  selection_metric: train_mse\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "selection_metric"):
                load_set_a_config(config_path)
            config_path.write_text(
                "feature_store_root: x\noutput_dir: y\nruntime: {}\nmodel: {}\n"
                "training:\n  holdout_fraction: 0.15\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "obsolete"):
                load_set_a_config(config_path)

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

    def test_full_train_is_optimized_and_validation_runs_without_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_ids = [f"train:{index}.wav" for index in range(10)]
            validation_ids = [f"validation:{index}.wav" for index in range(3)]
            train, validation = make_train_validation(
                root, train_ids=train_ids, validation_ids=validation_ids
            )
            model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
            trainer = SetAResidualTrainer(model, seed=37, device="cpu")
            active_split = {"name": "optimization"}
            forwards = []
            hook = model.register_forward_hook(
                lambda module, args, output: forwards.append(
                    (active_split["name"], module.training, torch.is_grad_enabled(), output[0].requires_grad,
                     output[0].shape[0])
                )
            )
            evaluate = trainer._evaluate_mse

            def track_evaluation(dataset, **kwargs):
                active_split["name"] = dataset.split
                try:
                    return evaluate(dataset, **kwargs)
                finally:
                    active_split["name"] = "optimization"

            trainer._evaluate_mse = track_evaluation
            history = trainer.fit(
                train,
                validation_dataset=validation,
                epochs=1,
                batch_size=2,
                output_dir=root / "run",
            )
            hook.remove()

            self.assertEqual(trainer.global_step, 5)
            self.assertEqual(sum(event[4] for event in forwards if event[0] == "optimization"), 10)
            validation_forwards = [event for event in forwards if event[0] == "validation"]
            self.assertEqual(sum(event[4] for event in validation_forwards), 3)
            self.assertTrue(all(not training and not grad and not has_graph
                                for _, training, grad, has_graph, _ in validation_forwards))
            self.assertEqual(
                set(history[0]),
                {"epoch", "train_mse", "validation_mse", "best_validation_mse", "learning_rate"},
            )

            latest = torch.load(root / "run" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            state = latest["training_state"]
            self.assertEqual(latest["format_version"], CHECKPOINT_FORMAT_VERSION)
            self.assertEqual(latest["data_provenance"], train.provenance)
            self.assertEqual(state["selection_policy"], "official_validation_mse_v1")
            self.assertEqual(state["train_sample_ids"], train_ids)
            self.assertEqual(state["validation_sample_ids"], validation_ids)
            self.assertEqual(state["validation_provenance"], validation.provenance)
            for obsolete in ("holdout_split", "fit_sample_ids", "holdout_sample_ids", "best_holdout_mse"):
                self.assertNotIn(obsolete, state)

    def test_training_entrypoint_loads_and_reports_both_official_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_ids = [f"train:{index}.wav" for index in range(3)]
            validation_ids = [f"validation:{index}.wav" for index in range(2)]
            train, validation = make_train_validation(
                root, train_ids=train_ids, validation_ids=validation_ids
            )
            config_path = root / "e5.yaml"
            config_path.write_text(
                f"feature_store_root: {root}\noutput_dir: {root / 'run'}\n"
                "source_manifest_paths: {}\nruntime:\n  device: cpu\n"
                "model:\n  hidden_dims: [8, 4]\n  dropout: 0.0\n"
                "training:\n  epochs: 1\n  batch_size: 2\n"
                "  selection_metric: official_validation_mse\n",
                encoding="utf-8",
            )
            from scripts import train_set_a_residual
            output = io.StringIO()
            with patch.object(sys, "argv", ["train_set_a_residual.py", "--config", str(config_path)]), \
                 redirect_stdout(output):
                self.assertEqual(train_set_a_residual.main(), 0)
            self.assertIn("E5 train samples: 3", output.getvalue())
            self.assertIn("E5 validation samples: 2", output.getvalue())
            checkpoint = torch.load(
                root / "run" / "expert5_latest.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["data_provenance"], train.provenance)
            self.assertEqual(checkpoint["training_state"]["validation_provenance"], validation.provenance)

    def test_best_checkpoint_and_early_stopping_use_validation_mse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=13,
                device="cpu",
                early_stopping_patience=1,
                min_delta=0.05,
            )
            with patch.object(trainer, "_evaluate_mse", side_effect=[0.4, 0.2, 0.3, 0.16]):
                history = trainer.fit(
                    train, validation_dataset=validation, epochs=10, batch_size=1,
                    output_dir=root / "run",
                )
            self.assertEqual(trainer.completed_epochs, 2)
            self.assertEqual([row["validation_mse"] for row in history], [0.2, 0.16])
            self.assertEqual([row["best_validation_mse"] for row in history], [0.2, 0.2])
            best = torch.load(root / "run" / "expert5_best.pt", map_location="cpu", weights_only=False)
            latest = torch.load(root / "run" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            final = torch.load(root / "run" / "expert5_final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["training_state"]["best_validation_mse"], 0.2)
            self.assertEqual(best["training_state"]["best_epoch"], 1)
            self.assertEqual(latest["training_state"]["epochs_since_improvement"], 1)
            self.assertEqual(final["checkpoint_role"], "final_best")
            for key, value in best["model_state_dict"].items():
                torch.testing.assert_close(final["model_state_dict"][key], value)

    def test_resume_preserves_policy_and_rejects_changed_validation_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            first = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=31, device="cpu",
            )
            first.fit(train, validation_dataset=validation, epochs=1, batch_size=1, output_dir=root / "run")

            changed_validation_root = root / "changed_validation"
            make_full9_store(
                changed_validation_root,
                split="validation",
                label_values=[5.0, 7.0],
            )
            changed_validation = SetACachedDataset(
                changed_validation_root, split="validation", require_labels=True
            )
            resumed = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=31, device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "validation provenance mismatch"):
                resumed.fit(
                    train,
                    validation_dataset=changed_validation,
                    epochs=2,
                    batch_size=1,
                    output_dir=root / "run",
                    resume_from=root / "run" / "expert5_latest.pt",
                )

            resumed = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=31, device="cpu",
            )
            history = resumed.fit(
                train,
                validation_dataset=validation,
                epochs=2,
                batch_size=1,
                output_dir=root / "run",
                resume_from=root / "run" / "expert5_latest.pt",
            )
            latest = torch.load(root / "run" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual([row["epoch"] for row in history], [1.0, 2.0])
            self.assertEqual(latest["training_state"]["selection_policy"], "official_validation_mse_v1")
            self.assertEqual(latest["training_state"]["validation_provenance"], validation.provenance)
            self.assertEqual(len(latest["training_state"]["metric_history"]), 2)

    def test_legacy_checkpoint_loads_for_inference_but_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=5, device="cpu",
            )
            trainer.fit(train, validation_dataset=validation, epochs=1, batch_size=2, output_dir=root / "run")
            checkpoint = torch.load(root / "run" / "expert5_latest.pt", map_location="cpu", weights_only=False)
            checkpoint["format_version"] = 2
            checkpoint["training_state"] = {
                "holdout_split": {"strategy": "sample_id_sha256_v1", "fingerprint": "a" * 64},
                "fit_sample_ids": train.sample_ids[:-1],
                "holdout_sample_ids": train.sample_ids[-1:],
                "best_holdout_mse": 0.3,
                "best_epoch": 1,
            }
            legacy = root / "legacy.pt"
            torch.save(checkpoint, legacy)
            restored = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
            loaded = load_set_a_checkpoint(legacy, model=restored, expected_dataset=train)
            self.assertEqual(loaded["format_version"], 2)

            checkpoint["checkpoint_role"] = "latest"
            legacy_latest = root / "run" / "expert5_latest.pt"
            torch.save(checkpoint, legacy_latest)
            resumed = SetAResidualTrainer(restored, seed=5, device="cpu")
            with self.assertRaisesRegex(ValueError, "legacy E5 holdout checkpoint"):
                resumed.fit(
                    train, validation_dataset=validation, epochs=2, batch_size=2,
                    output_dir=root / "run", resume_from=legacy_latest,
                )

    def test_checkpoint_loader_preserves_inference_and_validates_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            test_root = root / "test_cache"
            make_full9_store(test_root, split="test", labels=False)
            test = SetACachedDataset(test_root, split="test")
            model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(12, 6), dropout=0.0))
            trainer = SetAResidualTrainer(model, seed=5, device="cpu")
            trainer.fit(train, validation_dataset=validation, epochs=1, batch_size=2, output_dir=root / "run")

            batch = inference_batch(test)
            model.eval()
            expected_score, expected_features = model(batch)
            restored = SetAResidualAdapter(SetAResidualHead(hidden_dims=(12, 6), dropout=0.0))
            load_set_a_checkpoint(
                root / "run" / "expert5_final.pt", model=restored,
                expected_dataset=test, map_location="cpu",
            )
            restored.eval()
            actual_score, actual_features = restored(batch)
            torch.testing.assert_close(actual_score, expected_score)
            torch.testing.assert_close(actual_features, expected_features)

    def test_inference_cli_runs_with_auto_device_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                seed=7, device="cpu",
            )
            trainer.fit(train, validation_dataset=validation, epochs=1, batch_size=2, output_dir=root / "run")
            test_root = root / "test_cache"
            make_full9_store(test_root, split="test", labels=False)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"feature_store_root: {test_root}\noutput_dir: {root / 'run'}\n"
                "runtime:\n  device: auto\nmodel:\n  hidden_dims: [8, 4]\n  dropout: 0.0\n"
                "training:\n  epochs: 1\n  batch_size: 2\n  selection_metric: official_validation_mse\n",
                encoding="utf-8",
            )
            output = root / "scores.csv"
            argv = ["infer_set_a_residual.py", "--config", str(config_path), "--checkpoint",
                    str(root / "run" / "expert5_best.pt"), "--split", "test", "--output", str(output)]
            from scripts import infer_set_a_residual
            with patch.object(sys, "argv", argv), patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(infer_set_a_residual.main(), 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

    def test_training_requires_labeled_official_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, unlabeled_validation = make_train_validation(
                root, validation_labels=False
            )
            trainer = SetAResidualTrainer(
                SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "labeled official validation"):
                trainer.fit(
                    train, validation_dataset=unlabeled_validation,
                    epochs=1, batch_size=1, output_dir=root / "run",
                )

    def test_deterministic_seed_reproduces_cpu_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = make_train_validation(root)
            states = []
            for run in range(2):
                set_deterministic_seed(23)
                model = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
                trainer = SetAResidualTrainer(model, seed=23, device="cpu")
                trainer.fit(
                    train, validation_dataset=validation, epochs=1, batch_size=1,
                    output_dir=root / f"out{run}",
                )
                states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key])


if __name__ == "__main__":
    unittest.main()
