import csv
import tempfile
import unittest

import torch

from model.moe_gating_core import GatingNetwork
from training.moe_evaluator import MoeGateEvaluator, compute_metrics, write_submission_csv


class MoeValidationInferenceTests(unittest.TestCase):
    def test_known_metrics_and_undefined_spearman_are_explicit(self):
        metrics = compute_metrics(
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([2.0, 4.0, 6.0, 8.0]),
        )
        self.assertEqual(metrics["srcc"], 1.0)
        self.assertEqual(metrics["mse"], 7.5)
        self.assertEqual(metrics["mae"], 2.5)
        constant = compute_metrics(torch.ones(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertIsNone(constant["srcc"])
        self.assertEqual(constant["srcc_status"], "UNDEFINED_CONSTANT_INPUT")

    def test_inference_needs_no_labels_preserves_order_and_uses_raw_scale(self):
        torch.manual_seed(7)
        gate = GatingNetwork().train()
        evaluator = MoeGateEvaluator(gate)
        batch = {
            "split": "test",
            "sample_ids": ["test:a.wav", "test:b.wav"],
            "wave_file_names": ["a.wav", "b.wav"],
            "scores": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.4, 0.2]]),
            "features": torch.randn(2, 7936),
        }

        rows = evaluator.predict([batch])

        self.assertFalse(gate.training)
        self.assertEqual([row["sample_id"] for row in rows], ["test:a.wav", "test:b.wav"])
        self.assertEqual([row["wave_file_name"] for row in rows], ["a.wav", "b.wav"])
        self.assertTrue(all(0.0 <= row["pred_score"] <= 10.0 for row in rows))

    def test_validation_does_not_update_parameters_or_require_optimizer(self):
        torch.manual_seed(3)
        gate = GatingNetwork().train()
        before = [p.detach().clone() for p in gate.parameters()]
        evaluator = MoeGateEvaluator(gate)
        result = evaluator.evaluate(
            [{
                "split": "validation",
                "sample_ids": ["validation:a", "validation:b"],
                "wave_file_names": ["a.wav", "b.wav"],
                "scores": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]),
                "features": torch.randn(2, 7936),
                "labels_raw": torch.tensor([2.0, 8.0]),
                "labels_scale": "raw_0_to_10",
            }]
        )
        self.assertFalse(gate.training)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, gate.parameters())))
        self.assertEqual(result["prediction_normalized"].shape, (2,))
        self.assertEqual(result["prediction_raw"].shape, (2,))
        self.assertIn("mse", result["metrics"])

    def test_evaluator_accepts_streaming_dataloader_batches(self):
        gate = GatingNetwork().eval()
        evaluator = MoeGateEvaluator(gate)
        batch = {
            "split": "validation",
            "sample_ids": ["validation:one"],
            "wave_file_names": ["one.wav"],
            "scores": torch.zeros(1, 4),
            "features": torch.zeros(1, 7936),
            "labels_raw": torch.tensor([5.0]),
            "labels_scale": "raw_0_to_10",
        }
        result = evaluator.evaluate(iter([batch]))
        self.assertEqual(result["sample_ids"], ["validation:one"])

    def test_validation_rejects_original_class_balancing_broadcast_path(self):
        loss_config = {
            "loss_weights": [1.0, -0.5],
            "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": True},
            "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
        }
        batch = {
            "split": "validation",
            "sample_ids": ["validation:a", "validation:b"],
            "wave_file_names": ["a.wav", "b.wav"],
            "scores": torch.zeros(2, 4),
            "features": torch.zeros(2, 7936),
            "labels_raw": torch.tensor([2.0, 8.0]),
            "labels_scale": "raw_0_to_10",
            "num_class": torch.tensor([2, 3], dtype=torch.int64),
        }
        with self.assertRaisesRegex(ValueError, "broadcasts \[B,1\] weights"):
            MoeGateEvaluator(GatingNetwork(), loss_config).evaluate([batch])

    def test_submission_csv_has_verified_upstream_header_and_unique_rows(self):
        rows = [
            {"sample_id": "test:a.wav", "wave_file_name": "a.wav", "pred_score": 5.5},
            {"sample_id": "test:b.wav", "wave_file_name": "b.wav", "pred_score": 4.25},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/predictions.csv"
            write_submission_csv(rows, path)
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["wav_file_name", "pred_score"])
                written = list(reader)
            self.assertEqual([row["wav_file_name"] for row in written], ["a.wav", "b.wav"])
            self.assertEqual([float(row["pred_score"]) for row in written], [5.5, 4.25])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            write_submission_csv([rows[0], rows[0]], "/tmp/unused-moe.csv")


if __name__ == "__main__":
    unittest.main()
