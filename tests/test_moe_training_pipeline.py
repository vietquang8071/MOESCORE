import unittest

import torch

from model.moe_gating_core import GatingNetwork
from training.moe_trainer import MoeGateTrainer
from training.sampler import StatefulRandomSampler
from training.target_transform import (
    denormalize_predictions,
    normalize_targets,
    score_interval_feasibility,
)


def loss_config(*, cbl=False):
    return {
        "loss_weights": [1.0, -0.5],
        "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": cbl},
        "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": cbl},
    }


class TargetTransformTests(unittest.TestCase):
    def test_documented_score_boundaries_and_inverse(self):
        raw = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float32)
        normalized = normalize_targets(raw)
        torch.testing.assert_close(
            normalized,
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(denormalize_predictions(normalized), raw, atol=0, rtol=0)

    def test_scalar_dtype_device_and_input_immutability(self):
        raw = torch.tensor(7.5, dtype=torch.float64)
        original = raw.clone()
        result = normalize_targets(raw)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device, raw.device)
        torch.testing.assert_close(raw, original, atol=0, rtol=0)
        self.assertEqual(result.item(), 0.5)

    def test_original_clipped_mse_active_and_inactive_regions(self):
        from loss.loss_function import get_loss_function

        configured = get_loss_function({
            "loss_weights": [1.0, 0.0],
            "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
            "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
        })
        inactive = configured(torch.tensor([0.0, 0.0]), torch.tensor([0.1, -0.2]), None)
        active = configured(torch.tensor([0.0, 0.0]), torch.tensor([0.5, -0.5]), None)
        self.assertEqual(inactive.item(), 0.0)
        self.assertAlmostEqual(active.item(), 0.25, places=7)

    def test_invalid_raw_targets_are_rejected(self):
        for values in ([-0.1], [10.1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_targets(torch.tensor(values))

    def test_convex_score_interval_feasibility(self):
        scores = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]
        )
        targets = torch.tensor([0.25, -1.0, 4.0])

        result = score_interval_feasibility(scores, targets)

        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["outside_interval_count"], 2)
        self.assertEqual(result["below_interval_count"], 1)
        self.assertEqual(result["above_interval_count"], 1)
        self.assertAlmostEqual(result["outside_interval_fraction"], 2 / 3)
        self.assertAlmostEqual(result["mean_feasibility_distance"], 2 / 3, places=6)
        self.assertAlmostEqual(result["max_feasibility_distance"], 1.0, places=6)
        self.assertAlmostEqual(result["mean_squared_error_lower_bound"], 2 / 3, places=6)


class MoeGateTrainerTests(unittest.TestCase):
    def make_batch(self, batch_size):
        generator = torch.Generator().manual_seed(2027 + batch_size)
        return {
            "split": "train",
            "sample_ids": [f"train:{i}.wav" for i in range(batch_size)],
            "scores": torch.randn(batch_size, 4, generator=generator),
            "features": torch.randn(batch_size, 7936, generator=generator),
            "labels_raw": torch.linspace(0.0, 10.0, batch_size),
            "labels_scale": "raw_0_to_10",
        }

    def test_original_loss_boundary_and_one_gate_only_update(self):
        from loss.loss_function import get_loss_function

        torch.manual_seed(2027)
        gate = GatingNetwork()
        trainer = MoeGateTrainer(gate, loss_config(), learning_rate=1e-3)
        batch = self.make_batch(2)
        scores_before = batch["scores"].clone()
        features_before = batch["features"].clone()
        gate_before = [p.detach().clone() for p in gate.parameters()]

        result = trainer.train_step(batch)

        self.assertEqual(tuple(result["prediction"].shape), (2, 1))
        self.assertEqual(tuple(result["targets_normalized"].shape), (2,))
        self.assertTrue(torch.isfinite(result["total_loss"]))
        self.assertTrue(torch.isfinite(result["gradient_norm"]))
        self.assertGreater(result["updated_parameter_count"], 0)
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(gate_before, gate.parameters()))
        )
        torch.testing.assert_close(batch["scores"], scores_before, atol=0, rtol=0)
        torch.testing.assert_close(batch["features"], features_before, atol=0, rtol=0)
        self.assertIsNone(batch["scores"].grad)
        self.assertIsNone(batch["features"].grad)

        reference_loss = get_loss_function(loss_config())(
            result["prediction"].reshape(-1), result["targets_normalized"], None
        )
        torch.testing.assert_close(result["regression_loss"], reference_loss)

    def test_b1_b2_b4_loss_inputs_are_vectors_and_scalar_loss(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                trainer = MoeGateTrainer(GatingNetwork(), loss_config())
                result = trainer.train_step(self.make_batch(batch_size))
                self.assertEqual(result["prediction"].shape, (batch_size, 1))
                self.assertEqual(result["targets_normalized"].shape, (batch_size,))
                self.assertEqual(result["total_loss"].ndim, 0)
                self.assertTrue(torch.isfinite(result["total_loss"]))

    def test_class_balancing_requires_explicit_num_class_metadata(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config(cbl=True))
        with self.assertRaisesRegex(ValueError, "num_class"):
            trainer.train_step(self.make_batch(2))

    def test_original_pairwise_contrastive_values_use_vector_ranks(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config())
        batch = self.make_batch(4)
        scores, features, labels, classes = trainer._batch_tensors(batch, expected_split="train")
        result = trainer._losses(scores, features, labels, classes)
        prediction = result["prediction"].reshape(-1)
        target = result["targets_normalized"]
        target_difference = target.unsqueeze(1) - target.unsqueeze(0)
        prediction_difference = prediction.unsqueeze(1) - prediction.unsqueeze(0)
        expected_matrix = torch.maximum(
            torch.zeros_like(target_difference),
            torch.abs(prediction_difference - target_difference) - 0.1,
        )
        self.assertEqual(tuple(expected_matrix.shape), (4, 4))
        torch.testing.assert_close(result["contrastive_loss"], expected_matrix.mean() / 2)

    def test_source_clipped_loss_class_balancing_broadcast_is_rejected(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config(cbl=True))
        batch = self.make_batch(2)
        batch["num_class"] = torch.tensor([2, 3], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "ClippedMSELoss expands per-sample weights"):
            trainer.train_step(batch)

    def test_validation_and_test_batches_cannot_update_optimizer(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config())
        batch = self.make_batch(2)
        batch["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training split"):
            trainer.train_step(batch)

    def test_training_requires_raw_label_scale_marker(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config())
        batch = self.make_batch(2)
        batch["labels_scale"] = "normalized_minus1_to1"
        with self.assertRaisesRegex(ValueError, "raw_0_to_10"):
            trainer.train_step(batch)

    def test_validation_step_does_not_update_gate(self):
        trainer = MoeGateTrainer(GatingNetwork(), loss_config())
        trainer.gating_network.train()
        before = [p.detach().clone() for p in trainer.gating_network.parameters()]
        batch = self.make_batch(2)
        batch["split"] = "validation"
        result = trainer.validation_step(batch)
        self.assertFalse(trainer.gating_network.training)
        self.assertTrue(all(p.grad is None for p in trainer.gating_network.parameters()))
        self.assertTrue(
            all(torch.equal(old, new) for old, new in zip(before, trainer.gating_network.parameters()))
        )
        self.assertTrue(torch.isfinite(result["regression_loss"]))

    def test_sampler_resume_preserves_epoch_order_and_cursor(self):
        source = StatefulRandomSampler(range(11), seed=2027)
        iterator = iter(source)
        prefix = [next(iterator), next(iterator), next(iterator)]
        state = source.state_dict()
        restored = StatefulRandomSampler(range(11), seed=2027)
        restored.load_state_dict(state)
        suffix = list(restored)
        expected = list(iterator)
        self.assertEqual(suffix, expected)
        self.assertEqual(len(prefix + suffix), 11)
        self.assertEqual(len(set(prefix + suffix)), 11)


if __name__ == "__main__":
    unittest.main()
