import unittest

import torch
from torch import nn


class AdapterBaseTests(unittest.TestCase):
    def test_base_and_validator_are_importable(self):
        from model.expert_adapters.base import ExpertAdapter, validate_expert_output

        self.assertTrue(issubclass(ExpertAdapter, nn.Module))
        self.assertTrue(callable(validate_expert_output))

    def test_base_is_abstract(self):
        from model.expert_adapters.base import ExpertAdapter

        with self.assertRaises(TypeError):
            ExpertAdapter(feature_dim=3)

    def test_base_forward_validates_raw_outputs_from_concrete_adapters(self):
        from model.expert_adapters.base import ExpertAdapter

        class MalformedAdapter(ExpertAdapter):
            def _forward(self, batch):
                return torch.zeros(batch), torch.zeros(batch, self.feature_dim)

        adapter = MalformedAdapter(feature_dim=3)
        with self.assertRaisesRegex(ValueError, r"score.*\[B, 1\]"):
            adapter(2)

    def test_feature_dimension_must_be_a_positive_integer(self):
        from model.expert_adapters.base import ExpertAdapter

        class DummyAdapter(ExpertAdapter):
            def _forward(self, batch):
                raise NotImplementedError

        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "feature_dim"):
                    DummyAdapter(feature_dim=value)

        self.assertEqual(DummyAdapter(feature_dim=4).feature_dim, 4)

    def test_valid_outputs_preserve_tensors_for_single_and_multiple_samples(self):
        from model.expert_adapters.base import ExpertAdapter

        class DummyAdapter(ExpertAdapter):
            def _forward(self, batch):
                batch_size = batch
                score = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1)
                features = torch.ones(batch_size, self.feature_dim)
                return score, features

        adapter = DummyAdapter(feature_dim=3)
        self.assertEqual(list(adapter.parameters()), [])
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                score, features = adapter(batch_size)
                self.assertEqual(tuple(score.shape), (batch_size, 1))
                self.assertEqual(tuple(features.shape), (batch_size, 3))

    def test_validator_returns_the_original_tensors_without_reshaping(self):
        from model.expert_adapters.base import validate_expert_output

        score = torch.zeros(2, 1)
        features = torch.zeros(2, 5)
        actual_score, actual_features = validate_expert_output(
            score, features, feature_dim=5, expert_name="Dummy"
        )
        self.assertIs(actual_score, score)
        self.assertIs(actual_features, features)

    def test_validator_rejects_non_tensor_outputs(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(TypeError, "score.*Tensor"):
            validate_expert_output([0.0], torch.zeros(1, 2), feature_dim=2)
        with self.assertRaisesRegex(TypeError, "features.*Tensor"):
            validate_expert_output(torch.zeros(1, 1), [[0.0, 1.0]], feature_dim=2)

    def test_validator_requires_exact_score_shape(self):
        from model.expert_adapters.base import validate_expert_output

        for score in (torch.zeros(2), torch.zeros(2, 2), torch.zeros(2, 1, 1)):
            with self.subTest(shape=tuple(score.shape)):
                with self.assertRaisesRegex(ValueError, r"score.*\[B, 1\]"):
                    validate_expert_output(score, torch.zeros(2, 3), feature_dim=3)

    def test_validator_requires_rank_two_features_and_declared_width(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(ValueError, r"features.*\[B, D\]"):
            validate_expert_output(torch.zeros(2, 1), torch.zeros(2, 3, 1), feature_dim=3)
        with self.assertRaisesRegex(ValueError, "feature_dim=3"):
            validate_expert_output(torch.zeros(2, 1), torch.zeros(2, 4), feature_dim=3)

    def test_validator_requires_matching_batch_dimensions(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(ValueError, "batch dimension"):
            validate_expert_output(torch.zeros(2, 1), torch.zeros(3, 4), feature_dim=4)

    def test_validator_requires_positive_feature_width(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(ValueError, "feature_dim"):
            validate_expert_output(torch.zeros(1, 1), torch.zeros(1, 0), feature_dim=0)

    def test_validator_rejects_non_finite_values(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(ValueError, "score.*finite"):
            validate_expert_output(torch.tensor([[float("nan")]]), torch.zeros(1, 2), feature_dim=2)
        with self.assertRaisesRegex(ValueError, "features.*finite"):
            validate_expert_output(torch.zeros(1, 1), torch.tensor([[0.0, float("inf")]]), feature_dim=2)

    def test_validator_requires_floating_outputs_on_the_same_device(self):
        from model.expert_adapters.base import validate_expert_output

        with self.assertRaisesRegex(TypeError, "floating-point"):
            validate_expert_output(torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 2), feature_dim=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
