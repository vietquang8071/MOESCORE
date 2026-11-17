"""Unit tests for shared paired audio/text interaction features."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from model.expert_adapters.paired_cross_features import build_paired_cross_features


class PairedCrossFeatureTests(unittest.TestCase):
    def test_five_blocks_have_exact_order_and_values(self):
        audio = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
        text = torch.tensor([[0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        features = build_paired_cross_features(audio, text)

        self.assertEqual(tuple(features.shape), (2, 10))
        self.assertEqual(features.dtype, torch.float32)
        audio_f, text_f, difference, product, repeated_cosine = features.split(2, dim=-1)
        torch.testing.assert_close(audio_f, audio.float(), rtol=0, atol=0)
        torch.testing.assert_close(text_f, text.float(), rtol=0, atol=0)
        audio_norm = F.normalize(audio.float(), p=2, dim=-1)
        text_norm = F.normalize(text.float(), p=2, dim=-1)
        expected_cosine = (audio_norm * text_norm).sum(-1, keepdim=True)
        torch.testing.assert_close(difference, torch.abs(audio_norm - text_norm))
        torch.testing.assert_close(product, audio_norm * text_norm)
        torch.testing.assert_close(repeated_cosine, expected_cosine.expand(-1, 2))
        self.assertTrue(torch.equal(repeated_cosine[:, :1], repeated_cosine[:, 1:]))

    def test_zero_embeddings_are_finite_and_keep_first_blocks(self):
        audio = torch.zeros(2, 3, dtype=torch.float16)
        text = torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
        features = build_paired_cross_features(audio, text)
        self.assertEqual(tuple(features.shape), (2, 15))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[:, :3], audio.float(), rtol=0, atol=0)
        torch.testing.assert_close(features[:, 3:6], text.float(), rtol=0, atol=0)
        self.assertTrue(torch.equal(features[:, 12:], torch.zeros(2, 3)))

    def test_batch_size_and_dimension_must_be_positive_and_shapes_must_match(self):
        invalid = (
            (torch.zeros(2, 3), torch.zeros(2, 4)),
            (torch.zeros(2, 3, 1), torch.zeros(2, 3, 1)),
            (torch.zeros(0, 3), torch.zeros(0, 3)),
            (torch.zeros(2, 0), torch.zeros(2, 0)),
        )
        for audio, text in invalid:
            with self.subTest(audio_shape=tuple(audio.shape), text_shape=tuple(text.shape)):
                with self.assertRaises((TypeError, ValueError)):
                    build_paired_cross_features(audio, text)

    def test_rejects_integer_nonfinite_and_mismatched_device_inputs(self):
        with self.assertRaisesRegex(TypeError, "floating-point"):
            build_paired_cross_features(torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2))
        for bad_value in (float("nan"), float("inf")):
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    build_paired_cross_features(torch.tensor([[bad_value]]), torch.zeros(1, 1))
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "same device"):
                build_paired_cross_features(torch.zeros(1, 2), torch.zeros(1, 2, device="cuda"))

    def test_builder_is_stateless_and_does_not_add_trainable_parameters(self):
        self.assertFalse(isinstance(build_paired_cross_features, torch.nn.Module))
        first = build_paired_cross_features(torch.ones(1, 2), torch.ones(1, 2))
        second = build_paired_cross_features(torch.ones(1, 2), torch.ones(1, 2))
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
