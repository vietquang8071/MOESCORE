import unittest

import torch

from scripts.verify_expert4_runtime import compare_state_after_finetuned_overlay


class Expert4RuntimeVerificationTests(unittest.TestCase):
    def test_pretrained_comparison_excludes_only_verified_finetuned_overlays(self):
        base = {
            "frozen.weight": torch.tensor([1.0]),
            "trained.weight": torch.tensor([2.0]),
        }
        restored = {
            "frozen.weight": torch.tensor([1.0]),
            "trained.weight": torch.tensor([3.0]),
        }
        overlay = {"trained.weight": torch.tensor([3.0])}

        result = compare_state_after_finetuned_overlay(restored, base, overlay)

        self.assertTrue(result["passed"])
        self.assertEqual(result["pretrained_values_compared"], 1)
        self.assertEqual(result["fine_tuned_overlay_keys"], 1)
        self.assertEqual(result["mismatched_pretrained_keys"], [])

    def test_pretrained_comparison_still_rejects_frozen_tensor_mismatch(self):
        base = {
            "frozen.weight": torch.tensor([1.0]),
            "trained.weight": torch.tensor([2.0]),
        }
        restored = {
            "frozen.weight": torch.tensor([9.0]),
            "trained.weight": torch.tensor([3.0]),
        }
        overlay = {"trained.weight": torch.tensor([3.0])}

        result = compare_state_after_finetuned_overlay(restored, base, overlay)

        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatched_pretrained_keys"], ["frozen.weight"])

    def test_pretrained_comparison_rejects_overlay_keys_outside_base_state(self):
        base = {"frozen.weight": torch.tensor([1.0])}
        restored = {"frozen.weight": torch.tensor([1.0])}
        overlay = {"missing.weight": torch.tensor([3.0])}

        result = compare_state_after_finetuned_overlay(restored, base, overlay)

        self.assertFalse(result["passed"])
        self.assertEqual(result["overlay_missing_from_base"], ["missing.weight"])

    def test_pretrained_comparison_reports_checkpoint_only_keys_as_unexpected(self):
        restored = {"weight": torch.tensor([1.0])}
        pretrained = {
            "weight": torch.tensor([1.0]),
            "checkpoint_only_buffer": torch.tensor([0]),
        }

        result = compare_state_after_finetuned_overlay(restored, pretrained, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["unexpected_keys"], ["checkpoint_only_buffer"])


if __name__ == "__main__":
    unittest.main()
