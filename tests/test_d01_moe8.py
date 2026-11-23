import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnostics.d01_moe8 import (
    FULL8_GATE_ARCHITECTURE,
    FULL8_INFERENCE_COMPAT_FORMAT_VERSION,
    FULL8_INFERENCE_COMPAT_SOURCE_FILES,
    analyze_split,
    build_prediction_rows,
    denormalize_scores,
    normalize_raw_scores,
    prepare_format7_gate_payload,
    write_prediction_csv,
)


class D01Moe8Tests(unittest.TestCase):
    def _format7_payload(self, *, overlap=False):
        digest = "a" * 64
        validation_digest = digest if overlap else "b" * 64
        current = {name: f"{index:064x}" for index, name in enumerate(FULL8_INFERENCE_COMPAT_SOURCE_FILES, 1)}
        recorded = dict(current)
        recorded["training/checkpoint_manager.py"] = "c" * 64
        payload = {
            "format_version": FULL8_INFERENCE_COMPAT_FORMAT_VERSION,
            "expert_profile": "full_8",
            "expert_order": [f"expert{i}" for i in range(1, 9)],
            "expert_feature_dimensions": {
                "expert1": 2049, "expert2": 4097, "expert3": 3840, "expert4": 1024,
                "expert5": 11267, "expert6": 4609, "expert7": 3073, "expert8": 3073,
            },
            "gate_input_dim": 33032,
            "gate_expert_count": 8,
            "gate_architecture": FULL8_GATE_ARCHITECTURE,
            "source_code_sha256": recorded,
            "feature_store_identity": {
                "train": "d" * 64,
                "validation": "e" * 64,
                "preprocessing_contract": {"profile": "full_8"},
                "score_review_policy": "EXPERIMENTAL_SKIPPED",
                "source_sample_ids_by_split": {
                    "train": ["train:a.wav"],
                    "validation": ["validation:b.wav"],
                },
                "source_wav_sha256_by_split": {
                    "train": [digest],
                    "validation": [validation_digest],
                },
            },
            "config": {"data": {"profile": "full_8", "score_review_policy": "EXPERIMENTAL_SKIPPED"}},
        }
        return payload, current

    def test_convex_weighted_predictions_stay_inside_expert_interval(self):
        scores = np.array([[-0.8, 0.1, 0.4, 0.9], [-0.3, 0.6, 0.2, 1.2]])
        weights = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
        result = analyze_split(scores, weights)
        prediction = result["envelope"]["moe_prediction_normalized"]
        self.assertTrue(np.all(result["envelope"]["expert_min_normalized"] <= prediction))
        self.assertTrue(np.all(prediction <= result["envelope"]["expert_max_normalized"]))

    def test_oracle_error_is_zero_inside_interval(self):
        result = analyze_split(np.array([[-0.5, 0.5]]), np.array([[0.5, 0.5]]), labels_raw=[5.0])
        self.assertEqual(result["global_metrics"]["mean_minimum_absolute_error_normalized"], 0.0)
        self.assertEqual(result["global_metrics"]["convex_mse_lower_bound_normalized"], 0.0)

    def test_oracle_error_above_interval_is_distance_to_maximum(self):
        result = analyze_split(np.array([[-0.5, 0.5]]), np.array([[0.5, 0.5]]), labels_raw=[9.0])
        self.assertAlmostEqual(result["envelope"]["expert_max_raw"][0], 7.5)
        self.assertAlmostEqual(result["envelope"]["minimum_absolute_error_raw"][0], 1.5)

    def test_oracle_error_below_interval_is_distance_to_minimum(self):
        result = analyze_split(np.array([[-0.5, 0.5]]), np.array([[0.5, 0.5]]), labels_raw=[1.0])
        self.assertAlmostEqual(result["envelope"]["expert_min_raw"][0], 2.5)
        self.assertAlmostEqual(result["envelope"]["minimum_absolute_error_raw"][0], 1.5)

    def test_raw_and_normalized_score_conversion_is_exact(self):
        np.testing.assert_array_equal(normalize_raw_scores([0.0, 5.0, 10.0]), [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(denormalize_scores([-1.0, 0.0, 1.0]), [0.0, 5.0, 10.0])

    def test_exact_ten_is_in_the_final_ground_truth_bin(self):
        result = analyze_split(np.array([[0.8, 1.0]]), np.array([[0.5, 0.5]]), labels_raw=[10.0])
        self.assertEqual(result["per_bin_metrics"][-1]["ground_truth_bin"], "[9,10]")
        self.assertEqual(result["per_bin_metrics"][-1]["sample_count"], 1)

    def test_vectorized_mse_lower_bound_matches_manual_example(self):
        scores = np.array([[-0.2, 0.2], [-0.8, -0.4]])
        weights = np.array([[0.25, 0.75], [0.5, 0.5]])
        result = analyze_split(scores, weights, labels_raw=[5.5, 5.0])
        metrics = result["global_metrics"]
        self.assertAlmostEqual(metrics["convex_mse_lower_bound_normalized"], (0.0 + 0.16) / 2)
        self.assertAlmostEqual(metrics["actual_moe_mse_normalized"], (0.0 + 0.36) / 2)
        self.assertAlmostEqual(metrics["convex_mse_lower_bound_raw"], 25 * (0.16 / 2))
        self.assertAlmostEqual(metrics["actual_moe_mse_raw"], 25 * (0.36 / 2))

    def test_envelope_supports_arbitrary_expert_counts(self):
        for expert_count in (4, 8, 10):
            with self.subTest(expert_count=expert_count):
                scores = np.linspace(-0.9, 0.8, expert_count)[None, :]
                weights = np.full_like(scores, 1 / expert_count)
                result = analyze_split(scores, weights)
                self.assertEqual(result["envelope"]["expert_scores_normalized"].shape, (1, expert_count))

    def test_missing_labels_do_not_create_oracle_metrics(self):
        scores = np.array([[-0.5, 0.5]])
        weights = np.array([[0.5, 0.5]])
        result = analyze_split(scores, weights, labels_raw=None)
        rows = build_prediction_rows(["test:1.wav"], "test", scores, weights, result)
        self.assertFalse(result["labels_present"])
        self.assertIsNone(result["global_metrics"])
        self.assertIsNone(result["structural_vs_gating"])
        self.assertEqual(rows[0]["label_raw"], "")
        self.assertNotIn("convex_mse_lower_bound_raw", rows[0])

    def test_invalid_scores_or_weights_are_rejected(self):
        cases = [
            ([[np.nan, 0.0]], [[0.5, 0.5]]),
            ([[np.inf, 0.0]], [[0.5, 0.5]]),
            ([[0.0, 1.0]], [[0.5]]),
            ([[0.0, 1.0]], [[0.5, 0.4]]),
            ([[0.0, 1.0]], [[-0.01, 1.01]]),
        ]
        for scores, weights in cases:
            with self.subTest(scores=scores, weights=weights), self.assertRaises((TypeError, ValueError)):
                analyze_split(scores, weights)

    def test_prediction_csv_preserves_sample_order(self):
        ids = ["validation:second.wav", "validation:first.wav"]
        scores = np.array([[0.1, 0.2], [-0.2, 0.4]])
        weights = np.array([[0.8, 0.2], [0.25, 0.75]])
        analysis = analyze_split(scores, weights, labels_raw=[5.0, 6.0])
        rows = build_prediction_rows(ids, "validation", scores, weights, analysis, labels_raw=[5.0, 6.0])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            write_prediction_csv(output, rows)
            with output.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual([row["sample_id"] for row in saved], ids)
        self.assertAlmostEqual(float(saved[0]["expert1_score_normalized"]), 0.1)
        self.assertAlmostEqual(float(saved[1]["expert2_gate_weight"]), 0.75)

    def test_format7_gate_payload_adapts_and_records_overlap_policy(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                payload, current = self._format7_payload(overlap=overlap)
                adapted, diagnostic, audit, origin = prepare_format7_gate_payload(
                    payload, current_source_hashes=current
                )
                self.assertEqual(adapted["format_version"], 5)
                self.assertEqual(diagnostic["overlapping_sha256_count"], int(overlap))
                self.assertEqual(
                    adapted["config"]["data"]["allow_train_validation_audio_overlap"], overlap
                )
                self.assertEqual(audit["different_or_unavailable_files"], ["training/checkpoint_manager.py"])
                self.assertEqual(origin, "reconstructed_from_checkpoint_source_hashes_format7")

    def test_format7_gate_rejects_changed_gate_source(self):
        payload, current = self._format7_payload()
        payload["source_code_sha256"]["model/moe_gating_core.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "inference source code is incompatible"):
            prepare_format7_gate_payload(payload, current_source_hashes=current)

    def test_kaggle_notebook_is_public_only_and_diagnostic_scoped(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        metadata = json.loads((root / "kaggle/d01_moe8/kernel-metadata.json").read_text())
        notebook_text = json.dumps(notebook)
        self.assertNotIn("XACLE2027_evaluation_dataset", notebook_text)
        self.assertIn("--allow-public-split-audio-overlap", notebook_text)
        self.assertEqual(metadata["id"], "qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(set(metadata["dataset_sources"]), {
            "qnv8071/moe8-bestgate", "qnv8071/recovery", "qnv8071/xacle-weights",
            "qnv8071/xacle-2027-data", "qnv8071/moescore-full9-feature-cache",
        })
        packaged = json.loads((root / "kaggle/d01_moe8/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        self.assertEqual(packaged, notebook)


if __name__ == "__main__":
    unittest.main()
