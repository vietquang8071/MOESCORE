"""Focused tests for the real-data preprocessing verification helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from scripts.verify_unified_dataset_runtime import (
    classify_failure,
    compare_e4_representation,
    compare_tensors,
    e1_crop_start,
    scan_auxiliary_csv,
    scan_split_metadata,
    write_json_report,
)


class WaveformComparisonTests(unittest.TestCase):
    def test_equal_and_tolerance_accepted_tensors_report_numeric_metrics(self):
        native = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
        exact = compare_tensors(native, native.clone(), atol=1e-6, rtol=1e-5)
        self.assertTrue(exact["exact_equal"])
        self.assertTrue(exact["allclose"])
        self.assertEqual(exact["mismatch_count"], 0)
        close = compare_tensors(native, native + 1e-7, atol=1e-6, rtol=1e-5)
        self.assertFalse(close["exact_equal"])
        self.assertTrue(close["allclose"])
        self.assertGreater(close["max_absolute_error"], 0.0)

    def test_shape_dtype_and_nonfinite_differences_are_explicit(self):
        native = torch.tensor([1.0, 2.0], dtype=torch.float32)
        shape = compare_tensors(native, torch.ones(1, 2), atol=1e-6, rtol=1e-5)
        self.assertFalse(shape["shape_match"])
        self.assertIsNone(shape["allclose"])
        dtype = compare_tensors(native, native.double(), atol=1e-6, rtol=1e-5)
        self.assertFalse(dtype["dtype_match"])
        nonfinite = compare_tensors(native, torch.tensor([1.0, float("nan")]), atol=1e-6, rtol=1e-5)
        self.assertFalse(nonfinite["finite"])
        self.assertFalse(nonfinite["allclose"])

    def test_expert4_valid_prefix_and_fixed_padding_are_checked_separately(self):
        native = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        unified = torch.zeros((1, 8), dtype=torch.float32)
        unified[0, :3] = native
        result = compare_e4_representation(native, unified, valid_length=3)
        self.assertTrue(result["valid_prefix"]["exact_equal"])
        self.assertTrue(result["padding_is_zero"])
        self.assertTrue(result["length_matches_native"])
        unified[0, 7] = 0.1
        result = compare_e4_representation(native, unified, valid_length=3)
        self.assertFalse(result["padding_is_zero"])

    def test_expert4_rejects_length_or_prefix_mismatch(self):
        native = torch.ones(3, dtype=torch.float32)
        unified = torch.zeros((1, 8), dtype=torch.float32)
        result = compare_e4_representation(native, unified, valid_length=2)
        self.assertFalse(result["length_matches_native"])
        self.assertFalse(result["valid_prefix"]["exact_equal"])


class CropAndMetadataTests(unittest.TestCase):
    def test_e1_crop_offset_is_reproducible_epoch_sensitive_and_inclusive(self):
        import random

        global_state = random.getstate()
        first = e1_crop_start(seed=2027, epoch=0, index=7, last_start=1000)
        self.assertEqual(first, e1_crop_start(seed=2027, epoch=0, index=7, last_start=1000))
        self.assertNotEqual(first, e1_crop_start(seed=2027, epoch=1, index=7, last_start=1000))
        self.assertEqual(e1_crop_start(seed=1, epoch=0, index=0, last_start=0), 0)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 1000)
        self.assertEqual(random.getstate(), global_state)

    def test_auxiliary_metadata_schema_is_scanned_without_decoding_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["wav_file_name", "text", "listener_score"])
                writer.writeheader()
                writer.writerow({"wav_file_name": "a.wav", "text": "caption", "listener_score": "4"})
                writer.writerow({"wav_file_name": "b.wav", "text": "another", "listener_score": ""})
            result = scan_auxiliary_csv(path)
            self.assertEqual(result["headers"], ["wav_file_name", "text", "listener_score"])
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["blank_field_counts"]["listener_score"], 1)

    def test_metadata_scan_reports_rows_duplicates_and_missing_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "wav"
            audio.mkdir()
            wavfile.write(audio / "a.wav", 16_000, np.zeros(16, dtype=np.int16))
            csv_path = root / "validation_average.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["wav_file_name", "text", "average_score", "model_ID", "model_name"],
                )
                writer.writeheader()
                writer.writerow({"wav_file_name": "a.wav", "text": "caption", "average_score": "7", "model_ID": "m1", "model_name": "Model"})
                writer.writerow({"wav_file_name": "a.wav", "text": "caption", "average_score": "7", "model_ID": "m1", "model_name": "Model"})
                writer.writerow({"wav_file_name": "missing.wav", "text": "text", "average_score": "nan", "model_ID": "m2", "model_name": "Other"})
            result = scan_split_metadata(csv_path, audio, "validation", inspect_wav_headers=True)
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["duplicate_wave_filenames"], ["a.wav"])
            self.assertEqual(result["missing_audio_files"], ["missing.wav"])
            self.assertEqual(result["nonfinite_labels"], 1)
            self.assertEqual(result["model_id_counts"]["m1"], 2)
            self.assertEqual(result["subtype_counts"]["PCM_16"], 2)
            self.assertEqual(result["container_format_counts"]["WAV"], 2)
            self.assertEqual(len(result["rows"]), 3)

    def test_failure_classification_and_json_report_preserve_deferred_checks(self):
        self.assertEqual(classify_failure(FileNotFoundError("wav missing")), "MISSING_AUDIO_FILE")
        self.assertEqual(classify_failure(AssertionError("numerical mismatch")), "WAVEFORM_NUMERICAL_PARITY_FAILURE")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = {
                "status": "PARTIAL",
                "model_construction": False,
                "checkpoint_deserialization": False,
                "gpu_inference": False,
                "real_caption_parity": "PASS",
            }
            write_json_report(report, path)
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
