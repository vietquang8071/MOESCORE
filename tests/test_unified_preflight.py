"""Tests for the model-free unified dataset diagnostic."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_unified_dataset import run_preflight


class UnifiedPreflightTests(unittest.TestCase):
    def test_synthetic_preflight_reports_batch_contract_and_defers_real_parity(self):
        report = run_preflight(
            argparse.Namespace(
                config=None,
                split="validation",
                metadata_csv=None,
                audio_dir=None,
                offline=True,
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["synthetic_tests"]), {"1", "2", "4"})
        self.assertFalse(report["model_construction"])
        self.assertFalse(report["checkpoint_deserialization"])
        self.assertEqual(report["real_waveform_parity"], "NOT_VERIFIED_LOCAL_09_1")
        self.assertEqual(report["real_caption_parity"], "NOT_VERIFIED_LOCAL_09_1")

    def test_metadata_preflight_reads_schema_and_checks_audio_directory_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "validation_average.csv"
            csv_path.write_text(
                "wave_file_name,text,average_semantic_alignement_score,model_ID,model_name\n",
                encoding="utf-8",
            )
            audio_dir = root / "wav" / "validation"
            audio_dir.mkdir(parents=True)
            report = run_preflight(
                argparse.Namespace(
                    config=None,
                    split="validation",
                    metadata_csv=csv_path,
                    audio_dir=audio_dir,
                    offline=True,
                )
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["dataset_schema"]["required_columns_status"], "PASS")
        self.assertIn("model_ID", report["dataset_schema"]["optional_model_metadata"])
        self.assertFalse(report["split_configuration"]["audio_decoding_performed"])

    def test_missing_explicit_dataset_resource_is_reported_as_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = run_preflight(
                argparse.Namespace(
                    config=None,
                    split="validation",
                    metadata_csv=root / "missing.csv",
                    audio_dir=root / "missing_audio",
                    offline=True,
                )
            )
        self.assertEqual(report["status"], "BLOCKED_MISSING_DATA_RESOURCE")
        self.assertTrue(report["errors"])
        self.assertFalse(report["model_construction"])

    def test_json_report_is_machine_readable_and_has_no_heavy_loading_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "report.json"
            report = run_preflight(
                argparse.Namespace(
                    config=None,
                    split="validation",
                    metadata_csv=None,
                    audio_dir=None,
                    offline=True,
                )
            )
            target.write_text(json.dumps(report), encoding="utf-8")
            loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded["status"], "PASS")
        self.assertIs(loaded["model_construction"], False)
        self.assertIs(loaded["checkpoint_deserialization"], False)


if __name__ == "__main__":
    unittest.main()
