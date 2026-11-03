"""Lightweight tests for the real pairwise expert runtime verifier."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest

import torch

from scripts.verify_multi_expert_runtime import (
    ScenarioReport,
    assess_four_expert_resource_safety,
    classify_failure,
    compare_tensors,
    inspect_module_origins,
    memory_snapshot,
    parse_order,
)


class ScenarioParsingTests(unittest.TestCase):
    def test_both_required_orders_are_accepted(self):
        self.assertEqual(parse_order("expert2,expert4"), ("expert2", "expert4"))
        self.assertEqual(parse_order("expert4,expert2"), ("expert4", "expert2"))

    def test_invalid_names_duplicate_and_incomplete_orders_are_rejected(self):
        for value in (
            "expert1,expert4",
            "expert2,expert2",
            "expert2",
            "expert2,expert4,expert3",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_order(value)


class StageReportTests(unittest.TestCase):
    def test_stage_order_and_partial_report_survive_failure(self):
        report = ScenarioReport(("expert2", "expert4"), device="cpu", seed=2027)
        report.record_stage("PROCESS_START", memory={"cpu": {"rss_bytes": 1}})
        report.record_stage("AFTER_FIRST_FACTORY", details={"expert": "expert2"})
        report.fail("second_factory", RuntimeError("synthetic factory failure"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "result.json"
            report.write(path)
            saved = json.loads(path.read_text())
        self.assertEqual(saved["status"], "FACTORY_INITIALIZATION_ERROR")
        self.assertEqual(saved["last_completed_stage"], "AFTER_FIRST_FACTORY")
        self.assertEqual(saved["failed_stage"], "second_factory")
        self.assertEqual([stage["name"] for stage in saved["stages"]], ["PROCESS_START", "AFTER_FIRST_FACTORY"])
        self.assertIn("traceback", saved["failure"])


class MemoryReportTests(unittest.TestCase):
    def test_cpu_memory_fields_are_measured_or_explicitly_unavailable(self):
        snapshot = memory_snapshot("cpu")
        self.assertIn("process_rss_bytes", snapshot["cpu"])
        self.assertIn("process_peak_rss_bytes", snapshot["cpu"])
        self.assertGreater(snapshot["cpu"]["process_rss_bytes"], 0)
        self.assertIn("system_memory_available_bytes", snapshot["cpu"])
        self.assertIn("cuda", snapshot)
        if not torch.cuda.is_available():
            self.assertIsNone(snapshot["cuda"]["allocated_bytes"])

    def test_cuda_metrics_are_not_reported_as_zero_when_unavailable(self):
        snapshot = memory_snapshot("cpu")
        if not torch.cuda.is_available():
            for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "free_bytes", "total_bytes"):
                self.assertIsNone(snapshot["cuda"][key])


class FailureClassificationTests(unittest.TestCase):
    def test_checkpoint_identity_restoration_import_and_oom_are_distinguished(self):
        self.assertEqual(classify_failure(ValueError("UniLM source fingerprint mismatch"), "factory"), "CHECKPOINT_IDENTITY_MISMATCH")
        self.assertEqual(classify_failure(RuntimeError("strict checkpoint state mismatch"), "factory"), "CHECKPOINT_RESTORATION_ERROR")
        self.assertEqual(classify_failure(ImportError("models origin conflict"), "factory"), "IMPORT_ORIGIN_MISMATCH")

        class OutOfMemoryError(RuntimeError):
            pass

        self.assertEqual(classify_failure(OutOfMemoryError("CUDA out of memory"), "factory"), "CUDA_OUT_OF_MEMORY")
        self.assertEqual(classify_failure(FileNotFoundError("missing local file"), "resource_preflight"), "MISSING_LOCAL_RESOURCE")
        self.assertEqual(classify_failure(AssertionError("tensor mismatch"), "standalone_adapter_parity"), "NUMERICAL_PARITY_FAILURE")


class NumericalParityTests(unittest.TestCase):
    def test_matching_tensors_and_tolerance_metrics(self):
        expected = torch.tensor([[1.0, -2.0]])
        actual = expected + torch.tensor([[2e-7, 0.0]])
        metrics = compare_tensors(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["max_abs_diff"], float((actual - expected).abs().max()))

    def test_shape_nonfinite_and_numeric_differences_fail(self):
        expected = torch.tensor([[1.0]])
        self.assertFalse(compare_tensors(torch.tensor([[2.0]]), expected)["passed"])
        self.assertFalse(compare_tensors(torch.tensor([1.0]), expected)["passed"])
        self.assertFalse(compare_tensors(torch.tensor([[float("nan")]]), expected)["passed"])


class ModuleOriginTests(unittest.TestCase):
    def test_expected_origin_is_accepted_and_wrong_origin_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected_root = Path(tmp) / "vendor"
            expected_root.mkdir()
            module_file = expected_root / "models.py"
            module_file.write_text("# fixture\n")
            module = types.ModuleType("models")
            module.__file__ = str(module_file)
            module.__spec__ = types.SimpleNamespace(origin=str(module_file))
            registry = {"models": module}
            before = dict(registry)
            result = inspect_module_origins(
                {"models": expected_root}, modules=registry
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(registry, before)

            wrong_root = Path(tmp) / "wrong"
            wrong_root.mkdir()
            with self.assertRaises(ImportError):
                inspect_module_origins({"models": wrong_root}, modules=registry)
            self.assertEqual(registry, before)


class FourExpertRiskTests(unittest.TestCase):
    def test_missing_memory_measurements_prevent_full_coexistence_claim(self):
        result = assess_four_expert_resource_safety(
            pairwise_snapshot=None,
            additional_gpu_peak_bytes=100,
            additional_host_bytes=100,
        )
        self.assertEqual(result["status"], "NOT_ATTEMPTED_RESOURCE_RISK")

    def test_measured_headroom_gates_four_expert_attempt(self):
        safe = assess_four_expert_resource_safety(
            pairwise_snapshot={"cuda": {"free_bytes": 10_000}, "cpu": {"system_memory_available_bytes": 10_000}},
            additional_gpu_peak_bytes=1_000,
            additional_host_bytes=1_000,
            safety_margin_bytes=1_000,
        )
        unsafe = assess_four_expert_resource_safety(
            pairwise_snapshot={"cuda": {"free_bytes": 1_500}, "cpu": {"system_memory_available_bytes": 10_000}},
            additional_gpu_peak_bytes=1_000,
            additional_host_bytes=1_000,
            safety_margin_bytes=1_000,
        )
        self.assertEqual(safe["status"], "SAFE_TO_ATTEMPT")
        self.assertEqual(unsafe["status"], "NOT_ATTEMPTED_RESOURCE_RISK")


if __name__ == "__main__":
    unittest.main()
