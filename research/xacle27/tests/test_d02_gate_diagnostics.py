import ast
import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import FULL8_EXPERT_ORDER, profile_contract
from model.moe_forward_integration import MoEForwardIntegration
from model.moe_gating_core import GatingNetwork
from research.xacle27.analysis.d02_gate_diagnostics import (
    build_per_sample_frame,
    compare_predictions,
    entropy_statistics,
    gate_weights_by_expert,
    generator_summary,
    infer_gate_weights,
    join_metadata_by_filename,
    score_bin_summary,
    validate_checkpoint_cache_provenance,
    validate_checkpoint_header,
    validate_full8_arrays,
)


class _FixedExpert(ExpertAdapter):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)

    def _forward(self, batch):
        raise AssertionError("forward_from_expert_outputs must not invoke adapters")


class D02GateDiagnosticsTests(unittest.TestCase):
    def test_uniform_weights_have_normalized_entropy_one(self):
        _, _, _ = profile_contract("full_8")
        stats = entropy_statistics(np.full((2, 8), 1 / 8, dtype=np.float64))
        self.assertTrue(np.allclose(stats["normalized_entropy"], 1.0, atol=1e-7))
        self.assertTrue(np.allclose(stats["effective_experts"], 8.0, atol=1e-6))

    def test_one_hot_weights_have_near_zero_normalized_entropy(self):
        weights = np.zeros((2, 8), dtype=np.float64)
        weights[:, 3] = 1.0
        stats = entropy_statistics(weights)
        self.assertTrue(np.all(stats["normalized_entropy"] < 1e-6))
        np.testing.assert_array_equal(stats["top1_expert_index"], [4, 4])

    def test_gate_weight_summary_contains_overall_and_score_bin_routing(self):
        rows = build_per_sample_frame(
            ["validation:a.wav", "validation:b.wav"], ["a.wav", "b.wav"], "validation",
            np.zeros((2, 8), dtype=np.float32),
            np.eye(8, dtype=np.float32)[[0, 7]], labels_raw=[0, 10],
        )
        summary = gate_weights_by_expert(rows)
        self.assertEqual(set(summary["group_type"]), {"overall", "score_bin"})
        overall = summary.loc[summary["group_type"] == "overall"]
        self.assertEqual(list(overall["expert"]), list(FULL8_EXPERT_ORDER))
        self.assertAlmostEqual(overall.loc[overall["expert"] == "expert1", "mean_weight"].iloc[0], 0.5)

    def test_full8_contract_uses_registry_width_order_and_eight_scores(self):
        order, _, width = profile_contract("full_8")
        self.assertEqual(tuple(order), tuple(FULL8_EXPERT_ORDER))
        features = np.zeros((2, width), dtype=np.float32)
        scores = np.zeros((2, 8), dtype=np.float32)
        weights = np.full((2, 8), 1 / 8, dtype=np.float32)
        validate_full8_arrays(features, scores, weights, expert_order=order)
        with self.assertRaisesRegex(ValueError, "expert order"):
            validate_full8_arrays(features, scores, weights, expert_order=tuple(reversed(order)))
        with self.assertRaisesRegex(ValueError, "feature width"):
            validate_full8_arrays(features[:, :-1], scores, weights)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_full8_arrays(features, scores[:, :7], weights)

    def test_full9_checkpoint_header_is_rejected(self):
        payload = {
            "expert_profile": "full_9",
            "expert_order": [*FULL8_EXPERT_ORDER, "expert9"],
            "gate_input_dim": 33032,
            "gate_expert_count": 9,
        }
        with self.assertRaisesRegex(ValueError, "Full8"):
            validate_checkpoint_header(payload)

    def test_checkpoint_cache_provenance_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_checkpoint_cache_provenance(
                {"expert_checkpoint_provenance": {"expert1": {"sha256": "a"}}},
                {"expert1": {"sha256": "b"}},
            )


    def test_gate_weights_are_probabilities_and_inference_does_not_update_parameters(self):
        gate = GatingNetwork(33032, 8, profile="full_8")
        before = {name: value.detach().clone() for name, value in gate.state_dict().items()}
        weights = infer_gate_weights(gate, torch.zeros((3, 33032)), batch_size=2, device="cpu")
        self.assertEqual(tuple(weights.shape), (3, 8))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(3), atol=1e-6, rtol=0))
        self.assertFalse(gate.training)
        self.assertTrue(all(parameter.grad is None for parameter in gate.parameters()))
        self.assertTrue(all(torch.equal(value, gate.state_dict()[name]) for name, value in before.items()))


    def test_learned_gate_prediction_matches_original_moe8_forward(self):
        order, dimensions, width = profile_contract("full_8")
        gate = GatingNetwork(width, 8, profile="full_8")
        experts = {name: _FixedExpert(dimensions[name]) for name in order}
        integration = MoEForwardIntegration(experts, gating_network=gate, profile="full_8").eval()
        features = torch.randn(2, width)
        scores = torch.linspace(-0.8, 0.8, 16).reshape(2, 8)
        outputs = {}
        offset = 0
        for index, name in enumerate(order):
            size = dimensions[name]
            outputs[name] = (scores[:, index : index + 1], features[:, offset : offset + size])
            offset += size
        original, _ = integration.forward_from_expert_outputs(outputs)
        weights = infer_gate_weights(gate, features, batch_size=2, device="cpu")
        self.assertTrue(torch.allclose(original[:, 0], (scores * weights).sum(1), atol=1e-6, rtol=1e-6))


    def test_controls_and_raw_score_conversion_are_single_pass(self):
        scores = np.zeros((2, 8), dtype=np.float32)
        scores[:, 0] = -1
        scores[:, 1] = 1
        weights = np.zeros((2, 8), dtype=np.float32)
        weights[0, 0] = 1
        weights[1, 1] = 1
        rows = build_per_sample_frame(
            ["validation:a.wav", "validation:b.wav"], ["a.wav", "b.wav"], "validation",
            scores, weights, labels_raw=[0, 10],
        )
        self.assertEqual(rows.loc[0, "predicted_score"], 0)
        self.assertEqual(rows.loc[1, "predicted_score"], 10)
        self.assertEqual(rows.loc[0, "expert_1_score"], 0)
        self.assertEqual(rows.loc[1, "expert_2_score"], 10)
        self.assertTrue(np.allclose(rows["uniform_prediction_score"], 5))
        self.assertEqual(rows.loc[0, "hard_top1_prediction_score"], 0)
        self.assertEqual(rows.loc[1, "hard_top1_prediction_score"], 10)
        self.assertEqual(rows.loc[0, "oracle_diagnostic_only_score"], 0)


    def test_score_ten_is_in_last_score_bin(self):
        rows = build_per_sample_frame(
            ["validation:a.wav"], ["a.wav"], "validation",
            np.ones((1, 8), dtype=np.float32), np.full((1, 8), 1 / 8, dtype=np.float32),
            labels_raw=[10],
        )
        summary = score_bin_summary(rows)
        last = summary.loc[summary["ground_truth_bin"] == "[9,10]"].iloc[0]
        self.assertEqual(last["sample_count"], 1)
        self.assertEqual(last["mean_ground_truth"], 10)


    def test_metadata_join_aligns_rows_and_rejects_missing_rows(self):
        predictions = pd.DataFrame({"sample_id": ["validation:a.wav"], "wave_file_name": ["a.wav"]})
        metadata = pd.DataFrame({"wav_file_name": ["a.wav"], "model_id": ["g1"]})
        joined = join_metadata_by_filename(predictions, metadata)
        self.assertEqual(joined.loc[0, "model_id"], "g1")
        with self.assertRaisesRegex(ValueError, "missing"):
            join_metadata_by_filename(predictions, metadata.assign(wav_file_name="missing.wav"))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            join_metadata_by_filename(
                predictions, pd.concat([metadata, metadata], ignore_index=True)
            )


    def test_constant_generator_labels_have_undefined_srcc(self):
        rows = build_per_sample_frame(
            [f"validation:{i}.wav" for i in range(3)], [f"{i}.wav" for i in range(3)], "validation",
            np.zeros((3, 8), dtype=np.float32), np.full((3, 8), 1 / 8, dtype=np.float32),
            labels_raw=[5, 5, 5], model_ids=["constant"] * 3,
        )
        summary = generator_summary(rows).iloc[0]
        self.assertTrue(pd.isna(summary["srcc"]))
        self.assertEqual(summary["srcc_status"], "undefined_constant_scores")

    def test_unlabeled_public_generator_summary_keeps_routing_and_skips_label_metrics(self):
        rows = build_per_sample_frame(
            ["test:a.wav", "test:b.wav"], ["a.wav", "b.wav"], "test",
            np.zeros((2, 8), dtype=np.float32), np.full((2, 8), 1 / 8, dtype=np.float32),
            model_ids=["g1", "g1"],
        )
        summary = generator_summary(rows).iloc[0]
        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["mean_normalized_entropy"], 1.0)
        self.assertTrue(pd.isna(summary["srcc"]))
        self.assertEqual(summary["srcc_status"], "undefined_no_labels")
        self.assertTrue(pd.isna(summary["mse"]))
        self.assertTrue(pd.isna(summary["mae"]))


    def test_comparison_uses_matched_controls_and_reports_gap_inversions(self):
        scores = np.zeros((3, 8), dtype=np.float32)
        scores[:, 0] = [-1, 0, 1]
        weights = np.zeros_like(scores)
        weights[:, 0] = 1
        rows = build_per_sample_frame(
            [f"validation:{i}.wav" for i in range(3)], [f"{i}.wav" for i in range(3)], "validation",
            scores, weights, labels_raw=[0, 5, 10],
        )
        metrics = compare_predictions(rows, bootstrap_samples=10, seed=5)
        self.assertGreaterEqual(set(metrics["prediction_method"]), {"learned", "uniform", "hard_top1"})
        learned = metrics.loc[metrics["prediction_method"] == "learned"].iloc[0]
        self.assertEqual(learned["sample_count"], 3)
        self.assertEqual(learned["inversion_rate_gap_1"], 0)


    def test_notebook_json_code_compiles_and_training_is_disabled(self):
        root = Path(__file__).parents[3]
        paths = [
            root / "notebooks/XACLE2027_D02_MOE8_Gate_Diagnostics_Kaggle.ipynb",
            root / "notebooks/kaggle/xacle2027-d02-moe8-gate-diagnostics/XACLE2027_D02_MOE8_Gate_Diagnostics_Kaggle.ipynb",
        ]
        sources = []
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertGreaterEqual(len(code_cells), 10)
            for cell in code_cells:
                compile("".join(cell["source"]), str(path), "exec")
            sources.append("\n".join("".join(cell["source"]) for cell in code_cells))
        self.assertEqual(sources[0], sources[1])
        source = sources[0]
        self.assertIn('USE_PRECOMPUTED_FULL8_STORE = True', source)
        self.assertIn("TRAIN_GATE = False", source)
        self.assertIn("TRAIN_EXPERTS = False", source)
        self.assertIn("RUN_FEATURE_EXTRACTION = False", source)
        self.assertIn("RUN_FULL7_ASSEMBLY = False", source)
        self.assertIn("RUN_FULL8_ASSEMBLY = False", source)
        self.assertIn('RUN_PUBLIC_TEST_ANALYSIS = True', source)
        self.assertNotIn("XACLE2027_evaluation_dataset", source)
        self.assertNotIn("extract_expert8_features.py", source)
        self.assertNotIn("build_full7_feature_store.py", source)
        self.assertNotIn("build_full8_feature_store.py", source)
        self.assertNotIn("snapshot_download", source)

    def test_notebook_uses_precomputed_full8_store_and_pinned_compatible_source(self):
        path = Path(__file__).parents[3] / "notebooks/XACLE2027_D02_MOE8_Gate_Diagnostics_Kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        self.assertIn('"qnv8071/moescore-full8-artifacts"', source)
        self.assertIn('MOE8_SOURCE_COMMIT = "68124d0e57005d29149f659780ab749d14e09532"', source)
        self.assertIn('D02_HELPER_COMMIT = "e337d5d34f0e3fd22c834be98fb62254b951e698"', source)
        self.assertIn("resolve_precomputed_full8_artifacts", source)
        self.assertIn("load_for_inference", source)
        self.assertIn("validate_feature_store", source)
        self.assertIn('("train", "validation", "test")', source)
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("gate.eval()", source)
        self.assertIn("SKIP_CROSS_SPLIT_AUDIO_OVERLAP_CHECK = True", source)
        self.assertIn('"cross_split_audio_overlap": {"status": "SKIPPED"', source)
        self.assertNotIn("_validate_full7_overlap_identity", source)

    def test_kaggle_kernel_metadata_updates_existing_public_d02_kernel_only(self):
        path = Path(__file__).parents[3] / "notebooks/kaggle/xacle2027-d02-moe8-gate-diagnostics/kernel-metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], "qnv8071/xacle2027-d02-moe8-gate-diagnostics")
        self.assertEqual(metadata["title"], "XACLE2027 D02 MOE8 Gate Diagnostics")
        self.assertEqual(metadata["kernel_type"], "notebook")
        self.assertEqual(
            set(metadata["dataset_sources"]),
            {"qnv8071/moescore-full8-artifacts", "qnv8071/xacle-2027-data"},
        )
        self.assertFalse(metadata["enable_gpu"])
        self.assertNotIn("machine_shape", metadata)

    def test_subprocess_helper_prints_success_failure_and_launch_errors(self):
        path = Path(__file__).parents[3] / "notebooks/XACLE2027_D02_MOE8_Gate_Diagnostics_Kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def run_subprocess(command" in "".join(cell["source"])
        )
        parsed = ast.parse(code)
        helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                      and node.name == "run_subprocess")
        module = ast.Module(body=[helper], type_ignores=[])
        namespace = {"subprocess": subprocess, "shlex": shlex, "sys": sys, "Path": Path}
        exec(compile(module, str(path), "exec"), namespace)
        run_subprocess = namespace["run_subprocess"]
        with tempfile.TemporaryDirectory() as temporary:
            stdout, stderr = io.StringIO(), io.StringIO()
            log_path = Path(temporary) / "command.log"
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                run_subprocess([sys.executable, "-c", "print('visible output')"], log_path=log_path)
            self.assertIn("visible output", stdout.getvalue())
            self.assertIn("visible output", log_path.read_text())

            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_subprocess([sys.executable, "-c", "import sys; print('command error', file=sys.stderr); sys.exit(2)"])
            self.assertIn("command error", stderr.getvalue())
            self.assertIn("exit code 2", stderr.getvalue())

            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                with self.assertRaises(FileNotFoundError):
                    run_subprocess([str(Path(temporary) / "missing-command")])
            self.assertIn("could not start", stderr.getvalue().lower())

