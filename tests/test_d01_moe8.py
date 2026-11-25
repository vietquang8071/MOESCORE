import ast
import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.diagnostics.d01_moe8 import (
    FULL8_GATE_ARCHITECTURE,
    FULL8_INFERENCE_COMPAT_FORMAT_VERSION,
    FULL8_INFERENCE_COMPAT_SOURCE_FILES,
    analyze_split,
    build_prediction_rows,
    classification_counts_for_score_bin,
    denormalize_scores,
    load_gate_checkpoint_for_d01,
    normalize_raw_scores,
    prepare_format7_gate_payload,
    write_prediction_csv,
)
from scripts.diagnostics.d01_moe8 import (
    discover_frozen_artifacts,
    read_aligned_public_metadata,
    resolve_dataset_mount,
    validate_frozen_artifact_pair,
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

    def test_score_bin_class_counts_exclude_other_split_bins(self):
        scores = np.array([[-1.0, 1.0], [-0.1, 0.1]])
        weights = np.array([[1.0, 0.0], [0.5, 0.5]])
        analysis = analyze_split(scores, weights, labels_raw=[9.5, 5.0])
        self.assertEqual(
            classification_counts_for_score_bin(analysis, [9.5, 5.0], 9.0, 10.0, include_upper=True),
            {"A": 0, "B": 0, "C": 1, "D": 0},
        )

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

    def test_prediction_csv_preserves_sample_identity_order(self):
        ids = ["validation:second.wav", "validation:first.wav"]
        scores = np.array([[0.1, 0.2], [-0.2, 0.4]])
        weights = np.array([[0.8, 0.2], [0.25, 0.75]])
        analysis = analyze_split(scores, weights, labels_raw=[5.0, 6.0])
        rows = build_prediction_rows(
            ids, "validation", scores, weights, analysis, labels_raw=[5.0, 6.0],
            wave_file_names=["second.wav", "first.wav"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            write_prediction_csv(output, rows)
            with output.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual([row["sample_id"] for row in saved], ids)
        self.assertEqual([row["wave_file_name"] for row in saved], ["second.wav", "first.wav"])
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

    def test_gate_loader_accepts_notebook_compatibility_hash_argument(self):
        payload, checkpoint_source = self._format7_payload()
        payload["expert_checkpoint_provenance"] = {"expert8": "frozen-sha"}
        payload["config_sha256"] = "fixture-fingerprint"
        payload["feature_store_identity_sha256"] = "fixture-fingerprint"
        payload["expert_checkpoint_provenance_sha256"] = "fixture-fingerprint"
        payload["gating_state_dict"] = {"gate.weight": "frozen-gate-state"}
        runtime_source = dict(checkpoint_source)
        runtime_source["model/expert_adapters/expert8_m2d_clap_adapter.py"] = "f" * 64

        class Gate:
            def load_state_dict(self, state, *, strict):
                self.loaded_state = state
                self.strict = strict

        fake_torch = types.ModuleType("torch")
        fake_torch.load = lambda *_args, **_kwargs: payload
        fake_checkpoint_manager = types.ModuleType("training.checkpoint_manager")
        fake_checkpoint_manager.FULL8_CHECKPOINT_FORMAT_VERSION = 5
        fake_checkpoint_manager.MoeCheckpointManager = type("Manager", (), {})
        fake_checkpoint_manager.full8_source_code_sha256 = lambda: runtime_source
        validated_payloads = []
        fake_checkpoint_manager._validate_gate_payload = lambda value, _gate: validated_payloads.append(value)
        fake_checkpoint_manager._fingerprint = lambda *_args: "fixture-fingerprint"
        fake_training = types.ModuleType("training")
        fake_training.__path__ = []
        fake_training.checkpoint_manager = fake_checkpoint_manager
        gate = Gate()

        with patch.dict(sys.modules, {
            "torch": fake_torch,
            "training": fake_training,
            "training.checkpoint_manager": fake_checkpoint_manager,
        }), patch(
            "scripts.diagnostics.d01_moe8.prepare_format7_gate_payload",
            wraps=prepare_format7_gate_payload,
        ) as prepare:
            loaded, _, audit, origin = load_gate_checkpoint_for_d01(
                "frozen-gate.pt", gate=gate,
                expected_expert_checkpoint_provenance=payload["expert_checkpoint_provenance"],
                checkpoint_compat_source_hashes=checkpoint_source,
            )

        self.assertEqual(origin, "reconstructed_from_checkpoint_source_hashes_format7")
        self.assertEqual(gate.loaded_state, payload["gating_state_dict"])
        self.assertTrue(gate.strict)
        self.assertEqual(loaded["source_code_sha256"], payload["source_code_sha256"])
        self.assertEqual(validated_payloads[0]["source_code_sha256"], runtime_source)
        self.assertIn("model/expert_adapters/expert8_m2d_clap_adapter.py",
                      audit["different_or_unavailable_files"])
        self.assertIs(prepare.call_args.kwargs["checkpoint_compat_source_hashes"], checkpoint_source)

    def test_format7_gate_rejects_changed_gate_source(self):
        payload, current = self._format7_payload()
        payload["source_code_sha256"]["model/moe_gating_core.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "inference source code is incompatible"):
            prepare_format7_gate_payload(payload, current_source_hashes=current)

    def test_format7_gate_keeps_checkpoint_source_validation_with_runtime_loader_patch(self):
        payload, baseline = self._format7_payload()
        runtime = dict(baseline)
        runtime["model/expert_adapters/expert8_m2d_clap_adapter.py"] = "f" * 64
        _, _, audit, _ = prepare_format7_gate_payload(
            payload, current_source_hashes=runtime,
            checkpoint_compat_source_hashes=baseline,
        )
        self.assertIn(
            "model/expert_adapters/expert8_m2d_clap_adapter.py",
            audit["different_or_unavailable_files"],
        )
        bad_baseline = dict(baseline)
        bad_baseline["model/moe_gating_core.py"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "inference source code is incompatible"):
            prepare_format7_gate_payload(
                payload, current_source_hashes=runtime,
                checkpoint_compat_source_hashes=bad_baseline,
            )

    def test_notebook_is_frozen_artifact_only_public_split_inference(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        metadata = json.loads((root / "kaggle/d01_moe8/kernel-metadata.json").read_text())
        packaged = json.loads((root / "kaggle/d01_moe8/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        self.assertEqual(packaged, notebook)
        notebook_text = json.dumps(notebook)
        self.assertNotIn("XACLE2027_evaluation_dataset", notebook_text)
        self.assertNotIn("XACLE2027_evaluation_dataset", json.dumps(metadata))
        cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        combined = "\n".join(cells)
        for obsolete in ("moe8-bestgate", "moescore-full9-feature-cache", "xacle-weights", "recovery"):
            self.assertNotIn(obsolete, combined)
            self.assertNotIn(obsolete, json.dumps(metadata))
        self.assertIn("moescore-full8-artifacts", combined)
        self.assertIn("xacle-2027-data", combined)
        self.assertIn("discover_frozen_artifacts", combined)
        self.assertIn("MoeTrainingDataset", combined)
        self.assertIn("load_gate_checkpoint_for_d01", combined)
        self.assertIn("torch.inference_mode", combined)
        self.assertNotIn("scripts.extract", combined)
        self.assertNotIn("snapshot_download", combined)
        self.assertNotIn("AutoTokenizer", combined)
        self.assertNotIn("AutoModel", combined)
        self.assertNotIn("optimizer.step", combined)
        self.assertNotIn("loss.backward", combined)
        code_nodes = [ast.parse(source) for cell, source in zip(notebook["cells"], cells)
                      if cell["cell_type"] == "code"]
        imports = [node for tree in code_nodes for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = " ".join(
            (alias.name for node in imports for alias in node.names)
        ) + " " + " ".join(node.module or "" for node in imports if isinstance(node, ast.ImportFrom))
        self.assertNotIn("expert_adapters.expert", imported)
        self.assertNotIn("extract_expert", imported)
        self.assertEqual(metadata["dataset_sources"], [
            "qnv8071/moescore-full8-artifacts", "qnv8071/xacle-2027-data",
        ])
        self.assertEqual(metadata["id"], "qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_tpu"], "false")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertNotIn("machine_shape", metadata)

        source_cell = next(source for source in cells if "SOURCE_COMMIT =" in source)
        source_commit = next(ast.literal_eval(node.value) for tree in [ast.parse(source_cell)]
                             for node in ast.walk(tree)
                             if isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id == "SOURCE_COMMIT"
                                 for target in node.targets))
        self.assertEqual(len(source_commit), 40)
        source = subprocess.check_output(
            ["git", "show", f"{source_commit}:scripts/diagnostics/d01_moe8.py"], cwd=root, text=True
        )
        self.assertIn("def discover_frozen_artifacts(", source)

        parsed_calls = []
        for cell_index, source in enumerate(cells):
            if notebook["cells"][cell_index]["cell_type"] != "code":
                continue
            tree = ast.parse(source)
            parsed_calls.extend((cell_index, node) for node in ast.walk(tree) if isinstance(node, ast.Call))
        gate_calls = [(i, node) for i, node in parsed_calls
                      if isinstance(node.func, ast.Name) and node.func.id == "load_gate_checkpoint_for_d01"]
        self.assertEqual(len(gate_calls), 1)
        kwargs = {item.arg: item.value for item in gate_calls[0][1].keywords}
        self.assertIn("expected_expert_checkpoint_provenance", kwargs)
        self.assertIn("checkpoint_compat_source_hashes", kwargs)
        test_csv_write = next(i for i, node in parsed_calls if isinstance(node.func, ast.Name)
                              and node.func.id == "write_prediction_csv"
                              and "test_d01.csv" in ast.unparse(node))
        test_metadata_read = next(i for i, node in parsed_calls if isinstance(node.func, ast.Name)
                                  and node.func.id == "read_aligned_public_metadata"
                                  and any(isinstance(arg, ast.Constant) and arg.value == "test"
                                          for arg in node.args)
                                  and any(k.arg == "include_labels" and isinstance(k.value, ast.Constant)
                                          and k.value.value is True for k in node.keywords))
        self.assertLess(test_csv_write, test_metadata_read)

    def test_notebook_source_and_kernel_metadata_match_and_have_no_training_or_extraction_calls(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        packaged = json.loads((root / "kaggle/d01_moe8/XACLE2027_D01_MOE8_Kaggle.ipynb").read_text())
        self.assertEqual(packaged, notebook)


class D01FrozenArtifactTests(unittest.TestCase):
    def _write_artifacts(self, root, *, duplicate_gate=False):
        gate_root = root / "moe8_gate/moe8_gate"
        gate_root.mkdir(parents=True)
        (gate_root / "best_gate.pt").write_bytes(b"checkpoint")
        (gate_root / "moe_full8.yaml").write_text("data:\n  profile: full_8\n")
        (gate_root / "run_manifest.json").write_text("{}")
        if duplicate_gate:
            other = root / "duplicate"
            other.mkdir()
            (other / "best_gate.pt").write_bytes(b"other")
        store = root / "moescore_full8_store/full8_store"
        for split in ("train", "validation", "test"):
            (store / split).mkdir(parents=True)
            (store / split / "manifest.json").write_text(json.dumps({"expert_profile": "full_8"}))
        return gate_root, store

    def _paired_fixture(self):
        source = {"model/moe_gating_core.py": "a" * 64}
        provenance = {f"expert{i}": {"checkpoint_sha256": f"{i:064x}"} for i in range(1, 9)}
        identities = {"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64}
        contract = {"core_assembly_contract": "full8_v1"}
        checkpoint_sha = "c" * 64
        payload = {
            "format_version": 7, "expert_profile": "full_8",
            "expert_order": [f"expert{i}" for i in range(1, 9)],
            "expert_feature_dimensions": {
                "expert1": 2049, "expert2": 4097, "expert3": 3840, "expert4": 1024,
                "expert5": 11267, "expert6": 4609, "expert7": 3073, "expert8": 3073,
            },
            "gate_input_dim": 33032, "gate_expert_count": 8,
            "gate_architecture": FULL8_GATE_ARCHITECTURE,
            "source_code_sha256": source,
            "expert_checkpoint_provenance": provenance,
            "feature_store_identity": {**identities, "preprocessing_contract": contract},
        }
        run = {
            "git_commit": "d" * 40, "expert_profile": "full_8",
            "gate_checkpoint_sha256": checkpoint_sha,
            "source_code_sha256": source,
            "expert_checkpoint_provenance": provenance,
            "train_store_identity_sha256": identities["train"],
            "validation_store_identity_sha256": identities["validation"],
        }
        manifests = {
            split: {
                "split": split, "expert_profile": "full_8",
                "expert_order": [f"expert{i}" for i in range(1, 9)],
                "feature_width": 33032, "score_width": 8,
                "feature_dimensions": {
                    "expert1": 2049, "expert2": 4097, "expert3": 3840, "expert4": 1024,
                    "expert5": 11267, "expert6": 4609, "expert7": 3073, "expert8": 3073,
                },
                "store_identity_sha256": identities[split],
                "source_manifest_sha256": {"train": "e", "validation": "f", "test": "0"}[split] * 64,
                "labels_present": split != "test",
                "checkpoint_provenance": provenance,
                "preprocessing_contract": contract,
                "sample_count": count,
                "shards": [{"count": count}],
                "assembly_identity": {"ordered_source_fingerprints": [
                    {"sample_id": f"{split}:same.wav", "wav_sha256": "a" * 64},
                ]},
            }
            for split, count in (("train", 7500), ("validation", 3000), ("test", 3000))
        }
        contracts = {split: contract for split in manifests}
        return checkpoint_sha, payload, run, manifests, source, contracts

    def test_resolves_only_the_requested_dataset_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "datasets/qnv8071/moescore-full8-artifacts"
            expected.mkdir(parents=True)
            self.assertEqual(resolve_dataset_mount(root, "moescore-full8-artifacts"), expected)

    def test_rejects_missing_or_ambiguous_dataset_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                resolve_dataset_mount(root, "moescore-full8-artifacts")
            (root / "moescore-full8-artifacts").mkdir()
            (root / "qnv8071/moescore-full8-artifacts").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_dataset_mount(root, "moescore-full8-artifacts")

    def test_discovers_nested_checkpoint_config_run_and_full8_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, store = self._write_artifacts(root)
            found = discover_frozen_artifacts(root)
            self.assertEqual(found, {
                "gate_checkpoint": gate / "best_gate.pt",
                "gate_config": gate / "moe_full8.yaml",
                "run_manifest": gate / "run_manifest.json",
                "feature_store": store,
            })

    def test_artifact_discovery_rejects_missing_or_ambiguous_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifacts(root)
            (root / "moe8_gate/moe8_gate/best_gate.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                discover_frozen_artifacts(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifacts(root, duplicate_gate=True)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                discover_frozen_artifacts(root)

    def test_artifact_discovery_rejects_missing_or_ambiguous_full8_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifacts(root)
            for split in ("train", "validation", "test"):
                (root / "moescore_full8_store/full8_store" / split / "manifest.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Full8"):
                discover_frozen_artifacts(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifacts(root)
            duplicate = root / "duplicate_store"
            for split in ("train", "validation", "test"):
                folder = duplicate / split
                folder.mkdir(parents=True)
                (folder / "manifest.json").write_text(json.dumps({"expert_profile": "full_8"}))
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                discover_frozen_artifacts(root)

    def test_valid_paired_format7_gate_and_three_splits_are_accepted(self):
        values = self._paired_fixture()
        validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                      preprocessing_contracts=values[5])

    def test_gate_checkpoint_hash_and_store_identity_mismatches_are_rejected(self):
        values = list(self._paired_fixture())
        values[0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                          preprocessing_contracts=values[5])
        values = list(self._paired_fixture())
        values[3]["validation"]["store_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "validation.*identity"):
            validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                          preprocessing_contracts=values[5])

    def test_incompatible_gate_architecture_order_dimensions_and_provenance_fail(self):
        for field, value, message in (
            ("gate_expert_count", 10, "architecture"),
            ("gate_input_dim", 33031, "architecture"),
            ("expert_order", [f"expert{i}" for i in range(8, 0, -1)], "architecture"),
        ):
            values = list(self._paired_fixture())
            values[1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                              preprocessing_contracts=values[5])
        values = list(self._paired_fixture())
        values[3]["test"]["checkpoint_provenance"] = {
            **values[3]["test"]["checkpoint_provenance"],
            "expert8": {"checkpoint_sha256": "f" * 64},
        }
        with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
            validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                          preprocessing_contracts=values[5])

    def test_invalid_full8_split_contracts_fail_without_rejecting_overlap(self):
        values = list(self._paired_fixture())
        validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                      preprocessing_contracts=values[5])
        self.assertEqual(sum(m["shards"][0]["count"] for m in values[3].values()), 13500)
        for field, value in (("expert_profile", "full_9"), ("feature_width", 33031),
                             ("score_width", 10), ("labels_present", True)):
            values = list(self._paired_fixture())
            values[3]["test"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                              preprocessing_contracts=values[5])

    def test_source_identity_must_match_run_manifest_checkpoint_and_verified_commit(self):
        values = list(self._paired_fixture())
        values[4] = {"model/moe_gating_core.py": "b" * 64}
        with self.assertRaisesRegex(ValueError, "source-code hashes"):
            validate_frozen_artifact_pair(*values[:4], verified_source_hashes=values[4],
                                          preprocessing_contracts=values[5])

    def test_official_metadata_aligns_by_filename_and_rejects_mismatched_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text(
                "wav_file_name,average_score,model_ID,model_name\n"
                "00001.wav,4.5,gen-a,A\n00000.wav,3.0,gen-b,B\n",
                encoding="utf-8",
            )
            result = read_aligned_public_metadata(
                path, "train", ["train:00000.wav", "train:00001.wav"],
                ["00000.wav", "00001.wav"], include_labels=True,
            )
            self.assertEqual(result["labels_raw"], [3.0, 4.5])
            self.assertEqual(result["model_ids"], ["gen-b", "gen-a"])
            with self.assertRaisesRegex(ValueError, "sample alignment|WAV filenames"):
                read_aligned_public_metadata(
                    path, "train", ["train:00000.wav", "train:00001.wav"],
                    ["00000.wav", "missing.wav"], include_labels=True,
                )

    def test_test_metadata_can_be_joined_without_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text("wav_file_name,average_score,model_name\na.wav,9.0,G\n", encoding="utf-8")
            result = read_aligned_public_metadata(
                path, "test", ["test:a.wav"], ["a.wav"], include_labels=False,
            )
            self.assertIsNone(result["labels_raw"])
            self.assertEqual(result["model_names"], ["G"])


if __name__ == "__main__":
    unittest.main()
