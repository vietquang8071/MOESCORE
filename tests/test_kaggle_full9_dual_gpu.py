"""Bounded regression checks for device-aware dual-worker extraction."""
import json
from pathlib import Path
import tempfile
import threading
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import extract_moe_feature_store as extraction
from scripts import kaggle_full9_dual_gpu as dual
from scripts import kaggle_full9_artifacts as artifacts
from data.moe_expert_cache import build_expert_cache_identity
from test_kaggle_feature_extraction import audio_fixture
from scripts.kaggle_moe_pipeline import generate_configs


class DualGpuTests(unittest.TestCase):
    def test_python312_dependency_plan_uses_numpy1_and_native_model_requirements(self):
        requirements, bypass_legacy_metadata = dual.dependency_plan_for_python((3, 12))
        body = requirements.read_text()
        self.assertIn("numpy==1.26.4", body)
        self.assertIn("msclap==1.3.4", body)
        self.assertIn("laion-clap==1.1.7", body)
        self.assertFalse(bypass_legacy_metadata)
        requirements313, bypass313 = dual.dependency_plan_for_python((3, 13))
        self.assertTrue(bypass313)
        self.assertIn("numpy>=2.1,<3", requirements313.read_text())

    def test_runtime_accepts_python312_and_still_requires_two_gpus(self):
        with patch.object(dual.sys, "version_info", (3, 12, 15)), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_name", side_effect=["GPU0", "GPU1"]):
            self.assertEqual(dual.require_kaggle_dual_gpu(), {"cuda:0": "GPU0", "cuda:1": "GPU1"})

    def test_assignment_requires_two_real_gpu_devices_and_covers_every_expert(self):
        order = extraction.EXPERT_PROFILES["full_9"]
        mapping = dual.default_worker_devices(order)
        self.assertEqual(set(mapping), set(order))
        self.assertEqual(set(mapping.values()), {"cuda:0", "cuda:1"})
        with self.assertRaises(ValueError):
            dual.validate_worker_devices({name: "cuda:0" for name in order}, order, require_two=True)
        with self.assertRaises(ValueError):
            dual.validate_worker_devices({**mapping, "extra": "cuda:0"}, order, require_two=True)

    def test_scheduler_runs_one_expert_per_gpu_and_stops_after_failure(self):
        order = ("a", "b", "c", "d")
        mapping = {"a": "cuda:0", "b": "cuda:1", "c": "cuda:0", "d": "cuda:1"}
        calls = []
        def runner(expert, device):
            calls.append((expert, device))
            if expert == "a":
                raise RuntimeError("failed a")
            return {"status": "PASS", "expert": expert, "device": device}
        with self.assertRaisesRegex(RuntimeError, "failed a"):
            dual.run_assigned_workers(order, mapping, runner)
        self.assertNotIn("c", [item[0] for item in calls])
        self.assertEqual(len(calls), len(set(item[0] for item in calls)))

    def test_scheduler_uses_both_gpu_queues_concurrently(self):
        order = ("a", "b")
        mapping = {"a": "cuda:0", "b": "cuda:1"}
        barrier = threading.Barrier(2, timeout=2)
        seen = []
        def runner(expert, device):
            seen.append((expert, device))
            barrier.wait()
            return {"status": "PASS", "expert": expert}
        reports = dual.run_assigned_workers(order, mapping, runner)
        self.assertEqual(set(reports), set(order))
        self.assertEqual(set(device for _, device in seen), {"cuda:0", "cuda:1"})

    def test_worker_rejects_wrong_assigned_device(self):
        job = {"expert_order": ["expert1"], "worker_devices": {"expert1": "cuda:1"}}
        with self.assertRaisesRegex(ValueError, "assigned"):
            extraction.validate_worker_device(job, "expert1", "cuda:0")
        extraction.validate_worker_device(job, "expert1", "cuda:1")
        extraction.validate_worker_device({"expert_order": ["expert1"], "inference_device": "cpu"}, "expert1", "cpu")

    def test_cache_archive_restores_exact_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "source/feature_cache"
            (cache / "experts/expert1").mkdir(parents=True)
            (cache / "experts/expert1/shard.npz").write_bytes(b"validated shard bytes")
            upload = root / "upload"
            manifest = artifacts.package_feature_cache(cache.parent, upload)
            self.assertEqual(set(p.name for p in upload.iterdir()), {"feature_cache.tar", "upload_manifest.json"})
            restored = root / "restored"
            artifacts.restore_feature_cache(upload, restored)
            self.assertEqual((restored / "feature_cache/experts/expert1/shard.npz").read_bytes(), b"validated shard bytes")
            with (upload / "feature_cache.tar").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                artifacts.restore_feature_cache(upload, root / "other")

    def test_archive_rejects_unrelated_upload_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source/feature_cache").mkdir(parents=True)
            upload = root / "upload"
            upload.mkdir()
            (upload / "token.txt").write_text("secret")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                artifacts.package_feature_cache(root / "source", upload)

    def test_upload_checks_returned_remote_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source/feature_cache").mkdir(parents=True)
            upload = root / "upload"
            artifacts.package_feature_cache(root / "source", upload)
            calls = []
            fake = SimpleNamespace(
                dataset_upload=lambda **kwargs: calls.append(kwargs),
            )
            api = self._upload_verification_api(upload)
            with patch.dict(sys.modules, {"kagglehub": fake}):
                report = artifacts.upload_feature_cache("owner/slug", upload, version_notes="test", api=api)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(calls[0]["handle"], "owner/slug")
            self.assertEqual(calls[0]["local_dataset_dir"], str(upload))

    def test_upload_rejects_corrupted_remote_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source/feature_cache").mkdir(parents=True)
            upload = root / "upload"
            artifacts.package_feature_cache(root / "source", upload)
            corrupt = root / "corrupt.tar"
            corrupt.write_bytes(b"corrupt remote bytes")
            fake = SimpleNamespace(dataset_upload=lambda **kwargs: None)
            api = self._upload_verification_api(upload, corrupt_archive=corrupt)
            with patch.dict(sys.modules, {"kagglehub": fake}):
                with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                    artifacts.upload_feature_cache("owner/slug", upload, api=api)

    def test_existing_public_dataset_is_rejected_before_upload(self):
        api = SimpleNamespace(is_private=False)
        api.authenticate = lambda: None
        def dataset_metadata(handle, path):
            metadata = Path(path) / "dataset-metadata.json"
            metadata.write_text(json.dumps({"isPrivate": api.is_private}))
            return str(metadata)
        api.dataset_metadata = dataset_metadata
        with self.assertRaisesRegex(ValueError, "public"):
            artifacts.check_private_upload_target("owner/slug", api=api)
        api.is_private = True
        self.assertTrue(artifacts.check_private_upload_target("owner/slug", api=api))

    @staticmethod
    def _upload_verification_api(upload, corrupt_archive=None):
        api = SimpleNamespace(authenticate=lambda: None)

        def dataset_metadata(handle, path):
            metadata = Path(path) / "dataset-metadata.json"
            metadata.write_text(json.dumps({"isPrivate": True}))
            return str(metadata)

        def dataset_download_file(handle, file_name, path, force, quiet, unzip):
            source = corrupt_archive if file_name == "feature_cache.tar" and corrupt_archive else upload / file_name
            return Path(path, file_name).write_bytes(source.read_bytes())

        api.dataset_metadata = dataset_metadata
        api.dataset_download_file = dataset_download_file
        return api

    def test_prepared_job_fingerprints_each_experts_assigned_device(self):
        order = extraction.EXPERT_PROFILES["full_9"]
        mapping = dual.default_worker_devices(order)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_fixture(root / "data", count=1)
            for file in ("laion-clap.pt", "mga-clap.pt", "checkpoint-30.pth", "BEATs_iter3_plus_AS2M.pt", "CLAP_weights_2023.pth"):
                path = root / "weights/pretrained" / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.encode())
            for expert in ("expert1", "expert2", "expert3", "expert4"):
                path = root / "weights/finetuned_weights" / f"{expert}_best.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expert.encode())
            (root / "weights/best.pt").write_bytes(b"e34")
            config, _, _ = generate_configs(Path(__file__).resolve().parents[1], root / "data", root / "weights",
                root / "hf", root / "output", root / "runs", root / "configs", profile="full_9")
            observed = {}
            def identity(name, configuration, *, device):
                observed[name] = device
                return build_expert_cache_identity(expert_name=name,
                    checkpoint_provenance={"fixture":name}, implementation_provenance={"fixture":name},
                    preprocessing_provenance={"fixture":name}, input_contract={"fixture":name},
                    inference_policy={"device":{"requested_device":device}},
                    score_contract={"fixture":name}, relevant_dependencies={"fixture":name})
            with patch.object(extraction, "compute_expert_cache_identity", side_effect=identity):
                _, job, _ = extraction.prepare_job(config, root / "data", root / "output", limit=1,
                    profile="full_9", worker_devices=mapping, device="cuda:0")
            self.assertEqual(observed, mapping)
            self.assertEqual(job["worker_devices"], mapping)

    def test_notebook_has_executable_stages_and_single_editable_configuration(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/MOESCORE_Full9_Kaggle.ipynb"
        document = json.loads(notebook.read_text())
        headings = ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "markdown"]
        for index in range(17):
            self.assertTrue(any(h.startswith(f"## {index:02d}.") for h in headings), index)
        code = ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(sum(line.startswith("DATASET_ROOT = ") for cell in code for line in cell.splitlines()), 1)
        self.assertTrue(any("kaggle_full9_dual_gpu.py" in cell for cell in code))
        self.assertTrue(any("dependency_plan_for_python" in cell for cell in code))
        self.assertTrue(any("if legacy_no_deps:" in cell for cell in code))
        self.assertEqual(document["metadata"]["language_info"]["version"], "3.12")
        self.assertTrue(any("package_feature_cache" in cell for cell in code))
        self.assertTrue(any("upload_feature_cache" in cell for cell in code))
        self.assertNotIn("/home/vitquay1708", "\n".join(code))
        for cell in code:
            compile(cell, str(notebook), "exec")


if __name__ == "__main__":
    unittest.main()
