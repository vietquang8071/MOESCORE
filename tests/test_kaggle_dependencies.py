import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.install_kaggle_dependencies import build_install_command, main, protected, write_runtime_constraints

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"torch": "2.5.1+cu121", "torchaudio": "2.5.1+cu121", "torchvision": "0.20.1+cu121",
            "nvidia-cublas-cu12": "12.1.3.1", "triton": "3.1.0", "numpy": "1.26.4"}


class KaggleDependencyTests(unittest.TestCase):
    def test_requirements_cover_audio_experts_without_listing_the_gpu_stack(self):
        lines = (ROOT / "requirements-kaggle.txt").read_text().splitlines()
        names = {__import__("re").split(r"[<>=!~\[]", line)[0].lower()
                 for line in lines if line.strip() and not line.startswith("#")}
        self.assertTrue({"laion-clap", "torchlibrosa", "nnaudio", "transformers", "timm", "matplotlib"}.issubset(names))
        self.assertFalse(any(protected(name) for name in names))

    def test_constraints_preserve_torch_and_installed_cuda_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.txt"
            frozen = write_runtime_constraints(path, VERSIONS)
            self.assertEqual(frozen["torch"], VERSIONS["torch"])
            self.assertEqual(frozen["nvidia-cublas-cu12"], VERSIONS["nvidia-cublas-cu12"])
            self.assertNotIn("numpy", frozen)
            self.assertIn("torchaudio==2.5.1+cu121", path.read_text())
            with self.assertRaisesRegex(RuntimeError, "Missing preinstalled"):
                write_runtime_constraints(path, {"torch": "2.5.1"})

    def test_offline_install_uses_constraints_and_rejects_gpu_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = build_install_command(ROOT / "requirements-kaggle.txt", "runtime.txt", wheelhouse=tmp)
            self.assertIn("--no-index", command)
            self.assertEqual(command[command.index("--find-links") + 1], tmp)
            self.assertEqual(command[command.index("--constraint") + 1], "runtime.txt")
            self.assertNotIn("--upgrade", command)
            for spec in ("torch>=2.6", "torchvision==0.21", "nvidia-cublas-cu12", "triton", "--index-url"):
                with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "cannot replace"):
                    build_install_command(ROOT / "requirements-kaggle.txt", "runtime.txt", extras=[spec])

    def test_installer_reports_changes_and_checks_the_gpu_stack_after_pip(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--constraints", str(Path(tmp) / "runtime.txt"), "--report-json", str(Path(tmp) / "report.json")]
            after = {**VERSIONS, "nnaudio": "0.3.4"}
            with patch("scripts.install_kaggle_dependencies.installed_versions", side_effect=[VERSIONS, after]), \
                 patch("scripts.install_kaggle_dependencies.subprocess.run") as pip:
                self.assertEqual(main(args), 0)
                pip.assert_called_once()
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["changed_distributions"]["nnaudio"]["after"], "0.3.4")
            with patch("scripts.install_kaggle_dependencies.installed_versions", side_effect=[VERSIONS, {**after, "torch": "2.6.0"}]), \
                 patch("scripts.install_kaggle_dependencies.subprocess.run"):
                with self.assertRaisesRegex(RuntimeError, "GPU stack changed"):
                    main(args)


if __name__ == "__main__":
    unittest.main()
