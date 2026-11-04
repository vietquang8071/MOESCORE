import csv
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.moe_feature_store import MoeFeatureStoreWriter, _atomic_json
from model.expert_adapters.registry import EXPERT_ORDER
from scripts.extract_moe_feature_store import fingerprint
from scripts.kaggle_moe_pipeline import evaluate_best, match_test_labels
from scripts.train_moe_gating import run_training
from training.moe_evaluator import MoeGateEvaluator
from model.moe_gating_core import GatingNetwork
from test_moe_cloud_entrypoints import _config
from test_kaggle_feature_extraction import audio_fixture, ROOT


def write_store(root, split):
    ids = [f"{split}:{i}.wav" for i in range(4)]
    records = [{"sample_id": sid, "wave_file_name": f"{i}.wav", "raw_caption": f"Caption {i}"} for i, sid in enumerate(ids)]
    source = {"schema_version": 1, "split": split, "records": records}
    source["sha256"] = fingerprint(source)
    writer = MoeFeatureStoreWriter(root, split=split, expected_sample_ids=ids, source_manifest_sha256=source["sha256"],
        checkpoint_provenance={e: {"sha256": str(i + 1) * 64} for i, e in enumerate(EXPERT_ORDER)},
        preprocessing_identity={"contract": "unit-test-eval-v1"})
    labels = np.array([2, 4, 7, 9], np.float32)
    scores = ((labels - 5) / 5)[:, None] + np.array([-.1, 0, .1, .2], np.float32)[None, :]
    writer.write_shard(0, sample_ids=ids, wave_file_names=[f"{i}.wav" for i in range(4)], scores=scores,
        features=np.ones((4, 7936), np.float32), labels_raw=labels if split != "test" else None)
    writer.finalize()
    _atomic_json(root / split / "source_manifest.json", source)


class NotebookPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_notebook_format_and_every_code_cell_compiles(self):
        notebook = json.loads((ROOT / "notebooks/kaggle_train_moe_gating.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
                self.assertEqual(cell["outputs"], [])
        headings = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")]
        self.assertEqual(len(headings), 19)  # Introduction + numbered sections 2..20.

    def test_notebook_preflight_cells_execute_in_order_without_extraction(self):
        notebook = json.loads((ROOT / "notebooks/kaggle_train_moe_gating.ipynb").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audio_fixture(base / "input/data")
            for section, names in (("pretrained", ("laion-clap.pt", "mga-clap.pt", "m2d-base.pth", "beats.pt")),
                                   ("finetuned", tuple(f"expert{i}_best.pt" for i in range(1, 5)))):
                for name in names:
                    file = base / "input/moescore-checkpoints" / section / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(b"test-only-not-a-checkpoint")
            (base / "working").mkdir()
            scope = {"display": lambda *a: None}
            with patch.dict(os.environ), patch.object(sys, "path", list(sys.path)), \
                 patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="test GPU"), \
                 patch("torch.cuda.mem_get_info", return_value=(100, 200)), \
                 patch("scripts.kaggle_moe_pipeline.preflight", return_value={"scope": "UNIT_TEST"}), \
                 patch("scripts.kaggle_moe_pipeline.check_dependencies", return_value={}), \
                 patch("subprocess.run", wraps=__import__("subprocess").run) as run:
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"]).replace('RUN_MODE = "full"', 'RUN_MODE = "preflight"')
                        source = source.replace('INSTALL_REQUIREMENTS = True', 'INSTALL_REQUIREMENTS = False')
                        source = source.replace('"/kaggle/working/MOESCORE"', repr(str(ROOT)))
                        source = source.replace('/kaggle/working', str(base / "working")).replace('/kaggle/input', str(base / "input"))
                        exec(compile(source, "notebook-preflight", "exec"), scope)
                self.assertTrue(all("train_moe_gating.py" not in str(call) and "extract_moe_feature_store.py" not in str(call) for call in run.call_args_list))

    def test_full_precision_test_predictions_preserve_default_submission_rounding(self):
        gate = GatingNetwork().eval()
        batch = {"split": "test", "sample_ids": ["test:a.wav"], "wave_file_names": ["a.wav"],
                 "scores": torch.full((1, 4), .123456), "features": torch.zeros(1, 7936)}
        evaluator = MoeGateEvaluator(gate)
        precise = evaluator.predict([batch], round_scores=False)[0]["pred_score"]
        rounded = evaluator.predict([batch])[0]["pred_score"]
        self.assertEqual(rounded, round(precise, 2))
        self.assertGreater(abs(precise - rounded), 1e-4)

    def test_best_validation_checkpoint_then_posthoc_test_metrics_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stores = root / "stores"
            for split in ("train", "validation", "test"):
                write_store(stores, split)
            config = _config(stores, root / "output")
            config["training"]["loss"]["loss_weights"] = [1.0, 0.5]
            with patch("scripts.kaggle_moe_pipeline.match_test_labels", side_effect=AssertionError("Test labels entered training")):
                trained = run_training(config, device_override="cpu")
            self.assertIsNotNone(trained["best_checkpoint"])
            self.assertAlmostEqual(trained["best_srcc_raw_scale"], trained["history"][0]["validation_metrics_raw"]["srcc"])
            metadata = root / "dataset/meta_data"
            metadata.mkdir(parents=True)
            csv_path = metadata / "test_average.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["wav_file_name", "text", "average_score"])
                # Reversed CSV order: post-hoc labels must be joined by identity.
                for i in (3, 2, 1, 0):
                    writer.writerow([f"{i}.wav", f"Caption {i}", [2, 4, 7, 9][i]])
            before = Path(trained["best_checkpoint"]).read_bytes()
            result = evaluate_best(config, root / "dataset")
            for split in ("validation", "test"):
                self.assertEqual(result[split]["sample_count"], 4)
                for name in ("SRCC", "LCC", "KTAU", "MSE"):
                    self.assertIn(name, result[split])
                json.dumps(result[split], allow_nan=False)
                self.assertTrue((root / "output" / f"{split}_predictions.csv").exists())
            self.assertEqual(before, Path(trained["best_checkpoint"]).read_bytes())
            self.assertEqual(result["test"]["optimizer_steps"], 0)
            predicted = [{"sample_id": f"test:{i}.wav", "wave_file_name": f"{i}.wav"} for i in range(4)]
            source = [{"sample_id": f"test:{i}.wav", "raw_caption": f"wrong {i}"} for i in range(4)]
            with self.assertRaisesRegex(ValueError, "caption/filename"):
                match_test_labels(csv_path, predicted, source)
            manifest_path = stores / "test/manifest.json"
            changed = json.loads(manifest_path.read_text())
            changed["preprocessing_identity"]["unreviewed_change"] = True
            changed["store_identity_sha256"] = fingerprint({k: v for k, v in changed.items() if k != "store_identity_sha256"})
            _atomic_json(manifest_path, changed)
            with self.assertRaisesRegex(ValueError, "preprocessing identity"):
                evaluate_best(config, root / "dataset")


if __name__ == "__main__":
    unittest.main()
