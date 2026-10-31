import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import wavfile


class MetricsTest(unittest.TestCase):
    def test_metrics_use_raw_score_scale(self):
        from MOESCORE.expert2_finetune.Train_Expert2 import compute_metrics

        metrics = compute_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        self.assertAlmostEqual(metrics["SRCC"], 1.0)
        self.assertAlmostEqual(metrics["LCC"], 1.0)
        self.assertAlmostEqual(metrics["KTAU"], 1.0)
        self.assertAlmostEqual(metrics["MSE"], 0.25)
        self.assertEqual(metrics["N"], 3)

    def test_rejects_mismatched_predictions(self):
        from MOESCORE.expert2_finetune.Train_Expert2 import compute_metrics

        with self.assertRaises(ValueError):
            compute_metrics([1.0, 2.0], [1.0])

    def test_training_writes_artifacts_next_to_its_own_checkpoint(self):
        from MOESCORE.expert2_finetune.Train_Expert2 import train

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, batch):
                score = batch["mga_wavs"][:, :, :32000].mean((1, 2)) * self.scale
                return score[:, None], torch.zeros(len(score), 2048)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation", "test"):
                folder = root / "wav" / split
                folder.mkdir(parents=True)
                (root / "meta_data").mkdir(exist_ok=True)
                with (root / "meta_data" / f"{split}_average.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("wav_file_name", "text", "average_score"))
                    for n, score, amplitude in ((0, 2.0, 8192), (1, 8.0, 24576)):
                        name = f"{split}_{n}.wav"
                        writer.writerow((name, "water", score))
                        wavfile.write(folder / name, 16000, np.full(16000, amplitude, dtype=np.int16))
            output = root / "weights" / "full_best.pt"
            config = {
                "data_root": str(root), "expected_train_pairs": 2, "expected_validation_pairs": 2,
                "expected_test_pairs": 2,
                "device": "cpu", "seed": 42, "batch_size": 2, "val_batch_size": 2,
                "num_workers": 0, "epochs": 1, "patience": 1, "lr": 1e-5,
                "mga_config": "MGA-CLAP/settings/inference_example.yaml",
                "mga_source": "MGA-CLAP", "mga_checkpoint": "weights/mga-clap.pt",
                "output": str(output), "log_every": 1,
            }
            with patch("Train_Expert2.create_expert2", return_value=TinyModel()):
                train(config)
            self.assertTrue(output.is_file())
            self.assertTrue((root / "weights/full_run/best_metrics.json").is_file())
            self.assertTrue((root / "weights/full_run/validation_predictions.csv").is_file())
            self.assertTrue((root / "weights/full_run/last.pt").is_file())
            config["epochs"] = 2
            with patch("Train_Expert2.create_expert2", return_value=TinyModel()):
                train(config, resume=root / "weights/full_run/last.pt")
            resumed = torch.load(root / "weights/full_run/last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["epoch"], 1)
            from MOESCORE.expert2_finetune.Train_Expert2 import evaluate_checkpoint

            with patch("Train_Expert2.create_expert2", return_value=TinyModel()):
                baseline = evaluate_checkpoint(config)
            with patch("Train_Expert2.create_expert2", return_value=TinyModel()):
                fitted = evaluate_checkpoint(config, checkpoint=output)
            with patch("Train_Expert2.create_expert2", return_value=TinyModel()):
                test_metrics = evaluate_checkpoint(config, checkpoint=output, split="test")
            self.assertEqual(baseline["N"], 2)
            self.assertEqual(fitted["N"], 2)
            self.assertEqual(test_metrics["N"], 2)
            self.assertTrue((root / "weights/full_run/pretrained_predictions.csv").is_file())
            self.assertTrue((root / "weights/full_run/evaluation_predictions.csv").is_file())
            self.assertTrue((root / "weights/full_run/test_evaluation_metrics.json").is_file())
            self.assertTrue((root / "weights/full_run/test_evaluation_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
