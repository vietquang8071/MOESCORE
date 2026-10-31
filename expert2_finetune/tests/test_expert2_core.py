import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile


class DataTest(unittest.TestCase):
    def test_xacle_average_audio_and_score(self):
        from MOESCORE.expert2_finetune.expert2_data import Expert2Dataset, collate_expert2

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            wavfile.write(root / "a.wav", 16000, np.full(16000, 16384, dtype=np.int16))
            with (root / "labels.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["wav_file_name", "text", "average_score"])
                writer.writerow(["a.wav", "Water flowing", 7.5])
            dataset = Expert2Dataset(root / "labels.csv", root)
            item = dataset[0]
            self.assertEqual(item["audio"].shape, (320000,))
            self.assertAlmostEqual(float(item["audio"][1000]), 0.5, places=2)
            self.assertEqual(item["score"], 0.5)
            batch = collate_expert2([item])
            self.assertEqual(tuple(batch["mga_wavs"].shape), (1, 1, 320000))
            self.assertEqual(batch["caption_tokens"], ["Water flowing"])

    def test_rejects_listener_rows(self):
        from MOESCORE.expert2_finetune.expert2_data import Expert2Dataset

        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "train.csv"
            csv_path.write_text("wav_file_name,text,score\na.wav,water,8\n")
            with self.assertRaisesRegex(ValueError, "average_score"):
                Expert2Dataset(csv_path, Path(root))

    def test_stereo_int16_is_scaled_before_mono_mix(self):
        from MOESCORE.expert2_finetune.expert2_data import Expert2Dataset

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            wavfile.write(root / "s.wav", 32000, np.full((32000, 2), 16384, dtype=np.int16))
            (root / "labels.csv").write_text("wav_file_name,text,average_score\ns.wav,water,5\n")
            audio = Expert2Dataset(root / "labels.csv", root)[0]["audio"]
            self.assertAlmostEqual(float(audio[1000]), 0.5, places=5)


class LossTest(unittest.TestCase):
    def test_pairwise_shape_and_numeric_result(self):
        from MOESCORE.expert2_finetune.expert2_loss import hybrid_loss

        pred = torch.tensor([0.0, 0.5], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        total, mse, pair = hybrid_loss(pred, target)
        self.assertAlmostEqual(float(mse.detach()), 0.125)
        self.assertAlmostEqual(float(pair.detach()), 0.1)
        self.assertAlmostEqual(float(total.detach()), 0.175)
        total.backward()
        self.assertIsNotNone(pred.grad)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            hybrid_loss(pred[:, None], target)


if __name__ == "__main__":
    unittest.main()
