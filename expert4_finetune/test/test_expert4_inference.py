import csv
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from expert4_finetune.inference import predict_rows, export_predictions


class FixedModel(torch.nn.Module):
    def forward(self, batch):
        return torch.tensor([0.2] * len(batch['captions'])), torch.zeros(len(batch['captions']), 1024)


class InferenceTests(unittest.TestCase):
    def test_unlabeled_order_and_export(self):
        batches = [{'wavs': torch.zeros(2, 1, 8), 'wav_lens': torch.tensor([8, 8]),
                    'captions': ['second', 'first'], 'scores': None, 'scores_raw': None,
                    'sample_ids': ['b.wav', 'a.wav'], 'model_ids': ['2', '1'], 'model_names': ['b', 'a']}]
        rows = predict_rows(FixedModel(), batches, torch.device('cpu'))
        self.assertEqual([r['sample_id'] for r in rows], ['b.wav', 'a.wav'])
        self.assertAlmostEqual(rows[0]['prediction_raw'], 6.0, places=5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test_predictions.csv'
            export_predictions(rows, path)
            with path.open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([r['sample_id'] for r in saved], ['b.wav', 'a.wav'])
            self.assertNotIn('score_raw', saved[0])


if __name__ == '__main__':
    unittest.main()
