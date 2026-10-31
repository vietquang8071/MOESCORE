import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from expert4_finetune.dataset import Expert4Dataset, collate_expert4


class DatasetTests(unittest.TestCase):
    def test_unlabeled_test_metadata_keeps_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wavfile.write(root / 'a.wav', 16000, np.zeros(16000, dtype=np.int16))
            wavfile.write(root / 'b.wav', 16000, np.zeros(16000, dtype=np.int16))
            (root / 'test.csv').write_text('wav_file_name,text,model_ID,model_name\nb.wav,second,2,m\na.wav,first,1,n\n')
            ds = Expert4Dataset(root / 'test.csv', root, require_score=False)
            batch = collate_expert4([ds[0], ds[1]])
            self.assertEqual(batch['sample_ids'], ['b.wav', 'a.wav'])
            self.assertIsNone(batch['scores'])

    def test_actual_headers_audio_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wavfile.write(root / 'a.wav', 8000, np.ones(8000, dtype=np.int16) * 1000)
            with (root / 'data.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['wav_file_name', 'text', 'average_score', 'model_ID', 'model_name'])
                writer.writerow(['a.wav', 'sound', '7.5', '2', 'example'])
            dataset = Expert4Dataset(root / 'data.csv', root)
            batch = collate_expert4([dataset[0]])
            self.assertEqual(tuple(batch['wavs'].shape), (1, 1, 160000))
            self.assertEqual(batch['wav_lens'].item(), 16000)
            self.assertEqual(batch['scores'].item(), 0.5)
            self.assertEqual(batch['sample_ids'], ['a.wav'])
            self.assertEqual(batch['model_ids'], ['2'])

    def test_missing_headers_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data.csv').write_text('text,score\nhello,5\n')
            with self.assertRaisesRegex(ValueError, 'column'):
                Expert4Dataset(root / 'data.csv', root)

    def test_inconsistent_model_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wavfile.write(root / 'a.wav', 16000, np.zeros(100, dtype=np.int16))
            wavfile.write(root / 'b.wav', 16000, np.zeros(100, dtype=np.int16))
            (root / 'data.csv').write_text('wav_file_name,text,average_score,model_ID,model_name\na.wav,one,5,1,alpha\nb.wav,two,5,1,beta\n')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                Expert4Dataset(root / 'data.csv', root)


if __name__ == '__main__':
    unittest.main()
