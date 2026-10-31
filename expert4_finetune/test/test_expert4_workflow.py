import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from expert4_finetune.model import Expert4
from expert4_finetune.trainer import run_training, load_checkpoint, evaluate
from expert4_finetune.inference import run_prediction


class TinyExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, batch):
        score = torch.tanh(self.scale * batch['wavs'][:, 0, 0])
        return score, score[:, None].expand(-1, 1024)


class WorkflowTests(unittest.TestCase):
    def test_train_validate_checkpoint_reload_predict_unlabeled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split, count, labeled in [('train', 8, True), ('validation', 3, True), ('test', 2, False)]:
                directory = root / split
                directory.mkdir()
                with (root / f'{split}.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    header = ['wav_file_name', 'text'] + (['average_score'] if labeled else [])
                    writer.writerow(header)
                    for index in range(count):
                        name = f'{split}{index}.wav'
                        waveform = np.full(1600, (index + 1) * 1500, dtype=np.int16)
                        wavfile.write(directory / name, 16000, waveform)
                        writer.writerow([name, f'caption {index}'] + ([str(index + 1)] if labeled else []))
            for name in ('beats.pt', 'clap.pt'):
                (root / name).write_bytes(b'synthetic fixture')
            (root / 'unilm').mkdir()
            config = {'resources': {'train_csv': str(root / 'train.csv'), 'validation_csv': str(root / 'validation.csv'),
                'test_csv': str(root / 'test.csv'), 'train_audio_dir': str(root / 'train'),
                'validation_audio_dir': str(root / 'validation'), 'test_audio_dir': str(root / 'test'),
                'beats_checkpoint': str(root / 'beats.pt'), 'clap_checkpoint': str(root / 'clap.pt'),
                'unilm_source_dir': str(root / 'unilm'), 'output_dir': str(root / 'output')},
                'training': {'epochs': 2, 'amp': False}}
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            with patch.object(Expert4, 'from_pretrained', side_effect=lambda *args: TinyExpert()), \
                 patch('expert4_repro.trainer.assert_training_gradients', side_effect=lambda model: self.assertIsNotNone(model.scale.grad)):
                evaluation_count = 0
                def interrupt_after_first_epoch(model, loader, device):
                    nonlocal evaluation_count
                    evaluation_count += 1
                    if evaluation_count == 2:
                        raise RuntimeError('synthetic disconnect')
                    return evaluate(model, loader, device)
                with patch('expert4_repro.trainer.evaluate', side_effect=interrupt_after_first_epoch):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic disconnect'):
                        run_training(config_path)
                output_dir = root / 'output'
                protected = [output_dir / name for name in ('metrics.csv', 'training_config.yaml', 'environment.json', 'last_expert4.pt')]
                before_smoke = [path.read_bytes() for path in protected]
                run_training(config_path, smoke_only=True)
                self.assertEqual(before_smoke, [path.read_bytes() for path in protected])
                best = run_training(config_path, resume=root / 'output' / 'last_expert4.pt')
                self.assertTrue(best.is_file())
                self.assertTrue((root / 'output' / 'last_expert4.pt').is_file())
                self.assertEqual(load_checkpoint(root / 'output' / 'last_expert4.pt')['global_step'], 2)
                validation_rows, validation_metrics, validation_path = run_prediction(best, config_path, 'validation')
                self.assertEqual(len(validation_rows), 3)
                self.assertIn('srcc', validation_metrics)
                self.assertTrue(validation_path.is_file())
                rows, metrics, output = run_prediction(best, config_path, 'test')
                self.assertIsNone(metrics)
                self.assertEqual([r['sample_id'] for r in rows], ['test0.wav', 'test1.wav'])
                with output.open() as handle:
                    self.assertEqual(list(csv.DictReader(handle))[0]['sample_id'], 'test0.wav')


if __name__ == '__main__':
    unittest.main()
