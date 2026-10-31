import tempfile
import unittest
from pathlib import Path
from expert4_finetune.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_and_rejects_wrong_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            path.write_text('training:\n  batch_size: 8\n  learning_rate: 1.0e-5\n  epochs: 2\n')
            cfg = load_config(path, validate_paths=False)
            self.assertEqual(cfg['training']['batch_size'], 8)
            path.write_text('training:\n  batch_size: 4\n')
            with self.assertRaises(ValueError):
                load_config(path, validate_paths=False)


if __name__ == '__main__':
    unittest.main()
