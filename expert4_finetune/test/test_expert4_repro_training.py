import tempfile
import unittest
from pathlib import Path
import torch
from expert4_finetune.trainer import linear_schedule, save_checkpoint, load_checkpoint, _source_identity


class TrainingTests(unittest.TestCase):
    def test_source_fingerprint_changes_when_beats_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'beats'
            source.mkdir()
            for name in ('BEATs.py', 'backbone.py', 'modules.py'):
                (source / name).write_text(name)
            first = _source_identity(Path(temp))
            (source / 'BEATs.py').write_text('changed')
            self.assertNotEqual(first['sha256'], _source_identity(Path(temp))['sha256'])

    def test_schedule_and_resume(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = linear_schedule(optimizer, total_steps=10, warmup_steps=2)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.last_epoch, 1)
        expected = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'last.pt'
            save_checkpoint(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'scheduler': scheduler.state_dict(), 'epoch': 3, 'step': 1})
            with torch.no_grad():
                model.weight.zero_()
            state = load_checkpoint(path)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            self.assertTrue(torch.equal(model.weight, expected))
            self.assertEqual(state['epoch'], 3)
            self.assertEqual(scheduler.last_epoch, 1)


if __name__ == '__main__':
    unittest.main()
