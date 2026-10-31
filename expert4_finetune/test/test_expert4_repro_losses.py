import unittest

import torch

from expert4_finetune.losses import expert4_loss


class Expert4LossTests(unittest.TestCase):
    def test_released_pairwise_reduction_and_threshold(self):
        prediction = torch.tensor([0.0, 0.5])
        target = torch.tensor([0.0, 1.0])
        total, mse, contrast = expert4_loss(prediction, target)
        self.assertAlmostEqual(mse.item(), 0.125)
        self.assertAlmostEqual(contrast.item(), 0.1)
        self.assertAlmostEqual(total.item(), 0.175)

    def test_no_broadcasting(self):
        with self.assertRaises(ValueError):
            expert4_loss(torch.zeros(2, 1), torch.zeros(2))

    def test_threshold_is_strict(self):
        total, mse, _ = expert4_loss(torch.tensor([0.25]), torch.tensor([0.0]))
        self.assertEqual(mse.item(), 0.0)
        self.assertEqual(total.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
