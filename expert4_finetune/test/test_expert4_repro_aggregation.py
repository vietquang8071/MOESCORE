import unittest
import torch
from expert4_finetune.audio_encoder import aggregate_layer_outputs


class AggregationTests(unittest.TestCase):
    def test_excludes_initial_and_uses_softmax(self):
        results = [(torch.full((3, 2, 4), float(i)), None) for i in range(13)]
        weights = torch.nn.Parameter(torch.zeros(12))
        result = aggregate_layer_outputs(results, weights)
        self.assertEqual(tuple(result.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(result, torch.full_like(result, 6.5)))
        result.sum().backward()
        self.assertIsNotNone(weights.grad)

    def test_missing_layer_fails(self):
        with self.assertRaisesRegex(ValueError, 'layer'):
            aggregate_layer_outputs([(torch.ones(2, 1, 4), None)] * 12, torch.zeros(12))


if __name__ == '__main__':
    unittest.main()
