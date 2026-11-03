import unittest

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import EXPERT_FEATURE_DIMS, EXPERT_ORDER
from model.moe_forward_integration import MoEForwardIntegration


class _UnusedExpert(ExpertAdapter):
    def __init__(self, name):
        super().__init__(EXPERT_FEATURE_DIMS[name])
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _forward(self, batch):
        raise AssertionError("precomputed output path must not invoke an expert")


def _integration():
    return MoEForwardIntegration({name: _UnusedExpert(name) for name in EXPERT_ORDER}).eval()


class RealOutputGatingIntegrationTests(unittest.TestCase):
    def test_precomputed_expert_outputs_use_existing_assembler_and_gate(self):
        model = _integration()
        batch_size = 2
        outputs = {}
        for index, name in enumerate(EXPERT_ORDER):
            width = EXPERT_FEATURE_DIMS[name]
            score = torch.tensor([[index / 10.0], [index / 10.0 + 0.2]], dtype=torch.float32)
            feature = torch.full((batch_size, width), float(index + 1), dtype=torch.float32)
            outputs[name] = (score, feature)

        scores_before = {name: pair[0].clone() for name, pair in outputs.items()}
        features_before = {name: pair[1].clone() for name, pair in outputs.items()}
        prediction, weights = model.forward_from_expert_outputs(outputs)

        self.assertEqual(tuple(prediction.shape), (batch_size, 1))
        self.assertEqual(tuple(weights.shape), (batch_size, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights >= 0))
        torch.testing.assert_close(weights.sum(1), torch.ones(batch_size), atol=2e-7, rtol=0)
        reference = sum(outputs[name][0] * weights[:, index:index + 1]
                        for index, name in enumerate(EXPERT_ORDER))
        torch.testing.assert_close(prediction, reference, atol=1e-6, rtol=1e-5)
        for name in EXPERT_ORDER:
            torch.testing.assert_close(outputs[name][0], scores_before[name], atol=0, rtol=0)
            torch.testing.assert_close(outputs[name][1], features_before[name], atol=0, rtol=0)

    def test_precomputed_output_path_rejects_missing_expert(self):
        model = _integration()
        outputs = {
            name: (
                torch.zeros((1, 1), dtype=torch.float32),
                torch.zeros((1, EXPERT_FEATURE_DIMS[name]), dtype=torch.float32),
            )
            for name in EXPERT_ORDER[:-1]
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            model.forward_from_expert_outputs(outputs)


if __name__ == "__main__":
    unittest.main()
