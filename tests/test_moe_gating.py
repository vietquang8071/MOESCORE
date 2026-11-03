import copy
import unittest

import torch
from torch import nn

from model.expert_adapters.registry import EXPECTED_FEATURE_WIDTH, EXPERT_ORDER
from model.moe_gating_core import GatingNetwork


class GatingNetworkTests(unittest.TestCase):
    def make_controlled_gate(self):
        gate = GatingNetwork()
        with torch.no_grad():
            first = gate.net[0]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1.0
            first.weight[1, 0] = -1.0

            norm = gate.net[1]
            norm.weight.fill_(1.0)
            norm.bias.zero_()

            final = gate.net[4]
            final.weight.zero_()
            final.bias.zero_()
            final.weight[0, 0] = 1.0
            final.weight[1, 1] = 1.0
        return gate

    def test_original_architecture_consumes_registry_width_and_four_experts(self):
        gate = GatingNetwork()

        self.assertEqual(EXPECTED_FEATURE_WIDTH, 7936)
        self.assertEqual(tuple(type(layer) for layer in gate.net), (
            nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear, nn.Softmax
        ))
        self.assertEqual(gate.net[0].in_features, 7936)
        self.assertEqual(gate.net[0].out_features, 512)
        self.assertEqual(gate.net[1].normalized_shape, (512,))
        self.assertTrue(gate.net[2].inplace)
        self.assertEqual(gate.net[3].p, 0.1)
        self.assertEqual(gate.net[4].out_features, len(EXPERT_ORDER))
        self.assertEqual(gate.net[5].dim, -1)

    def test_softmax_weights_and_batch_shapes_for_b1_b2_b4(self):
        gate = GatingNetwork().eval()
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                weights = gate(torch.randn(batch_size, EXPECTED_FEATURE_WIDTH))
                self.assertEqual(tuple(weights.shape), (batch_size, 4))
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue(torch.all(weights >= 0))
                torch.testing.assert_close(
                    # Summing four float32 softmax probabilities can round by
                    # one float32 ULP (1.1920929e-7).
                    weights.sum(dim=1), torch.ones(batch_size), atol=2e-7, rtol=0
                )

    def test_controlled_features_produce_sample_dependent_weights(self):
        gate = self.make_controlled_gate().eval()
        features = torch.zeros(2, EXPECTED_FEATURE_WIDTH)
        features[0, 0] = 1.0
        features[1, 0] = -1.0

        weights = gate(features)

        self.assertFalse(torch.allclose(weights[0], weights[1]))
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())
        self.assertGreater(weights[1, 1].item(), weights[1, 0].item())

    def test_eval_output_is_independent_of_other_batch_members_and_order(self):
        gate = self.make_controlled_gate().eval()
        sample_a = torch.zeros(1, EXPECTED_FEATURE_WIDTH)
        sample_a[0, 0] = 1.0
        sample_b = torch.zeros(1, EXPECTED_FEATURE_WIDTH)
        sample_b[0, 0] = -1.0

        alone = gate(sample_a)
        together = gate(torch.cat((sample_a, sample_b), dim=0))[0:1]
        reversed_batch = gate(torch.cat((sample_b, sample_a), dim=0))[1:2]

        torch.testing.assert_close(alone, together, atol=1e-7, rtol=0)
        torch.testing.assert_close(alone, reversed_batch, atol=1e-7, rtol=0)

    def test_gate_gradients_flow_to_trainable_parameters(self):
        gate = self.make_controlled_gate().train()
        features = torch.zeros(2, EXPECTED_FEATURE_WIDTH)
        features[:, 0] = torch.tensor([1.0, -1.0])
        loss = gate(features)[:, 0].sum()

        loss.backward()

        gradients = [p.grad for p in gate.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in gradients))
        self.assertTrue(all(p.requires_grad for p in gate.parameters()))

    def test_invalid_width_rank_dtype_and_nonfinite_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "7936"):
            GatingNetwork(input_feature_dim=7168)
        with self.assertRaisesRegex(ValueError, "four"):
            GatingNetwork(num_experts=3)

        gate = GatingNetwork()
        with self.assertRaisesRegex(ValueError, "shape"):
            gate(torch.zeros(2, EXPECTED_FEATURE_WIDTH, 1))
        with self.assertRaisesRegex(ValueError, "width"):
            gate(torch.zeros(2, EXPECTED_FEATURE_WIDTH - 1))
        with self.assertRaisesRegex(TypeError, "floating"):
            gate(torch.zeros(2, EXPECTED_FEATURE_WIDTH, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "finite"):
            gate(torch.full((1, EXPECTED_FEATURE_WIDTH), float("nan")))
        with self.assertRaisesRegex(TypeError, "dtype"):
            gate(torch.zeros(1, EXPECTED_FEATURE_WIDTH, dtype=torch.float64))

    def test_state_dict_round_trip_preserves_eval_outputs(self):
        source = GatingNetwork().eval()
        restored = GatingNetwork().eval()
        restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
        features = torch.linspace(-1, 1, EXPECTED_FEATURE_WIDTH).reshape(1, -1)

        torch.testing.assert_close(source(features), restored(features), atol=0, rtol=0)
        self.assertEqual(source.state_dict().keys(), restored.state_dict().keys())

    def test_legacy_7168_input_checkpoint_is_rejected_strictly(self):
        gate = GatingNetwork()
        legacy_state = copy.deepcopy(gate.state_dict())
        legacy_state["net.0.weight"] = torch.zeros(512, 7168)

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            gate.load_state_dict(legacy_state, strict=True)

    def test_eval_is_deterministic_and_train_enables_original_dropout(self):
        gate = GatingNetwork()
        features = torch.ones(2, EXPECTED_FEATURE_WIDTH)
        self.assertEqual(gate.net[3].p, 0.1)

        gate.eval()
        first = gate(features)
        second = gate(features)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

        gate.train()
        self.assertTrue(gate.net[3].training)


if __name__ == "__main__":
    unittest.main()
