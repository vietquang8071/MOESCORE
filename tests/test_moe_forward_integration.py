import unittest

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import (
    EXPECTED_FEATURE_WIDTH,
    EXPERT_FEATURE_DIMS,
    EXPERT_ORDER,
    validate_and_assemble_expert_outputs,
)
from model.moe_forward_integration import MoEForwardIntegration


class CountingExpertAdapter(ExpertAdapter):
    def __init__(self, name, feature_dim=None):
        resolved_width = EXPERT_FEATURE_DIMS[name] if feature_dim is None else feature_dim
        super().__init__(resolved_width)
        self.name = name
        self.audio_key = f"{name}_wavs"
        self.feature_width = resolved_width
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(float(EXPERT_ORDER.index(name)) / 10.0))
        self.dropout = nn.Dropout(0.9)
        self.calls = 0
        self.grad_enabled_on_forward = []
        self.last_input = None
        self.last_pair = None
        self.fail = False

    def _forward(self, batch):
        self.calls += 1
        self.grad_enabled_on_forward.append(torch.is_grad_enabled())
        if self.fail:
            raise RuntimeError(f"{self.name} synthetic failure")
        audio = batch[self.audio_key]
        self.last_input = audio.detach().clone()
        signal = self.dropout(audio.flatten(1).mean(dim=1))
        score = (signal * self.scale + self.bias).unsqueeze(1)
        feature_offset = torch.arange(
            self.feature_dim, dtype=audio.dtype, device=audio.device
        ).unsqueeze(0)
        features = feature_offset + signal.unsqueeze(1) * self.scale
        self.last_pair = (score.detach().clone(), features.detach().clone())
        return score, features


def make_experts(order=EXPERT_ORDER):
    return {name: CountingExpertAdapter(name) for name in order}


def make_batch(batch_size):
    batch = {}
    for index, name in enumerate(EXPERT_ORDER):
        values = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 1, 2)
        batch[f"{name}_wavs"] = values + index
    batch["captions_raw"] = [f"caption {index}" for index in range(batch_size)]
    batch["labels"] = torch.linspace(0.0, 10.0, batch_size)
    batch["model_ids"] = ["metadata-only"] * batch_size
    return batch


class MoEForwardIntegrationTests(unittest.TestCase):
    def test_four_frozen_adapters_integrate_in_registry_order_for_b1_b2_b4(self):
        experts = make_experts(tuple(reversed(EXPERT_ORDER)))
        model = MoEForwardIntegration(experts).eval()

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                batch = make_batch(batch_size)
                originals = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
                final_score, weights = model(batch)
                outputs = {name: experts[name].last_pair for name in EXPERT_ORDER}
                scores, features = validate_and_assemble_expert_outputs(outputs, batch_size=batch_size)
                expected_weights = model.gating_network(features)
                reference = (
                    scores[:, 0:1] * expected_weights[:, 0:1]
                    + scores[:, 1:2] * expected_weights[:, 1:2]
                    + scores[:, 2:3] * expected_weights[:, 2:3]
                    + scores[:, 3:4] * expected_weights[:, 3:4]
                )

                self.assertEqual(tuple(scores.shape), (batch_size, 4))
                self.assertEqual(tuple(features.shape), (batch_size, EXPECTED_FEATURE_WIDTH))
                self.assertEqual(tuple(final_score.shape), (batch_size, 1))
                self.assertEqual(tuple(weights.shape), (batch_size, 4))
                torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
                torch.testing.assert_close(final_score, reference, atol=1e-6, rtol=1e-5)
                for key, value in originals.items():
                    torch.testing.assert_close(batch[key], value, atol=0, rtol=0)

        for name in EXPERT_ORDER:
            self.assertEqual(experts[name].calls, 3)
            self.assertEqual(experts[name].grad_enabled_on_forward, [False, False, False])

    def test_only_gating_parameters_are_trainable_and_receive_gradients(self):
        experts = make_experts()
        model = MoEForwardIntegration(experts)
        model.train()
        gate_before = [parameter.detach().clone() for parameter in model.gating_network.parameters()]
        expert_before = {
            name: [parameter.detach().clone() for parameter in adapter.parameters()]
            for name, adapter in experts.items()
        }

        optimizer = torch.optim.SGD(model.gating_network.parameters(), lr=0.01)
        optimizer.zero_grad()
        final_score, weights = model(make_batch(4))
        (final_score.square().mean() + weights[:, 0].mean()).backward()

        gate_grads = [p.grad for p in model.gating_network.parameters() if p.grad is not None]
        self.assertTrue(gate_grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gate_grads))
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in gate_grads))
        for name, adapter in experts.items():
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in adapter.parameters()))
            self.assertTrue(torch.equal(expert_before[name][0], adapter.scale))
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(gate_before, model.gating_network.parameters())
        ))

    def test_parent_train_keeps_all_expert_modules_eval_and_gate_training(self):
        experts = make_experts()
        model = MoEForwardIntegration(experts)

        model.train()
        self.assertTrue(model.gating_network.training)
        for adapter in experts.values():
            self.assertFalse(adapter.training)
            self.assertFalse(adapter.dropout.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

        model.train(False)
        self.assertFalse(model.gating_network.training)
        self.assertTrue(all(not adapter.training and not adapter.dropout.training for adapter in experts.values()))
        model.eval()
        self.assertFalse(model.gating_network.training)
        self.assertTrue(all(not adapter.training for adapter in experts.values()))

    def test_expert_output_and_gate_inputs_are_deterministic_in_eval(self):
        model = MoEForwardIntegration(make_experts()).eval()
        batch = make_batch(2)
        first_score, first_weights = model(batch)
        batch["labels"] = torch.tensor([99.0, -99.0])
        batch["model_ids"] = ["different", "metadata"]
        second_score, second_weights = model(batch)

        torch.testing.assert_close(first_score, second_score, atol=0, rtol=0)
        torch.testing.assert_close(first_weights, second_weights, atol=0, rtol=0)

    def test_state_dict_round_trip_preserves_forward_outputs(self):
        first = MoEForwardIntegration(make_experts()).eval()
        second = MoEForwardIntegration(make_experts()).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        batch = make_batch(2)

        score_a, weights_a = first(batch)
        score_b, weights_b = second(batch)
        torch.testing.assert_close(score_a, score_b, atol=0, rtol=0)
        torch.testing.assert_close(weights_a, weights_b, atol=0, rtol=0)

    def test_bad_expert_registry_and_feature_metadata_are_rejected(self):
        experts = make_experts()
        with self.assertRaisesRegex(ValueError, "missing"):
            MoEForwardIntegration({name: adapter for name, adapter in experts.items() if name != "expert4"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            MoEForwardIntegration({**experts, "other": CountingExpertAdapter("expert1")})
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            MoEForwardIntegration({**experts, "expert4": CountingExpertAdapter("expert4", feature_dim=1023)})

    def test_expert_failure_propagates_without_gating(self):
        experts = make_experts()
        experts["expert3"].fail = True
        model = MoEForwardIntegration(experts)

        with self.assertRaisesRegex(RuntimeError, "expert3 synthetic failure"):
            model(make_batch(1))
        self.assertEqual(experts["expert4"].calls, 0)


if __name__ == "__main__":
    unittest.main()
