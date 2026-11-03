import tempfile
import unittest

import torch

from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager
from training.moe_trainer import MoeGateTrainer


LOSS_CONFIG = {
    "loss_weights": [1.0, -0.5],
    "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
    "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
}
PROVENANCE = {f"expert{i}": {"sha256": f"{i}" * 64} for i in range(1, 5)}
CONFIG = {"seed": 2027, "loss": LOSS_CONFIG, "feature_dim": 7936}


def batch(index):
    generator = torch.Generator().manual_seed(500 + index)
    return {
        "split": "train",
        "sample_ids": [f"train:{index}-a.wav", f"train:{index}-b.wav"],
        "scores": torch.randn(2, 4, generator=generator),
        "features": torch.randn(2, 7936, generator=generator),
        "labels_raw": torch.tensor([2.0, 8.0]),
        "labels_scale": "raw_0_to_10",
    }


class MoeCheckpointResumeTests(unittest.TestCase):
    def build(self):
        gate = GatingNetwork()
        trainer = MoeGateTrainer(gate, LOSS_CONFIG, learning_rate=1e-3)
        return gate, trainer

    def test_gate_only_checkpoint_restores_optimizer_scheduler_and_step(self):
        torch.manual_seed(2027)
        gate, trainer = self.build()
        trainer.train_step(batch(0))
        trainer.global_step = 1
        trainer.scheduler_step(0.75)
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/gate.pt"
            manager = MoeCheckpointManager()
            manager.save(
                path,
                gating_network=gate,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=0,
                global_step=trainer.global_step,
                config=CONFIG,
                feature_store_identity={"train": "a" * 64},
                expert_checkpoint_provenance=PROVENANCE,
                sampler_state={"epoch": 0, "cursor": 2, "order": [0, 1, 2]},
            )
            restored_gate, restored_trainer = self.build()
            state = manager.load(
                path,
                gating_network=restored_gate,
                optimizer=restored_trainer.optimizer,
                scheduler=restored_trainer.scheduler,
                expected_config=CONFIG,
                expected_feature_store_identity={"train": "a" * 64},
                expected_expert_checkpoint_provenance=PROVENANCE,
            )
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(state["sampler_state"]["cursor"], 2)
            self.assertEqual(
                trainer.scheduler.state_dict(), restored_trainer.scheduler.state_dict()
            )
            for name, parameter in gate.state_dict().items():
                torch.testing.assert_close(parameter, restored_gate.state_dict()[name], atol=0, rtol=0)
            self.assertTrue(all(key.startswith("net.") for key in state["gating_state_dict"]))
            self.assertFalse(any("expert" in key for key in state["gating_state_dict"]))

    def test_resume_matches_uninterrupted_steps_with_dropout_rng(self):
        torch.manual_seed(1234)
        uninterrupted_gate, uninterrupted = self.build()
        uninterrupted_losses = [uninterrupted.train_step(batch(i))["total_loss"].item() for i in range(3)]
        uninterrupted_parameters = [p.detach().clone() for p in uninterrupted_gate.parameters()]

        torch.manual_seed(1234)
        first_gate, first = self.build()
        first.train_step(batch(0))
        first.global_step = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/resume.pt"
            manager = MoeCheckpointManager()
            manager.save(
                path,
                gating_network=first_gate,
                optimizer=first.optimizer,
                scheduler=first.scheduler,
                epoch=0,
                global_step=1,
                config=CONFIG,
                feature_store_identity={"train": "a" * 64},
                expert_checkpoint_provenance=PROVENANCE,
            )
            resumed_gate, resumed = self.build()
            manager.load(
                path,
                gating_network=resumed_gate,
                optimizer=resumed.optimizer,
                scheduler=resumed.scheduler,
                expected_config=CONFIG,
                expected_feature_store_identity={"train": "a" * 64},
                expected_expert_checkpoint_provenance=PROVENANCE,
            )
            resumed.global_step = 1
            resumed_losses = [resumed.train_step(batch(i))["total_loss"].item() for i in (1, 2)]
            torch.testing.assert_close(
                torch.tensor(uninterrupted_losses[1:]), torch.tensor(resumed_losses), atol=0, rtol=0
            )
            for expected, actual in zip(uninterrupted_parameters, resumed_gate.parameters()):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_manifest_provenance_configuration_and_legacy_width_mismatches_fail(self):
        gate, trainer = self.build()
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/gate.pt"
            manager = MoeCheckpointManager()
            manager.save(
                path,
                gating_network=gate,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=0,
                global_step=0,
                config=CONFIG,
                feature_store_identity={"train": "a" * 64},
                expert_checkpoint_provenance=PROVENANCE,
            )
            for kwargs, text in (
                ({"expected_config": {**CONFIG, "seed": 9}}, "configuration"),
                ({"expected_feature_store_identity": {"train": "b" * 64}}, "feature store"),
                ({"expected_expert_checkpoint_provenance": {**PROVENANCE, "expert1": {"sha256": "f" * 64}}}, "provenance"),
            ):
                restored, other = self.build()
                args = {
                    "expected_config": CONFIG,
                    "expected_feature_store_identity": {"train": "a" * 64},
                    "expected_expert_checkpoint_provenance": PROVENANCE,
                }
                args.update(kwargs)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                    manager.load(
                        path,
                        gating_network=restored,
                        optimizer=other.optimizer,
                        scheduler=other.scheduler,
                        **args,
                    )

    def test_inference_load_restores_gate_without_optimizer_or_training_metadata(self):
        gate, trainer = self.build()
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/gate.pt"
            manager = MoeCheckpointManager()
            manager.save(
                path,
                gating_network=gate,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=0,
                global_step=0,
                config=CONFIG,
                feature_store_identity={"train": "a" * 64},
                expert_checkpoint_provenance=PROVENANCE,
            )
            restored = GatingNetwork()
            metadata = manager.load_for_inference(
                path,
                gating_network=restored,
                expected_expert_checkpoint_provenance=PROVENANCE,
            )
            self.assertEqual(metadata["feature_store_identity"], {"train": "a" * 64})
            for name, parameter in gate.state_dict().items():
                torch.testing.assert_close(parameter, restored.state_dict()[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
