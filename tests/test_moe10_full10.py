"""Full10 registry, config, feature-store and gate contract tests."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from configs.moe_config import MoeConfigError, validate_moe_config
from data.moe_feature_store import MoeFeatureStoreWriter, validate_feature_store
from model.expert_adapters.registry import (
    EXPECTED_FULL10_FEATURE_WIDTH,
    EXPERT_PROFILES,
    FULL10_EXPERT_ORDER,
    profile_contract,
    validate_and_assemble_expert_outputs,
)
from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager


class Full10RegistryConfigStoreTests(unittest.TestCase):
    def _full10_config(self):
        path = Path(__file__).resolve().parents[1] / "configs/moe_full9.example.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["expert_profile"] = "full_10"
        config["experts"]["m2d_lora_coral"] = {
            "base_checkpoint": "weights/pretrained/checkpoint-30.pth",
            "finetuned_checkpoint": "weights/m2d-lora-Nam.pt",
        }
        config["gate"]["expert_order"] = list(config["gate"]["expert_order"]) + ["m2d_lora_coral"]
        return config

    def test_full10_registry_has_exact_order_and_width_without_changing_full9(self):
        order, dims, width = profile_contract("full_10")
        self.assertEqual(order, FULL10_EXPERT_ORDER)
        self.assertEqual(order, (
            "expert1", "expert2", "expert3", "expert4", "top2_m2d", "top2_msclap",
            "top2_laion", "top2_mga", "m2d_lora", "m2d_lora_coral",
        ))
        self.assertEqual(len(order), 10)
        self.assertEqual(dims["m2d_lora_coral"], 3073)
        self.assertEqual(width, 30472)
        self.assertEqual(width, EXPECTED_FULL10_FEATURE_WIDTH)
        self.assertEqual(profile_contract("full_9")[2], 27399)
        self.assertEqual(profile_contract("top1_4")[2], 11010)
        self.assertIn("full_9", EXPERT_PROFILES)
        self.assertIn("full_10", EXPERT_PROFILES)

    def test_full10_config_requires_ten_experts_and_profile_specific_order(self):
        config = self._full10_config()
        validate_moe_config(config)
        missing = copy.deepcopy(config)
        del missing["experts"]["m2d_lora_coral"]
        with self.assertRaisesRegex(MoeConfigError, "exactly"):
            validate_moe_config(missing)
        reordered = copy.deepcopy(config)
        reordered["gate"]["expert_order"][-1] = "m2d_lora"
        with self.assertRaisesRegex(MoeConfigError, "expert_order"):
            validate_moe_config(reordered)

    def test_full10_assembly_and_gate_emit_ten_scores_and_simplex_weights(self):
        order, dimensions, width = profile_contract("full_10")
        outputs = {
            name: (
                torch.full((2, 1), float(index) / 10, dtype=torch.float32),
                torch.full((2, dimensions[name]), float(index), dtype=torch.float32),
            )
            for index, name in enumerate(order)
        }
        scores, features = validate_and_assemble_expert_outputs(outputs, profile="full_10")
        self.assertEqual(tuple(scores.shape), (2, 10))
        self.assertEqual(tuple(features.shape), (2, 30472))
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(width, features.shape[1])
        gate = GatingNetwork(input_feature_dim=width, num_experts=10, profile="full_10")
        weights = gate(features)
        prediction = (scores * weights).sum(dim=1, keepdim=True)
        self.assertEqual(tuple(weights.shape), (2, 10))
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(bool((weights >= 0).all()))
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        with self.assertRaises(ValueError):
            GatingNetwork(input_feature_dim=27399, num_experts=10, profile="full_10")

    def test_full10_feature_store_uses_schema_v2_and_label_free_test_rows(self):
        order, dimensions, width = profile_contract("full_10")
        source_hash = "a" * 64
        cache_ids = {name: f"{index + 1:064x}" for index, name in enumerate(order)}
        from data.moe_expert_cache import build_assembly_identity

        assembly = build_assembly_identity(
            expert_profile="full_10", expert_order=order,
            expert_cache_identities=cache_ids, feature_dimensions=dimensions,
            source_manifest_identity=source_hash,
        )
        provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"}
                      for index, name in enumerate(order)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation", "test"):
                writer = MoeFeatureStoreWriter(
                    root, split=split, expected_sample_ids=[f"{split}:sample.wav"],
                    source_manifest_sha256=source_hash, checkpoint_provenance=provenance,
                    preprocessing_identity={"policy": "full10-fixture"}, profile="full_10",
                    per_expert_cache_identities=cache_ids, assembly_identity=assembly,
                )
                writer.write_shard(
                    0, sample_ids=[f"{split}:sample.wav"], wave_file_names=["sample.wav"],
                    scores=np.full((1, 10), 0.5, dtype=np.float32),
                    features=np.full((1, width), 0.25, dtype=np.float32),
                    labels_raw=None if split == "test" else np.asarray([5.0], dtype=np.float32),
                )
                writer.finalize()
                manifest = validate_feature_store(root, split=split, require_labels=split != "test")
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["expert_profile"], "full_10")
                self.assertEqual(manifest["expert_order"], list(order))
                self.assertEqual(manifest["feature_dimensions"], dimensions)
                self.assertEqual(manifest["feature_width"], 30472)
                self.assertEqual(manifest["checkpoint_provenance"], provenance)
                self.assertEqual(manifest["per_expert_cache_identities"], cache_ids)
                self.assertEqual(manifest["assembly_identity"], assembly)
                if split == "test":
                    self.assertFalse(manifest["labels_present"])

    def test_full10_gate_checkpoint_round_trips_and_rejects_full9_cross_load(self):
        order, dimensions, width = profile_contract("full_10")
        provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"}
                      for index, name in enumerate(order)}
        identity = {
            "train": "train-store-id", "validation": "validation-store-id",
            "preprocessing_contract": {name: f"{index + 1:064x}" for index, name in enumerate(order)},
        }
        config = {"learning_rate": 0.001, "profile": "full_10"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full10-gate.pt"
            gate = GatingNetwork(input_feature_dim=width, num_experts=10, profile="full_10")
            optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            manager = MoeCheckpointManager()
            manager.save(
                path, gating_network=gate, optimizer=optimizer, scheduler=scheduler,
                epoch=3, global_step=41, config=config,
                feature_store_identity=identity, expert_checkpoint_provenance=provenance,
                best_srcc=0.42, patience=2, seed=7,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["expert_profile"], "full_10")
            self.assertEqual(payload["expert_order"], list(order))
            self.assertEqual(payload["expert_feature_dimensions"], dimensions)
            self.assertEqual(payload["gate_input_dim"], 30472)
            self.assertEqual(payload["gate_expert_count"], 10)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["global_step"], 41)
            self.assertTrue(payload["optimizer_state_dict"])
            self.assertIsInstance(payload["scheduler_state_dict"], dict)
            self.assertIsInstance(payload["torch_rng_state"], torch.Tensor)

            restored = GatingNetwork(input_feature_dim=width, num_experts=10, profile="full_10")
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
            manager.load(
                path, gating_network=restored, optimizer=restored_optimizer,
                scheduler=restored_scheduler, expected_config=config,
                expected_feature_store_identity=identity,
                expected_expert_checkpoint_provenance=provenance,
            )
            for name, value in gate.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)

            full9_order, _, full9_width = profile_contract("full_9")
            full9 = GatingNetwork(input_feature_dim=full9_width, num_experts=9, profile="full_9")
            full9_optimizer = torch.optim.AdamW(full9.parameters(), lr=0.001)
            full9_scheduler = torch.optim.lr_scheduler.StepLR(full9_optimizer, step_size=1)
            with self.assertRaisesRegex(ValueError, "profile|width|count"):
                manager.load_for_inference(
                    path, gating_network=full9,
                    expected_expert_checkpoint_provenance={
                        name: {"checkpoint_sha256": "f" * 64} for name in full9_order
                    },
                )

    def test_full10_checkpoint_rejects_missing_or_invalid_resume_state(self):
        order, dimensions, width = profile_contract("full_10")
        provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"}
                      for index, name in enumerate(order)}
        identity = {
            "train": "train-store-id", "validation": "validation-store-id",
            "preprocessing_contract": {name: f"{index + 1:064x}" for index, name in enumerate(order)},
        }
        config = {"learning_rate": 0.001, "profile": "full_10"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full10-gate.pt"
            gate = GatingNetwork(input_feature_dim=width, num_experts=len(order), profile="full_10")
            optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            manager = MoeCheckpointManager()
            manager.save(
                path, gating_network=gate, optimizer=optimizer, scheduler=scheduler,
                epoch=0, global_step=1, config=config,
                feature_store_identity=identity, expert_checkpoint_provenance=provenance,
                best_srcc=0.2,
            )
            original = torch.load(path, map_location="cpu", weights_only=True)
            corruptions = (
                lambda payload: payload.pop("optimizer_state_dict"),
                lambda payload: payload.pop("scheduler_state_dict"),
                lambda payload: payload.update(epoch=-1),
                lambda payload: payload.update(global_step=-1),
                lambda payload: payload.pop("torch_rng_state"),
                lambda payload: payload.update(torch_rng_state=torch.zeros(4, dtype=torch.float32)),
                lambda payload: payload.update(cuda_rng_states="invalid"),
            )
            for mutate in corruptions:
                corrupted = copy.deepcopy(original)
                mutate(corrupted)
                torch.save(corrupted, path)
                with self.subTest(mutation=mutate), self.assertRaisesRegex(
                    ValueError, "full_10 checkpoint resume state"
                ):
                    manager.load_for_inference(
                        path, gating_network=gate,
                        expected_expert_checkpoint_provenance=provenance,
                    )


if __name__ == "__main__":
    unittest.main()
