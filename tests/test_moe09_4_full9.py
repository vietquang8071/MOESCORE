"""Bounded integration coverage for the nine-expert feature-store gate."""

import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from data.moe_expert_cache import build_assembly_identity, build_expert_cache_identity, build_sample_identity
from data.moe_feature_store import MoeFeatureStoreWriter, validate_feature_store
from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from model.expert_adapters.registry import (
    EXPERT_PROFILES, profile_contract, resolve_expert_adapter_class,
    validate_and_assemble_expert_outputs,
)
from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager
from training.moe_evaluator import MoeGateEvaluator
from training.moe_trainer import MoeGateTrainer
from scripts.extract_moe_feature_store import (
    assemble_stores, assembly_store_root, chunk_path, completed_stores_match, fingerprint, save_chunk,
)
from configs.moe_config import load_moe_config, resolve_moe_paths
from scripts.kaggle_moe_pipeline import generate_configs


LOSS = {
    "loss_weights": [1.0, 0.0],
    "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
    "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
}


class FullNineIntegrationTests(unittest.TestCase):
    def test_full9_example_and_generated_config(self):
        project = Path(__file__).resolve().parents[1]
        example = project / "configs" / "moe_full9.example.yaml"
        config = resolve_moe_paths(load_moe_config(example), example)
        self.assertEqual(config["expert_profile"], "full_9")
        self.assertEqual(tuple(config["gate"]["expert_order"]), EXPERT_PROFILES["full_9"])
        self.assertEqual(config["experts"]["m2d_lora"]["finetuned_checkpoint"], project / "weights" / "best.pt")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _, resolved = generate_configs(project, root / "data", root / "weights",
                root / "cache", root / "stores", root / "out", root / "configs", profile="full_9",
                unilm_source_dir=root / "verified_unilm")
            self.assertEqual(load_moe_config(config_path)["expert_profile"], "full_9")
            self.assertEqual(resolved["experts"]["expert4"]["unilm_source_dir"], root / "verified_unilm")

    def test_cached_full9_sequential_assembly_and_corruption_rejection(self):
        order, dims, width = profile_contract("full_9")
        identities = {name: build_expert_cache_identity(
            expert_name=name, checkpoint_provenance={"checkpoint": name},
            implementation_provenance={"module": name}, preprocessing_provenance={"policy": "eval"},
            input_contract={"audio_key": name}, inference_policy={"frozen": True},
            score_contract={"feature_dim": dims[name]}, relevant_dependencies={"torch": "test"})
            for name in order}
        manifests = {}
        for split in ("train", "validation", "test"):
            record = {"sample_id": f"{split}:0.wav", "wave_file_name": "0.wav",
                      "raw_caption": "A caption", "wav_sha256": "a" * 64}
            if split != "test":
                record["label_raw"] = 5.0
            manifest = {"schema_version": 1, "split": split, "records": [record]}
            manifest["sha256"] = fingerprint(manifest)
            manifests[split] = manifest
        assemblies = {split: build_assembly_identity(expert_profile="full_9", expert_order=order,
            expert_cache_identities=identities, feature_dimensions=dims,
            source_manifest_identity=manifest["sha256"]) for split, manifest in manifests.items()}
        job = {"expert_profile": "full_9", "expert_order": list(order), "feature_dimensions": dims,
               "expert_cache_identities": identities, "checkpoint_provenance": {
                   name: {"checkpoint": name} for name in order}, "shard_size": 1,
               "manifests": {split: manifest["sha256"] for split, manifest in manifests.items()},
               "assembly_identities": assemblies,
               "assembly_ids": {split: value["assembly_id"] for split, value in assemblies.items()}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name_index, name in enumerate(order):
                identity = identities[name]
                for split, manifest in manifests.items():
                    records = manifest["records"]
                    sample = build_sample_identity(split, records, {
                        "expert_cache_id": identity["cache_id"], "input_contract": identity["input_contract"],
                        "preprocessing_provenance": identity["preprocessing_provenance"]}, dims[name])
                    path = chunk_path(root, name, split, 0, cache_id=identity["cache_id"],
                                      sample_identity_sha256=sample["sample_identity_sha256"])
                    save_chunk(path, expert=name, records=records, identity=identity, sample_identity=sample,
                               scores=np.full((1, 1), name_index / 10, dtype=np.float32),
                               features=np.full((1, dims[name]), name_index, dtype=np.float32),
                               factory_restoration="PASS_FROM_EXISTING_SOURCE_FACTORY", frozen_eval=True)
            # Prepared jobs are persisted as canonical JSON before workers or
            # standalone assembly read them; mapping key order can change.
            persisted_job = json.loads(json.dumps(job, sort_keys=True))
            result = assemble_stores(persisted_job, manifests, root)
            self.assertEqual(set(result), set(manifests))
            self.assertTrue(completed_stores_match(persisted_job, manifests, root))
            final = assembly_store_root(root, persisted_job)
            store = MoeTrainingDataset(str(final), split="train")
            self.assertEqual(tuple(store[0]["scores"].shape), (9,))
            self.assertEqual(tuple(store[0]["features"].shape), (width,))
            self.assertTrue(torch.equal(store[0]["scores"], torch.arange(9).float() / 10))
            shard = final / "train" / "shards" / "shard_000000" / "features.npy"
            with shard.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_feature_store(final, split="train")

    def test_registry_assembly_and_gate(self):
        order, dims, width = profile_contract("full_9")
        self.assertEqual(order, EXPERT_PROFILES["full_9"])
        self.assertEqual(width, 32257)
        self.assertEqual(profile_contract("top1_4")[2], 12544)
        for name in order:
            self.assertTrue(callable(resolve_expert_adapter_class(name).from_config))
        outputs = {
            name: (torch.full((2, 1), float(index), dtype=torch.float32),
                   torch.full((2, dims[name]), float(index), dtype=torch.float32))
            for index, name in enumerate(order)
        }
        scores, features = validate_and_assemble_expert_outputs(outputs, profile="full_9")
        self.assertEqual(tuple(scores.shape), (2, 9))
        self.assertEqual(tuple(features.shape), (2, 32257))
        offset = 0
        for index, name in enumerate(order):
            self.assertTrue(torch.all(features[:, offset:offset + dims[name]] == index))
            offset += dims[name]
        gate = GatingNetwork(input_feature_dim=width, num_experts=9, profile="full_9")
        weights = gate(features)
        prediction = (scores * weights).sum(dim=1, keepdim=True)
        self.assertEqual(tuple(weights.shape), (2, 9))
        self.assertEqual(tuple(prediction.shape), (2, 1))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        self.assertTrue(bool((weights >= 0).all()))
        torch.testing.assert_close(prediction[:, 0], sum(scores[:, i] * weights[:, i] for i in range(9)))
        with self.assertRaises(ValueError):
            validate_and_assemble_expert_outputs({key: value for key, value in outputs.items() if key != "m2d_lora"}, profile="full_9")
        with self.assertRaises(ValueError):
            GatingNetwork(input_feature_dim=12544, num_experts=9, profile="full_9")

    def test_full9_store_train_checkpoint_and_infer(self):
        order, dims, width = profile_contract("full_9")
        cache_ids = {name: f"{index + 1:064x}" for index, name in enumerate(order)}
        source = "a" * 64
        assembly = build_assembly_identity(expert_profile="full_9", expert_order=order,
            expert_cache_identities=cache_ids, feature_dimensions=dims, source_manifest_identity=source)
        provenance = {name: {"checkpoint_sha256": f"{index + 1:064x}"} for index, name in enumerate(order)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation", "test"):
                writer = MoeFeatureStoreWriter(root, split=split,
                    expected_sample_ids=[f"{split}:0.wav", f"{split}:1.wav"],
                    source_manifest_sha256=source, checkpoint_provenance=provenance,
                    preprocessing_identity={"policy": "synthetic"}, profile="full_9",
                    per_expert_cache_identities=cache_ids, assembly_identity=assembly)
                writer.write_shard(0, sample_ids=[f"{split}:0.wav", f"{split}:1.wav"],
                    wave_file_names=["0.wav", "1.wav"],
                    scores=np.full((2, 9), 0.2, dtype=np.float32),
                    features=np.full((2, width), 0.01, dtype=np.float32),
                    labels_raw=None if split == "test" else np.array([5.0, 6.0], dtype=np.float32))
                writer.finalize()
                manifest = validate_feature_store(root, split=split)
                self.assertEqual(manifest["expert_profile"], "full_9")
                self.assertEqual(manifest["feature_width"], width)
            train = MoeTrainingDataset(str(root), split="train")
            batch = collate_moe_features([train[0], train[1]])
            gate = GatingNetwork(input_feature_dim=width, num_experts=9, profile="full_9")
            trainer = MoeGateTrainer(gate, LOSS)
            result = trainer.train_step(batch)
            self.assertEqual(tuple(result["prediction"].shape), (2, 1))
            self.assertTrue(bool(torch.isfinite(result["total_loss"])))
            self.assertGreater(result["updated_parameter_count"], 0)
            manager = MoeCheckpointManager()
            path = root / "gate.pt"
            identity = {"train": train.identity, "validation": validate_feature_store(root, split="validation")["store_identity_sha256"],
                        "preprocessing_contract": cache_ids}
            manager.save(path, gating_network=gate, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
                epoch=0, global_step=1, config={"test": True}, feature_store_identity=identity,
                expert_checkpoint_provenance=provenance)
            restored = GatingNetwork(input_feature_dim=width, num_experts=9, profile="full_9")
            restored_trainer = MoeGateTrainer(restored, LOSS)
            manager.load(path, gating_network=restored, optimizer=restored_trainer.optimizer,
                scheduler=restored_trainer.scheduler, expected_config={"test": True},
                expected_feature_store_identity=identity, expected_expert_checkpoint_provenance=provenance)
            validation = MoeTrainingDataset(str(root), split="validation")
            validation_batch = collate_moe_features([validation[0], validation[1]])
            evaluated = MoeGateEvaluator(restored).evaluate([validation_batch])
            self.assertEqual(evaluated["metrics"]["sample_count"], 2)
            with self.assertRaises(ValueError):
                manager.load_for_inference(path, gating_network=GatingNetwork(),
                    expected_expert_checkpoint_provenance=provenance)
            test = MoeTrainingDataset(str(root), split="test", require_labels=False)
            test_batch = collate_moe_features([test[0], test[1]])
            self.assertNotIn("labels_raw", test_batch)
            rows = MoeGateEvaluator(restored).predict([test_batch])
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["sample_id"] for row in rows], test.store.sample_ids)
            with self.assertRaises(ValueError):
                collate_moe_features([train[0], test[0]])


if __name__ == "__main__":
    unittest.main()
