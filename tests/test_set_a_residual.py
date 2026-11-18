from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data.moe_expert_cache import build_assembly_identity, build_expert_cache_identity
from data.moe_feature_store import MoeFeatureStoreWriter
from data.set_a_cached_dataset import (
    SetACachedDataset,
    write_set_a_cache,
)
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.registry import (
    FULL9_EXPERT_ORDER,
    TOP2_EXPERT_INPUT_CONTRACTS,
    profile_contract,
)
from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_DIMS,
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_DIM,
    SetAResidualHead,
)
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter


REPRESENTATION = {
    "feature_representation": "paired_cross_4d_plus_1_v2",
    "feature_blocks": [
        "audio_embedding",
        "text_embedding",
        "absolute_difference_l2",
        "elementwise_product_l2",
        "cosine_l2_scalar",
    ],
    "feature_width_formula": "4D+1",
}


def make_component_outputs(batch_size: int = 2):
    result = {}
    for index, name in enumerate(SET_A_COMPONENT_ORDER):
        dim = SET_A_COMPONENT_DIMS[name]
        generator = torch.Generator().manual_seed(120 + index)
        audio = torch.randn(batch_size, dim, generator=generator)
        text = torch.randn(batch_size, dim, generator=generator)
        features = build_paired_cross_features(audio, text)
        result[name] = {
            "score": features[:, -1:],
            "features": features,
        }
    return result


def make_full9_store(root: Path, *, split: str = "train", representation=REPRESENTATION,
                     ids: list[str] | None = None, labels: bool = True, mga_sample_rate: int = 32_000,
                     label_values: list[float] | None = None,
                     stored_label_values: list[float] | None = None,
                     mga_input_contract: dict | None = None,
                     group_ids: list[str] | None = None):
    order, dimensions, total_width = profile_contract("full_9")
    ids = ids or [f"{split}:{index}.wav" for index in range(2)]
    if label_values is None:
        label_values = [5.0, 8.0] if len(ids) == 2 else [5.0] * len(ids)
    if stored_label_values is None:
        stored_label_values = list(label_values)
    if len(label_values) != len(ids) or len(stored_label_values) != len(ids):
        raise ValueError("synthetic source and stored labels must match the sample count")
    if group_ids is not None and len(group_ids) != len(ids):
        raise ValueError("synthetic group IDs must match the sample count")
    source_records = []
    for index, sample_id in enumerate(ids):
        record = {
            "sample_id": sample_id,
            "wave_file_name": sample_id.split(":", 1)[1],
            "raw_caption": f"caption {index}",
            "wav_sha256": f"{index + 1:064x}",
        }
        if split != "test" and labels:
            record["label_raw"] = label_values[index]
        if group_ids is not None:
            record["group_id"] = group_ids[index]
        source_records.append(record)
    source_body = {"schema_version": 1, "split": split, "records": source_records}
    source_hash = hashlib.sha256(
        json.dumps(source_body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    source_manifest = {**source_body, "sha256": source_hash}
    cache_identities = {}
    checkpoint_provenance = {}
    for index, name in enumerate(order):
        score_contract = {
            "score_shape": [None, 1],
            "feature_shape": [None, dimensions[name]],
            "score_dtype": "float32",
            "feature_dtype": "float32",
        }
        if name in SET_A_COMPONENT_ORDER:
            score_contract.update(representation)
        input_contract = asdict(TOP2_EXPERT_INPUT_CONTRACTS[name]) if name in SET_A_COMPONENT_ORDER else {
            "audio_key": "expert3_wavs", "sample_rate_hz": 16_000, "max_samples": 160_000
        }
        if name == "top2_mga":
            input_contract["sample_rate_hz"] = mga_sample_rate
            if mga_input_contract:
                input_contract.update(mga_input_contract)
        cache_identities[name] = build_expert_cache_identity(
            expert_name=name,
            checkpoint_provenance={"sha256": f"{index + 1:064x}"},
            implementation_provenance={"source": "synthetic-test"},
            preprocessing_provenance={"policy": "synthetic-test"},
            input_contract=input_contract,
            inference_policy={"module_mode": "eval"},
            score_contract=score_contract,
            relevant_dependencies={"torch": "test"},
        )
        checkpoint_provenance[name] = {"sha256": f"{index + 1:064x}"}
    assembly = build_assembly_identity(
        expert_profile="full_9",
        expert_order=order,
        expert_cache_identities=cache_identities,
        feature_dimensions=dimensions,
        source_manifest_identity=source_hash,
    )

    sample_count = len(ids)
    scores = np.full((sample_count, len(order)), 0.25, dtype=np.float32)
    features = np.zeros((sample_count, total_width), dtype=np.float32)
    offsets = {}
    offset = 0
    for name in order:
        offsets[name] = offset
        offset += dimensions[name]
    for name in SET_A_COMPONENT_ORDER:
        component = make_component_outputs(sample_count)[name]
        block = component["features"].numpy()
        start = offsets[name]
        features[:, start:start + dimensions[name]] = block
        scores[:, order.index(name)] = block[:, -1]

    writer = MoeFeatureStoreWriter(
        root,
        split=split,
        expected_sample_ids=ids,
        source_manifest_sha256=source_hash,
        checkpoint_provenance=checkpoint_provenance,
        preprocessing_identity={"policy": "synthetic-test", "source_manifest_sha256": source_hash},
        profile="full_9",
        per_expert_cache_identities=cache_identities,
        assembly_identity=assembly,
    )
    writer.write_shard(
        0,
        sample_ids=ids,
        wave_file_names=[sample_id.split(":", 1)[1] for sample_id in ids],
        scores=scores,
        features=features,
        labels_raw=np.asarray(stored_label_values, dtype=np.float32) if labels else None,
    )
    writer.finalize()
    source_path = root / split / "source_manifest.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    return cache_identities, checkpoint_provenance


class SetAResidualHeadTests(unittest.TestCase):
    def test_head_uses_exact_arithmetic_mean_affine_calibrator_and_residual(self):
        head = SetAResidualHead(hidden_dims=(16, 8), dropout=0.0)
        features = torch.randn(2, SET_A_FEATURE_DIM)
        component_scores = torch.tensor([[0.1, 0.3, 0.8], [-0.4, 0.2, 0.9]])
        with torch.no_grad():
            head.scale.fill_(1.7)
            head.bias.fill_(-0.2)
            head.residual_mlp[-1].bias.fill_(0.35)
        score = head(component_scores, features)
        mean = (component_scores[:, 0:1] + component_scores[:, 1:2] + component_scores[:, 2:3]) / 3.0
        expected = head.scale * mean + head.bias + head.residual_mlp(features)
        torch.testing.assert_close(score, expected)
        self.assertEqual(tuple(score.shape), (2, 1))

    def test_residual_head_starts_with_zero_correction(self):
        head = SetAResidualHead(hidden_dims=(12, 6), dropout=0.0)
        features = torch.randn(1, SET_A_FEATURE_DIM)
        scores = torch.tensor([[0.15, 0.45, -0.3]])
        with torch.no_grad():
            output = head(scores, features)
            baseline = (scores[:, 0:1] + scores[:, 1:2] + scores[:, 2:3]) / 3.0
        torch.testing.assert_close(output, baseline)
        torch.testing.assert_close(head.residual_mlp[-1].weight, torch.zeros_like(head.residual_mlp[-1].weight))
        torch.testing.assert_close(head.residual_mlp[-1].bias, torch.zeros_like(head.residual_mlp[-1].bias))

    def test_calibration_and_regression_parameters_receive_gradients(self):
        head = SetAResidualHead(hidden_dims=(12, 6), dropout=0.0)
        features = torch.randn(3, SET_A_FEATURE_DIM)
        scores = torch.randn(3, 3)
        head(scores, features).square().mean().backward()
        self.assertIsNotNone(head.scale.grad)
        self.assertIsNotNone(head.bias.grad)
        self.assertIsNotNone(head.residual_mlp[-1].weight.grad)

    def test_head_rejects_invalid_width_nonfinite_values_and_batch_mismatch(self):
        head = SetAResidualHead(hidden_dims=(8, 4), dropout=0.0)
        with self.assertRaisesRegex(ValueError, "feature width"):
            head(torch.zeros(1, 3), torch.zeros(1, SET_A_FEATURE_DIM - 1))
        bad_features = torch.zeros(1, SET_A_FEATURE_DIM)
        bad_features[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            head(torch.zeros(1, 3), bad_features)
        with self.assertRaisesRegex(ValueError, "batch"):
            head(torch.zeros(2, 3), torch.zeros(1, SET_A_FEATURE_DIM))


class SetAResidualAdapterTests(unittest.TestCase):
    def test_shared_4d_plus_1_helper_preserves_embeddings_and_normalizes_interactions(self):
        audio = torch.tensor([[3.0, 4.0]])
        text = torch.tensor([[1.0, 0.0]])
        features = build_paired_cross_features(audio, text)
        self.assertEqual(tuple(features.shape), (1, 9))
        torch.testing.assert_close(features[:, :2], audio)
        torch.testing.assert_close(features[:, 2:4], text)
        torch.testing.assert_close(features[:, 4:6], torch.tensor([[0.4, 0.8]]))
        torch.testing.assert_close(features[:, 6:8], torch.tensor([[0.6, 0.0]]))
        torch.testing.assert_close(features[:, 8:], torch.tensor([[0.6]]))

    def test_returns_exact_ordered_uncompressed_features_and_expert_adapter_pair(self):
        outputs = make_component_outputs(batch_size=3)
        adapter = SetAResidualAdapter(SetAResidualHead(hidden_dims=(16, 8), dropout=0.0))
        score, features = adapter({"component_outputs": outputs})
        expected_features = torch.cat([outputs[name]["features"] for name in SET_A_COMPONENT_ORDER], dim=-1)
        torch.testing.assert_close(features, expected_features)
        self.assertEqual(tuple(score.shape), (3, 1))
        self.assertEqual(tuple(features.shape), (3, 11267))
        self.assertEqual(adapter.feature_dim, 11267)
        self.assertEqual(len((score, features)), 2)

    def test_batch_one_and_component_cosine_consistency(self):
        adapter = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
        outputs = make_component_outputs(batch_size=1)
        score, features = adapter({"component_outputs": outputs})
        self.assertEqual(tuple(score.shape), (1, 1))
        self.assertEqual(tuple(features.shape), (1, 11267))
        for name in SET_A_COMPONENT_ORDER:
            component = outputs[name]
            torch.testing.assert_close(component["score"], component["features"][:, -1:])

    def test_adapter_rejects_missing_extra_width_nonfinite_and_inconsistent_score(self):
        adapter = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
        outputs = make_component_outputs()
        with self.assertRaisesRegex(ValueError, "component names"):
            adapter({"component_outputs": {name: value for name, value in outputs.items() if name != "top2_mga"}})
        bad = make_component_outputs()
        bad["top2_mga"]["features"] = bad["top2_mga"]["features"][:, :-1]
        with self.assertRaisesRegex(ValueError, "width"):
            adapter({"component_outputs": bad})
        bad = make_component_outputs()
        bad["top2_m2d"]["features"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter({"component_outputs": bad})
        bad = make_component_outputs()
        bad["top2_msclap"]["score"] = bad["top2_msclap"]["score"] + 0.1
        with self.assertRaisesRegex(ValueError, "cosine"):
            adapter({"component_outputs": bad})

    def test_adapter_rejects_mixed_devices_and_dtypes(self):
        adapter = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
        outputs = make_component_outputs()
        outputs["top2_m2d"] = {
            "score": torch.empty((2, 1), device="meta"),
            "features": torch.empty((2, 3073), device="meta"),
        }
        with self.assertRaisesRegex(ValueError, "device"):
            adapter({"component_outputs": outputs})
        outputs = make_component_outputs()
        outputs["top2_m2d"]["score"] = outputs["top2_m2d"]["score"].double()
        with self.assertRaisesRegex(TypeError, "float32"):
            adapter({"component_outputs": outputs})

    def test_adapter_detaches_frozen_component_outputs_but_trains_head(self):
        outputs = make_component_outputs()
        for component in outputs.values():
            component["score"].requires_grad_()
            component["features"].requires_grad_()
        adapter = SetAResidualAdapter(SetAResidualHead(hidden_dims=(8, 4), dropout=0.0))
        adapter({"component_outputs": outputs})[0].sum().backward()
        self.assertIsNotNone(adapter.head.scale.grad)
        for component in outputs.values():
            self.assertIsNone(component["score"].grad)
            self.assertIsNone(component["features"].grad)


class SetACachedDatasetTests(unittest.TestCase):
    def test_uses_verified_full9_store_order_and_hides_nontrain_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_full9_store(root, split="train")
            train = SetACachedDataset(root, split="train")
            self.assertEqual(train.sample_ids, ["train:0.wav", "train:1.wav"])
            item = train[0]
            self.assertEqual(tuple(item["component_outputs"]), SET_A_COMPONENT_ORDER)
            self.assertEqual(tuple(item["component_outputs"]["top2_m2d"]["features"].shape), (3073,))
            self.assertEqual(tuple(item["component_outputs"]["top2_msclap"]["features"].shape), (4097,))
            self.assertEqual(tuple(item["component_outputs"]["top2_mga"]["features"].shape), (4097,))
            expected_scores = train.store[0]["scores"]
            offset = 0
            for name in FULL9_EXPERT_ORDER:
                width = profile_contract("full_9")[1][name]
                if name in SET_A_COMPONENT_ORDER:
                    output = item["component_outputs"][name]
                    torch.testing.assert_close(output["score"], expected_scores[FULL9_EXPERT_ORDER.index(name):FULL9_EXPERT_ORDER.index(name) + 1])
                offset += width
            self.assertEqual(tuple(train.component_model_identities), SET_A_COMPONENT_ORDER)
            self.assertAlmostEqual(float(item["target_normalized"]), 0.0)
            make_full9_store(root, split="test", labels=False)
            test = SetACachedDataset(root, split="test")
            self.assertNotIn("label_raw", test[0])
            self.assertNotIn("target_normalized", test[0])

    def test_rejects_old_or_missing_feature_representation_provenance(self):
        bad_representation = dict(REPRESENTATION)
        bad_representation["feature_representation"] = "paired_cross_5d_legacy"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_full9_store(root, representation=bad_representation)
            with self.assertRaisesRegex(ValueError, "representation"):
                SetACachedDataset(root, split="train")

    def test_accepts_full9_mga_32khz_contract_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "valid"
            make_full9_store(root)
            dataset = SetACachedDataset(root, split="train")
            contract = dataset.component_cache_identities["top2_mga"]["input_contract"]
            self.assertEqual(contract["audio_key"], "expert2_wavs")
            self.assertEqual(contract["audio_shape"], [None, 1, 320000])
            self.assertEqual(contract["sample_rate_hz"], 32000)
            self.assertEqual(contract["max_samples"], 320000)

            invalid_contracts = (
                {"sample_rate_hz": 16_000},
                {"max_samples": 160_000},
                {"audio_key": "expert3_wavs"},
                {"audio_shape": [None, 1, 160000]},
            )
            for index, changes in enumerate(invalid_contracts):
                bad_root = Path(temporary) / f"bad-{index}"
                make_full9_store(bad_root, mga_input_contract=changes)
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Full9.*32 kHz"):
                    SetACachedDataset(bad_root, split="train")

    def test_writes_and_reads_standalone_e5_cache_with_verified_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            identities, models = make_full9_store(source_root, split="train")
            source_path = source_root / "train" / "source_manifest.json"
            component_outputs = make_component_outputs(batch_size=2)
            sample_ids = ["train:0.wav", "train:1.wav"]
            output_root = root / "e5_cache"
            set_a_identities = {name: identities[name] for name in SET_A_COMPONENT_ORDER}
            set_a_models = {name: models[name] for name in SET_A_COMPONENT_ORDER}
            write_set_a_cache(
                output_root,
                split="train",
                source_manifest_path=source_path,
                component_outputs=component_outputs,
                component_sample_ids={name: sample_ids for name in SET_A_COMPONENT_ORDER},
                component_cache_identities=set_a_identities,
                component_model_identities=set_a_models,
            )
            dataset = SetACachedDataset(output_root, split="train")
            self.assertEqual(dataset.sample_ids, sample_ids)
            self.assertEqual(tuple(dataset[1]["component_outputs"]), SET_A_COMPONENT_ORDER)
            self.assertEqual(tuple(dataset[0]["component_outputs"]["top2_mga"]["features"].shape), (4097,))
            self.assertAlmostEqual(float(dataset[0]["target_normalized"]), 0.0)

    def test_decimal_labels_round_trip_in_full9_and_standalone_caches(self):
        values = [6.7, 4.43, 5.9, 7.1, 0.0, 10.0]
        ids = [f"train:{index}.wav" for index in range(len(values))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full9_root = root / "full9"
            identities, models = make_full9_store(
                full9_root,
                ids=ids,
                label_values=values,
            )
            full9 = SetACachedDataset(full9_root, split="train")
            for index, expected in enumerate(values):
                self.assertEqual(
                    np.float32(full9[index]["label_raw"]).item(),
                    np.float32(expected).item(),
                )

            standalone_root = root / "standalone"
            write_set_a_cache(
                standalone_root,
                split="train",
                source_manifest_path=full9_root / "train" / "source_manifest.json",
                component_outputs=make_component_outputs(len(values)),
                component_sample_ids={name: ids for name in SET_A_COMPONENT_ORDER},
                component_cache_identities={name: identities[name] for name in SET_A_COMPONENT_ORDER},
                component_model_identities={name: models[name] for name in SET_A_COMPONENT_ORDER},
            )
            standalone = SetACachedDataset(standalone_root, split="train")
            for index, expected in enumerate(values):
                self.assertEqual(
                    np.float32(standalone[index]["label_raw"]).item(),
                    np.float32(expected).item(),
                )

    def test_rejects_genuinely_changed_full9_and_standalone_labels(self):
        ids = ["train:0.wav", "train:1.wav"]
        values = [6.7, 4.43]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full9_root = root / "full9"
            make_full9_store(
                full9_root,
                ids=ids,
                label_values=values,
                stored_label_values=[6.71, 4.43],
            )
            with self.assertRaisesRegex(ValueError, "source manifest"):
                SetACachedDataset(full9_root, split="train")[0]

            valid_root = root / "valid"
            identities, models = make_full9_store(valid_root, ids=ids, label_values=values)
            standalone_root = root / "standalone"
            write_set_a_cache(
                standalone_root,
                split="train",
                source_manifest_path=valid_root / "train" / "source_manifest.json",
                component_outputs=make_component_outputs(2),
                component_sample_ids={name: ids for name in SET_A_COMPONENT_ORDER},
                component_cache_identities={name: identities[name] for name in SET_A_COMPONENT_ORDER},
                component_model_identities={name: models[name] for name in SET_A_COMPONENT_ORDER},
            )
            split_root = standalone_root / "train"
            labels_path = split_root / "labels_raw.npy"
            labels = np.load(labels_path, allow_pickle=False)
            labels[0] = np.float32(6.71)
            np.save(labels_path, labels, allow_pickle=False)
            manifest_path = split_root / "set_a_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["array_file_sha256"]["labels_raw.npy"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
            body = {key: value for key, value in manifest.items() if key != "set_a_cache_identity_sha256"}
            manifest["set_a_cache_identity_sha256"] = hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source manifest"):
                SetACachedDataset(standalone_root, split="train")[0]

    def test_writer_rejects_component_row_order_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            identities, models = make_full9_store(source_root, split="train")
            sample_ids = ["train:0.wav", "train:1.wav"]
            ids_by_component = {name: list(sample_ids) for name in SET_A_COMPONENT_ORDER}
            ids_by_component["top2_mga"].reverse()
            set_a_identities = {name: identities[name] for name in SET_A_COMPONENT_ORDER}
            set_a_models = {name: models[name] for name in SET_A_COMPONENT_ORDER}
            with self.assertRaisesRegex(ValueError, "sample IDs/order"):
                write_set_a_cache(
                    root / "e5_cache",
                    split="train",
                    source_manifest_path=source_root / "train" / "source_manifest.json",
                    component_outputs=make_component_outputs(batch_size=2),
                    component_sample_ids=ids_by_component,
                    component_cache_identities=set_a_identities,
                    component_model_identities=set_a_models,
                )

    def test_full9_features_and_scores_are_read_from_one_store_without_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_full9_store(root, split="train")
            dataset = SetACachedDataset(root, split="train")
            stored = dataset.store[0]
            sample = dataset[0]
            order, dimensions, _ = profile_contract("full_9")
            score_blocks = []
            feature_blocks = []
            offset = 0
            for name in order:
                width = dimensions[name]
                if name in SET_A_COMPONENT_ORDER:
                    component = sample["component_outputs"][name]
                    expected_score = stored["scores"][order.index(name):order.index(name) + 1]
                    torch.testing.assert_close(component["score"], expected_score)
                    torch.testing.assert_close(component["features"], stored["features"][offset:offset + width])
                    score_blocks.append(component["score"])
                    feature_blocks.append(component["features"])
                offset += width
            self.assertEqual(torch.stack(score_blocks).shape, (3, 1))
            self.assertEqual(torch.cat(feature_blocks, dim=-1).shape, (11267,))
            self.assertEqual(dataset.identity, dataset.store.identity)

    def test_store_rejects_duplicate_or_missing_sample_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                make_full9_store(Path(temporary), ids=["train:0.wav", "train:0.wav"])

    def test_rejects_source_manifest_with_missing_sample_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_full9_store(root, split="train")
            source_path = root / "train" / "source_manifest.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source["records"].pop()
            source_body = {key: value for key, value in source.items() if key != "sha256"}
            source["sha256"] = hashlib.sha256(
                json.dumps(source_body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source manifest identity"):
                SetACachedDataset(root, split="train")


if __name__ == "__main__":
    unittest.main()
