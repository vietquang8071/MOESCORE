"""Regression tests for MOE-09.2 cache identity and deterministic MS-CLAP input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data.moe_expert_cache import (
    build_assembly_identity,
    build_expert_cache_identity,
    build_sample_identity,
    load_expert_shard,
    resolve_manifest_wav_paths,
    save_expert_shard,
    validate_assembly_identity,
)
from model.expert_adapters.top2_msclap_adapter import (
    _deterministic_load_audio_into_tensor,
)
from model.expert_adapters.registry import TOP2_EXPERT_INPUT_CONTRACTS
from scripts.extract_moe_feature_store import (
    _text_snapshot_identity,
    prepare_expert_inputs,
    projected_dataset,
)


def _expert_identity(name, *, checkpoint="a", policy=None):
    return build_expert_cache_identity(
        expert_name=name,
        checkpoint_provenance={"sha256": hashlib.sha256(checkpoint.encode()).hexdigest()},
        implementation_provenance={"adapter.py": "implementation-1"},
        preprocessing_provenance={"dataset_method": "selected-preprocessor-1"},
        input_contract={"audio_key": "audio", "requires_wav_paths": False},
        inference_policy=policy or {"mode": "eval", "crop": "none"},
        score_contract={"score_shape": [None, 1], "feature_dim": 3, "score": "adapter-defined"},
        relevant_dependencies={"torch": "test-version"},
    )


def _records(root: Path):
    audio = root / "train" / "a.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"wav-one")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    row = {
        "sample_id": "train:a.wav",
        "wave_file_name": "a.wav",
        "captions_raw": " A caption ",
        "audio_path": audio,
        "split": "train",
    }
    record = {
        "sample_id": row["sample_id"],
        "wave_file_name": row["wave_file_name"],
        "raw_caption": row["captions_raw"],
        "wav_sha256": digest,
    }
    return audio.parent, row, record


class PerExpertIdentityTests(unittest.TestCase):
    def test_huggingface_text_identity_binds_snapshot_files_and_selected_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            model_root = cache / "models--publisher--text-model"
            refs = model_root / "refs"
            snapshots = model_root / "snapshots"
            refs.mkdir(parents=True)
            for revision in ("rev-a", "rev-b"):
                path = snapshots / revision
                path.mkdir(parents=True)
                (path / "tokenizer.json").write_text(revision)
            ref_file = refs / "main"
            ref_file.write_text("rev-a")

            first = _text_snapshot_identity(
                "publisher/text-model", {"cache_dir": str(cache)}, {}
            )
            ref_file.write_text("rev-b")
            second = _text_snapshot_identity(
                "publisher/text-model", {"cache_dir": str(cache)}, {}
            )

            self.assertNotEqual(first, second)
            self.assertEqual(first["cache_views"][0]["refs"]["main"], "rev-a")
            self.assertEqual(second["cache_views"][0]["refs"]["main"], "rev-b")

    def test_registering_more_experts_does_not_change_existing_cache_ids(self):
        top1 = {name: _expert_identity(name) for name in ("expert1", "expert2", "expert3", "expert4")}
        expanded = {
            name: _expert_identity(name)
            for name in (
                "expert1", "expert2", "expert3", "expert4", "top2_m2d",
                "top2_msclap", "top2_laion", "top2_mga", "m2d_lora",
            )
        }

        for name, identity in top1.items():
            self.assertEqual(identity["cache_id"], expanded[name]["cache_id"])

    def test_checkpoint_and_msclap_crop_changes_invalidate_only_that_expert(self):
        expert1_before = _expert_identity("expert1")
        m2d_before = _expert_identity("top2_m2d")
        msclap_before = _expert_identity(
            "top2_msclap", policy={"audio_crop_policy": "center", "duration": 7}
        )

        expert1_after = _expert_identity("expert1")
        m2d_after = _expert_identity("top2_m2d", checkpoint="changed")
        msclap_after = _expert_identity(
            "top2_msclap", policy={"audio_crop_policy": "random", "duration": 7}
        )

        self.assertEqual(expert1_before["cache_id"], expert1_after["cache_id"])
        self.assertNotEqual(m2d_before["cache_id"], m2d_after["cache_id"])
        self.assertNotEqual(msclap_before["cache_id"], msclap_after["cache_id"])

    def test_assembly_identity_depends_on_profile_order_and_manifest(self):
        names = ["expert1", "expert2", "expert3", "expert4"]
        identities = {name: _expert_identity(name) for name in names}
        dims = {name: 3 for name in names}
        base = build_assembly_identity(
            expert_profile="top1_4",
            expert_order=names,
            expert_cache_identities=identities,
            feature_dimensions=dims,
            source_manifest_identity="a" * 64,
        )

        reordered = build_assembly_identity(
            expert_profile="top1_4",
            expert_order=list(reversed(names)),
            expert_cache_identities=identities,
            feature_dimensions=dims,
            source_manifest_identity="a" * 64,
        )
        expanded_names = names + ["top2_m2d", "top2_msclap", "top2_laion", "top2_mga", "m2d_lora"]
        expanded = {**identities, **{name: _expert_identity(name) for name in expanded_names[4:]}}
        full = build_assembly_identity(
            expert_profile="full_9",
            expert_order=expanded_names,
            expert_cache_identities=expanded,
            feature_dimensions={name: 3 for name in expanded_names},
            source_manifest_identity="a" * 64,
        )
        other_data = build_assembly_identity(
            expert_profile="top1_4",
            expert_order=names,
            expert_cache_identities=identities,
            feature_dimensions=dims,
            source_manifest_identity="b" * 64,
        )

        self.assertNotEqual(base["assembly_id"], reordered["assembly_id"])
        self.assertNotEqual(base["assembly_id"], full["assembly_id"])
        self.assertNotEqual(base["assembly_id"], other_data["assembly_id"])
        validate_assembly_identity(base, expected_expert_order=names)
        with self.assertRaisesRegex(ValueError, "order"):
            validate_assembly_identity(base, expected_expert_order=list(reversed(names)))

    def test_sample_identity_binds_order_caption_audio_and_expected_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, record = _records(Path(temporary))
            sample = build_sample_identity(
                split="train",
                records=[record],
                preprocessing_input_identity={"contract": "expert2-32khz"},
                feature_dim=3,
            )
            self.assertEqual(sample["expected_sample_count"], 1)
            self.assertEqual(sample["expected_score_shape"], [1, 1])
            self.assertEqual(sample["expected_feature_shape"], [1, 3])
            self.assertEqual(sample["ordered_samples"][0]["caption"], " A caption ")
            caption_change = {**record, "raw_caption": "different caption"}
            wav_change = {**record, "wav_sha256": "0" * 64}
            self.assertNotEqual(
                sample["sample_identity_sha256"],
                build_sample_identity("train", [caption_change], {"contract": "expert2-32khz"}, 3)["sample_identity_sha256"],
            )
            self.assertNotEqual(
                sample["sample_identity_sha256"],
                build_sample_identity("train", [wav_change], {"contract": "expert2-32khz"}, 3)["sample_identity_sha256"],
            )

    def test_batch_size_is_not_part_of_expert_output_identity(self):
        one = _expert_identity("expert1")
        another_extraction_run = _expert_identity("expert1")
        self.assertEqual(one["cache_id"], another_extraction_run["cache_id"])

    def test_atomic_shards_reuse_only_after_identity_and_content_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, record = _records(root)
            expert = _expert_identity("expert1")
            sample = build_sample_identity("train", [record], {"dataset": "v1"}, 3)
            path = root / "shard_000000.npz"
            scores = np.asarray([[0.25]], dtype=np.float32)
            features = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
            save_expert_shard(path, expert_identity=expert, sample_identity=sample,
                              scores=scores, features=features,
                              inference_metadata={
                                  "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY",
                                  "frozen_eval": True,
                              })
            loaded = load_expert_shard(path, expert_identity=expert, sample_identity=sample, feature_dim=3)
            np.testing.assert_array_equal(loaded["scores"], scores)
            np.testing.assert_array_equal(loaded["features"], features)

            changed = {**expert, "cache_id": "f" * 64}
            with self.assertRaisesRegex(ValueError, "expert cache identity"):
                load_expert_shard(path, expert_identity=changed, sample_identity=sample, feature_dim=3)
            path.write_bytes(b"corrupt npz")
            with self.assertRaisesRegex(ValueError, "hash|corrupt"):
                load_expert_shard(path, expert_identity=expert, sample_identity=sample, feature_dim=3)

    def test_wav_path_resolution_checks_root_identity_hash_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root, row, record = _records(root)
            dataset = SimpleNamespace(audio_dir=audio_root, rows=[row])
            paths = resolve_manifest_wav_paths(dataset, [record], split="train", start=0)
            self.assertEqual(paths, [str((audio_root / "a.wav").resolve())])

            wrong_hash = {**record, "wav_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_manifest_wav_paths(dataset, [wrong_hash], split="train", start=0)
            wrong_name = {**record, "wave_file_name": "../a.wav"}
            with self.assertRaisesRegex(ValueError, "filename|identity"):
                resolve_manifest_wav_paths(dataset, [wrong_name], split="train", start=0)
            dataset.rows.append(dict(row))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resolve_manifest_wav_paths(dataset, [record, record], split="train", start=0)

    def test_wav_path_resolution_rejects_escape_and_missing_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root, row, record = _records(root)
            outside = root / "outside.wav"
            outside.write_bytes(b"wav-one")
            row["audio_path"] = outside
            dataset = SimpleNamespace(audio_dir=audio_root, rows=[row])
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_manifest_wav_paths(dataset, [record], split="train", start=0)

            row["audio_path"] = audio_root / "missing.wav"
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                resolve_manifest_wav_paths(dataset, [record], split="train", start=0)


class MSCLAPCenterCropTests(unittest.TestCase):
    class Reader:
        def __init__(self, waveform, sample_rate=44_100):
            self.waveform = waveform
            self.sample_rate = sample_rate
            self.calls = []

        def read_audio(self, path, resample=True):
            self.calls.append((path, resample))
            return self.waveform, self.sample_rate

    def test_short_audio_repeats_then_truncates_like_upstream(self):
        waveform = torch.arange(5, dtype=torch.float32).reshape(1, 5)
        reader = self.Reader(waveform)
        actual = _deterministic_load_audio_into_tensor(reader, "short.wav", 7 / 44_100, resample=True)
        torch.testing.assert_close(actual, torch.tensor([0, 1, 2, 3, 4, 0, 1], dtype=torch.float32))

    def test_exact_duration_is_unchanged_and_output_is_float32(self):
        waveform = torch.arange(7, dtype=torch.float64).reshape(1, 7)
        actual = _deterministic_load_audio_into_tensor(self.Reader(waveform), "exact.wav", 7 / 44_100)
        torch.testing.assert_close(actual, torch.arange(7, dtype=torch.float32))
        self.assertEqual(actual.dtype, torch.float32)

    def test_long_audio_uses_floor_center_start_and_expected_samples(self):
        waveform = torch.arange(13, dtype=torch.float32).reshape(1, 13)
        actual = _deterministic_load_audio_into_tensor(self.Reader(waveform), "long.wav", 8 / 44_100)
        torch.testing.assert_close(actual, torch.arange(2, 10, dtype=torch.float32))

    def test_loader_preserves_upstream_read_and_resample_call(self):
        reader = self.Reader(torch.arange(10, dtype=torch.float32).reshape(1, 10), sample_rate=10)
        _deterministic_load_audio_into_tensor(reader, "resampled.wav", 1.0, resample=True)
        self.assertEqual(reader.calls, [("resampled.wav", True)])

    def test_empty_audio_is_rejected(self):
        reader = self.Reader(torch.empty((1, 0), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "empty"):
            _deterministic_load_audio_into_tensor(reader, "empty.wav", 1.0)

    def test_crop_result_is_identical_in_a_fresh_process(self):
        repo = Path(__file__).resolve().parents[1]
        script = (
            "import json, torch; "
            "from model.expert_adapters.top2_msclap_adapter import _center_crop_or_repeat; "
            "print(json.dumps(_center_crop_or_repeat(torch.arange(15), 10, 1).tolist()))"
        )
        first = subprocess.run([sys.executable, "-c", script], cwd=repo, capture_output=True, text=True, check=True)
        second = subprocess.run([sys.executable, "-c", script], cwd=repo, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(first.stdout), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(first.stdout, second.stdout)

    def test_mock_adapter_inference_is_identical_across_fresh_processes(self):
        repo = Path(__file__).resolve().parents[1]
        script = r'''
import json
from pathlib import Path
import tempfile
import torch
from model.expert_adapters.top2_msclap_adapter import Top2MSCLAPAdapter

class API:
    def __init__(self):
        self.clap = torch.nn.Linear(1, 1)
        self.args = type("Args", (), {"d_proj": 4})()
    def get_audio_embeddings(self, paths):
        return torch.tensor([[float(len(Path(path).name)), 1., 2., 3.] for path in paths])
    def get_text_embeddings(self, captions):
        return torch.tensor([[float(len(text)), 3., 2., 1.] for text in captions])

with tempfile.TemporaryDirectory() as directory:
    audio = Path(directory) / "fixed.wav"
    audio.write_bytes(b"fixture")
    adapter = Top2MSCLAPAdapter(API(), embedding_dim=4).eval()
    score, features = adapter({
        "expert2_wavs": torch.zeros(1, 1, 320000),
        "captions_raw": ["fixed caption"],
        "wav_paths": [str(audio)],
    })
    print(json.dumps({"score": score.tolist(), "features": features.tolist()}))
'''
        first = subprocess.run(
            [sys.executable, "-c", script], cwd=repo, capture_output=True,
            text=True, check=True,
        )
        second = subprocess.run(
            [sys.executable, "-c", script], cwd=repo, capture_output=True,
            text=True, check=True,
        )
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))

    def test_unified_extraction_batch_contains_trusted_original_msclap_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root = root / "wav" / "train"
            audio_root.mkdir(parents=True)
            metadata = root / "meta_data" / "train_average.csv"
            metadata.parent.mkdir(parents=True)
            audio = audio_root / "source.wav"
            with __import__("wave").open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                wav.writeframes(np.full(1600, 20, dtype=np.int16).tobytes())
            metadata.write_text("wav_file_name,text,average_score\nsource.wav,Exact caption,5\n")
            dataset, _ = projected_dataset(root, "train", root / "projection", limit=1)
            row = dataset.rows[0]
            record = {
                "sample_id": row["sample_id"],
                "wave_file_name": row["wave_file_name"],
                "raw_caption": row["captions_raw"],
                "wav_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            }
            contract = TOP2_EXPERT_INPUT_CONTRACTS["top2_msclap"]
            self.assertTrue(contract.requires_wav_paths)
            batch = prepare_expert_inputs(
                dataset, "top2_msclap", [record], start=0, device="cpu"
            )
            self.assertEqual(tuple(batch["expert2_wavs"].shape), (1, 1, 320_000))
            self.assertEqual(batch["captions_raw"], ["Exact caption"])
            self.assertEqual(batch["wav_paths"], [str(audio.resolve())])
            self.assertEqual(batch["wave_file_names"], ["source.wav"])
            self.assertEqual(batch["split"], "train")

    def test_adapter_outputs_are_stable_across_repeats_batching_and_order(self):
        from model.expert_adapters.top2_msclap_adapter import Top2MSCLAPAdapter

        class DeterministicAPI:
            def __init__(self):
                self.clap = torch.nn.Linear(1, 1)
                self.args = SimpleNamespace(d_proj=4)

            def get_audio_embeddings(self, wav_paths):
                return torch.stack([
                    torch.tensor([float(len(path)), 1.0, 2.0, 3.0]) for path in wav_paths
                ])

            def get_text_embeddings(self, captions):
                return torch.stack([
                    torch.tensor([float(len(text)), 3.0, 2.0, 1.0]) for text in captions
                ])

        adapter = Top2MSCLAPAdapter(DeterministicAPI(), embedding_dim=4).eval()
        with tempfile.TemporaryDirectory() as temporary:
            audio_paths = []
            for name in ("a.wav", "bb.wav", "ccc.wav"):
                path = Path(temporary) / name
                path.write_bytes(b"fixture")
                audio_paths.append(str(path))
            all_batch = {
                "expert2_wavs": torch.zeros(3, 1, 320_000),
                "captions_raw": ["first", "second caption", "third"],
                "wav_paths": audio_paths,
            }
            score1, features1 = adapter(all_batch)
            score2, features2 = adapter(all_batch)
            torch.testing.assert_close(score1, score2)
            torch.testing.assert_close(features1, features2)
            one_by_one = [
                adapter({
                    "expert2_wavs": all_batch["expert2_wavs"][index:index + 1],
                    "captions_raw": all_batch["captions_raw"][index:index + 1],
                    "wav_paths": all_batch["wav_paths"][index:index + 1],
                })
                for index in range(3)
            ]
            torch.testing.assert_close(torch.cat([item[0] for item in one_by_one]), score1)
            torch.testing.assert_close(torch.cat([item[1] for item in one_by_one]), features1)
            reverse = adapter({
                "expert2_wavs": all_batch["expert2_wavs"].flip(0),
                "captions_raw": list(reversed(all_batch["captions_raw"])),
                "wav_paths": list(reversed(all_batch["wav_paths"])),
            })
            torch.testing.assert_close(reverse[0], score1.flip(0))
            torch.testing.assert_close(reverse[1], features1.flip(0))


if __name__ == "__main__":
    unittest.main()
