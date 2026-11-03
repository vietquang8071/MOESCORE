import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from model.expert_adapters.registry import EXPERT_FEATURE_DIMS
from scripts.verify_real_expert_to_gate import (
    _classify_exception,
    _preflight_expert3_text_cache,
    load_output_artifact,
    make_sample_manifest,
    select_sample_records,
    write_output_artifact,
)


class RealExpertOutputManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for index, frames in enumerate((157504, 160000, 163872, 160000)):
            path = self.root / f"{index:05}.wav"
            path.write_bytes(f"wav-fixture-{index}".encode())
            self.rows.append({
                "sample_id": f"train:{path.name}",
                "split": "train",
                "wave_file_name": path.name,
                "audio_path": path,
                "captions_raw": f"caption {index}",
                "label_value": float(index),
                "model_id": str(index % 2),
                "model_name": "fixture-model",
                "row_index": index,
                "csv_row_number": index + 2,
                "_fixture_frames": frames,
            })

    def tearDown(self):
        self.temp.cleanup()

    def _manifest(self):
        selected = select_sample_records(
            self.rows,
            seed=2027,
            max_samples=4,
            audio_info_fn=lambda row: {
                "samplerate": 16000,
                "frames": row["_fixture_frames"],
                "channels": 1,
            },
        )
        return make_sample_manifest(selected, seed=2027, split="train")

    def test_selection_is_seeded_and_covers_available_duration_groups(self):
        first = self._manifest()
        second = self._manifest()
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            [record["sample_id"] for record in first["samples"]],
            [record["sample_id"] for record in second["samples"]],
        )
        groups = {record["duration_group"] for record in first["samples"]}
        self.assertEqual(groups, {"short_under_10s", "exactly_10s", "over_10s"})
        self.assertEqual(len({record["sample_id"] for record in first["samples"]}), 4)

    def test_expert3_cache_preflight_checks_factory_cache_dir_explicitly(self):
        import transformers

        class Tokenizer:
            def __len__(self):
                return 30522

        cache_dir = self.root / "huggingface" / "hub"
        config = SimpleNamespace(model_type="bert", hidden_size=768)
        with (
            mock.patch.object(transformers.AutoConfig, "from_pretrained", return_value=config) as config_load,
            mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=Tokenizer()) as tokenizer_load,
        ):
            result = _preflight_expert3_text_cache(cache_dir)

        for loader in (config_load, tokenizer_load):
            self.assertEqual(loader.call_args.args, ("google-bert/bert-base-uncased",))
            self.assertEqual(loader.call_args.kwargs["cache_dir"], str(cache_dir.resolve()))
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cache_dir"], str(cache_dir.resolve()))

    def test_duplicate_sample_identity_is_rejected(self):
        duplicated = [dict(self.rows[0]), dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_sample_records(
                duplicated,
                seed=2027,
                max_samples=2,
                audio_info_fn=lambda row: {"samplerate": 16000, "frames": 160000, "channels": 1},
            )

    def test_artifact_round_trip_preserves_fp32_values_and_identity_without_pickle(self):
        manifest = self._manifest()
        expert = "expert3"
        dim = EXPERT_FEATURE_DIMS[expert]
        scores = torch.linspace(-0.4, 0.6, len(manifest["samples"])).reshape(-1, 1)
        features = torch.arange(len(manifest["samples"]) * dim, dtype=torch.float32).reshape(-1, dim)
        path = self.root / "expert3_outputs.npz"
        write_output_artifact(
            path,
            expert=expert,
            scores=scores,
            features=features,
            sample_records=manifest["samples"],
            manifest_sha256=manifest["manifest_sha256"],
        )

        loaded = load_output_artifact(
            path,
            expert=expert,
            expected_dim=dim,
            expected_manifest_sha256=manifest["manifest_sha256"],
            sample_records=manifest["samples"],
        )
        np.testing.assert_array_equal(loaded["scores"], scores.numpy())
        np.testing.assert_array_equal(loaded["features"], features.numpy())
        self.assertEqual(loaded["features"].dtype, np.float32)
        self.assertEqual(loaded["sample_ids"].tolist(), [x["sample_id"] for x in manifest["samples"]])

    def test_artifact_rejects_manifest_identity_mismatch(self):
        manifest = self._manifest()
        path = self.root / "expert1_outputs.npz"
        dim = EXPERT_FEATURE_DIMS["expert1"]
        write_output_artifact(
            path,
            expert="expert1",
            scores=torch.zeros((4, 1), dtype=torch.float32),
            features=torch.zeros((4, dim), dtype=torch.float32),
            sample_records=manifest["samples"],
            manifest_sha256=manifest["manifest_sha256"],
        )
        records = [dict(item) for item in manifest["samples"]]
        records[0]["sample_id"] = "train:wrong.wav"
        with self.assertRaisesRegex(ValueError, "sample_id|identity|manifest"):
            load_output_artifact(
                path,
                expert="expert1",
                expected_dim=dim,
                expected_manifest_sha256=manifest["manifest_sha256"],
                sample_records=records,
            )

    def test_artifact_rejects_wrong_expert_and_nonfinite_values(self):
        manifest = self._manifest()
        dim = EXPERT_FEATURE_DIMS["expert2"]
        path = self.root / "expert2_outputs.npz"
        with self.assertRaisesRegex(ValueError, "finite"):
            write_output_artifact(
                path,
                expert="expert2",
                scores=torch.tensor([[float("nan")]] * 4, dtype=torch.float32),
                features=torch.zeros((4, dim), dtype=torch.float32),
                sample_records=manifest["samples"],
                manifest_sha256=manifest["manifest_sha256"],
            )
        write_output_artifact(
            path,
            expert="expert2",
            scores=torch.zeros((4, 1), dtype=torch.float32),
            features=torch.zeros((4, dim), dtype=torch.float32),
            sample_records=manifest["samples"],
            manifest_sha256=manifest["manifest_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "expert"):
            load_output_artifact(
                path,
                expert="expert1",
                expected_dim=EXPERT_FEATURE_DIMS["expert1"],
                expected_manifest_sha256=manifest["manifest_sha256"],
                sample_records=manifest["samples"],
            )

    def test_artifact_serialization_failure_has_its_own_category(self):
        self.assertEqual(
            _classify_exception(OSError("disk write failed"), "ARTIFACT_SERIALIZATION"),
            "OUTPUT_ARTIFACT_FAILURE",
        )
        self.assertEqual(
            _classify_exception(OSError("couldn't find them in the cached files"), "FACTORY"),
            "MISSING_TEXT_MODEL_CACHE",
        )


if __name__ == "__main__":
    unittest.main()
