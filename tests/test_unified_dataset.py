"""Source-grounded dataset tests using temporary synthetic WAV fixtures only."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from data.unified_dataset import UnifiedXACLEDataset, audit_split_overlaps


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = ["wav_file_name", "text", "average_score", "model_ID", "model_name"]


def _write_wav(path: Path, *, sample_rate=16_000, samples=16_000, stereo=False, value=None):
    if value is None:
        base = ((np.arange(samples, dtype=np.int32) % 1000) - 500).astype(np.int16)
    else:
        base = np.full(samples, value, dtype=np.int16)
    audio = np.column_stack((base, base // 2)) if stereo else base
    wavfile.write(path, sample_rate, audio)
    return audio


def _write_manifest(path: Path, rows, *, columns=RAW_COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(name="sample.wav", caption="A caption", score="7.5", **extra):
    return {
        "wav_file_name": name,
        "text": caption,
        "average_score": score,
        "model_ID": "model-7",
        "model_name": "generator-seven",
        **extra,
    }


class UnifiedDatasetMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio_dir = self.root / "wav" / "validation"
        self.audio_dir.mkdir(parents=True)
        self.csv_path = self.root / "validation_average.csv"
        _write_wav(self.audio_dir / "sample.wav")
        _write_manifest(self.csv_path, [_row(caption="  Keep raw spaces  ")])

    def tearDown(self):
        self.temporary.cleanup()

    def make_dataset(self, *, split="validation", **kwargs):
        return UnifiedXACLEDataset(
            metadata_csv=self.csv_path,
            audio_dir=self.audio_dir,
            split=split,
            **kwargs,
        )

    def test_preserves_raw_caption_raw_score_identity_and_optional_metadata(self):
        item = self.make_dataset()[0]
        self.assertEqual(item["captions_raw"], "  Keep raw spaces  ")
        self.assertEqual(float(item["label"]), 7.5)
        self.assertEqual(item["label"].dtype, torch.float32)
        self.assertEqual(item["wave_file_name"], "sample.wav")
        self.assertEqual(item["split"], "validation")
        self.assertEqual(item["model_id"], "model-7")
        self.assertEqual(item["model_name"], "generator-seven")
        self.assertEqual(item["sample_id"], "validation:sample.wav")

    def test_observed_spelling_and_documented_score_aliases_are_supported(self):
        for filename_column, score_column in (
            ("wave_file_name", "average_semantic_alignement_score"),
            ("wav_file_name", "average_semantic_alignment_score"),
        ):
            with self.subTest(filename_column=filename_column, score_column=score_column):
                columns = [filename_column, "text", score_column]
                csv_path = self.root / f"{score_column}.csv"
                _write_manifest(
                    csv_path,
                    [{filename_column: "sample.wav", "text": "caption", score_column: "3.25"}],
                    columns=columns,
                )
                ds = UnifiedXACLEDataset(csv_path, self.audio_dir, split="validation")
                self.assertEqual(float(ds[0]["label"]), 3.25)
                self.assertNotIn("model_id", ds[0])
                self.assertEqual(ds.label_column, score_column)

    def test_raw_column_aliases_are_resolved_by_name_not_position(self):
        columns = ["model_name", "text", "model_ID", "average_score", "wav_file_name"]
        path = self.root / "reordered.csv"
        _write_manifest(
            path,
            [{"model_name": "g", "text": "c", "model_ID": "id", "average_score": "4", "wav_file_name": "sample.wav"}],
            columns=columns,
        )
        ds = UnifiedXACLEDataset(path, self.audio_dir, split="validation")
        item = ds[0]
        self.assertEqual(float(item["label"]), 4.0)
        self.assertEqual(item["model_id"], "id")
        self.assertEqual(item["model_name"], "g")

    def test_absent_optional_model_metadata_is_not_manufactured(self):
        path = self.root / "without_model_metadata.csv"
        _write_manifest(path, [{"wav_file_name": "sample.wav", "text": "c", "average_score": "5"}], columns=["wav_file_name", "text", "average_score"])
        item = UnifiedXACLEDataset(path, self.audio_dir, split="validation")[0]
        self.assertNotIn("model_id", item)
        self.assertNotIn("model_name", item)

    def test_test_split_can_preserve_an_unlabeled_record_without_inventing_a_label(self):
        path = self.root / "test.csv"
        _write_manifest(path, [{"wav_file_name": "sample.wav", "text": "test caption"}], columns=["wav_file_name", "text"])
        item = UnifiedXACLEDataset(path, self.audio_dir, split="test")[0]
        self.assertNotIn("label", item)
        self.assertEqual(item["captions_raw"], "test caption")

    def test_missing_required_columns_are_reported_by_name(self):
        path = self.root / "bad.csv"
        _write_manifest(path, [{"wav_file_name": "sample.wav"}], columns=["wav_file_name"])
        with self.assertRaisesRegex(ValueError, "text"):
            UnifiedXACLEDataset(path, self.audio_dir, split="validation")

    def test_duplicate_audio_identity_is_rejected_even_when_caption_differs(self):
        path = self.root / "duplicates.csv"
        _write_manifest(path, [_row(), _row(caption="other caption")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            UnifiedXACLEDataset(path, self.audio_dir, split="validation")

    def test_path_traversal_and_missing_audio_are_rejected(self):
        for filename in ("../outside.wav", "/etc/passwd", "nested/file.wav"):
            with self.subTest(filename=filename):
                path = self.root / "unsafe.csv"
                _write_manifest(path, [_row(name=filename)])
                with self.assertRaisesRegex(ValueError, "basename|path"):
                    UnifiedXACLEDataset(path, self.audio_dir, split="validation")
        path = self.root / "missing.csv"
        _write_manifest(path, [_row(name="missing.wav")])
        with self.assertRaisesRegex(FileNotFoundError, "missing.wav"):
            UnifiedXACLEDataset(path, self.audio_dir, split="validation")

    def test_labels_must_be_finite_and_in_the_original_zero_to_ten_scale(self):
        for score in ("nan", "inf", "-0.1", "10.1", "bad"):
            with self.subTest(score=score):
                path = self.root / "bad_score.csv"
                _write_manifest(path, [_row(score=score)])
                with self.assertRaisesRegex(ValueError, "score|label"):
                    UnifiedXACLEDataset(path, self.audio_dir, split="validation")

    def test_empty_manifest_and_invalid_caption_are_rejected(self):
        path = self.root / "empty.csv"
        _write_manifest(path, [])
        with self.assertRaisesRegex(ValueError, "empty|no rows"):
            UnifiedXACLEDataset(path, self.audio_dir, split="validation")
        path = self.root / "empty_caption.csv"
        _write_manifest(path, [_row(caption="  ")])
        with self.assertRaisesRegex(ValueError, "caption"):
            UnifiedXACLEDataset(path, self.audio_dir, split="validation")

    def test_split_is_required_and_restricted(self):
        with self.assertRaises(TypeError):
            UnifiedXACLEDataset(self.csv_path, self.audio_dir)
        with self.assertRaisesRegex(ValueError, "split"):
            self.make_dataset(split="dev")

    def test_from_dataset_root_resolves_official_split_layout(self):
        root = self.root / "official"
        (root / "meta_data").mkdir(parents=True)
        audio_dir = root / "wav" / "train"
        audio_dir.mkdir(parents=True)
        _write_wav(audio_dir / "sample.wav")
        _write_manifest(root / "meta_data" / "train_average.csv", [_row()])
        ds = UnifiedXACLEDataset.from_dataset_root(root, split="train")
        self.assertEqual(ds.audio_dir, audio_dir.resolve())
        self.assertEqual(ds.metadata_csv, (root / "meta_data" / "train_average.csv").resolve())

    def test_from_resolved_config_uses_each_split_csv_and_audio_subdirectory(self):
        from configs.moe_config import load_moe_config, resolve_moe_paths

        config_path = REPO_ROOT / "configs" / "moe_local.example.yaml"
        config = load_moe_config(config_path)
        config["data"]["train_csv"] = str(self.root / "train.csv")
        config["data"]["validation_csv"] = str(self.csv_path)
        config["data"]["test_csv"] = str(self.root / "test.csv")
        config["data"]["audio_root"] = str(self.root / "wav")
        resolved = resolve_moe_paths(config, config_path, environ={})
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                audio_dir = self.root / "wav" / split
                audio_dir.mkdir(parents=True, exist_ok=True)
                if split != "validation":
                    csv_path = self.root / f"{split}.csv"
                    _write_manifest(csv_path, [_row()])
                    _write_wav(audio_dir / "sample.wav")
                ds = UnifiedXACLEDataset.from_config(resolved, split=split)
                self.assertEqual(ds.split, split)
                self.assertEqual(ds.audio_dir, audio_dir.resolve())

    def test_corrupt_and_nonfinite_audio_fail_with_sample_context(self):
        corrupt = self.audio_dir / "corrupt.wav"
        corrupt.write_bytes(b"not a wave file")
        path = self.root / "corrupt.csv"
        _write_manifest(path, [_row(name="corrupt.wav")])
        ds = UnifiedXACLEDataset(path, self.audio_dir, split="validation")
        with self.assertRaisesRegex(ValueError, "corrupt.wav"):
            ds[0]

        nan_path = self.audio_dir / "nan.wav"
        wavfile.write(nan_path, 16_000, np.array([0.0, float("nan")], dtype=np.float32))
        path = self.root / "nan.csv"
        _write_manifest(path, [_row(name="nan.wav")])
        ds = UnifiedXACLEDataset(path, self.audio_dir, split="validation")
        with self.assertRaisesRegex(ValueError, "finite|nan.wav"):
            ds[0]


class UnifiedAudioProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio_dir = self.root / "audio"
        self.audio_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def dataset_for(self, *, sample_rate=16_000, samples=16_000, value=None, training=None, stereo=False):
        audio = self.audio_dir / "clip.wav"
        _write_wav(audio, sample_rate=sample_rate, samples=samples, value=value, stereo=stereo)
        manifest = self.root / "manifest.csv"
        _write_manifest(manifest, [_row(name="clip.wav")])
        return UnifiedXACLEDataset(
            manifest,
            self.audio_dir,
            split="train" if training else "validation",
            seed=11,
        )

    def test_short_audio_repeats_only_for_expert1_and_zero_pads_others(self):
        item = self.dataset_for(samples=16_000, value=1200)[0]
        self.assertEqual(tuple(item["expert1_wavs"].shape), (1, 480_000))
        self.assertEqual(tuple(item["expert2_wavs"].shape), (1, 320_000))
        self.assertEqual(tuple(item["expert3_wavs"].shape), (1, 160_000))
        self.assertEqual(tuple(item["expert4_wavs"].shape), (1, 160_000))
        self.assertEqual(item["expert4_wav_lens"], 16_000)
        self.assertTrue(torch.all(item["expert1_wavs"] != 0))
        self.assertTrue(torch.equal(item["expert1_wavs"][0, :48_000], item["expert1_wavs"][0, 48_000:96_000]))
        self.assertTrue(torch.all(item["expert2_wavs"][0, 32_000:] == 0))
        self.assertTrue(torch.all(item["expert3_wavs"][0, 16_000:] == 0))
        self.assertTrue(torch.all(item["expert4_wavs"][0, 16_000:] == 0))

    def test_exact_ten_second_audio_has_no_crop_or_padding_and_e4_length_is_full(self):
        item = self.dataset_for(samples=160_000, value=1800)[0]
        self.assertEqual(item["expert4_wav_lens"], 160_000)
        for key in ("expert1_wavs", "expert2_wavs", "expert3_wavs", "expert4_wavs"):
            self.assertTrue(torch.all(item[key] != 0), key)

    def test_stereo_non_target_rate_uses_each_source_resampler_and_tracks_e4_valid_length(self):
        dataset = self.dataset_for(sample_rate=8_000, samples=4_000, training=False, stereo=True)
        item = dataset[0]
        self.assertEqual(tuple(item["expert1_wavs"].shape), (1, 480_000))
        self.assertEqual(tuple(item["expert2_wavs"].shape), (1, 320_000))
        self.assertEqual(tuple(item["expert3_wavs"].shape), (1, 160_000))
        self.assertEqual(tuple(item["expert4_wavs"].shape), (1, 160_000))
        self.assertEqual(item["expert4_wav_lens"], 8_000)
        # E3 and E4 use the same 16 kHz polyphase conversion for this source;
        # their source decoders both yield normalized float PCM here.
        self.assertTrue(torch.equal(item["expert3_wavs"], item["expert4_wavs"]))

    def test_long_audio_uses_e1_center_crop_for_eval_and_first_crop_for_e2_e3_e4(self):
        item = self.dataset_for(samples=176_000)[0]
        self.assertEqual(tuple(item["expert1_wavs"].shape), (1, 480_000))
        self.assertEqual(item["expert4_wav_lens"], 160_000)
        # The deterministic eval crop for E1 begins halfway through the excess.
        source, source_rate = __import__("torchaudio").load(self.audio_dir / "clip.wav")
        source = source.mean(dim=0)
        source = __import__("torchaudio").functional.resample(source, source_rate, 48_000)
        offset = (source.numel() - 480_000) // 2
        self.assertTrue(torch.equal(item["expert1_wavs"][0], source[offset:offset + 480_000]))
        expected_16k = torch.from_numpy(
            ((((np.arange(176_000, dtype=np.int32) % 1000) - 500).astype(np.float32))
             / 32768.0)[:160_000]
        )
        expected_32k = resample_poly(
            (((np.arange(176_000, dtype=np.int32) % 1000) - 500).astype(np.float32)
             / 32768.0),
            2,
            1,
        )[:320_000]
        self.assertTrue(torch.equal(item["expert2_wavs"][0], torch.from_numpy(expected_32k)))
        self.assertTrue(torch.equal(item["expert3_wavs"][0], expected_16k))
        self.assertTrue(torch.equal(item["expert3_wavs"], item["expert4_wavs"]))

    def test_training_crop_is_local_deterministic_and_changes_with_epoch(self):
        ds = self.dataset_for(samples=176_000, training=True)
        first = ds[0]["expert1_wavs"].clone()
        self.assertTrue(torch.equal(first, ds[0]["expert1_wavs"]))
        ds.set_epoch(1)
        second = ds[0]["expert1_wavs"]
        self.assertFalse(torch.equal(first, second))

    def test_import_is_lightweight_and_does_not_import_expert_models_or_mutate_path(self):
        script = (
            "import sys; before=list(sys.path); "
            "import data.unified_dataset; "
            "assert sys.path==before; "
            "assert not any(name.startswith(('expert1_finetune','expert2_finetune',"
            "'expert3_finetune','expert4_finetune','laion_clap')) for name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SplitOverlapTests(unittest.TestCase):
    def test_overlap_audit_reports_wave_and_pair_overlap_without_decoding_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            validation = root / "validation.csv"
            _write_manifest(train, [_row(name="same.wav", caption="Train caption")])
            _write_manifest(validation, [_row(name="same.wav", caption="Validation caption")])
            report = audit_split_overlaps({"train": train, "validation": validation})
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["wave_file_name_overlaps"]["train:validation"], ["same.wav"])
            self.assertEqual(report["audio_text_pair_overlaps"]["train:validation"], [])

    def test_overlap_audit_marks_unavailable_split_resources_unverified(self):
        report = audit_split_overlaps({"validation": Path("/missing/validation.csv")})
        self.assertEqual(report["status"], "NOT_VERIFIED")
        self.assertFalse(report["model_id_used_as_input"])


if __name__ == "__main__":
    unittest.main()
