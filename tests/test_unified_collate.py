"""Contract tests for unified CPU collation."""

from __future__ import annotations

import unittest

import torch

from data.unified_collate import collate_unified


def _sample(index: int, *, length: int | None = None):
    length = 160_000 if length is None else length
    return {
        "sample_id": f"validation:{index:05d}.wav",
        "wave_file_name": f"{index:05d}.wav",
        "split": "validation",
        "captions_raw": f" caption {index} ",
        "expert1_wavs": torch.full((1, 480_000), float(index)),
        "expert2_wavs": torch.full((1, 320_000), float(index + 10)),
        "expert3_wavs": torch.full((1, 160_000), float(index + 20)),
        "expert4_wavs": torch.cat(
            (torch.full((1, length), float(index + 30)), torch.zeros((1, 160_000 - length))),
            dim=1,
        ),
        "expert4_wav_lens": length,
        "label": torch.tensor(float(index), dtype=torch.float32),
        "model_id": f"generator-{index}",
        "model_name": f"Generator {index}",
    }


class UnifiedCollateTests(unittest.TestCase):
    def test_batch_sizes_one_two_and_four_preserve_shapes_order_and_values(self):
        for batch_size in (1, 2, 4):
            samples = [_sample(i, length=80_000 + i * 20_000) for i in range(batch_size)]
            batch = collate_unified(samples)
            self.assertEqual(tuple(batch["expert1_wavs"].shape), (batch_size, 1, 480_000))
            self.assertEqual(tuple(batch["expert2_wavs"].shape), (batch_size, 1, 320_000))
            self.assertEqual(tuple(batch["expert3_wavs"].shape), (batch_size, 1, 160_000))
            self.assertEqual(tuple(batch["expert4_wavs"].shape), (batch_size, 1, 160_000))
            self.assertEqual(tuple(batch["expert4_wav_lens"].shape), (batch_size,))
            self.assertEqual(batch["expert4_wav_lens"].dtype, torch.int64)
            self.assertEqual(batch["captions_raw"], [f" caption {i} " for i in range(batch_size)])
            self.assertEqual(batch["wave_file_names"], [f"{i:05d}.wav" for i in range(batch_size)])
            self.assertEqual(batch["sample_ids"], [f"validation:{i:05d}.wav" for i in range(batch_size)])
            self.assertEqual(batch["labels"].tolist(), [float(i) for i in range(batch_size)])
            self.assertEqual(batch["expert4_wav_lens"].tolist(), [80_000 + i * 20_000 for i in range(batch_size)])
            self.assertTrue(torch.equal(batch["expert1_wavs"][:, 0, 0], torch.arange(batch_size, dtype=torch.float32)))
            self.assertTrue(torch.equal(batch["expert2_wavs"][:, 0, 0], torch.arange(10, 10 + batch_size, dtype=torch.float32)))
            self.assertTrue(all(value.device.type == "cpu" for value in batch.values() if isinstance(value, torch.Tensor)))

    def test_collation_is_deterministic_and_does_not_change_caption_or_waveform(self):
        sample = _sample(3, length=80_000)
        first = collate_unified([sample])
        second = collate_unified([sample])
        self.assertEqual(first["captions_raw"], [" caption 3 "])
        self.assertTrue(torch.equal(first["expert4_wavs"], second["expert4_wavs"]))
        self.assertTrue(torch.equal(first["expert4_wav_lens"], torch.tensor([80_000])))

    def test_optional_metadata_is_preserved_but_not_fabricated(self):
        sample = _sample(0)
        batch = collate_unified([sample])
        self.assertEqual(batch["model_ids"], ["generator-0"])
        del sample["model_id"]
        del sample["model_name"]
        batch = collate_unified([sample])
        self.assertNotIn("model_ids", batch)
        self.assertNotIn("model_names", batch)

    def test_unlabeled_test_batch_has_no_synthetic_labels(self):
        sample = _sample(0)
        del sample["label"]
        batch = collate_unified([sample])
        self.assertNotIn("labels", batch)

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_unified([])

    def test_missing_field_invalid_shape_dtype_and_mixed_splits_are_rejected(self):
        sample = _sample(0)
        del sample["expert2_wavs"]
        with self.assertRaisesRegex(ValueError, "expert2_wavs"):
            collate_unified([sample])

        sample = _sample(0)
        sample["expert1_wavs"] = torch.zeros(480_000)
        with self.assertRaisesRegex(ValueError, "expert1_wavs"):
            collate_unified([sample])

        sample = _sample(0)
        sample["expert2_wavs"] = sample["expert2_wavs"].to(torch.float64)
        with self.assertRaisesRegex(ValueError, "CPU torch.float32"):
            collate_unified([sample])

        sample = _sample(0)
        sample["expert4_wav_lens"] = torch.tensor(80_000, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "expert4_wav_lens"):
            collate_unified([sample])

        samples = [_sample(0), _sample(1)]
        samples[1]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "split"):
            collate_unified(samples)

        sample = _sample(0)
        sample["captions_raw"] = "   "
        with self.assertRaisesRegex(ValueError, "captions_raw"):
            collate_unified([sample])

    def test_bad_lengths_nonfinite_values_mixed_labels_and_metadata_are_rejected(self):
        for invalid_length in (0, -1, 160_001, 80_000.0):
            sample = _sample(0)
            sample["expert4_wav_lens"] = invalid_length
            with self.subTest(length=invalid_length):
                with self.assertRaisesRegex(ValueError, "expert4_wav_lens"):
                    collate_unified([sample])

        sample = _sample(0)
        sample["expert3_wavs"][0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            collate_unified([sample])

        samples = [_sample(0), _sample(1)]
        del samples[1]["label"]
        with self.assertRaisesRegex(ValueError, "labeled|label"):
            collate_unified(samples)

        samples = [_sample(0), _sample(1)]
        del samples[1]["model_id"]
        with self.assertRaisesRegex(ValueError, "model"):
            collate_unified(samples)


if __name__ == "__main__":
    unittest.main()
