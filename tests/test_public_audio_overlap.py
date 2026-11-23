import unittest

from data.audio_overlap import (
    build_public_split_audio_overlap_diagnostic,
    raise_for_public_split_audio_overlap,
)


class PublicAudioOverlapTests(unittest.TestCase):
    def test_diagnostic_mode_records_all_pair_groups_and_preserves_source_records(self):
        records = {
            "train": [
                {"sample_id": "train:tv.wav", "wav_sha256": "a" * 64},
                {"sample_id": "train:tt.wav", "wav_sha256": "b" * 64},
                {"sample_id": "train:unique.wav", "wav_sha256": "c" * 64},
            ],
            "validation": [
                {"sample_id": "validation:tv.wav", "wav_sha256": "a" * 64},
                {"sample_id": "validation:vt.wav", "wav_sha256": "d" * 64},
                {"sample_id": "validation:unique.wav", "wav_sha256": "e" * 64},
            ],
            "test": [
                {"sample_id": "test:tt.wav", "wav_sha256": "b" * 64},
                {"sample_id": "test:vt.wav", "wav_sha256": "d" * 64},
                {"sample_id": "test:unique.wav", "wav_sha256": "f" * 64},
            ],
        }
        original = {split: list(rows) for split, rows in records.items()}
        diagnostic = build_public_split_audio_overlap_diagnostic(
            records,
            allow_public_split_audio_overlap=True,
            allow_train_validation_audio_overlap=True,
        )
        self.assertEqual(records, original)
        self.assertEqual(diagnostic["status"], "WARNING")
        self.assertEqual(diagnostic["total_overlapping_sha256_groups"], 3)
        self.assertEqual(diagnostic["pair_overlap_group_counts"], {
            "train_validation": 1, "train_test": 1, "validation_test": 1,
        })
        self.assertEqual(
            diagnostic["overlaps"][0]["sample_ids_by_split"],
            {"train": ["train:tv.wav"], "validation": ["validation:tv.wav"], "test": []},
        )
        self.assertTrue(diagnostic["allow_public_split_audio_overlap"])
        self.assertIn("may bias evaluation", diagnostic["warning"])

    def test_default_mode_rejects_public_test_duplicate(self):
        records = {
            "train": [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64}],
            "validation": [{"sample_id": "validation:b.wav", "wav_sha256": "b" * 64}],
            "test": [{"sample_id": "test:c.wav", "wav_sha256": "a" * 64}],
        }
        diagnostic = build_public_split_audio_overlap_diagnostic(records)
        self.assertEqual(diagnostic["status"], "ERROR")
        self.assertEqual(diagnostic["pair_overlap_group_counts"]["train_test"], 1)
        with self.assertRaisesRegex(ValueError, "test"):
            raise_for_public_split_audio_overlap(diagnostic, stage="Full7 builder")


if __name__ == "__main__":
    unittest.main()
