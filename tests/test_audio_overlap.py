import hashlib
import json

import pytest

from data.audio_overlap import (
    build_public_split_audio_overlap_diagnostic,
    build_train_validation_audio_overlap_diagnostic,
    raise_for_public_split_audio_overlap,
)


def test_audio_overlap_diagnostic_is_complete_and_deterministic():
    shared = "0de242afa78769b6f6baf5b552a4d80fbea05fb5c615f1868a695e57b91eb1c2"
    train = [
        {"sample_id": "train:03768.wav", "wav_sha256": shared},
        {"sample_id": "train:other.wav", "wav_sha256": "a" * 64},
    ]
    validation = [
        {"sample_id": "validation:07603.wav", "wav_sha256": shared},
        {"sample_id": "validation:other.wav", "wav_sha256": "b" * 64},
    ]

    result = build_train_validation_audio_overlap_diagnostic(
        train, validation, allow_train_validation_audio_overlap=True,
    )
    unsigned = {key: value for key, value in result.items() if key != "diagnostic_sha256"}
    expected = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    assert result["status"] == "WARNING"
    assert result["policy"] == "experimental_train_validation_audio_overlap"
    assert result["overlapping_sha256_count"] == 1
    assert result["affected_train_rows"] == result["affected_validation_rows"] == 1
    assert result["overlaps"] == [{
        "wav_sha256": shared,
        "train_sample_ids": ["train:03768.wav"],
        "validation_sample_ids": ["validation:07603.wav"],
    }]
    assert result["diagnostic_sha256"] == expected
    assert build_train_validation_audio_overlap_diagnostic(
        train, validation, allow_train_validation_audio_overlap=True,
    ) == result


def test_audio_overlap_diagnostic_counts_all_members_for_full_official_splits():
    shared = "c" * 64
    train = [{"sample_id": f"train:{index:05d}.wav", "wav_sha256": shared if index == 4 else f"{index:064x}"}
             for index in range(7500)]
    validation = [{"sample_id": f"validation:{index:05d}.wav", "wav_sha256": shared if index == 8 else f"{index + 7500:064x}"}
                  for index in range(3000)]
    diagnostic = build_train_validation_audio_overlap_diagnostic(
        train, validation, allow_train_validation_audio_overlap=True,
    )
    assert len(train) == 7500
    assert len(validation) == 3000
    assert diagnostic["affected_train_rows"] == diagnostic["affected_validation_rows"] == 1
    assert diagnostic["overlaps"][0]["train_sample_ids"] == ["train:00004.wav"]
    assert diagnostic["overlaps"][0]["validation_sample_ids"] == ["validation:00008.wav"]


def test_audio_overlap_diagnostic_keeps_all_hashes_and_bounds_examples():
    digests = [f"{index + 1:064x}" for index in range(8)]
    train = [{"sample_id": f"train:{index}.wav", "wav_sha256": digest}
             for index, digest in enumerate(digests)]
    validation = [{"sample_id": f"validation:{index}.wav", "wav_sha256": digest}
                  for index, digest in enumerate(digests)]
    diagnostic = build_train_validation_audio_overlap_diagnostic(
        train, validation, allow_train_validation_audio_overlap=True,
    )
    assert diagnostic["overlapping_sha256_count"] == 8
    assert len(diagnostic["overlaps"]) == 8
    assert len(diagnostic["examples"]) == 5


@pytest.mark.parametrize("records", [
    [{"sample_id": "train:a.wav", "wav_sha256": "z" * 64}],
    [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64},
     {"sample_id": "train:a.wav", "wav_sha256": "b" * 64}],
])
def test_audio_overlap_diagnostic_rejects_invalid_hashes_and_duplicate_ids(records):
    with pytest.raises(ValueError):
        build_train_validation_audio_overlap_diagnostic(
            records, [], allow_train_validation_audio_overlap=True,
        )


def test_public_split_diagnostic_records_every_pair_hash_and_sample_id_without_mutation():
    records = {
        "train": [
            {"sample_id": "train:0.wav", "wav_sha256": "a" * 64},
            {"sample_id": "train:1.wav", "wav_sha256": "b" * 64},
            {"sample_id": "train:2.wav", "wav_sha256": "d" * 64},
            {"sample_id": "train:unique.wav", "wav_sha256": "e" * 64},
        ],
        "validation": [
            {"sample_id": "validation:0.wav", "wav_sha256": "a" * 64},
            {"sample_id": "validation:2.wav", "wav_sha256": "c" * 64},
            {"sample_id": "validation:3.wav", "wav_sha256": "d" * 64},
        ],
        "test": [
            {"sample_id": "test:1.wav", "wav_sha256": "b" * 64},
            {"sample_id": "test:2.wav", "wav_sha256": "c" * 64},
            {"sample_id": "test:3.wav", "wav_sha256": "d" * 64},
        ],
    }
    original = json.loads(json.dumps(records))
    diagnostic = build_public_split_audio_overlap_diagnostic(
        records, allow_public_split_audio_overlap=True,
        allow_train_validation_audio_overlap=True,
    )
    assert records == original
    assert diagnostic["status"] == "WARNING"
    assert diagnostic["total_overlapping_sha256_groups"] == 4
    assert diagnostic["pair_overlap_group_counts"] == {
        "train_validation": 2, "train_test": 2, "validation_test": 2,
    }
    assert [row["wav_sha256"] for row in diagnostic["overlaps"]] == ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
    triple = diagnostic["overlaps"][-1]
    assert triple["sample_ids_by_split"] == {
        "train": ["train:2.wav"], "validation": ["validation:3.wav"], "test": ["test:3.wav"],
    }
    assert "may bias evaluation" in diagnostic["warning"]


def test_public_split_default_is_strict_for_train_test_duplicates():
    records = {
        "train": [{"sample_id": "train:a.wav", "wav_sha256": "a" * 64}],
        "validation": [{"sample_id": "validation:b.wav", "wav_sha256": "b" * 64}],
        "test": [{"sample_id": "test:c.wav", "wav_sha256": "a" * 64}],
    }
    diagnostic = build_public_split_audio_overlap_diagnostic(records)
    assert diagnostic["status"] == "ERROR"
    assert diagnostic["pair_overlap_group_counts"]["train_test"] == 1
    with pytest.raises(ValueError, match="test"):
        raise_for_public_split_audio_overlap(diagnostic, stage="Full7 builder")
