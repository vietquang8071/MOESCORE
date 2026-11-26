import csv
import json

import pytest

from training.logo_protocol import align_metadata, aggregate_predictions, fold_indices, load_metadata


def _row(split, name, model_id, score="5", text="caption"):
    return {"sample_id": f"{split}:{name}", "source_split": split,
            "wave_file_name": name, "model_id": model_id, "model_name": model_id,
            "ground_truth": float(score), "text": text}


def _manifest(split, names):
    import hashlib
    return {"shards": [{"sample_ids": [f"{split}:{n}" for n in names],
                         "wave_file_names": names, "count": len(names)}],
            "assembly_identity": {"ordered_source_fingerprints": [
                {"sample_id": f"{split}:{n}", "caption_sha256": hashlib.sha256(b"caption").hexdigest()}
                for n in names]}}


def test_alignment_uses_identity_not_csv_order():
    rows = {"train": [_row("train", "b.wav", "b"), _row("train", "a.wav", "a")]}
    aligned = align_metadata(rows, {"train": _manifest("train", ["a.wav", "b.wav"])})
    assert [r["model_id"] for r in aligned["train"]] == ["a", "b"]


def test_load_metadata_accepts_official_wav_file_name_column(tmp_path):
    for split in ("train", "validation", "test"):
        with (tmp_path / f"{split}_average.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["wav_file_name", "text", "average_score", "model_ID", "model_name"])
            writer.writeheader()
            writer.writerow({"wav_file_name": "one.wav", "text": "caption", "average_score": "5",
                             "model_ID": "1", "model_name": "generator"})
    assert load_metadata(tmp_path)["train"][0]["sample_id"] == "train:one.wav"


def test_alignment_rejects_missing_duplicate_and_caption_mismatch():
    manifest = {"train": _manifest("train", ["a.wav", "b.wav"])}
    with pytest.raises(ValueError, match="coverage"):
        align_metadata({"train": [_row("train", "a.wav", "a")]}, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        align_metadata({"train": [_row("train", "a.wav", "a")] * 2}, {"train": _manifest("train", ["a.wav"])})
    with pytest.raises(ValueError, match="caption"):
        align_metadata({"train": [_row("train", "a.wav", "a", text="changed")]},
                       {"train": _manifest("train", ["a.wav"])})


def test_fold_excludes_generator_and_preserves_control():
    rows = {"train": [_row("train", "a.wav", "a"), _row("train", "b.wav", "b"),
                      _row("train", "c.wav", "control")],
            "validation": [_row("validation", "v.wav", "a")],
            "test": [_row("test", "t.wav", "a")]}
    fold = fold_indices(rows, "a")
    assert fold == {"optimization": [1, 2], "internal_holdout": [0], "validation": [0], "public_test": [0]}


def test_aggregate_uses_pooled_rows_and_ignores_undefined_fold_srcc():
    rows = [
        {"sample_id": "train:a1", "fold_id": "a", "ground_truth": 1, "predicted_score": 1},
        {"sample_id": "train:a2", "fold_id": "a", "ground_truth": 2, "predicted_score": 2},
        {"sample_id": "train:b1", "fold_id": "b", "ground_truth": 5, "predicted_score": 4},
        {"sample_id": "train:b2", "fold_id": "b", "ground_truth": 5, "predicted_score": 5},
    ]
    result = aggregate_predictions(rows, eligible_folds=3, total_samples=6)
    assert result["logo_macro_srcc"] == pytest.approx(1)
    assert result["logo_weighted_srcc"] == pytest.approx(1)
    assert result["logo_worst_srcc"] == pytest.approx(1)
    assert result["logo_srcc_std"] == pytest.approx(0)
    assert result["logo_pooled_srcc"] < 1
    assert result["evaluated_samples"] == 4
    assert result["successful_folds"] == 2
    assert result["valid_srcc_folds"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_predictions(rows + [rows[0]], eligible_folds=3, total_samples=6)
