"""Metadata alignment and research-only LOGO summaries for cached Full8 rows."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SOURCES = ("train", "validation", "test")
ROLES = {"train": "internal_holdout", "validation": "validation", "test": "public_test"}


def load_metadata(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(root)
    result = {}
    for split in SOURCES:
        matches = list(root.rglob(f"{split}_average.csv"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {split}_average.csv under {root}; found {len(matches)}")
        with matches[0].open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            id_field = "model_id" if "model_id" in fields else "model_ID"
            name_field = "wav_file_name" if "wav_file_name" in fields else "wave_file_name"
            score_field = "average_score" if "average_score" in fields else "average_semantic_alignement_score"
            required = {name_field, "text", "model_name", id_field, score_field}
            if not required <= fields:
                raise ValueError(f"{matches[0]} lacks required columns: {sorted(required-fields)}")
            rows = []
            for row in reader:
                name = row[name_field]
                model_id = row[id_field].strip()
                model_name = row["model_name"].strip()
                if not name or Path(name).name != name or not model_id or not model_name:
                    raise ValueError(f"missing or ambiguous generator/sample identity in {matches[0]}")
                try:
                    score = float(row[score_field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"missing ground truth for {split}:{name}") from exc
                if not math.isfinite(score) or not 0 <= score <= 10:
                    raise ValueError(f"invalid ground truth for {split}:{name}")
                rows.append({"sample_id": f"{split}:{name}", "source_split": split,
                             "wave_file_name": name, "model_id": model_id,
                             "model_name": model_name, "ground_truth": score,
                             "text": row["text"]})
        if not rows:
            raise ValueError(f"{split} metadata is empty")
        result[split] = rows
    return result


def align_metadata(rows_by_split: dict, manifests: dict) -> dict[str, list[dict]]:
    aligned = {}
    names_by_id: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        manifest = manifests[split]
        by_id = {}
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in by_id:
                raise ValueError(f"duplicate metadata sample ID: {sample_id}")
            if row["source_split"] != split or not sample_id.startswith(split + ":"):
                raise ValueError(f"source split mismatch for {sample_id}")
            score = float(row["ground_truth"])
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError(f"invalid ground truth for {sample_id}")
            model_id, model_name = str(row["model_id"]).strip(), str(row["model_name"]).strip()
            if not model_id or not model_name:
                raise ValueError(f"missing generator ID/name for {sample_id}")
            if model_id in names_by_id and names_by_id[model_id] != model_name:
                raise ValueError(f"ambiguous generator name for ID {model_id}")
            names_by_id[model_id] = model_name
            by_id[sample_id] = row
        shards = manifest["shards"]
        ids = [sample_id for shard in shards for sample_id in shard["sample_ids"]]
        filenames = [name for shard in shards for name in shard["wave_file_names"]]
        fingerprints = manifest["assembly_identity"]["ordered_source_fingerprints"]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError(f"{split} metadata-to-cache coverage or duplicate identity failure")
        if len(fingerprints) != len(ids):
            raise ValueError(f"{split} cache fingerprint coverage failure")
        ordered = []
        for sample_id, name, fingerprint in zip(ids, filenames, fingerprints):
            row = by_id[sample_id]
            if row["wave_file_name"] != name or fingerprint["sample_id"] != sample_id:
                raise ValueError(f"{split} metadata-to-cache identity mismatch: {sample_id}")
            if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != fingerprint["caption_sha256"]:
                raise ValueError(f"{split} metadata-to-cache caption mismatch: {sample_id}")
            ordered.append(row)
        aligned[split] = ordered
    return aligned


def fold_indices(rows_by_split: dict, generator_id: str) -> dict[str, list[int]]:
    train = rows_by_split["train"]
    result = {
        "optimization": [i for i, row in enumerate(train) if row["model_id"] != generator_id],
        "internal_holdout": [i for i, row in enumerate(train) if row["model_id"] == generator_id],
        "validation": [i for i, row in enumerate(rows_by_split["validation"]) if row["model_id"] == generator_id],
        "public_test": [i for i, row in enumerate(rows_by_split["test"]) if row["model_id"] == generator_id],
    }
    if not result["optimization"] or not result["internal_holdout"]:
        raise ValueError(f"generator {generator_id} has no optimization or holdout rows")
    if (set(result["optimization"]) & set(result["internal_holdout"]) or
            sorted(result["optimization"] + result["internal_holdout"]) != list(range(len(train)))):
        raise ValueError("optimization and holdout fail the official-train partition contract")
    return result


def aggregate_predictions(rows: list[dict], *, eligible_folds: int, total_samples: int) -> dict[str, Any]:
    from training.moe_evaluator import compute_metrics

    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample identity in pooled LOGO predictions")
    by_fold: dict[str, list[dict]] = {}
    for row in rows:
        by_fold.setdefault(str(row["fold_id"]), []).append(row)
    fold_metrics = {fold: compute_metrics([float(r["predicted_score"]) for r in group],
                                          [float(r["ground_truth"]) for r in group])
                    for fold, group in by_fold.items()}
    result: dict[str, Any] = {"eligible_folds": eligible_folds, "successful_folds": len(by_fold),
                              "evaluated_samples": len(rows), "total_samples": total_samples,
                              "sample_coverage": len(rows) / total_samples if total_samples else None}
    valid = [(fold_metrics[fold]["srcc"], len(group)) for fold, group in by_fold.items()
             if fold_metrics[fold]["srcc"] is not None]
    result.update({"valid_srcc_folds": len(valid),
                   "logo_macro_srcc": mean(r for r, _ in valid) if valid else None,
                   "logo_weighted_srcc": sum(r*n for r, n in valid)/sum(n for _, n in valid) if valid else None,
                   "logo_worst_srcc": min(r for r, _ in valid) if valid else None,
                   "logo_srcc_std": pstdev(r for r, _ in valid) if valid else None})
    pooled = compute_metrics([float(r["predicted_score"]) for r in rows],
                             [float(r["ground_truth"]) for r in rows]) if rows else {}
    for field in ("srcc", "lcc", "ktau", "mse"):
        values = [metric[field] for metric in fold_metrics.values() if metric[field] is not None]
        result[f"macro_{field}"] = mean(values) if values else None
        result[f"pooled_{field}"] = pooled.get(field)
        result[f"valid_{field}_folds"] = len(values)
    result["logo_pooled_srcc"] = result["pooled_srcc"]
    return result
