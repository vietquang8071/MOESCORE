"""Join verified Full7 rows with the independent frozen Expert8 cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data.full8_contract import build_expert8_score_contract
from data.moe_expert_cache import build_sample_identity, load_expert_shard, sha256_file
from data.moe_feature_store import (
    MoeFeatureStoreDataset,
    MoeFeatureStoreWriter,
    _canonical_json,
    _sha256_bytes,
    _safe_shard_path,
    preprocessing_contract,
    validate_feature_store,
)
from model.expert_adapters.registry import FULL7_EXPERT_ORDER, FULL8_EXPERT_ORDER


def _load_e8_manifest(cache_root: Path, split: str, cache_id: str | None = None) -> tuple[dict[str, Any], Path]:
    base = cache_root / "expert8"
    pattern = f"{cache_id}/{split}/manifest.json" if cache_id else f"*/{split}/manifest.json"
    candidates = sorted(base.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one completed Expert8 {split} cache manifest under {base}; found {len(candidates)}"
        )
    path = candidates[0]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Expert8 cache manifest is unreadable: {path}: {exc}") from exc
    return manifest, path.parent


def _ordered_full7_rows(dataset: MoeFeatureStoreDataset) -> tuple[list[str], list[str], list[dict[str, str]]]:
    sample_ids = [sample_id for shard in dataset.manifest["shards"] for sample_id in shard["sample_ids"]]
    filenames = [name for shard in dataset.manifest["shards"] for name in shard["wave_file_names"]]
    fingerprints = dataset.manifest["assembly_identity"]["ordered_source_fingerprints"]
    if len(sample_ids) != len(filenames) or len(sample_ids) != len(fingerprints):
        raise ValueError("Full7 ordered sample, filename and source-fingerprint counts differ")
    return sample_ids, filenames, fingerprints


def _sample_identity(split: str, records, cache_identity):
    return build_sample_identity(
        split,
        [{key: row[key] for key in ("sample_id", "wave_file_name", "raw_caption", "wav_sha256")}
         for row in records],
        {"expert_cache_id": cache_identity["cache_id"],
         "input_contract": cache_identity["input_contract"],
         "preprocessing_provenance": cache_identity["preprocessing_provenance"]},
        3073,
    )


def build_full8_split(
    full7_root: str | Path,
    expert8_cache_root: str | Path,
    output_root: str | Path,
    *, split: str, expert8_cache_id: str | None = None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    full7 = MoeFeatureStoreDataset(
        full7_root, split=split, require_labels=split != "test"
    )
    if full7.manifest.get("expert_profile") != "full_7":
        raise ValueError("Full8 assembly requires a verified Full7 source store")
    e8_manifest, e8_split_root = _load_e8_manifest(
        Path(expert8_cache_root).resolve(), split, cache_id=expert8_cache_id
    )
    cache_identity = e8_manifest.get("expert_cache_identity")
    records = e8_manifest.get("source_records")
    from scripts.extract_expert8_features import validate_expert8_cache_manifest
    validate_expert8_cache_manifest(
        e8_manifest, split=split, expected_records=records, cache_identity=cache_identity
    )
    sample_ids, filenames, full7_fingerprints = _ordered_full7_rows(full7)
    e8_ids = [row.get("sample_id") for row in records]
    e8_names = [row.get("wave_file_name") for row in records]
    if e8_ids != sample_ids or e8_names != filenames:
        raise ValueError("Expert8 cache sample IDs/filenames are not exactly aligned with Full7")
    if len(records) != len(full7_fingerprints):
        raise ValueError("Expert8 cache and Full7 source row counts differ")
    for e8_row, full7_row in zip(records, full7_fingerprints):
        if (e8_row.get("wav_sha256") != full7_row.get("wav_sha256")
                or e8_row.get("caption_sha256") != full7_row.get("caption_sha256")
                or hashlib.sha256(e8_row["raw_caption"].encode("utf-8")).hexdigest()
                != full7_row.get("caption_sha256")):
            raise ValueError(f"Expert8 WAV/caption identity differs from Full7 at {e8_row.get('sample_id')}")

    checkpoint_provenance = dict(full7.manifest["checkpoint_provenance"])
    checkpoint_provenance["expert8"] = cache_identity["checkpoint_provenance"]
    lora_sha = cache_identity["checkpoint_provenance"]["finetuned_checkpoint"]["sha256"]
    full7_assembly = full7.manifest["assembly_identity"]
    synthetic = bool(full7_assembly.get("synthetic_provenance", False))
    score_contracts = dict(full7_assembly["score_contracts"])
    score_contracts["expert8"] = build_expert8_score_contract(lora_sha, synthetic=synthetic)
    assembly = {
        "schema_version": 1,
        "expert_profile": "full_8",
        "split": split,
        "source_manifest_sha256": full7.manifest["source_manifest_sha256"],
        "full7_store_identity_sha256": full7.identity,
        "full7_assembly_contract_sha256": full7.manifest["core_assembly_contract_sha256"],
        "expert8_cache_identity": cache_identity,
        "expert8_cache_id": cache_identity["cache_id"],
        "expert8_cache_manifest_sha256": e8_manifest["manifest_sha256"],
        "ordered_source_fingerprints": full7_fingerprints,
        "score_contracts": score_contracts,
        "shared_source_contract": full7_assembly["shared_source_contract"],
        "head_membership": full7_assembly["head_membership"],
        "synthetic_provenance": synthetic,
        "core_assembly_contract": "full8_v1",
    }
    output_root = Path(output_root).expanduser().resolve()
    source_contract = {
        "full7_preprocessing_contract": preprocessing_contract(full7.manifest),
        "expert8_cache_identity": cache_identity,
        "full8_assembly_contract": "full8_v1",
    }
    writer = MoeFeatureStoreWriter(
        output_root, split=split, expected_sample_ids=sample_ids,
        source_manifest_sha256=full7.manifest["source_manifest_sha256"],
        checkpoint_provenance=checkpoint_provenance,
        preprocessing_identity=source_contract,
        profile="full_8", assembly_identity=assembly,
    )
    source_position = 0
    for entry in e8_manifest["shards"]:
        count = entry["count"]
        selected_records = records[source_position:source_position + count]
        sample_identity = _sample_identity(split, selected_records, cache_identity)
        if entry.get("sample_identity_sha256") != sample_identity["sample_identity_sha256"]:
            raise ValueError("Expert8 cache shard is bound to a different row order")
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("Expert8 cache shard path is missing")
        path = _safe_shard_path(e8_split_root, relative)
        if sha256_file(path) != entry.get("file_sha256"):
            raise ValueError(f"Expert8 cache shard file hash mismatch: {path}")
        arrays = load_expert_shard(
            path, expert_identity=cache_identity, sample_identity=sample_identity,
            feature_dim=3073,
        )
        rows = [full7[index] for index in range(source_position, source_position + count)]
        if ([row["sample_id"] for row in rows] != [row["sample_id"] for row in selected_records]
                or [row["wave_file_name"] for row in rows] != [row["wave_file_name"] for row in selected_records]):
            raise ValueError("Full7 row order changed while assembling Full8")
        scores7 = torch.stack([row["scores"] for row in rows])
        features7 = torch.stack([row["features"] for row in rows])
        scores8 = torch.cat((scores7, torch.from_numpy(arrays["scores"])), dim=1)
        features8 = torch.cat((features7, torch.from_numpy(arrays["features"])), dim=1)
        if (tuple(scores8.shape) != (count, 8) or tuple(features8.shape) != (count, 33032)
                or scores8.dtype != torch.float32 or features8.dtype != torch.float32):
            raise ValueError("Full8 concatenated score/feature tensor contract mismatch")
        labels = None
        if split != "test":
            labels = np.asarray([row["label_raw"].item() for row in rows], dtype=np.float32)
        writer.write_shard(
            entry["index"], sample_ids=[row["sample_id"] for row in selected_records],
            wave_file_names=[row["wave_file_name"] for row in selected_records],
            scores=scores8.numpy(), features=features8.numpy(), labels_raw=labels,
        )
        source_position += count
    if source_position != len(full7):
        raise ValueError("Expert8 shards do not cover the entire Full7 split")
    manifest_path = writer.finalize()
    manifest = validate_feature_store(output_root, split=split, require_labels=split != "test")
    return {
        "split": split, "sample_count": len(full7), "manifest": str(manifest_path),
        "store_identity_sha256": manifest["store_identity_sha256"],
        "full7_store_identity_sha256": full7.identity,
        "expert8_cache_manifest_sha256": e8_manifest["manifest_sha256"],
        "feature_width": manifest["feature_width"], "score_width": manifest["score_width"],
    }


def build_full8_stores(
    full7_root: str | Path,
    expert8_cache_root: str | Path,
    output_root: str | Path,
    *, allow_train_validation_audio_overlap: bool = False,
    allow_public_split_audio_overlap: bool = False,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    expert8_cache_id: str | None = None,
    skip_cross_split_audio_overlap_check: bool = False,
) -> dict[str, Any]:
    if (type(allow_train_validation_audio_overlap) is not bool
            or type(allow_public_split_audio_overlap) is not bool
            or type(skip_cross_split_audio_overlap_check) is not bool):
        raise ValueError("audio overlap policies must be boolean")
    splits = tuple(dict.fromkeys(splits))
    if not splits or any(split not in {"train", "validation", "test"} for split in splits):
        raise ValueError("splits must contain train, validation, and/or test")
    selected_splits = splits if skip_cross_split_audio_overlap_check else ("train", "validation", "test")
    full7_manifests = {
        split: validate_feature_store(full7_root, split=split, require_labels=split != "test")
        for split in selected_splits
    }
    diagnostic = public_diagnostic = None
    if not skip_cross_split_audio_overlap_check:
        train_rows = full7_manifests["train"]["assembly_identity"]["ordered_source_fingerprints"]
        val_rows = full7_manifests["validation"]["assembly_identity"]["ordered_source_fingerprints"]
        test_rows = full7_manifests["test"]["assembly_identity"]["ordered_source_fingerprints"]
        from data.audio_overlap import (
            build_public_split_audio_overlap_diagnostic,
            build_train_validation_audio_overlap_diagnostic,
            emit_audio_overlap_warning,
            emit_public_split_audio_overlap_warning,
            raise_for_audio_overlap,
            raise_for_public_split_audio_overlap,
        )
        diagnostic = build_train_validation_audio_overlap_diagnostic(
            train_rows, val_rows,
            allow_train_validation_audio_overlap=(allow_train_validation_audio_overlap
                                                  or allow_public_split_audio_overlap),
        )
        if diagnostic["overlapping_sha256_count"]:
            if not (allow_train_validation_audio_overlap or allow_public_split_audio_overlap):
                raise_for_audio_overlap(diagnostic, stage="Full8 builder")
            emit_audio_overlap_warning(diagnostic, stage="Full8 builder")
        public_diagnostic = build_public_split_audio_overlap_diagnostic(
            {"train": train_rows, "validation": val_rows, "test": test_rows},
            allow_public_split_audio_overlap=allow_public_split_audio_overlap,
            allow_train_validation_audio_overlap=allow_train_validation_audio_overlap,
        )
        if public_diagnostic["status"] == "ERROR":
            raise_for_public_split_audio_overlap(public_diagnostic, stage="Full8 builder")
        if public_diagnostic["status"] == "WARNING":
            emit_public_split_audio_overlap_warning(public_diagnostic, stage="Full8 builder")
    reports = [build_full8_split(full7_root, expert8_cache_root, output_root, split=split,
                                 expert8_cache_id=expert8_cache_id)
               for split in splits]
    result = {"status": "PASS", "expert_profile": "full_8", "splits": reports,
              "allow_train_validation_audio_overlap": allow_train_validation_audio_overlap,
              "allow_public_split_audio_overlap": allow_public_split_audio_overlap,
              "run_purpose": "research_only"}
    if skip_cross_split_audio_overlap_check:
        result["cross_split_audio_overlap"] = {
            "status": "SKIPPED",
            "reason": "Explicit D02 research-only configuration",
            "scope": list(splits),
        }
    else:
        result["audio_overlap_diagnostic"] = diagnostic
        result["public_split_audio_overlap_diagnostic"] = public_diagnostic
    return result


__all__ = ["build_full8_split", "build_full8_stores"]
