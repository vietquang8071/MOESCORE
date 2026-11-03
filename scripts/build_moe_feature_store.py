#!/usr/bin/env python3
"""Validate sequential expert artifacts and import them into split shards.

This utility does not construct models or deserialize checkpoints. It checks
the LOCAL-10.1 numeric artifacts and their reports, then writes a split-bound
feature store suitable for gating-only training/evaluation/inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_feature_store import MoeFeatureStoreWriter
from model.expert_adapters.registry import EXPERT_FEATURE_DIMS, EXPERT_ORDER
from scripts.verify_real_expert_to_gate import (
    load_output_artifact,
    validate_sample_manifest,
)


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path, split: str) -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("unsupported source manifest schema")
    if manifest.get("split") != split:
        raise ValueError("source manifest split does not match the requested split")
    if tuple(manifest.get("expert_order", ())) != EXPERT_ORDER:
        raise ValueError("source manifest expert order mismatch")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source manifest must contain samples")
    ids = [item.get("sample_id") if isinstance(item, Mapping) else None for item in samples]
    if any(not isinstance(value, str) or not value.startswith(split + ":") for value in ids):
        raise ValueError("source manifest sample IDs are missing or belong to another split")
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate sample IDs")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    calculated = _canonical_sha256(body)
    recorded = manifest.get("manifest_sha256")
    if recorded != calculated:
        raise ValueError("source manifest fingerprint does not match its contents")
    if split == "train":
        if validate_sample_manifest(manifest) != calculated:
            raise ValueError("LOCAL-10.1 manifest validator returned a different identity")
    return manifest, calculated


def _verified_report(artifact_dir: Path, expert: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = artifact_dir / f"{expert}_report.json"
    artifact_path = artifact_dir / f"{expert}_outputs.npz"
    if not report_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(f"missing verified LOCAL-10.1 report or output for {expert}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("expert") != expert:
        raise ValueError(f"{expert} runtime report is not a matching PASS report")
    model = report.get("model")
    if not isinstance(model, Mapping) or model.get("factory_restoration") != "PASS_FROM_EXISTING_SOURCE_FACTORY":
        raise ValueError(f"{expert} report does not prove source factory checkpoint restoration")
    if not model.get("all_parameters_frozen") or not model.get("all_modules_eval"):
        raise ValueError(f"{expert} real model was not reported frozen and in eval mode")
    before, after = model.get("checkpoint_metadata_before"), model.get("checkpoint_metadata_after")
    if not isinstance(before, Mapping) or before != after:
        raise ValueError(f"{expert} checkpoint metadata changed across the verified run")
    for item in after.values():
        if not isinstance(item, Mapping) or not item.get("exists") or not item.get("readable"):
            raise ValueError(f"{expert} checkpoint provenance is incomplete")
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"verified checkpoint is no longer present: {path}")
        stat = path.stat()
        if stat.st_size != item.get("size_bytes") or stat.st_mtime_ns != item.get("mtime_ns"):
            raise ValueError(f"verified checkpoint metadata no longer matches: {path}")
    if report.get("artifact_sha256") != _sha256_file(artifact_path):
        raise ValueError(f"{expert} numeric artifact hash differs from its runtime report")
    return report, {key: dict(value) for key, value in after.items()}


def _extract_raw_labels(records: Sequence[Mapping[str, Any]], split: str) -> tuple[bool, list[float | None]]:
    """Preserve an all-present or all-absent label field; reject partial rows."""
    raw_values = [record.get("label_raw_0_to_10", record.get("label_raw")) for record in records]
    present = [value is not None for value in raw_values]
    if any(present) and not all(present):
        raise ValueError(f"{split} source manifest has labels for only some samples")
    labels_present = all(present)
    if split == "train" and not labels_present:
        raise ValueError("training output manifest is missing raw labels")
    labels: list[float | None] = []
    for record, value in zip(records, raw_values):
        if value is None:
            labels.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 10.0:
            sample_id = record.get("sample_id", "<unknown>")
            raise ValueError(f"raw label is invalid for {sample_id}")
        labels.append(numeric)
    return labels_present, labels


def import_verified_outputs(
    *,
    artifact_dir: str | Path,
    output_root: str | Path,
    split: str,
    shard_size: int = 256,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    manifest, manifest_sha = _read_manifest(artifact_dir / "manifest.json", split)
    records = manifest["samples"]
    sample_ids = [record["sample_id"] for record in records]
    checkpoint_provenance: dict[str, Any] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    report_ids: dict[str, list[str]] = {}
    for expert in EXPERT_ORDER:
        report, checkpoint_provenance[expert] = _verified_report(artifact_dir, expert)
        output_report = report.get("output", {})
        report_ids[expert] = output_report.get("sample_ids", [])
        if report_ids[expert] != sample_ids:
            raise ValueError(f"{expert} report sample sequence differs from the shared manifest")
        artifacts[expert] = load_output_artifact(
            artifact_dir / f"{expert}_outputs.npz",
            expert=expert,
            expected_dim=EXPERT_FEATURE_DIMS[expert],
            expected_manifest_sha256=manifest_sha,
            sample_records=records,
        )
    if len({tuple(values) for values in report_ids.values()}) != 1:
        raise ValueError("expert artifact sample identity sequences differ")

    labels_present, raw_labels = _extract_raw_labels(records, split)

    if dataset_root is not None:
        from data.unified_dataset import UnifiedXACLEDataset

        dataset = UnifiedXACLEDataset.from_dataset_root(
            dataset_root,
            split=split,
            require_labels=labels_present,
        )
        rows = {row["sample_id"]: row for row in dataset.rows}
        for record, label in zip(records, raw_labels):
            row = rows.get(record["sample_id"])
            if row is None or row["wave_file_name"] != record.get("wave_file_name"):
                raise ValueError(f"source record identity mismatch for {record['sample_id']}")
            if row["captions_raw"] != record.get("raw_caption"):
                raise ValueError(f"source caption mismatch for {record['sample_id']}")
            if labels_present and float(row["label_value"]) != label:
                raise ValueError(f"source raw label mismatch for {record['sample_id']}")
            audio_path = Path(row["audio_path"])
            expected_wav_hash = record.get("wav_sha256")
            if expected_wav_hash and _sha256_file(audio_path) != expected_wav_hash:
                raise ValueError(f"source WAV hash mismatch for {record['sample_id']}")

    scores = np.concatenate(
        [artifacts[name]["scores"] for name in EXPERT_ORDER], axis=1
    ).astype(np.float32, copy=False)
    features = np.concatenate(
        [artifacts[name]["features"] for name in EXPERT_ORDER], axis=1
    ).astype(np.float32, copy=False)
    if scores.shape != (len(records), 4) or features.shape != (len(records), 7936):
        raise ValueError("assembled feature-store shape does not match LOCAL-08/10 contracts")
    writer = MoeFeatureStoreWriter(
        output_root,
        split=split,
        expected_sample_ids=sample_ids,
        source_manifest_sha256=manifest_sha,
        checkpoint_provenance=checkpoint_provenance,
        preprocessing_identity={
            "contract": "UnifiedXACLEDataset-LOCAL-09.1",
            "source_manifest_sha256": manifest_sha,
            "expert_order": list(EXPERT_ORDER),
        },
    )
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        stop = min(start + shard_size, len(records))
        labels = np.asarray(raw_labels[start:stop], dtype=np.float32) if labels_present else None
        class_values = [record.get("num_class") for record in records[start:stop]]
        classes = None
        if any(value is not None for value in class_values):
            if not all(value is not None for value in class_values):
                raise ValueError("num_class metadata must be present for every record or none")
            classes = np.asarray(class_values, dtype=np.int64)
        writer.write_shard(
            shard_index,
            sample_ids=sample_ids[start:stop],
            wave_file_names=[record["wave_file_name"] for record in records[start:stop]],
            scores=scores[start:stop],
            features=features[start:stop],
            labels_raw=labels,
            num_class=classes,
        )
    manifest_path = writer.finalize()
    return {
        "status": "PASS",
        "split": split,
        "sample_count": len(records),
        "source_manifest_sha256": manifest_sha,
        "store_manifest": str(manifest_path),
        "store_identity_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))["store_identity_sha256"],
        "scores_shape": list(scores.shape),
        "features_shape": list(features.shape),
        "scores_dtype": str(scores.dtype),
        "features_dtype": str(features.dtype),
        "checkpoint_provenance_evidence": "runtime report checkpoint path/size/mtime and artifact hashes validated; no checkpoint contents reloaded",
        "model_construction": False,
        "checkpoint_deserialization": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = import_verified_outputs(
            artifact_dir=args.artifact_dir,
            output_root=args.output_root,
            split=args.split,
            shard_size=args.shard_size,
            dataset_root=args.dataset_root,
        )
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.report_json:
        path = Path(args.report_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
