"""Narrow train/validation WAV-overlap policy and deterministic diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_WARNING = (
    "Validation results may be optimistically biased because identical WAV audio "
    "appears in both training and validation."
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validated_audio_rows(records: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{split} source audio records must be a sequence")
    result = []
    sample_ids = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError(f"{split} source audio record is invalid")
        sample_id = row.get("sample_id")
        digest = row.get("wav_sha256")
        if not isinstance(sample_id, str) or not sample_id.startswith(split + ":"):
            raise ValueError(f"{split} source sample ID is missing or invalid")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate {split} source sample ID: {sample_id}")
        sample_ids.add(sample_id)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{split} source audio SHA-256 is missing or invalid")
        result.append({"sample_id": sample_id, "wav_sha256": digest.lower()})
    return result


def validate_source_audio_records(records: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, str]]:
    """Public split-aware WAV/sample identity validation for source manifests."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported source split: {split}")
    return _validated_audio_rows(records, split=split)


def build_train_validation_audio_overlap_diagnostic(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    allow_train_validation_audio_overlap: bool,
    example_limit: int = 5,
) -> dict[str, Any]:
    """Validate source rows and describe every train/validation WAV hash overlap."""
    if type(allow_train_validation_audio_overlap) is not bool:
        raise ValueError("allow_train_validation_audio_overlap must be boolean")
    if type(example_limit) is not int or example_limit < 0:
        raise ValueError("example_limit must be a non-negative integer")
    train = validate_source_audio_records(train_records, split="train")
    validation = validate_source_audio_records(validation_records, split="validation")

    train_by_hash: dict[str, list[str]] = defaultdict(list)
    validation_by_hash: dict[str, list[str]] = defaultdict(list)
    for row in train:
        train_by_hash[row["wav_sha256"]].append(row["sample_id"])
    for row in validation:
        validation_by_hash[row["wav_sha256"]].append(row["sample_id"])

    overlaps = [
        {
            "wav_sha256": digest,
            "train_sample_ids": sorted(train_by_hash[digest]),
            "validation_sample_ids": sorted(validation_by_hash[digest]),
        }
        for digest in sorted(set(train_by_hash) & set(validation_by_hash))
    ]
    affected_train_ids = {sample_id for row in overlaps for sample_id in row["train_sample_ids"]}
    affected_validation_ids = {sample_id for row in overlaps for sample_id in row["validation_sample_ids"]}
    allowed = allow_train_validation_audio_overlap
    diagnostic: dict[str, Any] = {
        "schema_version": 1,
        "status": "WARNING" if overlaps and allowed else "ERROR" if overlaps else "CLEAR",
        "policy": "experimental_train_validation_audio_overlap" if allowed
                  else "strict_train_validation_audio_overlap",
        "allow_train_validation_audio_overlap": allowed,
        "overlapping_sha256_count": len(overlaps),
        "affected_train_rows": len(affected_train_ids),
        "affected_validation_rows": len(affected_validation_ids),
        "overlaps": overlaps,
        "examples": overlaps[:example_limit],
    }
    if overlaps and allowed:
        diagnostic["warning"] = _WARNING
    diagnostic["diagnostic_sha256"] = hashlib.sha256(_canonical(diagnostic)).hexdigest()
    return diagnostic


def validate_audio_overlap_diagnostic(value: Any) -> dict[str, Any]:
    """Validate the diagnostic shape, membership totals, policy, and stable digest."""
    if not isinstance(value, dict):
        raise ValueError("audio-overlap diagnostic must be a mapping")
    required = {
        "schema_version", "status", "policy", "allow_train_validation_audio_overlap",
        "overlapping_sha256_count", "affected_train_rows", "affected_validation_rows",
        "overlaps", "examples", "diagnostic_sha256",
    }
    if (not required.issubset(value) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1):
        raise ValueError("audio-overlap diagnostic schema is incomplete")
    allowed = value["allow_train_validation_audio_overlap"]
    if type(allowed) is not bool:
        raise ValueError("audio-overlap diagnostic policy flag is invalid")
    overlaps = value["overlaps"]
    examples = value["examples"]
    if not isinstance(overlaps, list) or not isinstance(examples, list):
        raise ValueError("audio-overlap diagnostic membership is invalid")
    hashes = []
    train_ids = set()
    validation_ids = set()
    for row in overlaps:
        if (not isinstance(row, dict) or set(row) != {"wav_sha256", "train_sample_ids", "validation_sample_ids"}
                or not isinstance(row.get("wav_sha256"), str)
                or _SHA256.fullmatch(row["wav_sha256"]) is None
                or not isinstance(row.get("train_sample_ids"), list)
                or not isinstance(row.get("validation_sample_ids"), list)
                or not row["train_sample_ids"] or not row["validation_sample_ids"]):
            raise ValueError("audio-overlap diagnostic hash membership is invalid")
        hashes.append(row["wav_sha256"])
        for split, ids, seen in (
            ("train", row["train_sample_ids"], train_ids),
            ("validation", row["validation_sample_ids"], validation_ids),
        ):
            if (any(not isinstance(sample_id, str) or not sample_id.startswith(split + ":")
                    for sample_id in ids)
                    or ids != sorted(set(ids))
                    or seen.intersection(ids)):
                raise ValueError(f"audio-overlap diagnostic {split} membership is invalid")
            seen.update(ids)
    if hashes != sorted(set(hashes)):
        raise ValueError("audio-overlap diagnostic hash order is invalid")
    expected_policy = ("experimental_train_validation_audio_overlap" if allowed
                       else "strict_train_validation_audio_overlap")
    expected_status = "WARNING" if overlaps and allowed else "ERROR" if overlaps else "CLEAR"
    if (any(type(value[field]) is not int or value[field] < 0 for field in (
                "overlapping_sha256_count", "affected_train_rows", "affected_validation_rows"))
            or value["policy"] != expected_policy or value["status"] != expected_status
            or value["overlapping_sha256_count"] != len(overlaps)
            or value["affected_train_rows"] != len(train_ids)
            or value["affected_validation_rows"] != len(validation_ids)
            or examples != overlaps[:len(examples)] or len(examples) > 5):
        raise ValueError("audio-overlap diagnostic policy or totals are inconsistent")
    if overlaps and allowed and value.get("warning") != _WARNING:
        raise ValueError("audio-overlap diagnostic is missing its validation-bias warning")
    if (not overlaps or not allowed) and "warning" in value:
        raise ValueError("audio-overlap diagnostic contains an unexpected warning")
    unsigned = {key: item for key, item in value.items() if key != "diagnostic_sha256"}
    expected_digest = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if value.get("diagnostic_sha256") != expected_digest:
        raise ValueError("audio-overlap diagnostic digest mismatch")
    return value


def emit_audio_overlap_warning(diagnostic: Mapping[str, Any], *, stage: str, example_limit: int = 5) -> None:
    """Emit a bounded console warning while preserving full diagnostic in artifacts."""
    validated = validate_audio_overlap_diagnostic(dict(diagnostic))
    if validated["status"] != "WARNING":
        return
    console = {
        "status": "WARNING",
        "stage": stage,
        "policy": validated["policy"],
        "allow_train_validation_audio_overlap": True,
        "overlapping_sha256_count": validated["overlapping_sha256_count"],
        "affected_train_rows": validated["affected_train_rows"],
        "affected_validation_rows": validated["affected_validation_rows"],
        "examples": validated["overlaps"][:example_limit],
        "diagnostic_sha256": validated["diagnostic_sha256"],
        "warning": validated["warning"],
    }
    warnings.warn(json.dumps(console, sort_keys=True, ensure_ascii=False), UserWarning, stacklevel=2)


def raise_for_audio_overlap(diagnostic: Mapping[str, Any], *, stage: str) -> None:
    """Fail strict-mode overlaps while keeping a stable, searchable error message."""
    validated = validate_audio_overlap_diagnostic(dict(diagnostic))
    if validated["overlapping_sha256_count"]:
        first = validated["overlaps"][0]
        raise ValueError(
            f"{stage} train/validation source audio overlap (cross-split duplicate WAV SHA-256): "
            f"{first['wav_sha256']} "
            f"train={first['train_sample_ids']} validation={first['validation_sample_ids']}"
        )
