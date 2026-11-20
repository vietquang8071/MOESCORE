"""Validation for ordered source rows within one official dataset split."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SPLITS = {"train", "validation", "test"}


def validate_source_fingerprints(
    records: Sequence[Mapping[str, Any]], *, split: str,
) -> list[dict[str, str]]:
    """Validate WAV hashes and unique IDs within *one* split, without cross-split checks."""
    if split not in _SPLITS:
        raise ValueError(f"unsupported source split: {split}")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{split} source fingerprints must be a sequence")
    result = []
    sample_ids = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError(f"{split} source fingerprint is invalid")
        sample_id = row.get("sample_id")
        digest = row.get("wav_sha256")
        if not isinstance(sample_id, str) or not sample_id.startswith(split + ":"):
            raise ValueError(f"{split} source sample ID is missing or invalid")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate {split} source sample ID: {sample_id}")
        sample_ids.add(sample_id)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{split} source WAV SHA-256 is missing or invalid")
        result.append({"sample_id": sample_id, "wav_sha256": digest.lower()})
    return result
