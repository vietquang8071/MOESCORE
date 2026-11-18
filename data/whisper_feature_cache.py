"""Atomic, split-aware cache for paired Whisper component features.

This cache deliberately has no score field: Whisper is an E6 feature
component, not an independent scoring expert.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


CACHE_SCHEMA_VERSION = 1
FEATURE_EXTRACTION_VERSION = "top2-whisper-large-v2-meanpool-v1"
FEATURE_DIMENSIONS = {"audio": 1280, "text": 1280, "total": 2560}
_SPLITS = {"train", "validation", "test"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    try:
        return json.loads(_canonical_json(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_whisper_dataset_identity(records: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    """Fingerprint ordered sample IDs, exact captions and source audio bytes.

    Audio hashing is done before cache reuse so a reused filename cannot hide
    changed audio. Caption hashes preserve whitespace and row order.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}; got {split!r}")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("dataset records must be a non-empty ordered sequence")
    seen: set[str] = set()
    digest = hashlib.sha256()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"dataset record {index} must be a mapping")
        sample_id = record.get("sample_id")
        caption = record.get("captions_raw")
        audio_path = record.get("audio_path")
        record_split = record.get("split", split)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"dataset record {index} has an invalid sample_id")
        if not sample_id.startswith(f"{split}:"):
            raise ValueError(f"dataset sample ID must begin with the selected split {split!r}: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID in Whisper dataset identity: {sample_id}")
        if record_split != split:
            raise ValueError(f"dataset record {sample_id} belongs to split {record_split!r}, expected {split!r}")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"dataset record {sample_id} has an empty raw caption")
        if not isinstance(audio_path, (str, os.PathLike)):
            raise TypeError(f"dataset record {sample_id} has an invalid audio_path")
        path = Path(audio_path).expanduser().resolve(strict=True)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"dataset audio is missing or unreadable: {path}")
        seen.add(sample_id)
        row_identity = {
            "sample_id": sample_id,
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "audio_sha256": _sha256_file(path),
        }
        digest.update(_canonical_json(row_identity))
        digest.update(b"\n")
    return {
        "split": split,
        "sample_count": len(records),
        "ordered_samples_sha256": digest.hexdigest(),
    }


def validate_whisper_cache_resume(
    dataset_sample_ids: Sequence[str],
    cached_sample_ids: Sequence[str],
    *,
    batch_size: int,
) -> int:
    """Require cached IDs to be a whole-batch prefix of dataset row order."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for name, values in (("dataset_sample_ids", dataset_sample_ids), ("cached_sample_ids", cached_sample_ids)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence of sample IDs")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} must contain non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate sample IDs in {name}")
    cached_count = len(cached_sample_ids)
    if cached_count > len(dataset_sample_ids) or list(cached_sample_ids) != list(dataset_sample_ids[:cached_count]):
        raise ValueError("Whisper cache sample IDs must be an ordered prefix of current split metadata")
    if cached_count < len(dataset_sample_ids) and cached_count % batch_size != 0:
        raise ValueError(
            "Whisper cache is not on an extraction batch boundary; original decoder padding context is unavailable"
        )
    return cached_count


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_npz(path: Path, *, sample_ids: list[str], audio: np.ndarray, text: np.ndarray) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(
                stream,
                sample_ids=np.asarray(sample_ids, dtype=np.str_),
                whisper_audio_features=audio,
                whisper_text_features=text,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _float32_matrix(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
        if hasattr(value, "float") and callable(value.float):
            value = value.float()
        if hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(value.numpy):
            value = value.numpy()
    try:
        matrix = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a numeric matrix or tensor") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 1280:
        raise ValueError(f"{name} must have shape [B,1280]; got {matrix.shape}")
    if matrix.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain numeric values; got {matrix.dtype}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(matrix, dtype=np.float32)


class WhisperFeatureCacheWriter:
    """Append and load source-identified Whisper feature shards for one split.

    A cache directory contains ``manifest.json`` and zero or more atomic
    ``shard-XXXXXXXX.npz`` files. Each NPZ carries sample IDs and the two
    float32 feature matrices in matching row order. Reopening validates the
    complete manifest and all committed shards before exposing IDs for resume.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        split: str,
        source_checkpoint_identity: Mapping[str, Any],
        preprocessing_identity: Mapping[str, Any],
        dataset_identity: Mapping[str, Any],
        feature_extraction_version: str = FEATURE_EXTRACTION_VERSION,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}; got {split!r}")
        if not isinstance(feature_extraction_version, str) or not feature_extraction_version.strip():
            raise ValueError("feature_extraction_version must be a non-empty string")
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=False)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.split = split
        self.source_checkpoint_identity = _identity_mapping(
            source_checkpoint_identity, name="source_checkpoint_identity"
        )
        self.preprocessing_identity = _identity_mapping(
            preprocessing_identity, name="preprocessing_identity"
        )
        self.dataset_identity = _identity_mapping(dataset_identity, name="dataset_identity")
        if set(self.dataset_identity) != {"split", "sample_count", "ordered_samples_sha256"}:
            raise ValueError("dataset_identity must contain split, sample_count, and ordered_samples_sha256")
        if self.dataset_identity["split"] != split:
            raise ValueError("dataset identity split mismatch")
        if type(self.dataset_identity["sample_count"]) is not int or self.dataset_identity["sample_count"] <= 0:
            raise ValueError("dataset_identity.sample_count must be a positive integer")
        dataset_digest = self.dataset_identity["ordered_samples_sha256"]
        if not isinstance(dataset_digest, str) or len(dataset_digest) != 64:
            raise ValueError("dataset_identity.ordered_samples_sha256 must be a SHA-256 digest")
        try:
            int(dataset_digest, 16)
        except ValueError as exc:
            raise ValueError("dataset_identity.ordered_samples_sha256 must be hexadecimal") from exc
        self.feature_extraction_version = feature_extraction_version
        self._manifest = self._identity_header()
        self._sample_ids: list[str] = []

        if self.manifest_path.exists():
            self._read_and_validate_existing()
        else:
            self._write_manifest()

    def _identity_header(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "split": self.split,
            "source_checkpoint_identity": self.source_checkpoint_identity,
            "preprocessing_identity": self.preprocessing_identity,
            "dataset_identity": self.dataset_identity,
            "feature_extraction_version": self.feature_extraction_version,
            "feature_dimensions": FEATURE_DIMENSIONS,
            "shards": [],
        }

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """IDs already committed in the cache, in feature-row order."""
        return tuple(self._sample_ids)

    def _write_manifest(self) -> None:
        manifest = dict(self._manifest)
        manifest["metadata_sha256"] = _identity_sha256(manifest)
        _atomic_json(self.manifest_path, manifest)

    def _read_and_validate_existing(self) -> None:
        try:
            stored = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Whisper cache manifest is missing or corrupt: {self.manifest_path}") from exc
        expected_keys = {
            "schema_version",
            "split",
            "source_checkpoint_identity",
            "preprocessing_identity",
            "dataset_identity",
            "feature_extraction_version",
            "feature_dimensions",
            "shards",
            "metadata_sha256",
        }
        if not isinstance(stored, dict) or set(stored) != expected_keys:
            raise ValueError(f"Whisper cache manifest schema is incomplete or unsupported: {self.manifest_path}")
        saved_digest = stored.pop("metadata_sha256")
        if saved_digest != _identity_sha256(stored):
            raise ValueError(f"Whisper cache manifest SHA-256 mismatch: {self.manifest_path}")
        if stored.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported Whisper cache schema version: {stored.get('schema_version')!r}")
        if stored.get("split") != self.split:
            raise ValueError(f"Whisper cache split mismatch: {stored.get('split')!r} != {self.split!r}")
        if stored.get("source_checkpoint_identity") != self.source_checkpoint_identity:
            raise ValueError("Whisper cache source checkpoint identity mismatch")
        if stored.get("preprocessing_identity") != self.preprocessing_identity:
            raise ValueError("Whisper cache preprocessing identity mismatch")
        if stored.get("dataset_identity") != self.dataset_identity:
            raise ValueError("Whisper cache dataset identity mismatch")
        if stored.get("feature_extraction_version") != self.feature_extraction_version:
            raise ValueError("Whisper cache feature extraction version mismatch")
        if stored.get("feature_dimensions") != FEATURE_DIMENSIONS:
            raise ValueError("Whisper cache feature dimensions mismatch")
        if not isinstance(stored.get("shards"), list):
            raise ValueError("Whisper cache shards metadata must be a list")

        self._manifest = stored
        seen: set[str] = set()
        for shard in stored["shards"]:
            ids = self._validate_shard(shard)
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate sample IDs within a Whisper cache shard")
            duplicate = seen.intersection(ids)
            if duplicate:
                raise ValueError(f"duplicate sample IDs across Whisper cache shards: {sorted(duplicate)[:3]}")
            seen.update(ids)
            self._sample_ids.extend(ids)

    def _validate_shard(self, shard: Any) -> list[str]:
        if not isinstance(shard, dict) or set(shard) != {"file", "sample_ids", "sample_count", "sha256"}:
            raise ValueError("Whisper cache shard metadata is incomplete")
        filename = shard["file"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError(f"invalid Whisper cache shard filename: {filename!r}")
        sample_ids = shard["sample_ids"]
        if (
            not isinstance(sample_ids, list)
            or not sample_ids
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        ):
            raise ValueError(f"invalid sample IDs in Whisper cache shard metadata: {filename}")
        if shard["sample_count"] != len(sample_ids):
            raise ValueError(f"sample count mismatch in Whisper cache shard metadata: {filename}")
        path = self.cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Whisper cache shard is missing: {path}")
        if _sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Whisper cache shard SHA-256 mismatch: {path}")
        result = self._read_shard_arrays(path)
        if result["sample_ids"] != sample_ids:
            raise ValueError(f"Whisper cache sample IDs do not match shard metadata: {path}")
        if result["whisper_audio_features"].shape[0] != len(sample_ids):
            raise ValueError(f"Whisper cache feature count mismatch: {path}")
        return sample_ids

    @staticmethod
    def _read_shard_arrays(path: Path) -> dict[str, Any]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "sample_ids",
                    "whisper_audio_features",
                    "whisper_text_features",
                }:
                    raise ValueError(f"unexpected arrays in Whisper cache shard: {path}")
                raw_ids = archive["sample_ids"].copy()
                audio = archive["whisper_audio_features"].copy()
                text = archive["whisper_text_features"].copy()
        except (OSError, EOFError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("unexpected arrays"):
                raise
            raise ValueError(f"Whisper cache shard is corrupt: {path}: {exc}") from exc
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "US":
            raise ValueError(f"Whisper cache sample_ids must be a one-dimensional string array: {path}")
        sample_ids = [str(value) for value in raw_ids.tolist()]
        if not sample_ids or any(not sample_id for sample_id in sample_ids):
            raise ValueError(f"Whisper cache shard contains empty sample IDs: {path}")
        for name, features in (("audio", audio), ("text", text)):
            if features.dtype != np.dtype(np.float32) or features.shape != (len(sample_ids), 1280):
                raise ValueError(f"Whisper cache {name} features must be float32 [N,1280]: {path}")
            if not np.isfinite(features).all():
                raise ValueError(f"Whisper cache {name} features contain non-finite values: {path}")
        return {
            "sample_ids": sample_ids,
            "whisper_audio_features": audio,
            "whisper_text_features": text,
        }

    def append(self, sample_ids: Sequence[str], audio_features: Any, text_features: Any) -> Path:
        if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence) or not sample_ids:
            raise ValueError("sample_ids must be a non-empty sequence of IDs")
        ids = list(sample_ids)
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
            raise ValueError("sample_ids must contain non-empty strings")
        if any(not sample_id.startswith(f"{self.split}:") for sample_id in ids):
            raise ValueError(f"sample IDs in a {self.split!r} cache must use the same split prefix")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample IDs within Whisper cache append batch")
        if len(self._sample_ids) + len(ids) > self.dataset_identity["sample_count"]:
            raise ValueError("Whisper cache append exceeds the dataset identity sample count")
        duplicate = set(ids).intersection(self._sample_ids)
        if duplicate:
            raise ValueError(f"duplicate sample IDs already exist in Whisper cache: {sorted(duplicate)[:3]}")
        audio = _float32_matrix(audio_features, name="whisper_audio_features")
        text = _float32_matrix(text_features, name="whisper_text_features")
        if audio.shape[0] != len(ids) or text.shape[0] != len(ids):
            raise ValueError("sample IDs and Whisper audio/text feature row counts must match")

        filename = f"shard-{len(self._manifest['shards']):08d}.npz"
        path = self.cache_dir / filename
        _atomic_npz(path, sample_ids=ids, audio=audio, text=text)
        shard_metadata = {
            "file": filename,
            "sample_ids": ids,
            "sample_count": len(ids),
            "sha256": _sha256_file(path),
        }
        updated = dict(self._manifest)
        updated["shards"] = [*self._manifest["shards"], shard_metadata]
        previous = self._manifest
        self._manifest = updated
        try:
            self._write_manifest()
        except Exception:
            self._manifest = previous
            raise
        self._sample_ids.extend(ids)
        return path

    def load(self) -> dict[str, Any]:
        """Verify and concatenate all committed shards in manifest order."""
        arrays = []
        for shard in self._manifest["shards"]:
            self._validate_shard(shard)
            arrays.append(self._read_shard_arrays(self.cache_dir / shard["file"]))
        if not arrays:
            return {
                "sample_ids": [],
                "whisper_audio_features": np.empty((0, 1280), dtype=np.float32),
                "whisper_text_features": np.empty((0, 1280), dtype=np.float32),
            }
        return {
            "sample_ids": [sample_id for shard in arrays for sample_id in shard["sample_ids"]],
            "whisper_audio_features": np.concatenate(
                [shard["whisper_audio_features"] for shard in arrays], axis=0
            ).astype(np.float32, copy=False),
            "whisper_text_features": np.concatenate(
                [shard["whisper_text_features"] for shard in arrays], axis=0
            ).astype(np.float32, copy=False),
        }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "FEATURE_DIMENSIONS",
    "FEATURE_EXTRACTION_VERSION",
    "WhisperFeatureCacheWriter",
    "build_whisper_dataset_identity",
    "validate_whisper_cache_resume",
]
