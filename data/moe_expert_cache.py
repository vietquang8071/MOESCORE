"""Schema-v2 per-expert cache identities and verified extraction shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


EXPERT_CACHE_SCHEMA_VERSION = 2
_SHA256_LENGTH = 64


def canonical_json(value: Any) -> bytes:
    """Serialize identity metadata deterministically and reject non-finite JSON."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(canonical_json(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be finite canonical-JSON data: {exc}") from exc


def _required_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty mapping")
    return _json_copy(value, field=field)


def build_expert_cache_identity(
    *,
    expert_name: str,
    checkpoint_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    preprocessing_provenance: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    inference_policy: Mapping[str, Any],
    score_contract: Mapping[str, Any],
    relevant_dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an identity containing only sources/resources that affect one expert."""
    if not isinstance(expert_name, str) or not expert_name.strip():
        raise ValueError("expert_name must be a non-empty string")
    body = {
        "schema_version": EXPERT_CACHE_SCHEMA_VERSION,
        "expert_name": expert_name,
        "checkpoint_provenance": _required_mapping(checkpoint_provenance, field="checkpoint_provenance"),
        "implementation_provenance": _required_mapping(implementation_provenance, field="implementation_provenance"),
        "preprocessing_provenance": _required_mapping(preprocessing_provenance, field="preprocessing_provenance"),
        "input_contract": _required_mapping(input_contract, field="input_contract"),
        "inference_policy": _required_mapping(inference_policy, field="inference_policy"),
        "score_contract": _required_mapping(score_contract, field="score_contract"),
        "relevant_dependencies": _required_mapping(relevant_dependencies, field="relevant_dependencies"),
    }
    return {**body, "cache_id": identity_sha256(body)}


def validate_expert_cache_identity(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping):
        raise TypeError("expert cache identity must be a mapping")
    expected_keys = {
        "schema_version", "expert_name", "checkpoint_provenance", "implementation_provenance",
        "preprocessing_provenance", "input_contract", "inference_policy", "score_contract",
        "relevant_dependencies", "cache_id",
    }
    if set(identity) != expected_keys or identity.get("schema_version") != EXPERT_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete expert cache identity schema")
    body = {key: identity[key] for key in expected_keys if key != "cache_id"}
    actual = identity_sha256(body)
    if identity.get("cache_id") != actual:
        raise ValueError("expert cache identity SHA-256 mismatch")
    return actual


def _expert_cache_id(value: Any, *, name: str) -> str:
    if isinstance(value, Mapping):
        cache_id = validate_expert_cache_identity(value)
    else:
        cache_id = value
    if not isinstance(cache_id, str) or len(cache_id) != _SHA256_LENGTH:
        raise ValueError(f"expert_cache_identities[{name!r}] must be a SHA-256 identity")
    try:
        int(cache_id, 16)
    except ValueError as exc:
        raise ValueError(f"expert_cache_identities[{name!r}] is not hexadecimal") from exc
    return cache_id.lower()


def build_assembly_identity(
    *,
    expert_profile: str,
    expert_order: Sequence[str],
    expert_cache_identities: Mapping[str, Any],
    feature_dimensions: Mapping[str, int],
    source_manifest_identity: str,
) -> dict[str, Any]:
    """Identify one assembled expert profile independently from its expert caches."""
    order = list(expert_order) if isinstance(expert_order, Sequence) and not isinstance(expert_order, (str, bytes)) else None
    if not order or any(not isinstance(name, str) or not name for name in order):
        raise ValueError("expert_order must be a non-empty sequence of names")
    if len(order) != len(set(order)):
        raise ValueError("expert_order contains duplicate names")
    if not isinstance(expert_profile, str) or not expert_profile:
        raise ValueError("expert_profile must be a non-empty string")
    if not isinstance(expert_cache_identities, Mapping) or set(expert_cache_identities) != set(order):
        raise ValueError("expert cache identity keys must match expert_order exactly")
    if not isinstance(feature_dimensions, Mapping) or set(feature_dimensions) != set(order):
        raise ValueError("feature_dimensions keys must match expert_order exactly")
    dimensions = {}
    cache_ids = {}
    for name in order:
        width = feature_dimensions[name]
        if type(width) is not int or width <= 0:
            raise ValueError(f"feature_dimensions[{name!r}] must be a positive integer")
        dimensions[name] = width
        cache_ids[name] = _expert_cache_id(expert_cache_identities[name], name=name)
    if not isinstance(source_manifest_identity, str) or len(source_manifest_identity) != _SHA256_LENGTH:
        raise ValueError("source_manifest_identity must be a SHA-256 digest")
    try:
        int(source_manifest_identity, 16)
    except ValueError as exc:
        raise ValueError("source_manifest_identity must be hexadecimal") from exc
    body = {
        "schema_version": EXPERT_CACHE_SCHEMA_VERSION,
        "expert_profile": expert_profile,
        "expert_order": order,
        "expert_cache_identities": cache_ids,
        "feature_dimensions": dimensions,
        "source_manifest_identity": source_manifest_identity.lower(),
    }
    return {**body, "assembly_id": identity_sha256(body)}


def validate_assembly_identity(
    identity: Mapping[str, Any], *, expected_expert_order: Sequence[str] | None = None
) -> str:
    if not isinstance(identity, Mapping):
        raise TypeError("assembly identity must be a mapping")
    body_keys = {
        "schema_version", "expert_profile", "expert_order", "expert_cache_identities",
        "feature_dimensions", "source_manifest_identity",
    }
    if set(identity) != body_keys | {"assembly_id"} or identity.get("schema_version") != EXPERT_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete assembly identity schema")
    order = identity.get("expert_order")
    if expected_expert_order is not None and list(expected_expert_order) != order:
        raise ValueError("assembly expert order differs from the requested expert order")
    actual = identity_sha256({key: identity[key] for key in body_keys})
    if identity.get("assembly_id") != actual:
        raise ValueError("assembly identity SHA-256 mismatch")
    return actual


def build_sample_identity(
    split: str,
    records: Sequence[Mapping[str, Any]],
    preprocessing_input_identity: Mapping[str, Any],
    feature_dim: int,
) -> dict[str, Any]:
    """Bind a shard to ordered IDs, WAV bytes, exact captions, and output shapes."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid source split {split!r}")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("sample identity requires at least one manifest record")
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    ordered = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        sample_id = record.get("sample_id")
        filename = record.get("wave_file_name")
        caption = record.get("raw_caption")
        wav_hash = record.get("wav_sha256")
        if not isinstance(sample_id, str) or not sample_id.startswith(split + ":"):
            raise ValueError(f"records[{index}] has a mismatched sample ID for {split}")
        if (
            not isinstance(filename, str) or not filename or Path(filename).name != filename
            or "/" in filename or "\\" in filename or filename in {".", ".."}
        ):
            raise ValueError(f"records[{index}] has an invalid WAV filename")
        if sample_id != f"{split}:{filename}":
            raise ValueError(f"records[{index}] sample ID and WAV filename do not match")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"records[{index}] caption is missing or empty")
        if not isinstance(wav_hash, str) or len(wav_hash) != _SHA256_LENGTH:
            raise ValueError(f"records[{index}] WAV SHA-256 is invalid")
        try:
            int(wav_hash, 16)
        except ValueError as exc:
            raise ValueError(f"records[{index}] WAV SHA-256 is not hexadecimal") from exc
        if sample_id in seen_ids or filename in seen_names:
            raise ValueError("sample identity contains duplicate sample IDs or WAV filenames")
        seen_ids.add(sample_id)
        seen_names.add(filename)
        ordered.append({
            "sample_id": sample_id,
            "wave_file_name": filename,
            "wav_sha256": wav_hash.lower(),
            "caption": caption,
        })
    preprocess = _json_copy(preprocessing_input_identity, field="preprocessing_input_identity")
    body = {
        "schema_version": EXPERT_CACHE_SCHEMA_VERSION,
        "split": split,
        "ordered_samples": ordered,
        "preprocessing_input_identity": preprocess,
        "expected_sample_count": len(ordered),
        "expected_score_shape": [len(ordered), 1],
        "expected_feature_shape": [len(ordered), feature_dim],
    }
    return {**body, "sample_identity_sha256": identity_sha256(body)}


def _validate_sample_identity(sample_identity: Mapping[str, Any]) -> str:
    if not isinstance(sample_identity, Mapping):
        raise TypeError("sample_identity must be a mapping")
    expected = {
        "schema_version", "split", "ordered_samples", "preprocessing_input_identity",
        "expected_sample_count", "expected_score_shape", "expected_feature_shape",
        "sample_identity_sha256",
    }
    if set(sample_identity) != expected or sample_identity.get("schema_version") != EXPERT_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete per-shard sample identity")
    body = {key: sample_identity[key] for key in expected if key != "sample_identity_sha256"}
    actual = identity_sha256(body)
    if sample_identity.get("sample_identity_sha256") != actual:
        raise ValueError("per-shard sample identity SHA-256 mismatch")
    return actual


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _numeric_output(value: Any, *, name: str, shape: list[int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.float32) or list(value.shape) != shape:
        raise ValueError(f"{name} must be float32 with shape {shape}; got {value.dtype} {list(value.shape)}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def save_expert_shard(
    path: str | os.PathLike[str],
    *,
    expert_identity: Mapping[str, Any],
    sample_identity: Mapping[str, Any],
    scores: np.ndarray,
    features: np.ndarray,
    inference_metadata: Mapping[str, Any],
) -> Path:
    """Atomically write one verified float32 output shard and its provenance."""
    cache_id = validate_expert_cache_identity(expert_identity)
    sample_id = _validate_sample_identity(sample_identity)
    if not isinstance(inference_metadata, Mapping):
        raise TypeError("inference_metadata must be a mapping")
    if inference_metadata.get("factory_restoration") != "PASS_FROM_EXISTING_SOURCE_FACTORY":
        raise ValueError("shard requires successful source-factory restoration evidence")
    if inference_metadata.get("frozen_eval") is not True:
        raise ValueError("shard requires a frozen evaluation-mode model")
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError("expert shard path must end in .npz")
    metadata_path = target.with_suffix(".json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing or incomplete expert shard: {target}")
    score_array = _numeric_output(scores, name="scores", shape=sample_identity["expected_score_shape"])
    feature_array = _numeric_output(features, name="features", shape=sample_identity["expected_feature_shape"])
    extra_metadata = _json_copy(inference_metadata, field="inference_metadata")
    reserved_metadata = {
        "schema_version", "expert_name", "expert_cache_id", "sample_identity",
        "sample_identity_sha256", "sample_count", "score_shape", "feature_shape",
        "score_dtype", "feature_dtype", "file_sha256", "metadata_sha256",
    }
    if reserved_metadata.intersection(extra_metadata):
        raise ValueError("inference_metadata cannot override integrity or shape metadata")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, scores=score_array, features=feature_array)
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    metadata = {
        **extra_metadata,
        "schema_version": EXPERT_CACHE_SCHEMA_VERSION,
        "expert_name": expert_identity["expert_name"],
        "expert_cache_id": cache_id,
        "sample_identity": _json_copy(sample_identity, field="sample_identity"),
        "sample_identity_sha256": sample_id,
        "sample_count": sample_identity["expected_sample_count"],
        "score_shape": sample_identity["expected_score_shape"],
        "feature_shape": sample_identity["expected_feature_shape"],
        "score_dtype": "float32",
        "feature_dtype": "float32",
        "file_sha256": sha256_file(target),
    }
    metadata["metadata_sha256"] = identity_sha256(metadata)
    _atomic_json(metadata_path, metadata)
    return target


def load_expert_shard(
    path: str | os.PathLike[str],
    *,
    expert_identity: Mapping[str, Any],
    sample_identity: Mapping[str, Any],
    feature_dim: int,
) -> dict[str, np.ndarray]:
    """Validate provenance, file hash, shapes, dtypes and finite arrays before reuse."""
    target = Path(path)
    metadata_path = target.with_suffix(".json")
    if not target.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"expert cache shard or metadata is missing: {target}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"expert shard metadata is corrupt: {metadata_path}: {exc}") from exc
    metadata_digest = metadata.pop("metadata_sha256", None)
    if metadata_digest != identity_sha256(metadata):
        raise ValueError(f"expert shard metadata hash mismatch or corruption: {metadata_path}")
    cache_id = validate_expert_cache_identity(expert_identity)
    sample_id = _validate_sample_identity(sample_identity)
    if metadata.get("schema_version") != EXPERT_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported expert cache shard schema: {target}")
    if metadata.get("expert_name") != expert_identity["expert_name"] or metadata.get("expert_cache_id") != cache_id:
        raise ValueError(f"expert cache identity mismatch: {target}")
    if metadata.get("sample_identity_sha256") != sample_id or metadata.get("sample_identity") != sample_identity:
        raise ValueError(f"per-shard sample identity/order mismatch: {target}")
    if metadata.get("sample_count") != sample_identity["expected_sample_count"]:
        raise ValueError(f"per-shard sample count mismatch: {target}")
    if metadata.get("score_shape") != sample_identity["expected_score_shape"]:
        raise ValueError(f"per-shard score shape metadata mismatch: {target}")
    expected_feature_shape = [sample_identity["expected_sample_count"], feature_dim]
    if sample_identity["expected_feature_shape"] != expected_feature_shape or metadata.get("feature_shape") != expected_feature_shape:
        raise ValueError(f"per-shard feature shape metadata mismatch: {target}")
    if metadata.get("score_dtype") != "float32" or metadata.get("feature_dtype") != "float32":
        raise ValueError(f"per-shard output dtype metadata mismatch: {target}")
    if metadata.get("factory_restoration") != "PASS_FROM_EXISTING_SOURCE_FACTORY" or metadata.get("frozen_eval") is not True:
        raise ValueError(f"per-shard model restoration/eval evidence is invalid: {target}")
    if sha256_file(target) != metadata.get("file_sha256"):
        raise ValueError(f"expert shard file hash mismatch or corrupt shard: {target}")
    try:
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != {"scores", "features"}:
                raise ValueError(f"unexpected NPZ arrays in expert shard: {target}")
            result = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("unexpected NPZ"):
            raise
        raise ValueError(f"expert shard NPZ is corrupt: {target}: {exc}") from exc
    _numeric_output(result["scores"], name="scores", shape=sample_identity["expected_score_shape"])
    _numeric_output(result["features"], name="features", shape=expected_feature_shape)
    return result


def resolve_manifest_wav_paths(
    dataset: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    start: int,
) -> list[str]:
    """Resolve trusted UnifiedXACLEDataset rows and recheck path/sample/hash identity."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid requested split {split!r}")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative row offset")
    rows = getattr(dataset, "rows", None)
    audio_dir = getattr(dataset, "audio_dir", None)
    if not isinstance(rows, Sequence) or audio_dir is None:
        raise TypeError("dataset must expose trusted rows and its split audio_dir")
    root = Path(audio_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"expected split audio root is not a directory: {root}")
    paths: list[str] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for offset, record in enumerate(records):
        row_index = start + offset
        if row_index >= len(rows):
            raise ValueError("manifest shard range exceeds the trusted dataset rows")
        row = rows[row_index]
        if not isinstance(record, Mapping) or not isinstance(row, Mapping):
            raise TypeError("manifest record and dataset row must be mappings")
        sample_id = record.get("sample_id")
        filename = record.get("wave_file_name")
        if (
            not isinstance(filename, str) or not filename or Path(filename).name != filename
            or "/" in filename or "\\" in filename or filename in {".", ".."}
        ):
            raise ValueError(f"manifest WAV filename is invalid: {filename!r}")
        if sample_id != f"{split}:{filename}" or row.get("sample_id") != sample_id:
            raise ValueError("manifest sample ID, filename and dataset row identity differ")
        if row.get("split") != split or row.get("wave_file_name") != filename:
            raise ValueError("manifest split/filename differs from the trusted dataset row")
        if row.get("captions_raw") != record.get("raw_caption"):
            raise ValueError("manifest caption differs from the trusted dataset row")
        if sample_id in seen_ids:
            raise ValueError("duplicate sample identity in WAV path request")
        seen_ids.add(sample_id)
        try:
            path = Path(row["audio_path"]).expanduser().resolve(strict=True)
        except (KeyError, OSError, TypeError) as exc:
            raise FileNotFoundError(f"trusted dataset WAV is missing or unreadable for {sample_id}") from exc
        if not path.is_relative_to(root):
            raise ValueError(f"trusted dataset WAV path escapes the expected split root: {path}")
        if path.name != filename or not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f"trusted dataset WAV is missing or unreadable: {path}")
        if path in seen_paths:
            raise ValueError("duplicate WAV path in one MS-CLAP batch")
        seen_paths.add(path)
        observed_hash = sha256_file(path)
        if observed_hash != record.get("wav_sha256"):
            raise ValueError(f"WAV content SHA-256 mismatch for {sample_id}")
        paths.append(str(path))
    if not paths:
        raise ValueError("cannot resolve WAV paths for an empty shard")
    return paths


__all__ = [
    "EXPERT_CACHE_SCHEMA_VERSION",
    "build_assembly_identity",
    "build_expert_cache_identity",
    "build_sample_identity",
    "canonical_json",
    "identity_sha256",
    "load_expert_shard",
    "resolve_manifest_wav_paths",
    "save_expert_shard",
    "sha256_file",
    "validate_assembly_identity",
    "validate_expert_cache_identity",
]
