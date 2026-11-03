"""Verified, split-isolated storage for frozen-expert outputs.

Feature arrays are written as memory-mappable NPY files inside atomic shards;
the JSON index binds sample order, expert provenance, preprocessing identity,
and every shard's content hashes. No Python objects are unpickled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from model.expert_adapters.registry import (
    EXPECTED_FEATURE_WIDTH,
    EXPERT_FEATURE_DIMS,
    EXPERT_ORDER,
)


FEATURE_STORE_SCHEMA_VERSION = 1
_ARRAY_FILES = ("scores.npy", "features.npy", "labels_raw.npy", "num_class.npy")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_metadata(value: Any, field: str) -> Any:
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-compatible finite metadata") from exc
    return json.loads(encoded.decode("utf-8"))


def _ids(values: Sequence[str], *, split: str, field: str = "sample_ids") -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{field} must be a non-empty sequence of sample IDs")
    result = list(values)
    if any(not isinstance(item, str) or not item or item.strip() != item for item in result):
        raise ValueError(f"{field} must contain non-empty, trimmed strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate sample IDs")
    prefix = split + ":"
    if any(not item.startswith(prefix) for item in result):
        raise ValueError(f"sample IDs must belong to the requested split {split!r}")
    return result


def _filenames(values: Sequence[str], count: int) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("wave_file_names must be a sequence of basenames")
    result = list(values)
    if len(result) != count:
        raise ValueError("wave_file_names length must match sample_ids")
    if any(
        not isinstance(item, str)
        or not item
        or Path(item).name != item
        or "/" in item
        or "\\" in item
        for item in result
    ):
        raise ValueError("wave_file_names must contain safe file basenames")
    if len(result) != len(set(result)):
        raise ValueError("wave_file_names contains duplicates")
    return result


def _numeric_array(value: Any, field: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.from_numpy(np.empty((), dtype=dtype)).dtype:
            raise TypeError(f"{field} must use dtype {dtype}")
        if value.device.type != "cpu":
            raise ValueError(f"{field} must be stored on CPU")
        array = value.detach().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{field} must be a NumPy array or CPU tensor")
    if array.dtype != dtype:
        raise TypeError(f"{field} must use dtype {dtype}; got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return np.ascontiguousarray(array)


def _safe_shard_path(split_root: Path, relative: str) -> Path:
    candidate = (split_root / relative).resolve(strict=False)
    if not candidate.is_relative_to(split_root.resolve()):
        raise ValueError(f"feature-store shard path escapes its split directory: {relative!r}")
    return candidate


class MoeFeatureStoreWriter:
    """Write one logical split as idempotent, atomic numeric shards."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        split: str,
        expected_sample_ids: Sequence[str],
        source_manifest_sha256: str,
        checkpoint_provenance: Mapping[str, Any],
        preprocessing_identity: Any,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_root = self.root / split
        self.shards_root = self.split_root / "shards"
        self.expected_sample_ids = _ids(expected_sample_ids, split=split, field="expected_sample_ids")
        if (
            not isinstance(source_manifest_sha256, str)
            or len(source_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_manifest_sha256.lower())
        ):
            raise ValueError("source_manifest_sha256 must be a 64-character hexadecimal SHA-256")
        if not isinstance(checkpoint_provenance, Mapping) or tuple(checkpoint_provenance) != EXPERT_ORDER:
            raise ValueError(f"checkpoint_provenance keys must follow {EXPERT_ORDER}")
        self.checkpoint_provenance = _json_metadata(checkpoint_provenance, "checkpoint_provenance")
        if not all(isinstance(self.checkpoint_provenance[name], Mapping) for name in EXPERT_ORDER):
            raise ValueError("checkpoint provenance for each expert must be a non-empty mapping")
        if any(not self.checkpoint_provenance[name] for name in EXPERT_ORDER):
            raise ValueError("checkpoint provenance for each expert must be non-empty")
        self.preprocessing_identity = _json_metadata(preprocessing_identity, "preprocessing_identity")
        if not self.preprocessing_identity:
            raise ValueError("preprocessing_identity must be non-empty")
        self.source_manifest_sha256 = source_manifest_sha256.lower()
        existing_manifest_path = self.split_root / "manifest.json"
        if existing_manifest_path.is_file():
            existing_manifest = validate_feature_store(self.root, split=self.split)
            if (
                existing_manifest.get("source_manifest_sha256") != self.source_manifest_sha256
                or existing_manifest.get("checkpoint_provenance") != self.checkpoint_provenance
                or existing_manifest.get("preprocessing_identity") != self.preprocessing_identity
                or [sample_id for shard in existing_manifest["shards"] for sample_id in shard["sample_ids"]]
                != self.expected_sample_ids
            ):
                raise ValueError("existing feature-store identity/provenance differs; refusing resume")
        self.shards_root.mkdir(parents=True, exist_ok=True)

    def _existing_shard_ids(self) -> set[str]:
        seen: set[str] = set()
        for metadata_path in sorted(self.shards_root.glob("shard_*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ids = _ids(metadata.get("sample_ids", []), split=self.split)
            duplicates = seen.intersection(ids)
            if duplicates:
                raise ValueError(f"duplicate sample IDs across feature-store shards: {sorted(duplicates)}")
            seen.update(ids)
        return seen

    def write_shard(
        self,
        shard_index: int,
        *,
        sample_ids: Sequence[str],
        wave_file_names: Sequence[str],
        scores: Any,
        features: Any,
        labels_raw: Optional[Any] = None,
        num_class: Optional[Any] = None,
    ) -> Path:
        if type(shard_index) is not int or shard_index < 0:
            raise ValueError("shard_index must be a non-negative integer")
        selected_ids = _ids(sample_ids, split=self.split)
        count = len(selected_ids)
        names = _filenames(wave_file_names, count)
        if not set(selected_ids).issubset(set(self.expected_sample_ids)):
            raise ValueError("shard contains sample IDs outside the source manifest")
        score_array = _numeric_array(scores, "scores", np.dtype(np.float32), (count, len(EXPERT_ORDER)))
        feature_array = _numeric_array(features, "features", np.dtype(np.float32), (count, EXPECTED_FEATURE_WIDTH))
        label_array = None
        if labels_raw is None:
            if self.split == "train":
                raise ValueError("training feature-store shards require raw labels")
        else:
            label_array = _numeric_array(labels_raw, "labels_raw", np.dtype(np.float32), (count,))
            if np.any(label_array < 0.0) or np.any(label_array > 10.0):
                raise ValueError("labels_raw must remain on the original [0,10] scale")
        class_array = None
        if num_class is not None:
            class_array = _numeric_array(num_class, "num_class", np.dtype(np.int64), (count,))
            if np.any(class_array <= 0):
                raise ValueError("num_class values must be positive")

        target = self.shards_root / f"shard_{shard_index:06d}"
        arrays: dict[str, np.ndarray] = {"scores.npy": score_array, "features.npy": feature_array}
        if label_array is not None:
            arrays["labels_raw.npy"] = label_array
        if class_array is not None:
            arrays["num_class.npy"] = class_array
        content_hashes = {name: _sha256_bytes(array.tobytes(order="C")) for name, array in arrays.items()}

        if target.exists():
            metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            existing_hashes = metadata.get("array_content_sha256")
            if (
                metadata.get("sample_ids") != selected_ids
                or metadata.get("wave_file_names") != names
                or existing_hashes != content_hashes
            ):
                raise ValueError(f"existing shard {shard_index} differs; refusing non-idempotent resume")
            return target

        already_written = self._existing_shard_ids()
        duplicates = already_written.intersection(selected_ids)
        if duplicates:
            raise ValueError(f"sample IDs already exist in another shard: {sorted(duplicates)}")
        temp = Path(tempfile.mkdtemp(prefix=f"shard_{shard_index:06d}.", suffix=".tmp", dir=self.shards_root))
        try:
            for name, array in arrays.items():
                np.save(temp / name, array, allow_pickle=False)
            file_hashes = {name: _sha256_file(temp / name) for name in arrays}
            metadata = {
                "schema_version": FEATURE_STORE_SCHEMA_VERSION,
                "split": self.split,
                "shard_index": shard_index,
                "sample_ids": selected_ids,
                "wave_file_names": names,
                "array_files": sorted(arrays),
                "array_content_sha256": content_hashes,
                "file_sha256": file_hashes,
            }
            _atomic_json(temp / "metadata.json", metadata)
            os.replace(temp, target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return target

    def finalize(self) -> Path:
        shard_dirs = sorted(self.shards_root.glob("shard_[0-9][0-9][0-9][0-9][0-9][0-9]"))
        if not shard_dirs:
            raise ValueError("cannot finalize an empty feature store")
        entries: list[dict[str, Any]] = []
        ordered_ids: list[str] = []
        labels_present: Optional[bool] = None
        num_class_present: Optional[bool] = None
        for expected_index, shard_dir in enumerate(shard_dirs):
            metadata = json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("shard_index") != expected_index or metadata.get("split") != self.split:
                raise ValueError("shard indices must be contiguous and split identity must match")
            ids = _ids(metadata.get("sample_ids", []), split=self.split)
            ordered_ids.extend(ids)
            has_labels = "labels_raw.npy" in metadata.get("array_files", [])
            has_classes = "num_class.npy" in metadata.get("array_files", [])
            if labels_present is None:
                labels_present = has_labels
            elif labels_present != has_labels:
                raise ValueError("all shards in a split must either include labels or omit them")
            if num_class_present is None:
                num_class_present = has_classes
            elif num_class_present != has_classes:
                raise ValueError("all shards in a split must consistently include or omit num_class")
            entries.append({
                "index": expected_index,
                "directory": f"shards/{shard_dir.name}",
                "sample_ids": ids,
                "wave_file_names": metadata["wave_file_names"],
                "count": len(ids),
                "array_files": metadata["array_files"],
                "array_content_sha256": metadata["array_content_sha256"],
                "file_sha256": metadata["file_sha256"],
            })
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("feature-store shards contain duplicate sample IDs")
        if ordered_ids != self.expected_sample_ids:
            raise ValueError("finalized feature-store identity/order differs from the expected manifest")
        if self.split == "train" and not labels_present:
            raise ValueError("training feature store must include raw labels")
        body = {
            "schema_version": FEATURE_STORE_SCHEMA_VERSION,
            "split": self.split,
            "expert_order": list(EXPERT_ORDER),
            "feature_dimensions": dict(EXPERT_FEATURE_DIMS),
            "feature_width": EXPECTED_FEATURE_WIDTH,
            "score_dtype": "float32",
            "feature_dtype": "float32",
            "label_dtype": "float32",
            "source_manifest_sha256": self.source_manifest_sha256,
            "checkpoint_provenance": self.checkpoint_provenance,
            "preprocessing_identity": self.preprocessing_identity,
            "labels_present": bool(labels_present),
            "num_class_present": bool(num_class_present),
            "shards": entries,
        }
        body["store_identity_sha256"] = _sha256_bytes(_canonical_json(body))
        manifest_path = self.split_root / "manifest.json"
        _atomic_json(manifest_path, body)
        validate_feature_store(self.root, split=self.split, require_labels=self.split == "train")
        return manifest_path


def validate_feature_store(
    root: str | os.PathLike[str],
    *,
    split: str,
    require_labels: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    split_root = (Path(root).expanduser().resolve() / split).resolve()
    manifest_path = split_root / "manifest.json"
    if not manifest_path.is_file():
        available = [
            candidate.name
            for name in ("train", "validation", "test")
            if (candidate := Path(root).expanduser().resolve() / name / "manifest.json").is_file()
        ]
        if available:
            raise ValueError(
                f"feature-store split mismatch: requested {split!r}; available manifests: {available}"
            )
        raise FileNotFoundError(f"feature-store manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"feature-store manifest is unreadable: {exc}") from exc
    if manifest.get("schema_version") != FEATURE_STORE_SCHEMA_VERSION:
        raise ValueError("unsupported feature-store schema version")
    if manifest.get("split") != split:
        raise ValueError("feature-store split does not match the requested split")
    if tuple(manifest.get("expert_order", ())) != EXPERT_ORDER:
        raise ValueError("feature-store expert order mismatch")
    if manifest.get("feature_dimensions") != dict(EXPERT_FEATURE_DIMS) or manifest.get("feature_width") != EXPECTED_FEATURE_WIDTH:
        raise ValueError("feature-store feature dimensions do not match the verified registry")
    provenance = manifest.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping) or tuple(provenance) != EXPERT_ORDER or any(
        not isinstance(provenance[name], Mapping) or not provenance[name] for name in EXPERT_ORDER
    ):
        raise ValueError("feature-store expert checkpoint provenance is incomplete")
    if not manifest.get("preprocessing_identity"):
        raise ValueError("feature-store preprocessing identity is missing")
    source_manifest_hash = manifest.get("source_manifest_sha256")
    if not isinstance(source_manifest_hash, str) or len(source_manifest_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_manifest_hash.lower()
    ):
        raise ValueError("feature-store source manifest SHA-256 is invalid")
    if type(manifest.get("labels_present")) is not bool or type(manifest.get("num_class_present")) is not bool:
        raise ValueError("feature-store label metadata flags must be boolean")
    body = {key: value for key, value in manifest.items() if key != "store_identity_sha256"}
    if manifest.get("store_identity_sha256") != _sha256_bytes(_canonical_json(body)):
        raise ValueError("feature-store manifest identity hash mismatch")
    if require_labels and manifest.get("labels_present") is not True:
        raise ValueError(f"{split} feature store is missing required labels")
    if split == "train" and manifest.get("labels_present") is not True:
        raise ValueError("training feature store must include raw labels")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("feature store must contain at least one shard")
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for expected_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("index") != expected_index:
            raise ValueError("feature-store shard entries must be ordered and contiguous")
        directory = _safe_shard_path(split_root, shard.get("directory", ""))
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"feature-store shard metadata is missing: {metadata_path}")
        try:
            shard_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"feature-store shard metadata is corrupt: {exc}") from exc
        if shard_meta.get("split") != split or shard_meta.get("shard_index") != expected_index:
            raise ValueError("feature-store shard provenance mismatch")
        ids = _ids(shard_meta.get("sample_ids", []), split=split)
        names = _filenames(shard_meta.get("wave_file_names", []), len(ids))
        if ids != shard.get("sample_ids") or names != shard.get("wave_file_names") or len(ids) != shard.get("count"):
            raise ValueError("feature-store shard index identity mismatch")
        if seen_ids.intersection(ids) or seen_files.intersection(names):
            raise ValueError("duplicate sample IDs or filenames occur in feature store")
        seen_ids.update(ids)
        seen_files.update(names)
        expected_arrays = {"scores.npy", "features.npy"}
        if manifest.get("labels_present"):
            expected_arrays.add("labels_raw.npy")
        if "num_class.npy" in shard_meta.get("array_files", []):
            expected_arrays.add("num_class.npy")
        if manifest.get("num_class_present") is not ("num_class.npy" in shard_meta.get("array_files", [])):
            raise ValueError("feature-store num_class presence differs between manifest and shard")
        if set(shard_meta.get("array_files", [])) != expected_arrays:
            raise ValueError("feature-store shard numeric fields are incomplete or unexpected")
        if shard_meta.get("file_sha256") != shard.get("file_sha256"):
            raise ValueError("feature-store shard file-hash index mismatch")
        if shard_meta.get("array_content_sha256") != shard.get("array_content_sha256"):
            raise ValueError("feature-store shard content-hash index mismatch")
        arrays: dict[str, np.ndarray] = {}
        for filename in sorted(expected_arrays):
            path = directory / filename
            if not path.is_file() or _sha256_file(path) != shard_meta["file_sha256"].get(filename):
                raise ValueError(f"feature-store shard file hash mismatch or missing file: {path}")
            try:
                arrays[filename] = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise ValueError(f"feature-store shard array is corrupt: {path}: {exc}") from exc
            count = len(ids)
            expected_shape = (count, 4) if filename == "scores.npy" else (
                (count, EXPECTED_FEATURE_WIDTH) if filename == "features.npy" else (count,)
            )
            expected_dtype = np.dtype(np.int64) if filename == "num_class.npy" else np.dtype(np.float32)
            array = arrays[filename]
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(f"feature-store array contract mismatch for {path}: {array.shape} {array.dtype}")
            for start in range(0, count, 128):
                if not np.isfinite(array[start : start + 128]).all():
                    raise ValueError(f"feature-store array contains non-finite values: {path}")
            if filename == "labels_raw.npy" and (np.any(array < 0.0) or np.any(array > 10.0)):
                raise ValueError("raw labels must remain in [0,10]")
            if filename == "num_class.npy" and np.any(array <= 0):
                raise ValueError("num_class values must be positive")
        if manifest.get("labels_present") and "labels_raw.npy" not in arrays:
            raise ValueError("store manifest declares labels but a shard has no labels")
    if not seen_ids:
        raise ValueError("feature store has no sample identities")
    return manifest


class MoeFeatureStoreDataset(Dataset):
    """Lazily read one verified split; cache a small number of shards."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        split: str,
        require_labels: Optional[bool] = None,
        cache_shards: int = 2,
    ) -> None:
        from collections import OrderedDict

        if type(cache_shards) is not int or cache_shards <= 0:
            raise ValueError("cache_shards must be a positive integer")
        if require_labels is None:
            require_labels = split == "train"
        if type(require_labels) is not bool:
            raise TypeError("require_labels must be bool or None")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.manifest = validate_feature_store(self.root, split=split, require_labels=require_labels)
        self.identity = self.manifest["store_identity_sha256"]
        self.cache_shards = cache_shards
        self._cache = OrderedDict()
        self._refs: list[tuple[int, int]] = []
        for shard_index, shard in enumerate(self.manifest["shards"]):
            self._refs.extend((shard_index, row) for row in range(shard["count"]))
        self.sample_ids = [sample_id for shard in self.manifest["shards"] for sample_id in shard["sample_ids"]]

    def __len__(self) -> int:
        return len(self._refs)

    def _shard_arrays(self, index: int) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        from collections import OrderedDict

        entry = self.manifest["shards"][index]
        directory = _safe_shard_path(self.root / self.split, entry["directory"])
        arrays = {
            filename: np.load(directory / filename, mmap_mode="r", allow_pickle=False)
            for filename in entry["array_files"]
        }
        value = (arrays, entry["sample_ids"], entry["wave_file_names"])
        self._cache[index] = value
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        shard_index, row = self._refs[index]
        arrays, sample_ids, filenames = self._shard_arrays(shard_index)
        item: dict[str, Any] = {
            "sample_id": sample_ids[row],
            "wave_file_name": filenames[row],
            "split": self.split,
            "scores": torch.from_numpy(np.array(arrays["scores.npy"][row], copy=True)),
            "features": torch.from_numpy(np.array(arrays["features.npy"][row], copy=True)),
        }
        if "labels_raw.npy" in arrays:
            item["label_raw"] = torch.tensor(float(arrays["labels_raw.npy"][row]), dtype=torch.float32)
        if "num_class.npy" in arrays:
            item["num_class"] = torch.tensor(int(arrays["num_class.npy"][row]), dtype=torch.int64)
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state


__all__ = [
    "FEATURE_STORE_SCHEMA_VERSION",
    "MoeFeatureStoreDataset",
    "MoeFeatureStoreWriter",
    "validate_feature_store",
]
