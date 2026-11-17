"""Verified Set A cache writing and split-explicit cached datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.moe_expert_cache import validate_expert_cache_identity
from data.moe_feature_store import MoeFeatureStoreDataset
from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_DIMS,
    SET_A_COMPONENT_FEATURE_DIMS,
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_BLOCKS,
    SET_A_FEATURE_DIM,
    SET_A_FEATURE_REPRESENTATION,
)
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.registry import FULL9_EXPERT_ORDER, profile_contract


SET_A_CACHE_SCHEMA_VERSION = 1
_ARRAY_NAMES = ("scores.npy", "features.npy", "labels_raw.npy")


def _fingerprint(value: Any) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_manifest(path: Path, *, split: str, expected_sha256: str) -> list[dict[str, Any]]:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Set A source manifest is missing or unreadable: {path}: {exc}") from exc
    if not isinstance(source, Mapping) or source.get("schema_version") != 1:
        raise ValueError(f"unsupported Set A source manifest schema: {path}")
    if source.get("split") != split:
        raise ValueError("Set A source manifest split mismatch")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Set A source manifest must contain ordered records")
    body = {key: value for key, value in source.items() if key != "sha256"}
    observed_sha256 = _fingerprint(body)
    if source.get("sha256") != observed_sha256 or observed_sha256 != expected_sha256:
        raise ValueError("Set A source manifest identity/provenance mismatch")

    ids: list[str] = []
    names: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"source manifest record {index} must be a mapping")
        sample_id = record.get("sample_id")
        wave_file_name = record.get("wave_file_name")
        caption = record.get("raw_caption")
        wav_sha256 = record.get("wav_sha256")
        if not isinstance(sample_id, str) or not sample_id.startswith(split + ":"):
            raise ValueError(f"source manifest record {index} has an invalid sample ID")
        if (
            not isinstance(wave_file_name, str)
            or not wave_file_name
            or Path(wave_file_name).name != wave_file_name
            or "/" in wave_file_name
            or "\\" in wave_file_name
        ):
            raise ValueError(f"source manifest record {index} has an invalid WAV basename")
        if not isinstance(caption, str):
            raise ValueError(f"source manifest record {index} is missing the original caption")
        if not isinstance(wav_sha256, str) or len(wav_sha256) != 64:
            raise ValueError(f"source manifest record {index} is missing a WAV SHA-256")
        try:
            int(wav_sha256, 16)
        except ValueError as exc:
            raise ValueError(f"source manifest record {index} has an invalid WAV SHA-256") from exc
        ids.append(sample_id)
        names.append(wave_file_name)
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("Set A source manifest contains duplicate sample IDs or WAV identities")
    return [dict(record) for record in records]


def _validate_component_identities(
    cache_identities: Any,
    model_identities: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(cache_identities, Mapping) or set(cache_identities) != set(SET_A_COMPONENT_ORDER):
        raise ValueError("Set A component cache identity keys/order are incomplete")
    if not isinstance(model_identities, Mapping) or set(model_identities) != set(SET_A_COMPONENT_ORDER):
        raise ValueError("Set A component checkpoint identity keys/order are incomplete")
    normalized_cache: dict[str, dict[str, Any]] = {}
    normalized_models: dict[str, dict[str, Any]] = {}
    for name in SET_A_COMPONENT_ORDER:
        identity = cache_identities[name]
        if not isinstance(identity, Mapping):
            raise ValueError(f"Set A {name} cache identity is not inspectable")
        validate_expert_cache_identity(identity)
        if identity.get("expert_name") != name:
            raise ValueError(f"Set A {name} cache identity names a different expert")
        score_contract = identity.get("score_contract")
        if not isinstance(score_contract, Mapping):
            raise ValueError(f"Set A {name} score/cache representation provenance is missing")
        if (
            score_contract.get("feature_representation") != SET_A_FEATURE_REPRESENTATION
            or score_contract.get("feature_blocks") != list(SET_A_FEATURE_BLOCKS)
            or score_contract.get("feature_width_formula") != "4D+1"
            or score_contract.get("feature_shape") != [None, SET_A_COMPONENT_FEATURE_DIMS[name]]
        ):
            raise ValueError(f"Set A {name} feature representation/width provenance is not current 4D+1")
        model_identity = model_identities[name]
        if not isinstance(model_identity, Mapping) or not model_identity:
            raise ValueError(f"Set A {name} component checkpoint identity is missing")
        if dict(model_identity) != identity.get("checkpoint_provenance"):
            raise ValueError(f"Set A {name} checkpoint/cache provenance mismatch")
        if name == "top2_mga":
            input_contract = identity.get("input_contract")
            if (
                not isinstance(input_contract, Mapping)
                or input_contract.get("sample_rate_hz") != 16_000
                or input_contract.get("max_samples") != 160_000
            ):
                raise ValueError(
                    "Set A MGA cache must use the Top 2 source input contract: 16 kHz, 160000 samples; "
                    "the MOESCORE 32 kHz cache cannot be reused as source-equivalent"
                )
        normalized_cache[name] = dict(identity)
        normalized_models[name] = dict(model_identity)
    return normalized_cache, normalized_models


def _as_cpu_float32_tensor(value: Any, *, field: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.asarray(value))
    else:
        raise TypeError(f"{field} must be a torch.Tensor or NumPy array")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{field} must use float32; got {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{field} contains NaN or infinity")
    return tensor.contiguous()


def write_set_a_cache(
    root: str | os.PathLike[str],
    *,
    split: str,
    source_manifest_path: str | os.PathLike[str],
    component_outputs: Mapping[str, Mapping[str, Any]],
    component_sample_ids: Mapping[str, Sequence[str]],
    component_cache_identities: Mapping[str, Any],
    component_model_identities: Mapping[str, Any],
) -> Path:
    """Write aligned component outputs into a content-hashed E5 cache.

    Each component's sample ID sequence must exactly match the signed source
    manifest order. The helper intentionally accepts precomputed outputs only;
    callers can reuse M2D/MS-CLAP shards and supply the E5-specific 16 kHz MGA
    extraction without loading all three models together.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    normalized_cache, normalized_models = _validate_component_identities(
        component_cache_identities, component_model_identities
    )
    source_path = Path(source_manifest_path).expanduser().resolve(strict=True)
    raw_source = json.loads(source_path.read_text(encoding="utf-8"))
    records = _read_source_manifest(
        source_path,
        split=split,
        expected_sha256=raw_source.get("sha256", ""),
    )
    ordered_ids = [record["sample_id"] for record in records]
    ordered_names = [record["wave_file_name"] for record in records]
    if not isinstance(component_sample_ids, Mapping) or set(component_sample_ids) != set(SET_A_COMPONENT_ORDER):
        raise ValueError("component_sample_ids must have exactly the Set A component names")
    for name in SET_A_COMPONENT_ORDER:
        ids = component_sample_ids[name]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            raise TypeError(f"{name} sample IDs must be an ordered sequence")
        ids = list(ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} component cache contains duplicate sample IDs")
        if ids != ordered_ids:
            raise ValueError(f"{name} component sample IDs/order differ from the source manifest")

    if not isinstance(component_outputs, Mapping) or set(component_outputs) != set(SET_A_COMPONENT_ORDER):
        raise ValueError("component_outputs must have exactly the Set A component names")
    score_columns: list[torch.Tensor] = []
    feature_blocks: list[torch.Tensor] = []
    sample_count = len(ordered_ids)
    for name in SET_A_COMPONENT_ORDER:
        output = component_outputs[name]
        if not isinstance(output, Mapping) or set(output) != {"score", "features"}:
            raise TypeError(f"{name} output must contain exactly 'score' and 'features'")
        score = _as_cpu_float32_tensor(output["score"], field=f"{name}.score")
        feature = _as_cpu_float32_tensor(output["features"], field=f"{name}.features")
        width = SET_A_COMPONENT_FEATURE_DIMS[name]
        if tuple(score.shape) != (sample_count, 1):
            raise ValueError(f"{name}.score must have shape [{sample_count},1]")
        if tuple(feature.shape) != (sample_count, width):
            raise ValueError(f"{name}.features must have shape [{sample_count},{width}]")
        embedding_dim = SET_A_COMPONENT_DIMS[name]
        expected = build_paired_cross_features(
            feature[:, :embedding_dim], feature[:, embedding_dim : 2 * embedding_dim]
        )
        if not torch.allclose(feature, expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name}.features does not match the five-block 4D+1 representation")
        if not torch.allclose(score, expected[:, -1:], rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name}.score is inconsistent with its cosine feature scalar")
        score_columns.append(score)
        feature_blocks.append(feature)

    score_array = torch.cat(score_columns, dim=1).numpy()
    feature_array = torch.cat(feature_blocks, dim=1).numpy()
    label_values = [record.get("label_raw") for record in records]
    labels_present = all(value is not None for value in label_values)
    if any(value is None for value in label_values) and labels_present is False:
        if any(value is not None for value in label_values):
            raise ValueError(f"{split} source manifest has labels for only some rows")
    if split == "train" and not labels_present:
        raise ValueError("Set A train source manifest must include raw labels")
    if split == "test" and any(value is not None for value in label_values):
        raise ValueError("Set A test cache must not contain test labels")
    labels = None
    if labels_present:
        labels = np.asarray(label_values, dtype=np.float32)
        if not np.isfinite(labels).all() or np.any(labels < 0.0) or np.any(labels > 10.0):
            raise ValueError("Set A raw labels must be finite and on the [0,10] scale")

    root_path = Path(root).expanduser().resolve()
    target = root_path / split
    root_path.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Set A cache split already exists; refusing to overwrite: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f"{split}.", suffix=".tmp", dir=root_path))
    try:
        np.save(temporary / "scores.npy", score_array, allow_pickle=False)
        np.save(temporary / "features.npy", feature_array, allow_pickle=False)
        array_paths = [temporary / "scores.npy", temporary / "features.npy"]
        if labels is not None:
            np.save(temporary / "labels_raw.npy", labels, allow_pickle=False)
            array_paths.append(temporary / "labels_raw.npy")
        shutil.copyfile(source_path, temporary / "source_manifest.json")
        body = {
            "schema_version": SET_A_CACHE_SCHEMA_VERSION,
            "split": split,
            "component_order": list(SET_A_COMPONENT_ORDER),
            "component_feature_dims": dict(SET_A_COMPONENT_FEATURE_DIMS),
            "feature_dim": SET_A_FEATURE_DIM,
            "feature_representation": SET_A_FEATURE_REPRESENTATION,
            "feature_blocks": list(SET_A_FEATURE_BLOCKS),
            "sample_ids": ordered_ids,
            "wave_file_names": ordered_names,
            "source_manifest_sha256": raw_source["sha256"],
            "component_cache_identities": normalized_cache,
            "component_model_identities": normalized_models,
            "preprocessing_identity": {
                name: normalized_cache[name]["cache_id"] for name in SET_A_COMPONENT_ORDER
            },
            "score_shape": [sample_count, len(SET_A_COMPONENT_ORDER)],
            "feature_shape": [sample_count, SET_A_FEATURE_DIM],
            "score_dtype": "float32",
            "feature_dtype": "float32",
            "labels_present": labels is not None,
            "array_file_sha256": {path.name: _file_sha256(path) for path in array_paths},
        }
        manifest = {**body, "set_a_cache_identity_sha256": _fingerprint(body)}
        (temporary / "set_a_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "set_a_manifest.json"


def write_set_a_cache_from_full9(
    full9_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    split: str,
    mga_output: Mapping[str, Any],
    mga_sample_ids: Sequence[str],
    mga_cache_identity: Mapping[str, Any],
    mga_model_identity: Mapping[str, Any],
    source_manifest_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Reuse full9 M2D/MS-CLAP rows and overlay source-compatible MGA rows.

    The MGA row IDs must match the source store exactly and in order. The
    current full9 MGA shard is deliberately ignored because it uses the
    MOESCORE 32 kHz input contract.
    """
    source_store = MoeFeatureStoreDataset(
        str(full9_root), split=split, require_labels=(split == "train")
    )
    manifest = source_store.manifest
    if manifest.get("expert_profile") != "full_9":
        raise ValueError("Set A cache assembly requires a full_9 source store")
    order, dimensions, _ = profile_contract("full_9")
    if tuple(manifest.get("expert_order", ())) != tuple(order) or manifest.get("feature_dimensions") != dimensions:
        raise ValueError("Set A full9 source store has incompatible expert ordering or dimensions")
    sample_ids = list(source_store.sample_ids)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("full9 source store contains duplicate sample IDs")
    if isinstance(mga_sample_ids, (str, bytes)) or not isinstance(mga_sample_ids, Sequence):
        raise TypeError("mga_sample_ids must be an ordered sequence")
    mga_ids = list(mga_sample_ids)
    if len(mga_ids) != len(set(mga_ids)):
        raise ValueError("source-compatible MGA cache contains duplicate sample IDs")
    if mga_ids != sample_ids:
        raise ValueError("source-compatible MGA sample IDs/order differ from the full9 source store")

    source_path = (
        Path(source_manifest_path).expanduser().resolve(strict=True)
        if source_manifest_path is not None
        else Path(full9_root).expanduser().resolve() / split / "source_manifest.json"
    )
    records = _read_source_manifest(
        source_path,
        split=split,
        expected_sha256=manifest["source_manifest_sha256"],
    )
    source_ids = [record["sample_id"] for record in records]
    source_names = [record["wave_file_name"] for record in records]
    store_names = [name for shard in manifest["shards"] for name in shard["wave_file_names"]]
    if source_ids != sample_ids or source_names != store_names:
        raise ValueError("full9 source-store rows differ from the signed source manifest")

    source_cache_identities = manifest.get("per_expert_cache_identities")
    source_model_identities = manifest.get("checkpoint_provenance")
    if not isinstance(source_cache_identities, Mapping) or not isinstance(source_model_identities, Mapping):
        raise ValueError("full9 source store is missing per-expert provenance")
    component_cache_identities = {
        "top2_m2d": source_cache_identities.get("top2_m2d"),
        "top2_msclap": source_cache_identities.get("top2_msclap"),
        "top2_mga": mga_cache_identity,
    }
    component_model_identities = {
        "top2_m2d": source_model_identities.get("top2_m2d"),
        "top2_msclap": source_model_identities.get("top2_msclap"),
        "top2_mga": mga_model_identity,
    }
    _validate_component_identities(component_cache_identities, component_model_identities)

    if not isinstance(mga_output, Mapping) or set(mga_output) != {"score", "features"}:
        raise TypeError("mga_output must contain exactly 'score' and 'features'")
    mga_score = _as_cpu_float32_tensor(mga_output["score"], field="top2_mga.score")
    mga_features = _as_cpu_float32_tensor(mga_output["features"], field="top2_mga.features")
    if tuple(mga_score.shape) != (len(sample_ids), 1):
        raise ValueError(f"source-compatible MGA scores must have shape [{len(sample_ids)},1]")
    if tuple(mga_features.shape) != (len(sample_ids), SET_A_COMPONENT_FEATURE_DIMS["top2_mga"]):
        raise ValueError("source-compatible MGA feature width/shape mismatch")

    component_outputs: dict[str, dict[str, torch.Tensor]] = {
        "top2_mga": {"score": mga_score, "features": mga_features}
    }
    feature_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in order:
        end = offset + dimensions[name]
        feature_offsets[name] = (offset, end)
        offset = end
    score_indices = {name: order.index(name) for name in SET_A_COMPONENT_ORDER}
    for name in ("top2_m2d", "top2_msclap"):
        start, end = feature_offsets[name]
        score_rows = []
        feature_rows = []
        for row_index in range(len(source_store)):
            row = source_store[row_index]
            if row["sample_id"] != sample_ids[row_index]:
                raise ValueError("full9 source-store sample order changed during E5 cache assembly")
            score_rows.append(row["scores"][score_indices[name] : score_indices[name] + 1])
            feature_rows.append(row["features"][start:end])
        component_outputs[name] = {
            "score": torch.stack(score_rows, dim=0),
            "features": torch.stack(feature_rows, dim=0),
        }
    return write_set_a_cache(
        output_root,
        split=split,
        source_manifest_path=source_path,
        component_outputs=component_outputs,
        component_sample_ids={name: sample_ids for name in SET_A_COMPONENT_ORDER},
        component_cache_identities=component_cache_identities,
        component_model_identities=component_model_identities,
    )


class SetACachedDataset(Dataset):
    """Expose verified Set A outputs from a full9 store or a standalone E5 cache.

    A standalone E5 cache is required when an E5-specific component such as
    source-compatible 16 kHz MGA differs from the generic full9 extraction.
    Both formats bind rows to one content-hashed source manifest and current
    component/cache identities.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        require_labels: Optional[bool] = None,
        source_manifest_path: str | Path | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if split == "train" and require_labels is False:
            raise ValueError("Set A training split must require raw labels")
        if require_labels is None:
            require_labels = split == "train"
        self.root = Path(root).expanduser().resolve()
        self.split = split
        split_root = self.root / split
        if (split_root / "set_a_manifest.json").is_file():
            self.store = None
            self._init_standalone(split_root, require_labels, source_manifest_path)
        else:
            self._init_full9(require_labels, source_manifest_path)

    def _set_offsets(self) -> None:
        self._feature_offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in SET_A_COMPONENT_ORDER:
            end = offset + SET_A_COMPONENT_FEATURE_DIMS[name]
            self._feature_offsets[name] = (offset, end)
            offset = end

    def _init_component_identities(self, cache_identities: Any, model_identities: Any) -> None:
        self.component_cache_identities, self.component_model_identities = _validate_component_identities(
            cache_identities, model_identities
        )

    def _init_full9(self, require_labels: bool, source_manifest_path: str | Path | None) -> None:
        self.store = MoeFeatureStoreDataset(
            str(self.root), split=self.split, require_labels=require_labels
        )
        self.identity = self.store.identity
        self.manifest = self.store.manifest
        self.sample_ids = list(self.store.sample_ids)
        if self.manifest.get("expert_profile") != "full_9":
            raise ValueError("Set A requires a verified full_9 store or standalone E5 cache")
        order, dimensions, _ = profile_contract("full_9")
        if tuple(order) != tuple(FULL9_EXPERT_ORDER) or tuple(self.manifest.get("expert_order", ())) != tuple(order):
            raise ValueError("Set A full9 feature-store expert order mismatch")
        if self.manifest.get("feature_dimensions") != dimensions:
            raise ValueError("Set A full9 feature dimensions differ from the verified registry")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Set A feature store contains duplicate sample IDs")
        self._init_component_identities(
            {
                name: self.manifest["per_expert_cache_identities"][name]
                for name in SET_A_COMPONENT_ORDER
            }
            if isinstance(self.manifest.get("per_expert_cache_identities"), Mapping)
            else None,
            {
                name: self.manifest["checkpoint_provenance"][name]
                for name in SET_A_COMPONENT_ORDER
            }
            if isinstance(self.manifest.get("checkpoint_provenance"), Mapping)
            else None,
        )
        if self.split == "test" and self.manifest.get("labels_present") is True:
            raise ValueError("Set A test feature stores must be label-free")
        manifest_path = (
            Path(source_manifest_path).expanduser().resolve()
            if source_manifest_path is not None
            else self.root / self.split / "source_manifest.json"
        )
        self.source_records = _read_source_manifest(
            manifest_path,
            split=self.split,
            expected_sha256=self.manifest["source_manifest_sha256"],
        )
        indexed_ids = [sample_id for shard in self.manifest["shards"] for sample_id in shard["sample_ids"]]
        indexed_names = [name for shard in self.manifest["shards"] for name in shard["wave_file_names"]]
        source_ids = [record["sample_id"] for record in self.source_records]
        source_names = [record["wave_file_name"] for record in self.source_records]
        if indexed_ids != source_ids or self.sample_ids != source_ids:
            raise ValueError("Set A cache sample IDs/order differ from the source manifest")
        if indexed_names != source_names:
            raise ValueError("Set A cache WAV identities/order differ from the source manifest")
        self.source_manifest_identity = self.manifest["source_manifest_sha256"]
        self.preprocessing_identity = self.manifest["preprocessing_identity"]
        self._full9_order = order
        self._full9_dimensions = dimensions
        self._set_offsets()

    def _init_standalone(
        self,
        split_root: Path,
        require_labels: bool,
        source_manifest_path: str | Path | None,
    ) -> None:
        try:
            manifest = json.loads((split_root / "set_a_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Set A cache manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SET_A_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported standalone Set A cache schema")
        body = {key: value for key, value in manifest.items() if key != "set_a_cache_identity_sha256"}
        cache_identity = _fingerprint(body)
        if manifest.get("set_a_cache_identity_sha256") != cache_identity:
            raise ValueError("standalone Set A cache manifest identity mismatch")
        if manifest.get("split") != self.split:
            raise ValueError("standalone Set A cache split mismatch")
        if manifest.get("component_order") != list(SET_A_COMPONENT_ORDER):
            raise ValueError("standalone Set A component order mismatch")
        if manifest.get("component_feature_dims") != SET_A_COMPONENT_FEATURE_DIMS:
            raise ValueError("standalone Set A component feature width mismatch")
        if (
            manifest.get("feature_dim") != SET_A_FEATURE_DIM
            or manifest.get("feature_representation") != SET_A_FEATURE_REPRESENTATION
            or manifest.get("feature_blocks") != list(SET_A_FEATURE_BLOCKS)
        ):
            raise ValueError("standalone Set A feature width/representation mismatch")
        ids = manifest.get("sample_ids")
        names = manifest.get("wave_file_names")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value.startswith(self.split + ":") for value in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("standalone Set A cache sample IDs are missing, duplicate, or mixed-split")
        if (
            not isinstance(names, list)
            or len(names) != len(ids)
            or any(not isinstance(value, str) or not value or Path(value).name != value for value in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("standalone Set A cache WAV identity/order is invalid")
        if manifest.get("score_shape") != [len(ids), len(SET_A_COMPONENT_ORDER)]:
            raise ValueError("standalone Set A score width/shape mismatch")
        if manifest.get("feature_shape") != [len(ids), SET_A_FEATURE_DIM]:
            raise ValueError("standalone Set A feature width/shape mismatch")
        if manifest.get("score_dtype") != "float32" or manifest.get("feature_dtype") != "float32":
            raise ValueError("standalone Set A cache dtype mismatch")
        labels_present = manifest.get("labels_present")
        if type(labels_present) is not bool or (require_labels and not labels_present):
            raise ValueError("standalone Set A cache label provenance does not satisfy the requested split")
        if self.split == "test" and labels_present:
            raise ValueError("Set A test caches must be label-free")
        self._init_component_identities(
            manifest.get("component_cache_identities"),
            manifest.get("component_model_identities"),
        )
        if {
            name: self.component_cache_identities[name]["cache_id"] for name in SET_A_COMPONENT_ORDER
        } != {
            name: manifest["component_cache_identities"][name]["cache_id"]
            for name in SET_A_COMPONENT_ORDER
        }:
            raise ValueError("standalone Set A component cache provenance mismatch")
        source_hash = manifest.get("source_manifest_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("standalone Set A source manifest provenance mismatch")
        manifest_path = (
            Path(source_manifest_path).expanduser().resolve()
            if source_manifest_path is not None
            else split_root / "source_manifest.json"
        )
        self.source_records = _read_source_manifest(
            manifest_path,
            split=self.split,
            expected_sha256=manifest["source_manifest_sha256"],
        )
        source_ids = [record["sample_id"] for record in self.source_records]
        source_names = [record["wave_file_name"] for record in self.source_records]
        if ids != source_ids or names != source_names:
            raise ValueError("standalone Set A cache sample IDs/order differ from the source manifest")
        arrays = manifest.get("array_file_sha256")
        expected_arrays = {"scores.npy", "features.npy"} | ({"labels_raw.npy"} if labels_present else set())
        if not isinstance(arrays, Mapping) or set(arrays) != expected_arrays:
            raise ValueError("standalone Set A cache array provenance is incomplete")
        loaded_arrays = {}
        for filename, shape in (
            ("scores.npy", (len(ids), len(SET_A_COMPONENT_ORDER))),
            ("features.npy", (len(ids), SET_A_FEATURE_DIM)),
            *(([("labels_raw.npy", (len(ids),))] if labels_present else [])),
        ):
            path = split_root / filename
            if not path.is_file() or _file_sha256(path) != arrays.get(filename):
                raise ValueError(f"standalone Set A cache array hash mismatch: {filename}")
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise ValueError(f"standalone Set A cache array cannot be loaded: {filename}") from exc
            expected_dtype = np.dtype(np.float32)
            if value.shape != shape or value.dtype != expected_dtype:
                raise ValueError(f"standalone Set A cache array shape/dtype mismatch: {filename}")
            if not np.isfinite(value).all():
                raise ValueError(f"standalone Set A cache array contains non-finite values: {filename}")
            loaded_arrays[filename] = value
        if labels_present:
            labels = loaded_arrays["labels_raw.npy"]
            if np.any(labels < 0.0) or np.any(labels > 10.0):
                raise ValueError("standalone Set A raw labels must be in [0,10]")
            for index, record in enumerate(self.source_records):
                if record.get("label_raw") is None or float(labels[index]) != float(record["label_raw"]):
                    raise ValueError("standalone Set A labels differ from the source manifest")
        self.store = None
        self.identity = cache_identity
        self.manifest = dict(manifest)
        self.sample_ids = list(ids)
        self.source_manifest_identity = manifest["source_manifest_sha256"]
        self.preprocessing_identity = manifest["preprocessing_identity"]
        self._scores = loaded_arrays["scores.npy"]
        self._features = loaded_arrays["features.npy"]
        self._labels = loaded_arrays.get("labels_raw.npy")
        self._set_offsets()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "feature_store_identity": self.identity,
            "source_manifest_sha256": self.source_manifest_identity,
            "split": self.split,
            "sample_count": len(self),
            "ordered_sample_ids_sha256": _fingerprint(self.sample_ids),
            "preprocessing_identity": self.preprocessing_identity,
        }

    def __len__(self) -> int:
        if self.store is not None:
            return len(self.store)
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        source_record = self.source_records[index]
        if self.store is not None:
            sample = self.store[index]
            if sample["sample_id"] != source_record["sample_id"] or sample["wave_file_name"] != source_record["wave_file_name"]:
                raise ValueError("Set A row identity changed between cache and source manifest")
            scores, features = sample["scores"], sample["features"]
            component_scores = {
                name: scores[self._full9_order.index(name) : self._full9_order.index(name) + 1]
                for name in SET_A_COMPONENT_ORDER
            }
            component_features = {}
            for name in SET_A_COMPONENT_ORDER:
                start = sum(self._full9_dimensions[value] for value in self._full9_order[: self._full9_order.index(name)])
                end = start + self._full9_dimensions[name]
                component_features[name] = features[start:end]
            label_raw = sample.get("label_raw")
        else:
            scores = torch.from_numpy(np.array(self._scores[index], copy=True))
            features = torch.from_numpy(np.array(self._features[index], copy=True))
            component_scores = {
                name: scores[position : position + 1]
                for position, name in enumerate(SET_A_COMPONENT_ORDER)
            }
            component_features = {
                name: features[start:end]
                for name, (start, end) in self._feature_offsets.items()
            }
            label_raw = (
                torch.tensor(float(self._labels[index]), dtype=torch.float32)
                if self._labels is not None
                else None
            )
        components = {
            name: {"score": component_scores[name], "features": component_features[name]}
            for name in SET_A_COMPONENT_ORDER
        }
        result: dict[str, Any] = {
            "sample_id": source_record["sample_id"],
            "wave_file_name": source_record["wave_file_name"],
            "split": self.split,
            "component_outputs": components,
        }
        if self.split == "train":
            if label_raw is None:
                raise ValueError("Set A training cache row is missing its raw target")
            expected_label = source_record.get("label_raw")
            if expected_label is None or float(label_raw) != float(expected_label):
                raise ValueError("Set A training target differs from the source manifest")
            result["label_raw"] = label_raw
            result["target_normalized"] = (label_raw - 5.0) / 5.0
        return result


__all__ = [
    "SET_A_CACHE_SCHEMA_VERSION",
    "SetACachedDataset",
    "write_set_a_cache",
    "write_set_a_cache_from_full9",
]
