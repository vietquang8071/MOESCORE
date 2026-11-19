#!/usr/bin/env python3
"""Extract the frozen E8 CORAL expert into resumable, content-addressed shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_expert_cache import (
    build_sample_identity,
    identity_sha256,
    load_expert_shard,
    save_expert_shard,
    sha256_file,
)
from model.expert_adapters.registry import create_expert_adapter
from model.expert_adapters.expert8_m2d_clap_runtime import (
    CORAL_FEATURE_DIM,
    load_expert8_checkpoint,
    validate_base_m2d_checkpoint,
)
from scripts.extract_moe_feature_store import (
    EXPECTED_COUNTS,
    _atomic_json,
    compute_expert_cache_identity,
    projected_dataset,
)


def make_expert8_cache_manifest(
    *, split: str, records: Sequence[Mapping[str, Any]],
    cache_identity: Mapping[str, Any], shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    safe_records = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("E8 source records must be mappings")
        sample_id = row.get("sample_id")
        filename = row.get("wave_file_name")
        caption = row.get("raw_caption")
        wav_hash = row.get("wav_sha256")
        if (not isinstance(sample_id, str) or not sample_id.startswith(split + ":")
                or sample_id != f"{split}:{filename}"):
            raise ValueError("E8 source manifest has a split or sample ID mismatch")
        if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                or filename in seen_names or sample_id in seen_ids):
            raise ValueError("E8 source manifest has a duplicate or invalid WAV filename/sample ID")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("E8 source manifest has a missing raw caption")
        if (not isinstance(wav_hash, str) or len(wav_hash) != 64
                or any(c not in "0123456789abcdef" for c in wav_hash.lower())):
            raise ValueError("E8 source manifest has an invalid WAV SHA-256")
        seen_ids.add(sample_id)
        seen_names.add(filename)
        safe_records.append({
            "sample_id": sample_id, "wave_file_name": filename,
            "raw_caption": caption, "wav_sha256": wav_hash.lower(),
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        })
    if not safe_records:
        raise ValueError("E8 source manifest cannot be empty")
    cache_id = cache_identity.get("cache_id") if isinstance(cache_identity, Mapping) else None
    if not isinstance(cache_id, str) or len(cache_id) != 64:
        raise ValueError("E8 cache identity is invalid")
    body = {
        "schema_version": 1,
        "expert_name": "expert8",
        "split": split,
        "expert_cache_id": cache_id,
        "expert_cache_identity": dict(cache_identity),
        "sample_count": len(safe_records),
        "source_records": safe_records,
        "source_identity_sha256": identity_sha256([
            {key: row[key] for key in ("sample_id", "wave_file_name", "wav_sha256", "caption_sha256")}
            for row in safe_records
        ]),
        "shards": [dict(shard) for shard in shards],
    }
    return {**body, "manifest_sha256": identity_sha256(body)}


def validate_expert8_cache_manifest(
    manifest: Mapping[str, Any], *, split: str,
    expected_records: Sequence[Mapping[str, Any]], cache_identity: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("E8 cache manifest must be a mapping")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (manifest.get("manifest_sha256") != identity_sha256(body)
            or manifest.get("schema_version") != 1
            or manifest.get("expert_name") != "expert8"
            or manifest.get("split") != split
            or manifest.get("expert_cache_id") != cache_identity.get("cache_id")
            or manifest.get("expert_cache_identity") != dict(cache_identity)):
        raise ValueError("E8 cache manifest provenance or content hash is incompatible")
    expected = make_expert8_cache_manifest(
        split=split, records=expected_records, cache_identity=cache_identity,
        shards=manifest.get("shards", []),
    )
    for field in ("sample_count", "source_records", "source_identity_sha256"):
        if manifest.get(field) != expected[field]:
            raise ValueError(f"E8 cache source manifest identity/order mismatch: {field}")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("E8 cache shard index must be a list")
    previous_end = 0
    for index, shard in enumerate(shards):
        if (not isinstance(shard, Mapping) or shard.get("index") != index
                or shard.get("start") != previous_end or type(shard.get("count")) is not int
                or shard["count"] <= 0 or not isinstance(shard.get("sample_identity_sha256"), str)
                or len(shard["sample_identity_sha256"]) != 64):
            raise ValueError("E8 cache shard order or sample identity is invalid")
        previous_end += shard["count"]
    if shards and previous_end != len(expected_records):
        raise ValueError("E8 cache shards do not cover the exact source row count")


def _config(path: str | Path, *, device: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        body = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load Expert8 extraction configuration: {exc}") from exc
    if (not isinstance(body, dict) or not isinstance(body.get("runtime"), dict)
            or not isinstance(body.get("experts"), dict)
            or not isinstance(body["experts"].get("expert8"), dict)):
        raise ValueError("Expert8 config requires runtime and experts.expert8 mappings")
    section = body["experts"]["expert8"]
    for field in ("base_checkpoint", "finetuned_checkpoint"):
        if not isinstance(section.get(field), str) or not section[field]:
            raise ValueError(f"experts.expert8.{field} is required")
        section[field] = str((source.parent / section[field]).resolve()) if not Path(section[field]).is_absolute() else section[field]
    body["runtime"]["device"] = device
    offline = body["runtime"].get("offline", True)
    if type(offline) is not bool:
        raise ValueError("runtime.offline must be boolean")
    return body


def _source_rows(dataset_root: str | Path, split: str, work_root: Path, *, limit: int | None):
    dataset, _ = projected_dataset(dataset_root, split, work_root, limit=limit)
    if limit is None and len(dataset) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} rows, observed {len(dataset)}")
    rows = []
    for row in dataset.rows:
        path = Path(row["audio_path"]).resolve(strict=True)
        if not path.is_relative_to(Path(dataset.audio_dir).resolve(strict=True)):
            raise ValueError(f"Expert8 audio path escapes split directory: {path}")
        rows.append({
            "sample_id": row["sample_id"],
            "wave_file_name": row["wave_file_name"],
            "raw_caption": row["captions_raw"],
            "wav_sha256": sha256_file(path),
            "audio_path": str(path),
        })
    return dataset, rows


def _shard_sample_identity(split: str, records, cache_identity):
    identity_rows = [
        {key: row[key] for key in ("sample_id", "wave_file_name", "raw_caption", "wav_sha256")}
        for row in records
    ]
    return build_sample_identity(
        split, identity_rows,
        {"expert_cache_id": cache_identity["cache_id"],
         "input_contract": cache_identity["input_contract"],
         "preprocessing_provenance": cache_identity["preprocessing_provenance"]},
        CORAL_FEATURE_DIM,
    )


def extract_expert8_feature_cache(
    config_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *, device: str = "cuda:0", batch_size: int = 1, shard_size: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    """Preflight and extract train/validation/test outputs with atomic resumable shards."""
    if type(batch_size) is not int or batch_size <= 0 or type(shard_size) is not int or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive integers")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    config = _config(config_path, device=device)
    section = config["experts"]["expert8"]
    base_checkpoint = Path(section["base_checkpoint"]).expanduser().resolve(strict=True)
    lora_checkpoint = Path(section["finetuned_checkpoint"]).expanduser().resolve(strict=True)
    base_sha256 = validate_base_m2d_checkpoint(base_checkpoint)
    load_expert8_checkpoint(lora_checkpoint)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Expert8 requested {selected_device}, but CUDA is unavailable")
    cache_identity = compute_expert_cache_identity(
        "expert8", config, feature_dim=CORAL_FEATURE_DIM, device=str(selected_device)
    )
    if cache_identity["checkpoint_provenance"].get("base_checkpoint", {}).get("sha256") != base_sha256:
        raise ValueError("Expert8 base checkpoint changed after verified preflight")
    root = Path(output_root).expanduser().resolve()
    work_root = root / "source_projection"
    manifests = {}
    sources = {}
    cache_base = root / "expert8" / cache_identity["cache_id"]
    for split in ("train", "validation", "test"):
        dataset, records = _source_rows(dataset_root, split, work_root, limit=limit)
        sources[split] = (dataset, records)
        manifest_path = cache_base / split / "manifest.json"
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Expert8 cache manifest is corrupt: {manifest_path}") from exc
            validate_expert8_cache_manifest(
                existing, split=split, expected_records=records, cache_identity=cache_identity
            )
            manifests[split] = existing

    pending_splits = []
    for split, (dataset, records) in sources.items():
        existing = manifests.get(split)
        existing_shards = {item["index"]: item for item in existing["shards"]} if existing else {}
        missing = []
        for index, start in enumerate(range(0, len(records), shard_size)):
            selected = records[start:start + shard_size]
            sample_identity = _shard_sample_identity(split, selected, cache_identity)
            shard_dir = cache_base / split / "shards"
            shard_path = shard_dir / f"shard_{index:06d}_{sample_identity['sample_identity_sha256']}.npz"
            prior = existing_shards.get(index)
            if prior and (prior.get("sample_identity_sha256") != sample_identity["sample_identity_sha256"]
                          or prior.get("count") != len(selected)):
                raise ValueError(f"Expert8 cached shard {split}/{index} differs from source identity")
            if shard_path.exists() or shard_path.with_suffix(".json").exists():
                load_expert_shard(
                    shard_path, expert_identity=cache_identity, sample_identity=sample_identity,
                    feature_dim=CORAL_FEATURE_DIM,
                )
            else:
                missing.append((index, start, selected, sample_identity, shard_path))
        if missing:
            pending_splits.append((split, dataset, records, missing, existing))

    adapter = None
    if pending_splits:
        before = {key: (Path(section[key]).stat().st_size, Path(section[key]).stat().st_mtime_ns)
                  for key in ("base_checkpoint", "finetuned_checkpoint")}
        adapter = create_expert_adapter("expert8", config, device=selected_device)
        adapter.eval()
        if any(parameter.requires_grad for parameter in adapter.parameters()):
            raise RuntimeError("Expert8 adapter must remain frozen during extraction")
        for key, state in before.items():
            path = Path(section[key])
            if (path.stat().st_size, path.stat().st_mtime_ns) != state:
                raise ValueError(f"Expert8 checkpoint changed during model restoration: {path}")

    reports = {}
    for split, (dataset, records) in sources.items():
        existing = manifests.get(split)
        old_shards = {item["index"]: item for item in existing["shards"]} if existing else {}
        shard_entries = []
        for index, start in enumerate(range(0, len(records), shard_size)):
            selected = records[start:start + shard_size]
            sample_identity = _shard_sample_identity(split, selected, cache_identity)
            shard_dir = cache_base / split / "shards"
            shard_path = shard_dir / f"shard_{index:06d}_{sample_identity['sample_identity_sha256']}.npz"
            if shard_path.exists() or shard_path.with_suffix(".json").exists():
                arrays = load_expert_shard(
                    shard_path, expert_identity=cache_identity,
                    sample_identity=sample_identity, feature_dim=CORAL_FEATURE_DIM,
                )
            else:
                assert adapter is not None
                score_rows, feature_rows = [], []
                for batch_start in range(0, len(selected), batch_size):
                    batch_rows = selected[batch_start:batch_start + batch_size]
                    waveforms = torch.stack([
                        dataset._expert8_audio(Path(row["audio_path"])) for row in batch_rows
                    ], dim=0).to(selected_device)
                    captions = [row["raw_caption"] for row in batch_rows]
                    with torch.inference_mode():
                        scores, features = adapter({
                            "expert3_wavs": waveforms,
                            "captions_raw": captions,
                        })
                    score_rows.append(scores.float().cpu())
                    feature_rows.append(features.float().cpu())
                score_array = torch.cat(score_rows).numpy().astype(np.float32, copy=False)
                feature_array = torch.cat(feature_rows).numpy().astype(np.float32, copy=False)
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                save_expert_shard(
                    shard_path, expert_identity=cache_identity, sample_identity=sample_identity,
                    scores=score_array, features=feature_array,
                    inference_metadata={
                        "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY",
                        "frozen_eval": True,
                        "base_m2d_checkpoint_sha256": base_sha256,
                        "lora_checkpoint_sha256": cache_identity["checkpoint_provenance"]["finetuned_checkpoint"]["sha256"],
                        "sample_rate_hz": 16000,
                        "waveform_samples": 160000,
                    },
                )
                arrays = load_expert_shard(
                    shard_path, expert_identity=cache_identity,
                    sample_identity=sample_identity, feature_dim=CORAL_FEATURE_DIM,
                )
            prior = old_shards.get(index)
            entry = {
                "index": index, "start": start, "count": len(selected),
                "sample_identity_sha256": sample_identity["sample_identity_sha256"],
                "relative_path": str(shard_path.relative_to(cache_base / split)),
                "file_sha256": sha256_file(shard_path),
            }
            if prior and prior.get("file_sha256") != entry["file_sha256"]:
                raise ValueError(f"Regenerated Expert8 shard differs from its previous identity: {shard_path}")
            shard_entries.append(entry)
        manifest = make_expert8_cache_manifest(
            split=split, records=records, cache_identity=cache_identity, shards=shard_entries
        )
        manifest_path = cache_base / split / "manifest.json"
        if existing and existing != manifest:
            raise ValueError(f"Expert8 completed manifest changed; use a fresh cache root: {manifest_path}")
        _atomic_json(manifest_path, manifest)
        reports[split] = {
            "sample_count": len(records), "cache_manifest": str(manifest_path),
            "cache_manifest_sha256": manifest["manifest_sha256"],
            "validated_shards": len(shard_entries), "expert_cache_id": cache_identity["cache_id"],
        }
    return {
        "status": "PASS", "expert": "expert8", "expert_cache_identity": cache_identity,
        "base_m2d_checkpoint_sha256": base_sha256,
        "lora_checkpoint_sha256": cache_identity["checkpoint_provenance"]["finetuned_checkpoint"]["sha256"],
        "splits": reports,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    try:
        report = extract_expert8_feature_cache(
            args.config, args.dataset_root, args.output_root, device=args.device,
            batch_size=args.batch_size, shard_size=args.shard_size, limit=args.limit,
        )
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        target = Path(args.report_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
