#!/usr/bin/env python3
"""Bounded, resumable FP32 extraction using the verified expert factories.

The parent hashes inputs once. Each worker owns one expert for all splits and
exits before the next expert starts. Intermediate NPZ files are bounded
verification shards; final stores use MoeFeatureStoreWriter unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.moe_config import load_moe_config, resolve_moe_paths
from data.unified_dataset import UnifiedXACLEDataset
from data.unified_collate import collate_unified
from data.moe_feature_store import MoeFeatureStoreWriter, validate_feature_store, _atomic_json, _sha256_file
from model.expert_adapters.registry import (
    EXPERT_ORDER, EXPERT_FEATURE_DIMS, EXPERT_INPUT_CONTRACTS,
    create_expert_adapter, validate_and_assemble_expert_outputs,
)

EXPECTED_COUNTS = {"train": 7500, "validation": 3000, "test": 3000}
CHECKPOINT_FIELDS = {
    "expert1": ("pretrained_checkpoint", "finetuned_checkpoint"),
    "expert2": ("pretrained_checkpoint", "finetuned_checkpoint"),
    "expert3": ("base_checkpoint", "finetuned_checkpoint"),
    "expert4": ("beats_checkpoint", "clap_checkpoint", "finetuned_checkpoint"),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def job_identity(job):
    # Changing inference batch size after OOM does not change sample/source
    # identity or shard boundaries; completed FP32 arrays remain immutable.
    return fingerprint({key: value for key, value in job.items() if key != "batch_sizes"})


def bind_json(path, value):
    """Reject changes to an interrupted extraction's identity."""
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Provenance/manifest differs; use a new output directory: {path}")
    _atomic_json(path, value)


def checkpoint_provenance(config):
    cache = {}
    result = {}
    for expert in EXPERT_ORDER:
        result[expert] = {}
        for field in CHECKPOINT_FIELDS[expert]:
            path = Path(config["experts"][expert][field])
            if not path.is_file():
                raise FileNotFoundError(f"Missing approved checkpoint: {path}")
            if path not in cache:
                cache[path] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            result[expert][field] = cache[path]
    return result


def projected_dataset(dataset_root, split, work_root, *, limit=None):
    """Exclude test label columns before constructing UnifiedXACLEDataset.

    average_score values are never accessed for test. The label-free metadata
    file also prevents Unified Dataset's optional-label parser from seeing them.
    """
    root, work = Path(dataset_root), Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    source = root / "meta_data" / f"{split}_average.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        required = {"wav_file_name", "text"} | ({"average_score"} if split != "test" else set())
        if not required.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid average-score headers in {source}: {headers}")
        # Model metadata is deliberately omitted from model inputs.
        selected = ["wav_file_name", "text"] + (["average_score"] if split != "test" else [])
        rows = [{key: row[key] for key in selected} for row in reader]
    if limit is not None:
        rows = rows[:limit]
    path = work / f"{split}_inputs.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected)
        writer.writeheader()
        writer.writerows(rows)
    dataset = UnifiedXACLEDataset(path, root / "wav" / split, split=split,
                                  require_labels=split != "test")
    return dataset, headers


def prepare_job(config_path, dataset_root, output_root, *, batch_sizes=None, shard_size=128, limit=None):
    if shard_size <= 0 or (limit is not None and not 1 <= limit <= 4):
        raise ValueError("shard_size must be positive; smoke limit must be 1..4")
    config = resolve_moe_paths(load_moe_config(config_path), config_path)
    output = Path(output_root).resolve()
    work = output / "_extraction"
    batch_sizes = batch_sizes or dict.fromkeys(EXPERT_ORDER, 1)
    if set(batch_sizes) != set(EXPERT_ORDER) or any(type(n) is not int or n <= 0 for n in batch_sizes.values()):
        raise ValueError("Provide one positive extraction batch size per expert")
    # Hash preprocessing and Python/config sources; paths are relative so a
    # verified job/store can move between Kaggle sessions without new identity.
    source_files = {Path(__file__).resolve()}
    for directory in ("data", "model/expert_adapters", "expert1_finetune", "expert2_finetune",
                      "expert3_finetune/XACLE_2027_baselinemodel", "expert4_finetune"):
        source_files.update(p for p in (PROJECT_ROOT / directory).rglob("*")
                            if p.suffix in {".py", ".yaml", ".json"} and ".git" not in p.parts)
    source_identity = fingerprint({str(p.relative_to(PROJECT_ROOT)): _sha256_file(p)
                                   for p in sorted(source_files)})
    provenance = checkpoint_provenance(config)
    manifests = {}
    for split in EXPECTED_COUNTS:
        dataset, headers = projected_dataset(dataset_root, split, work, limit=limit)
        if limit is None and len(dataset) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} rows, observed {len(dataset)}")
        records = []
        for row in tqdm(dataset.rows, desc=f"Identity {split}"):
            record = {"sample_id": row["sample_id"], "wave_file_name": row["wave_file_name"],
                      "raw_caption": row["captions_raw"], "wav_sha256": _sha256_file(row["audio_path"])}
            if split != "test":
                record["label_raw"] = row["label_value"]
            records.append(record)
        manifest = {"schema_version": 1, "split": split, "records": records}
        manifest["sha256"] = fingerprint(manifest)
        bind_json(work / f"{split}_manifest.json", manifest)
        manifests[split] = manifest
        print(f"{split}: {len(records)} pairs; columns={headers}; labels_consumed={split != 'test'}", flush=True)
    job = {"schema_version": 1, "checkpoint_provenance": provenance,
           "preprocessing_identity": {"contract": "UnifiedXACLEDataset-LOCAL-09.1-eval",
                                      "source_sha256": source_identity, "expert_order": list(EXPERT_ORDER)},
           "manifests": {key: val["sha256"] for key, val in manifests.items()},
           "shard_size": shard_size, "batch_sizes": batch_sizes, "smoke": limit is not None}
    job_path = work / "job.json"
    if job_path.exists() and job_identity(json.loads(job_path.read_text())) != job_identity(job):
        raise ValueError("Extraction manifest/checkpoint/source provenance changed; use a new store root")
    _atomic_json(job_path, job)
    return config, job, manifests


def load_chunk(path, *, expert, records, identity):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    ids = [r["sample_id"] for r in records]
    if meta.get("identity") != identity or meta.get("expert") != expert or meta.get("sample_ids") != ids:
        raise ValueError(f"Extraction shard identity/order differs: {path}")
    if _sha256_file(path) != meta.get("sha256"):
        raise ValueError(f"Corrupt extraction shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"scores", "features"}:
            raise ValueError(f"Unexpected shard arrays: {path}")
        arrays = {name: data[name].copy() for name in data.files}
    for name, width in (("scores", 1), ("features", EXPERT_FEATURE_DIMS[expert])):
        array = arrays[name]
        if array.shape != (len(records), width) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid FP32 extraction tensor {name}: {path}")
    return arrays


def save_chunk(path, *, expert, records, identity, scores, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Validate before committing; never silently cast model output precision.
    for array, width in ((scores, 1), (features, EXPERT_FEATURE_DIMS[expert])):
        if array.dtype != np.float32 or array.shape != (len(records), width) or not np.isfinite(array).all():
            raise ValueError("Expert output must be finite FP32 with the registered width")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, scores=scores, features=features)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    _atomic_json(path.with_suffix(".json"), {"expert": expert, "identity": identity,
                 "sample_ids": [r["sample_id"] for r in records], "sha256": _sha256_file(path)})


def chunk_path(root, expert, split, index):
    return Path(root) / "_extraction" / expert / split / f"shard_{index:06d}.npz"


def validate_manifests(job, manifests):
    if set(manifests) != set(EXPECTED_COUNTS):
        raise ValueError("Missing or extra extraction split manifests")
    for split, manifest in manifests.items():
        body = {key: value for key, value in manifest.items() if key != "sha256"}
        if (manifest.get("split") != split or fingerprint(body) != manifest.get("sha256")
                or manifest["sha256"] != job["manifests"][split]):
            raise ValueError(f"Extraction source manifest identity changed: {split}")
        ids = [r["sample_id"] for r in manifest["records"]]
        if len(ids) != len(set(ids)) or any(not sid.startswith(split + ":") for sid in ids):
            raise ValueError("Duplicate or mixed-split extraction sample identities")


def extract_worker(config, job, manifests, dataset_root, output_root, expert, device):
    from scripts.verify_real_expert_to_gate import memory_snapshot, _validate_real_module_origins
    validate_manifests(job, manifests)
    identity = job_identity(job)
    report_path = Path(output_root) / "_extraction" / f"{expert}_runtime.json"
    pending = []
    for split, manifest in manifests.items():
        for index, start in enumerate(range(0, len(manifest["records"]), job["shard_size"])):
            records = manifest["records"][start:start + job["shard_size"]]
            path = chunk_path(output_root, expert, split, index)
            if path.with_suffix(".json").exists():
                load_chunk(path, expert=expert, records=records, identity=identity)
            else:
                pending.append((split, index, start, records, path))
    if not pending:
        report = json.loads(report_path.read_text())
        if report.get("job_identity") != identity or not report.get("frozen_eval") or report.get("factory_restoration") != "PASS_FROM_EXISTING_SOURCE_FACTORY":
            raise ValueError("Completed output shards lack matching source factory restoration evidence")
        report.update(status="PASS", reused_validated_shards=True)
        _atomic_json(report_path, report)
        return report
    # Detect changed checkpoint contents before any model is constructed.
    stats = {}
    for field, expected in job["checkpoint_provenance"][expert].items():
        path = Path(config["experts"][expert][field])
        if _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint changed since preflight: {path}")
        stats[field] = (path.stat().st_size, path.stat().st_mtime_ns)
    measurements = [memory_snapshot("before_factory", device)]
    adapter = create_expert_adapter(expert, config, device=device)
    for field, before in stats.items():
        path = Path(config["experts"][expert][field])
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Checkpoint file changed during source factory restoration")
    if any(p.requires_grad for p in adapter.parameters()) or any(m.training for m in adapter.modules()):
        raise ValueError("Source factory must return a fully frozen/eval expert")
    origins = _validate_real_module_origins(expert, config)
    measurements.append(memory_snapshot("after_factory_and_restore", device))
    report = {"status": "IN_PROGRESS", "expert": expert, "pid": os.getpid(),
              "factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY", "frozen_eval": True,
              "checkpoint_provenance": job["checkpoint_provenance"][expert],
              "batch_size": job["batch_sizes"][expert], "dtype": "float32", "device": str(device),
              "sample_count_by_split": {s: len(m["records"]) for s, m in manifests.items()},
              "model_class": type(getattr(adapter, "original_model", getattr(adapter, "model", adapter))).__name__,
              "module_origins": origins, "memory": measurements, "job_identity": identity}
    _atomic_json(report_path, report)
    datasets = {}
    for split in manifests:
        dataset, _ = projected_dataset(dataset_root, split, Path(output_root) / "_extraction",
                                       limit=len(manifests[split]["records"]) if job["smoke"] else None)
        # Eval-style center crop for E1; row IDs keep their original split.
        dataset.split = "validation"
        datasets[split] = dataset
    first = True
    for split, index, start, records, path in tqdm(pending, desc=expert):
        dataset = datasets[split]
        scores, features = [], []
        for offset in range(0, len(records), job["batch_sizes"][expert]):
            selected = records[offset:offset + job["batch_sizes"][expert]]
            for i, record in enumerate(selected):
                if _sha256_file(dataset.rows[start + offset + i]["audio_path"]) != record["wav_sha256"]:
                    raise ValueError("Source WAV changed after manifest selection")
            samples = [dataset[start + offset + i] for i in range(len(selected))]
            batch = collate_unified(samples)
            if batch["sample_ids"] != [r["sample_id"] for r in selected] or batch["captions_raw"] != [r["raw_caption"] for r in selected]:
                raise ValueError("Unified input identity/caption sequence differs from manifest")
            contract = EXPERT_INPUT_CONTRACTS[expert]
            inputs = {contract.audio_key: batch[contract.audio_key].to(device), "captions_raw": batch["captions_raw"]}
            if contract.lengths_key:
                inputs[contract.lengths_key] = batch[contract.lengths_key].to(device)
            with torch.inference_mode():
                score, feature = adapter(inputs)
            if score.dtype != torch.float32 or feature.dtype != torch.float32:
                raise ValueError("Expert returned a non-FP32 output")
            scores.append(score.detach().cpu().numpy().copy())
            features.append(feature.detach().cpu().numpy().copy())
            if first:
                measurements.append(memory_snapshot("after_first_inference", device))
                first = False
            del inputs, batch, samples, score, feature
        save_chunk(path, expert=expert, records=records, identity=identity,
                   scores=np.concatenate(scores), features=np.concatenate(features))
    measurements.append(memory_snapshot("after_cpu_transfer", device))
    report["status"] = "PASS"
    _atomic_json(report_path, report)
    return report


def assemble_stores(job, manifests, output_root):
    validate_manifests(job, manifests)
    reports = {}
    for split, manifest in manifests.items():
        records = manifest["records"]
        writer = MoeFeatureStoreWriter(output_root, split=split,
            expected_sample_ids=[r["sample_id"] for r in records], source_manifest_sha256=manifest["sha256"],
            checkpoint_provenance=job["checkpoint_provenance"], preprocessing_identity=job["preprocessing_identity"])
        for index, start in enumerate(range(0, len(records), job["shard_size"])):
            selected = records[start:start + job["shard_size"]]
            outputs = {}
            for expert in EXPERT_ORDER:
                arrays = load_chunk(chunk_path(output_root, expert, split, index), expert=expert,
                                    records=selected, identity=job_identity(job))
                outputs[expert] = (torch.from_numpy(arrays["scores"]), torch.from_numpy(arrays["features"]))
            scores, features = validate_and_assemble_expert_outputs(outputs)
            labels = np.asarray([r["label_raw"] for r in selected], dtype=np.float32) if split != "test" else None
            writer.write_shard(index, sample_ids=[r["sample_id"] for r in selected],
                               wave_file_names=[r["wave_file_name"] for r in selected],
                               scores=scores, features=features, labels_raw=labels)
        writer.finalize()
        _atomic_json(Path(output_root) / split / "source_manifest.json", manifest)
        reports[split] = {"sample_count": len(records), "status": "PASS",
                          "identity": validate_feature_store(output_root, split=split)["store_identity_sha256"]}
    return reports


def completed_stores_match(job, manifests, output_root):
    validate_manifests(job, manifests)
    for split, source in manifests.items():
        if not (Path(output_root) / split / "manifest.json").exists():
            return False
        existing = validate_feature_store(output_root, split=split, require_labels=split != "test")
        if (existing["checkpoint_provenance"] != job["checkpoint_provenance"]
                or existing["preprocessing_identity"] != job["preprocessing_identity"]
                or existing["source_manifest_sha256"] != source["sha256"]):
            raise ValueError(f"Completed {split} store provenance differs from current extraction")
        if split == "test" and existing["labels_present"]:
            raise ValueError("Test extraction store must be label-free")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,1,1,1")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=EXPERT_ORDER)
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    try:
        if args.worker:
            config = resolve_moe_paths(load_moe_config(args.config), args.config)
            job = json.loads((root / "_extraction/job.json").read_text())
            manifests = {s: json.loads((root / f"_extraction/{s}_manifest.json").read_text()) for s in EXPECTED_COUNTS}
            extract_worker(config, job, manifests, args.dataset_root, root, args.worker, args.device)
        else:
            sizes = dict(zip(EXPERT_ORDER, map(int, args.batch_sizes.split(","))))
            _, job, manifests = prepare_job(args.config, args.dataset_root, root, batch_sizes=sizes,
                                            shard_size=args.shard_size, limit=4 if args.smoke else None)
            if not completed_stores_match(job, manifests, root):
                for expert in EXPERT_ORDER:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", args.config,
                        "--dataset-root", args.dataset_root, "--output-root", args.output_root,
                        "--device", args.device, "--worker", expert], check=True)
                assemble_stores(job, manifests, root)
            print(json.dumps({"status": "PASS", "smoke": args.smoke, "output_root": str(root)}))
    except Exception as exc:
        _atomic_json(root / "_extraction/failure.json", {"status": "FAIL", "expert": args.worker,
                     "error": f"{type(exc).__name__}: {exc}", "resume": "Validated completed shards are retained"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
