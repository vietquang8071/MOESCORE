#!/usr/bin/env python3
"""Prepare a profile job, run one expert at a time on each of two GPUs, assemble.

This module never imports an expert model. Each model lives in its own worker
subprocess; the shared parent hashes resources and writes manifests once.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_feature_store import _atomic_json, validate_feature_store
from model.expert_adapters.registry import EXPERT_PROFILES, profile_contract
from scripts.extract_moe_feature_store import (
    assembly_store_root, assemble_stores, completed_stores_match, job_identity,
    prepare_job,
)


def default_worker_devices(order=EXPERT_PROFILES["full_9"]):
    """Stable alternating assignment; each GPU has pending work on a fresh run."""
    return {name: f"cuda:{index % 2}" for index, name in enumerate(order)}


def dependency_plan_for_python(version):
    """Select a checked requirements set without changing the CUDA stack."""
    selected = tuple(version[:2])
    if selected == (3, 12):
        return PROJECT_ROOT / "requirements-kaggle-py312.txt", False
    if selected == (3, 13):
        return PROJECT_ROOT / "requirements-kaggle-py313.txt", True
    raise RuntimeError(f"Kaggle dual-GPU extraction requires Python 3.12 or 3.13; active {selected}")


def validate_worker_devices(mapping, order, *, require_two=True):
    if set(mapping) != set(order):
        raise ValueError("Device mapping must name each selected expert exactly once")
    if any(value not in {"cuda:0", "cuda:1"} for value in mapping.values()):
        raise ValueError("Dual-GPU extraction accepts only cuda:0 and cuda:1")
    if require_two and set(mapping.values()) != {"cuda:0", "cuda:1"}:
        raise ValueError("Both cuda:0 and cuda:1 must receive expert work")
    return dict(mapping)


def require_kaggle_dual_gpu():
    import torch
    dependency_plan_for_python(sys.version_info)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle dual-GPU mode requires two visible CUDA devices")
    return {f"cuda:{index}": torch.cuda.get_device_name(index) for index in (0, 1)}


def run_assigned_workers(order, mapping, runner):
    """One sequential queue per GPU, two queues running concurrently.

    A failure stops scheduling later experts on either queue. Already running
    workers finish and their validated shards remain available for resume.
    """
    validate_worker_devices(mapping, order)
    stop = threading.Event()
    results = {}
    lock = threading.Lock()
    queues = {device: [name for name in order if mapping[name] == device]
              for device in ("cuda:0", "cuda:1")}

    def consume(device):
        for expert in queues[device]:
            if stop.is_set():
                break
            try:
                report = runner(expert, device)
                if report.get("status") != "PASS":
                    raise RuntimeError(f"{expert} on {device} returned {report.get('status')!r}")
                with lock:
                    results[expert] = report
            except BaseException:
                stop.set()
                raise

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume, device) for device in queues]
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError(str(errors[0])) from errors[0]
    if set(results) != set(order):
        raise RuntimeError("Not every expert completed; assembly is prohibited")
    return {name: results[name] for name in order}


def extract_dual_gpu(config_path, dataset_root, output_root, *, worker_devices=None,
                     batch_sizes=None, shard_size=128, smoke=False, require_runtime=True,
                     profile="full_9"):
    import torch
    if profile not in {"full_9", "full_10"}:
        raise ValueError("dual-GPU extraction supports the full_9 and full_10 profiles")
    order = EXPERT_PROFILES[profile]
    mapping = validate_worker_devices(worker_devices or default_worker_devices(order), order)
    gpu_names = require_kaggle_dual_gpu() if require_runtime else {
        f"cuda:{i}": torch.cuda.get_device_name(i) for i in range(2)
    }
    output_root = Path(output_root).resolve()
    config, job, manifests = prepare_job(
        config_path, dataset_root, output_root, batch_sizes=batch_sizes, shard_size=shard_size,
        limit=4 if smoke else None, device="cuda:0", profile=profile, worker_devices=mapping,
    )
    job_id = job_identity(job)
    final_root = assembly_store_root(output_root, job)
    if completed_stores_match(job, manifests, output_root):
        return {"status": "PASS", "reused_complete_store": True, "assembly_id": job_id,
                "feature_store_root": str(final_root), "worker_devices": mapping}
    script = PROJECT_ROOT / "scripts/extract_moe_feature_store.py"
    log_root = output_root / "logs" / job_id
    log_root.mkdir(parents=True, exist_ok=True)

    def worker(expert, device):
        start = datetime.now(timezone.utc).isoformat()
        command = [sys.executable, str(script), "--config", str(config_path),
                   "--dataset-root", str(dataset_root), "--output-root", str(output_root),
                   "--device", device, "--profile", profile, "--worker", expert,
                   "--assembly-id", job_id]
        log = log_root / f"{expert}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        identity = job["expert_cache_identities"][expert]
        runtime_path = output_root / "feature_cache/experts" / expert / identity["cache_id"] / "runtime.json"
        runtime = json.loads(runtime_path.read_text()) if result.returncode == 0 and runtime_path.is_file() else {}
        snapshots = [item.get("cuda", {}) for item in runtime.get("memory", [])]
        report = {"status": "PASS" if result.returncode == 0 and runtime.get("status") == "PASS" else "FAIL",
                  "expert": expert, "device": device, "gpu_name": gpu_names[device],
                  "checkpoint_provenance": identity["checkpoint_provenance"],
                  "batch_size": job["batch_sizes"][expert], "started_at_utc": start,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "sample_count_by_split": runtime.get("sample_count_by_split", {}),
                  "validated_shards": sum((len(source["records"]) + job["shard_size"] - 1) // job["shard_size"]
                                          for source in manifests.values()) if runtime.get("status") == "PASS" else 0,
                  "model_constructed": runtime.get("model_constructed"),
                  "peak_allocated_vram_bytes": max((v.get("peak_allocated_bytes", 0) for v in snapshots), default=0),
                  "peak_reserved_vram_bytes": max((v.get("peak_reserved_bytes", 0) for v in snapshots), default=0),
                  "log_path": str(log), "returncode": result.returncode}
        _atomic_json(log_root / f"{expert}.json", report)
        if report["status"] != "PASS":
            raise RuntimeError(f"{expert} failed on {device}; see {log}")
        return report

    try:
        worker_reports = run_assigned_workers(order, mapping, worker)
        assemble_stores(job, manifests, output_root)
        splits = {split: validate_feature_store(final_root, split=split, require_labels=split != "test")
                  for split in ("train", "validation", "test")}
        expected_width = profile_contract(profile)[2]
        if any(value["feature_width"] != expected_width or len(value["expert_order"]) != len(order) for value in splits.values()):
            raise ValueError(f"Assembled {profile} store has an invalid feature or score contract")
    except BaseException as exc:
        _atomic_json(log_root / "stage.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        raise
    report = {"status": "PASS", "assembly_id": job_id, "feature_store_root": str(final_root),
              "worker_devices": mapping, "gpu_names": gpu_names, "workers": worker_reports,
              "splits": {name: {"rows": len(manifests[name]["records"]),
                                  "store_identity": value["store_identity_sha256"]}
                         for name, value in splits.items()}}
    _atomic_json(log_root / "stage.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-devices-json")
    parser.add_argument("--profile", choices=("full_9", "full_10"), default="full_9")
    parser.add_argument("--batch-sizes", help="One positive integer per expert in profile order")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    order = EXPERT_PROFILES[args.profile]
    sizes = None if args.batch_sizes is None else [int(x) for x in args.batch_sizes.split(",")]
    if sizes is not None and (len(sizes) != len(order) or any(n <= 0 for n in sizes)):
        parser.error(f"--batch-sizes requires {len(order)} positive values for {args.profile}")
    mapping = json.loads(Path(args.worker_devices_json).read_text()) if args.worker_devices_json else None
    report = extract_dual_gpu(args.config, args.dataset_root, args.output_root,
        worker_devices=mapping, batch_sizes=dict(zip(order, sizes)) if sizes else None,
        shard_size=args.shard_size, smoke=args.smoke, profile=args.profile)
    if args.report_json:
        _atomic_json(Path(args.report_json), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
