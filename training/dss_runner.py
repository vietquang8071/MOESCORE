"""Dedicated MOE8-DSS gate-only runner; frozen expert caches are read-only."""

from __future__ import annotations

import math
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from data.dss_dataset import DSSDataset, collate_dss, EXPECTED_COUNTS
from data.moe_feature_store import preprocessing_contract
from loss.domain_spearman import HybridSoftSpearmanLoss
from model.expert_adapters.registry import profile_contract
from model.moe_gating_core import GatingNetwork
from scripts.diagnostics.d01_moe8 import discover_frozen_artifacts
from training.domain_balanced_sampler import DomainBalancedBatchSampler
from training.dss_evaluation import predict_rows, summarize_predictions, write_csv, write_final_evaluation
from training.target_transform import normalize_targets

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def best_epoch_update(srcc: float | None, previous: float | None) -> bool:
    return srcc is not None and math.isfinite(srcc) and (previous is None or srcc > previous)


def train_step(gate, optimizer, batch, device, *, objective=None):
    objective = objective or HybridSoftSpearmanLoss()
    features = batch["features"].to(device)
    scores = batch["scores"].to(device)
    targets = normalize_targets(batch["labels_raw"].to(device))
    domains = batch["model_ids"].to(device)
    weights = gate(features)
    predictions = (scores * weights).sum(dim=1)
    global_loss, domain_loss, total, groups = objective(predictions, targets, domains)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradients = [parameter.grad for parameter in gate.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("DSS gate gradients are absent or nonfinite")
    grad_norm = math.sqrt(sum(float(gradient.detach().square().sum()) for gradient in gradients))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("DSS gradient norm is nonfinite")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in gate.parameters()):
        raise FloatingPointError("DSS optimizer produced nonfinite gate parameters")
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean() / math.log(8)
    return {
        "global_loss": float(global_loss.detach()), "domain_loss": float(domain_loss.detach()),
        "total_loss": float(total.detach()), "gradient_norm": grad_norm,
        "gate_entropy": float(entropy.detach()), "valid_group_count": len(groups["valid_domains"]),
        "invalid_group_count": len(groups["invalid_domains"]),
        "valid_domains": groups["valid_domains"], "invalid_domains": groups["invalid_domains"],
        "per_domain_loss": groups["per_domain_loss"],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit():
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
    if dirty:
        raise RuntimeError("DSS source checkout is dirty; commit reviewed code before execution")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def _metadata_paths(dataset_root):
    candidates = sorted(Path(dataset_root).rglob("train_average.csv"))
    candidates = [path for path in candidates if path.parent.name == "meta_data"]
    if len(candidates) != 1:
        raise ValueError(f"expected one official meta_data/train_average.csv under {dataset_root}; got {candidates}")
    root = candidates[0].parent
    paths = {split: root / f"{split}_average.csv" for split in EXPECTED_COUNTS}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"official average-label metadata is incomplete under {root}")
    return paths


def verify_official_audio_hashes(metadata_paths, datasets):
    """Bind official WAV bytes to each cache source fingerprint without changing splits."""
    for split, dataset in datasets.items():
        wav_root = metadata_paths[split].parent.parent / "wav" / split
        fingerprints = dataset.manifest["assembly_identity"]["ordered_source_fingerprints"]
        for row, fingerprint in zip(dataset.metadata, fingerprints):
            path = wav_root / row["wave_file_name"]
            if not path.is_file():
                raise FileNotFoundError(f"{row['sample_id']}: official WAV is missing: {path}")
            if _sha256(path) != fingerprint["wav_sha256"]:
                raise ValueError(f"{row['sample_id']}: official WAV SHA-256 differs from frozen cache")


def preflight(dataset_root, artifacts_root):
    """Validate real Full8 shards and pair-level official labels before gate creation."""
    artifacts = discover_frozen_artifacts(artifacts_root)
    paths = _metadata_paths(dataset_root)
    datasets = {split: DSSDataset(artifacts["feature_store"], paths[split], split=split)
                for split in EXPECTED_COUNTS}
    verify_official_audio_hashes(paths, datasets)
    manifests = {split: dataset.manifest for split, dataset in datasets.items()}
    baseline = json.loads(artifacts["run_manifest"].read_text(encoding="utf-8"))
    if baseline.get("git_commit") != "68124d0e57005d29149f659780ab749d14e09532":
        raise ValueError("paired Full8 baseline source commit differs from expected cache provenance")
    if (baseline.get("train_store_identity_sha256") != manifests["train"]["store_identity_sha256"]
            or baseline.get("validation_store_identity_sha256") != manifests["validation"]["store_identity_sha256"]):
        raise ValueError("baseline gate run is not paired with the supplied Full8 cache")
    if baseline.get("gate_checkpoint_sha256") != _sha256(artifacts["gate_checkpoint"]):
        raise ValueError("baseline gate file differs from paired run-manifest checksum")
    source_hashes = baseline.get("source_code_sha256", {})
    critical_paths = (
        "model/moe_gating_core.py", "model/expert_adapters/registry.py",
        "data/moe_feature_store.py", "data/full8_contract.py",
        "training/target_transform.py", "training/moe_evaluator.py",
    )
    if any(source_hashes.get(path) != _sha256(PROJECT_ROOT / path) for path in critical_paths):
        raise ValueError("current Full8 gate, reader, registry, or score transform differs from cache source")
    order, dimensions, width = profile_contract("full_8")
    for split, manifest in manifests.items():
        if (manifest["expert_order"] != list(order) or manifest["feature_dimensions"] != dimensions
                or manifest["feature_width"] != width or manifest["score_width"] != 8):
            raise ValueError(f"{split}: ordered Full8 scores/features violate registered gate contract")
        if manifest["checkpoint_provenance"] != manifests["train"]["checkpoint_provenance"]:
            raise ValueError(f"{split}: expert checkpoint provenance differs across cache splits")
        if preprocessing_contract(manifest) != preprocessing_contract(manifests["train"]):
            raise ValueError(f"{split}: preprocessing or score contract differs across cache splits")
    wav_hashes = {split: {row["wav_sha256"] for row in manifests[split]["assembly_identity"]["ordered_source_fingerprints"]}
                  for split in EXPECTED_COUNTS}
    overlaps = {f"{left}_{right}": len(wav_hashes[left] & wav_hashes[right])
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    candidate_baseline_results = sorted(
        str(path.relative_to(artifacts_root)) for path in Path(artifacts_root).rglob("*")
        if path.is_file() and (path.suffix == ".csv" and "pred" in path.name.lower()
                               or path.suffix == ".json" and "metric" in path.name.lower())
    )
    report = {
        "status": "PASS", "source_commit": _source_commit(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "artifacts_root": str(Path(artifacts_root).resolve()),
        "baseline_source_commit": baseline["git_commit"],
        "baseline_checkpoint_sha256": baseline["gate_checkpoint_sha256"],
        "baseline_config_file_sha256": _sha256(artifacts["gate_config"]),
        "baseline_config_identity_sha256": baseline.get("config_sha256"),
        "critical_source_hashes_verified": list(critical_paths),
        "source_hash_differences_outside_runtime": [path for path, digest in source_hashes.items()
            if (PROJECT_ROOT / path).is_file() and _sha256(PROJECT_ROOT / path) != digest and path not in critical_paths],
        "feature_store_identities": {split: manifest["store_identity_sha256"] for split, manifest in manifests.items()},
        "source_manifest_sha256": {split: manifest["source_manifest_sha256"] for split, manifest in manifests.items()},
        "score_contracts_sha256": hashlib.sha256(json.dumps(manifests["train"]["score_contracts"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "expert_checkpoint_provenance_sha256": hashlib.sha256(json.dumps(
            manifests["train"]["checkpoint_provenance"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "metadata_sha256": {split: _sha256(path) for split, path in paths.items()},
        "sample_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "domain_counts": {split: EXPECTED_COUNTS[split] for split in EXPECTED_COUNTS},
        "cross_split_wav_sha256_overlap_counts": overlaps,
        "official_wav_sha256_verified_count": sum(len(dataset) for dataset in datasets.values()),
        "expert_order": list(order), "feature_width": width, "score_width": 8,
        "score_scale": "normalized_minus1_to1", "raw_label_scale": "raw_0_to_10",
        "baseline_validation_metrics": max(
            (row for row in baseline.get("history", []) if row["validation_metrics_raw"]["srcc"] is not None),
            key=lambda row: row["validation_metrics_raw"]["srcc"], default=None,
        ),
        "candidate_baseline_result_files": candidate_baseline_results,
    }
    return datasets, report


def _rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _effective_config(run_mode, device, source_commit):
    return {
        "experiment": "MOE8-DSS", "model_profile": "full_8", "source_commit": source_commit,
        "run_mode": run_mode, "seed": 2027, "device": str(device),
        "gate": {"input_dim": 33032, "hidden_dim": 512, "output_dim": 8, "fresh_initialization": True},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-5, "weight_decay": 1e-4, "scheduler": None},
        "batch": {"size": 128, "domains": 4, "samples_per_domain": 32, "steps_per_epoch": 59},
        "epochs": 2 if run_mode == "smoke" else 30,
        "loss": {"name": "hybrid_l2_soft_spearman", "global_weight": 0.5,
                 "domain_weight": 0.5, "regularization_strength": 0.1, "min_domain_group_size": 8,
                 "rank_backend": "internal_exact_pav_l2_v1"},
        "validation_selection": "full_split_hard_srcc_strict_max_earliest_tie",
        "inference_batch_size": 64,
    }


def _run_dss(dataset_root, artifacts_root, output_dir, *, run_mode="full", device_name="cuda:0", resume=None):
    if run_mode not in ("smoke", "full"):
        raise ValueError("run_mode must be smoke or full")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DSS requested CUDA but no GPU is available")
    root = Path(output_dir).resolve()
    if root.exists() and any((root / "checkpoints" / name).exists() for name in ("best.pt", "last.pt")) and resume is None:
        raise FileExistsError("DSS output already contains checkpoints; pass a verified resume or choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    datasets, preflight_report = preflight(dataset_root, artifacts_root)
    config = _effective_config(run_mode, device, preflight_report["source_commit"])
    identity = {"cache": preflight_report["feature_store_identities"],
                "metadata": preflight_report["metadata_sha256"], "config": config}
    (root / "preflight_report.json").write_text(json.dumps(preflight_report, indent=2))
    import yaml
    (root / "experiment_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2027)
    gate = GatingNetwork(33032, 8, profile="full_8").to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-5, weight_decay=1e-4)
    objective = HybridSoftSpearmanLoss(0.1, 8)
    sampler = DomainBalancedBatchSampler(datasets["train"].model_ids, seed=2027, steps_per_epoch=59)
    train_loader = DataLoader(datasets["train"], batch_sampler=sampler, num_workers=0, collate_fn=collate_dss)
    selected_epoch, best_srcc, steps = None, None, 0
    history, sampler_history = [], []
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("identity") != identity or checkpoint.get("run_mode") != run_mode:
            raise ValueError("DSS resume identity, source or run mode differs from current inputs")
        if Path(resume).resolve() != (root / "checkpoints" / "last.pt"):
            raise ValueError("DSS resume must be this run's last.pt")
        best_path = root / "checkpoints" / "best.pt"
        if checkpoint.get("best_epoch") is None:
            if best_path.exists() or checkpoint.get("best_checkpoint_sha256") is not None:
                raise ValueError("DSS resume has an unselected but present best checkpoint")
        elif not best_path.is_file() or _sha256(best_path) != checkpoint.get("best_checkpoint_sha256"):
            raise ValueError("DSS best checkpoint is missing or differs from last.pt resume state")
        gate.load_state_dict(checkpoint["gate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        sampler.load_state_dict(checkpoint["sampler"])
        _restore_rng(checkpoint["rng"])
        selected_epoch, best_srcc = checkpoint["best_epoch"], checkpoint["best_srcc"]
        steps = checkpoint["optimizer_steps"]
        history, sampler_history = checkpoint["history"], checkpoint["sampler_history"]
        if checkpoint["epoch"] != sampler.epoch or steps != sampler.epoch * 59:
            raise ValueError("DSS resume epoch and optimizer step counts are inconsistent")
    metadata_path = root / "experiment_metadata.json"
    metadata = {"status": "PARTIAL", "source_commit": config["source_commit"],
                "dataset_identifiers": ["qnv8071/xacle-2027-data", "qnv8071/moescore-full8-artifacts"],
                "dataset_versions": "not exposed by mounted files", "cache_fingerprints": identity["cache"],
                "dataset_root": preflight_report["dataset_root"],
                "artifacts_root": preflight_report["artifacts_root"],
                "metadata_fingerprints": identity["metadata"], "python_version": platform.python_version(),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "ranking_backend": "internal_exact_pav_l2_v1", "run_mode": run_mode}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    for epoch in range(sampler.epoch + 1, config["epochs"] + 1):
        gate.train()
        first_layer_at_epoch_start = gate.net[0].weight.detach().clone()
        records = []
        for batch in train_loader:
            records.append(train_step(gate, optimizer, batch, device, objective=objective))
            steps += 1
        if len(records) != 59 or steps != epoch * 59:
            raise RuntimeError(f"DSS epoch {epoch} did not produce exactly 59 optimizer updates")
        validation_rows = predict_rows(gate, datasets["validation"], device,
                                       checkpoint_sha256="epoch-validation")
        validation_metrics, _, validation_aggregate, _ = summarize_predictions(
            validation_rows, "validation", expected_count=3000
        )
        val_srcc = validation_metrics["srcc"]
        if best_epoch_update(val_srcc, best_srcc):
            selected_epoch, best_srcc = epoch, val_srcc
            _atomic_save({"gate": gate.state_dict(), "epoch": epoch, "validation_srcc": val_srcc,
                          "identity": identity, "run_mode": run_mode, "optimizer_steps": steps},
                         root / "checkpoints" / "best.pt")
        row = {"epoch": epoch, "optimizer_steps": steps,
               **{name: float(np.mean([record[name] for record in records]))
                  for name in ("global_loss", "domain_loss", "total_loss", "gradient_norm", "gate_entropy")},
               "valid_group_count": sum(record["valid_group_count"] for record in records),
               "invalid_group_count": sum(record["invalid_group_count"] for record in records),
               "validation_srcc": val_srcc, "validation_macro_srcc": validation_aggregate["macro_srcc"],
               "learning_rate": optimizer.param_groups[0]["lr"], "best_epoch": selected_epoch,
               "epoch_gate_update_l2": float((gate.net[0].weight.detach() - first_layer_at_epoch_start).norm()),
               "per_domain_loss": json.dumps({domain: float(np.mean([record["per_domain_loss"][domain]
                  for record in records if domain in record["per_domain_loss"]]))
                  for domain in range(9) if any(domain in record["per_domain_loss"] for record in records)})}
        history.append(row)
        exposure = sampler.last_history
        sampler_history.append({key: value for key, value in exposure.items() if not isinstance(value, dict)} |
                               {f"domain_{domain}_appearances": count for domain, count in exposure["domain_appearances"].items()} |
                               {f"domain_{domain}_draws": count for domain, count in exposure["per_domain_draws"].items()} |
                               {f"domain_{domain}_cumulative_unique": count for domain, count in exposure["cumulative_per_domain_unique_samples"].items()})
        best_path = root / "checkpoints" / "best.pt"
        _atomic_save({"gate": gate.state_dict(), "optimizer": optimizer.state_dict(),
                      "sampler": sampler.state_dict(), "rng": _rng_state(), "epoch": epoch,
                      "optimizer_steps": steps, "best_epoch": selected_epoch, "best_srcc": best_srcc,
                      "best_checkpoint_sha256": _sha256(best_path) if best_path.exists() else None,
                      "history": history, "sampler_history": sampler_history,
                      "identity": identity, "run_mode": run_mode}, root / "checkpoints" / "last.pt")
        write_csv(root / "training_history.csv", history)
        write_csv(root / "sampler_history.csv", sampler_history)
        metadata.update({"completed_epochs": epoch, "optimizer_steps": steps,
                         "best_epoch": selected_epoch, "best_validation_srcc": best_srcc,
                         "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(json.dumps(row), flush=True)
    if selected_epoch is None or best_srcc is None:
        raise RuntimeError("DSS had no finite full-validation SRCC for checkpoint selection")
    if run_mode == "smoke":
        metadata["smoke_status"] = "COMPLETE"
        metadata["best_checkpoint_sha256"] = _sha256(root / "checkpoints" / "best.pt")
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata
    if steps != 1770 or sampler.epoch != 30:
        raise RuntimeError("full DSS training must complete exactly 30 epochs and 1,770 updates")
    best_file = root / "checkpoints" / "best.pt"
    best = torch.load(best_file, map_location="cpu", weights_only=True)
    if best["identity"] != identity or best["epoch"] != selected_epoch or best["validation_srcc"] != best_srcc:
        raise ValueError("selected DSS checkpoint differs from recorded best validation result")
    gate.load_state_dict(best["gate"])
    best_hash = _sha256(best_file)
    predictions = {split: predict_rows(gate, dataset, device, checkpoint_sha256=best_hash)
                   for split, dataset in datasets.items()}
    global_metrics, per_domain, aggregates = write_final_evaluation(root, predictions, history)
    metadata.update({"best_checkpoint_sha256": best_hash,
                     "prediction_count": sum(len(rows) for rows in predictions.values()),
                     "global_metrics": global_metrics, "aggregate_metrics": aggregates})
    metadata["status"] = "COMPLETE"
    (root / "report.md").write_text(_report_markdown(metadata, preflight_report, per_domain))
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _report_markdown(metadata, preflight, per_domain):
    lines = ["# MOE8-DSS result", "", f"Status: {metadata['status']}", "",
             f"Source commit: `{metadata['source_commit']}`", "",
             f"Selected epoch: {metadata['best_epoch']} / 30; optimizer updates: {metadata['optimizer_steps']}",
             f"Best full-validation hard SRCC: {metadata['best_validation_srcc']:.6f}", "",
             "| Split | Samples | SRCC | LCC | KTAU | Raw MSE |", "|---|---:|---:|---:|---:|---:|"]
    for split, row in metadata["global_metrics"].items():
        lines.append(f"| {split} | {row['sample_count']} | {row['srcc']} | {row['lcc']} | {row['ktau']} | {row['mse']} |")
    lines += ["", "## Per-domain metrics", "", "| Split | Domain | Samples | SRCC | LCC | KTAU | Raw MSE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in per_domain:
        lines.append(f"| {row['split']} | {row['model_name']} | {row['sample_count']} | {row['srcc']} | {row['lcc']} | {row['ktau']} | {row['mse']} |")
    lines += ["", "## Baseline evidence", "", "The paired standard Full8 run manifest provides full-validation metrics.", ""]
    baseline = preflight["baseline_validation_metrics"]
    if baseline is not None:
        lines.append(f"Baseline selected validation epoch {baseline['epoch']}: SRCC {baseline['validation_metrics_raw']['srcc']}; DSS minus baseline: {metadata['best_validation_srcc'] - baseline['validation_metrics_raw']['srcc']:+.6f}.")
    candidates = preflight["candidate_baseline_result_files"]
    if candidates:
        lines.append(f"Additional baseline result candidates exist but were not compared without verified checkpoint, metric and sample alignment: {candidates}.")
    else:
        lines.append("No saved baseline prediction CSV or metric JSON was found in the mounted Full8 artifact bundle; public-test deltas are unavailable.")
    lines += ["", "All nine development domains participated in training. This is not an unseen-generator experiment. Pure ranking loss does not optimize raw-score calibration; the CSV metrics and diagnostics contain score and routing details.", ""]
    return "\n".join(lines)


def run_dss(dataset_root, artifacts_root, output_dir, *, run_mode="full", device_name="cuda:0", resume=None):
    try:
        return _run_dss(dataset_root, artifacts_root, output_dir,
                        run_mode=run_mode, device_name=device_name, resume=resume)
    except Exception as exc:
        path = Path(output_dir) / "experiment_metadata.json"
        if not path.is_file() or json.loads(path.read_text()).get("status") != "COMPLETE":
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(path.read_text()) if path.is_file() else {}
            existing.update({"status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"})
            path.write_text(json.dumps(existing, indent=2))
        raise
