"""Gate-only Full8 LOGO experiment over verified, read-only feature stores."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from data.audio_overlap import (build_public_split_audio_overlap_diagnostic,
                                build_train_validation_audio_overlap_diagnostic)
from data.moe_feature_store import preprocessing_contract
from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from model.expert_adapters.registry import profile_contract
from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager
from training.logo_protocol import align_metadata, aggregate_predictions, fold_indices, load_metadata
from training.moe_evaluator import MoeGateEvaluator, compute_metrics
from training.moe_trainer import MoeGateTrainer
from training.sampler import StatefulRandomSampler
from training.target_transform import denormalize_predictions


CLASSIFICATION = {
    "experiment": "MOE8-LOGO-Lite", "model_profile": "full_8",
    "protocol": "gate_only_target_selection", "benchmark_type": "GATE_ONLY_PROXY_NOT_STRICT_DG",
    "strict_domain_generalization": False, "reuse_expert_checkpoints": True,
    "reuse_feature_store": True, "retrain_experts": False, "retrain_baseline": False,
    "run_baseline_inference": False, "initialize_from_baseline_gate": False,
    "heldout_generator_used_for_gate_gradient_updates": False,
    "heldout_generator_used_for_checkpoint_selection": True,
    "expert_generator_exposure": "possible", "research_only": True,
}
SPLIT_NAMES = {"internal_holdout": "train", "validation": "validation", "public_test": "public_test"}
PREDICTION_FIELDS = ("sample_id", "source_split", "fold_id", "wave_file_name", "model_id",
                     "model_name", "ground_truth", "predicted_score", *[f"expert_{i}_score" for i in range(1, 9)],
                     *[f"expert_{i}_weight" for i in range(1, 9)], "gate_entropy")
SOURCE_PATHS = ("model/moe_gating_core.py", "training/moe_trainer.py", "training/moe_evaluator.py",
                "training/target_transform.py", "loss/loss_function.py", "data/moe_feature_store.py",
                "data/full8_contract.py", "model/expert_adapters/registry.py")


def atomic_json(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict], fields: tuple | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_best_epoch(current: tuple[float, int] | None, srcc: float | None,
                      epoch: int) -> tuple[float, int] | None:
    if srcc is None or not math.isfinite(float(srcc)):
        return current
    return (float(srcc), epoch) if current is None or srcc > current[0] else current


class LogoView(Dataset):
    """Index-only fold view; loader split role and official source split stay distinct."""

    def __init__(self, store: Dataset, rows: list[dict], indices: list[int], role: str):
        if role not in {"optimization", "internal_holdout", "validation", "public_test"}:
            raise ValueError(f"unsupported fold role: {role}")
        self.store, self.rows, self.indices, self.role = store, rows, indices, role

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        row = self.rows[index]
        sample = dict(self.store[index])
        if sample["sample_id"] != row["sample_id"] or sample["wave_file_name"] != row["wave_file_name"]:
            raise ValueError(f"metadata-to-cache row mismatch at {row['sample_id']}")
        if "label_raw" in sample and abs(float(sample["label_raw"]) - row["ground_truth"]) > 1e-5:
            raise ValueError(f"metadata-to-cache label mismatch at {row['sample_id']}")
        sample["label_raw"] = torch.tensor(row["ground_truth"], dtype=torch.float32)
        sample["source_split"] = row["source_split"]
        sample["fold_role"] = self.role
        sample["split"] = "train" if self.role == "optimization" else "validation"
        return sample


def discover_inputs(dataset_root: str | None, artifacts_root: str | None) -> tuple[Path, Path, Path, Path]:
    mount = Path("/kaggle/input")
    data = Path(dataset_root) if dataset_root else mount
    artifacts = Path(artifacts_root) if artifacts_root else mount
    metadata = [p.parent for p in data.rglob("train_average.csv") if
                (p.parent / "validation_average.csv").is_file() and (p.parent / "test_average.csv").is_file()]
    stores = [p.parent.parent for p in artifacts.rglob("train/manifest.json") if
              (p.parent.parent / "validation/manifest.json").is_file() and
              (p.parent.parent / "test/manifest.json").is_file()]
    configs = list(artifacts.rglob("moe_full8.yaml"))
    run_manifests = list(artifacts.rglob("run_manifest.json"))
    if len(metadata) != 1 or len(stores) != 1 or len(configs) != 1 or len(run_manifests) != 1:
        raise ValueError(f"ambiguous/missing inputs: metadata={metadata}, stores={stores}, "
                         f"baseline configs={configs}, baseline manifests={run_manifests}; supply path overrides")
    return metadata[0], stores[0], configs[0], run_manifests[0]


def preflight(config: dict, *, dataset_root: str | None = None,
              artifacts_root: str | None = None) -> dict:
    metadata_root, store_root, baseline_config_path, baseline_manifest_path = discover_inputs(dataset_root, artifacts_root)
    baseline = yaml.safe_load(baseline_config_path.read_text())
    provenance = json.loads(baseline_manifest_path.read_text())
    baseline_gate = baseline_manifest_path.parent / "best_gate.pt"
    if not baseline_gate.is_file() or file_sha256(baseline_gate) != provenance.get("gate_checkpoint_sha256"):
        raise ValueError("baseline gate checkpoint file/hash differs from its run manifest")
    if baseline.get("data", {}).get("profile") != "full_8" or provenance.get("expert_profile") != "full_8":
        raise ValueError("supplied baseline is not the full_8 profile")
    baseline_effective = json.loads(json.dumps(baseline))
    baseline_effective.pop("output_dir", None)
    baseline_effective["runtime"].pop("device", None)
    baseline_effective["data"].pop("feature_store_root", None)
    baseline_effective["data"]["score_review_policy"] = provenance.get("score_review_policy")
    if digest(baseline_effective) != provenance.get("config_sha256"):
        raise ValueError("baseline YAML does not reproduce the supplied gate run configuration hash")
    source_hashes = provenance.get("source_code_sha256", {})
    project_root = Path(__file__).resolve().parents[1]
    changed = [name for name in SOURCE_PATHS if hashlib.sha256((project_root/name).read_bytes()).hexdigest()
               != source_hashes.get(name)]
    if changed:
        raise ValueError(f"baseline-compatible runtime source mismatch: {changed}")
    if provenance.get("seed") != baseline.get("runtime", {}).get("seed"):
        raise ValueError("baseline run seed differs from supplied baseline config")
    for section in ("training", "validation", "inference"):
        if section not in baseline or not baseline[section]:
            raise ValueError(f"baseline configuration lacks essential {section} settings")
    stores = {split: MoeTrainingDataset(store_root, split=split, require_labels=split != "test")
              for split in ("train", "validation", "test")}
    manifests = {split: store.manifest for split, store in stores.items()}
    order, dims, width = profile_contract("full_8")
    for split, manifest in manifests.items():
        if (manifest["expert_order"] != list(order) or manifest["feature_dimensions"] != dims
                or manifest["feature_width"] != width or manifest["score_width"] != 8):
            raise ValueError(f"{split} Full8 dimensions/order mismatch")
        if manifest["checkpoint_provenance"] != manifests["train"]["checkpoint_provenance"]:
            raise ValueError(f"{split} expert checkpoint provenance mismatch")
        if preprocessing_contract(manifest) != preprocessing_contract(manifests["train"]):
            raise ValueError(f"{split} score/preprocessing contract mismatch")
    if (stores["train"].identity != provenance.get("train_store_identity_sha256") or
            stores["validation"].identity != provenance.get("validation_store_identity_sha256")):
        raise ValueError("supplied cache differs from baseline gate provenance")
    rows = align_metadata(load_metadata(metadata_root), manifests)
    for split in ("train", "validation"):
        for index, row in enumerate(rows[split]):
            cached = stores[split][index]
            if abs(float(cached["label_raw"]) - row["ground_truth"]) > 1e-5:
                raise ValueError(f"{split} metadata-to-cache label mismatch: {row['sample_id']}")
    controls = set(config.get("control_model_names", []))
    names = {row["model_id"]: row["model_name"] for split in rows for row in rows[split]}
    controls_found = {model_id for model_id, name in names.items() if name in controls}
    unknown_controls = controls - set(names.values())
    if unknown_controls:
        raise ValueError(f"configured non-generator controls absent from metadata: {sorted(unknown_controls)}")
    eligible = sorted({row["model_id"] for row in rows["train"]} - controls_found)
    if not eligible:
        raise ValueError("no eligible train generators")
    statistics = [{"model_id": model_id, "model_name": names[model_id],
                   "role": "control" if model_id in controls_found else
                           "eligible_generator" if model_id in eligible else "evaluation_only",
                   **{f"{split}_samples": sum(row["model_id"] == model_id for row in rows[split])
                      for split in rows}}
                  for model_id in sorted(names)]
    source_rows = {split: manifests[split]["assembly_identity"]["ordered_source_fingerprints"]
                   for split in manifests}
    overlap = build_public_split_audio_overlap_diagnostic(
        source_rows, allow_public_split_audio_overlap=config["allow_public_split_audio_overlap"],
        allow_train_validation_audio_overlap=config["allow_train_validation_audio_overlap"])
    if overlap["status"] == "ERROR":
        raise ValueError(f"disallowed cross-split WAV overlap: {overlap['pair_overlap_group_counts']}")
    return {"metadata_root": metadata_root, "store_root": store_root, "baseline_config_path": baseline_config_path,
            "baseline_manifest_path": baseline_manifest_path, "baseline": baseline, "baseline_provenance": provenance,
            "stores": stores, "manifests": manifests, "rows": rows, "statistics": statistics,
            "eligible": eligible, "controls": sorted(controls_found), "overlap": overlap}


def _loader(view: LogoView, size: int, *, sampler=None) -> DataLoader:
    return DataLoader(view, batch_size=size, sampler=sampler, shuffle=False, num_workers=0,
                      drop_last=False, collate_fn=collate_moe_features)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        name = "cuda:0"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {name}")
    return device


def _predict(gate: GatingNetwork, loader: DataLoader, metadata: dict[str, dict],
             generator_id: str) -> list[dict]:
    gate.eval()
    output = []
    device = next(gate.parameters()).device
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            scores = batch["scores"].to(device)
            weights = gate(features)
            raw = denormalize_predictions((scores * weights).sum(dim=1)).cpu().numpy()
            weights = weights.cpu().numpy()
            scores = scores.cpu().numpy()
            for index, sample_id in enumerate(batch["sample_ids"]):
                source = metadata[sample_id]
                if source["model_id"] != generator_id or not math.isfinite(float(raw[index])):
                    raise ValueError(f"final inference fold membership/prediction failure: {sample_id}")
                w = weights[index]
                if not np.isfinite(w).all() or not np.isclose(w.sum(), 1, atol=1e-6):
                    raise ValueError(f"invalid gate weights for {sample_id}")
                row = {"sample_id": sample_id, "source_split": source["source_split"],
                       "fold_id": generator_id, "wave_file_name": source["wave_file_name"],
                       "model_id": generator_id, "model_name": source["model_name"],
                       "ground_truth": source["ground_truth"], "predicted_score": float(raw[index]),
                       "gate_entropy": float(-(w * np.log(np.maximum(w, 1e-12))).sum() / math.log(8))}
                row.update({f"expert_{i+1}_score": float(scores[index, i]) for i in range(8)})
                row.update({f"expert_{i+1}_weight": float(w[i]) for i in range(8)})
                output.append(row)
    if len({row["sample_id"] for row in output}) != len(output):
        raise ValueError("duplicate final prediction sample ID")
    return output


def _prediction_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0, "status": "UNAVAILABLE_EMPTY_SUBSET", "srcc": None,
                "lcc": None, "ktau": None, "mse": None}
    metrics = compute_metrics([r["predicted_score"] for r in rows], [r["ground_truth"] for r in rows])
    predicted = np.asarray([r["predicted_score"] for r in rows], dtype=np.float64)
    truth = np.asarray([r["ground_truth"] for r in rows], dtype=np.float64)
    metrics.update({"status": "EVALUATED", "prediction_mean": float(predicted.mean()),
                    "prediction_std": float(predicted.std()), "ground_truth_mean": float(truth.mean()),
                    "ground_truth_std": float(truth.std()), "prediction_bias": float((predicted-truth).mean()),
                    "gate_entropy_mean": float(np.mean([float(r["gate_entropy"]) for r in rows]))})
    return metrics


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _identity(pre: dict, generator_id: str, folds: dict, seed: int) -> tuple[dict, dict]:
    manifests, rows = pre["manifests"], pre["rows"]
    base = pre["baseline"]
    fold_digest = digest({role: [rows["train" if role in {"optimization", "internal_holdout"}
                                   else "validation" if role == "validation" else "test"][i]["sample_id"]
                               for i in indices] for role, indices in folds.items()})
    train_val = build_train_validation_audio_overlap_diagnostic(
        manifests["train"]["assembly_identity"]["ordered_source_fingerprints"],
        manifests["validation"]["assembly_identity"]["ordered_source_fingerprints"],
        allow_train_validation_audio_overlap=pre["overlap"]["allow_train_validation_audio_overlap"])
    identity = {"train": pre["stores"]["train"].identity, "validation": pre["stores"]["train"].identity,
                "preprocessing_contract": preprocessing_contract(manifests["train"]),
                "source_wav_sha256_by_split": {split: [r["wav_sha256"] for r in
                    manifests[split]["assembly_identity"]["ordered_source_fingerprints"]]
                    for split in ("train", "validation")},
                "source_sample_ids_by_split": {split: [r["sample_id"] for r in
                    manifests[split]["assembly_identity"]["ordered_source_fingerprints"]]
                    for split in ("train", "validation")},
                "audio_overlap_diagnostic": train_val,
                "score_review_policy": pre["baseline_provenance"]["score_review_policy"],
                "logo_fold_membership_sha256": fold_digest,
                "logo_public_test_store_identity": pre["stores"]["test"].identity,
                "logo_validation_store_identity": pre["stores"]["validation"].identity}
    if pre["baseline_provenance"].get("score_review_sha256"):
        identity["score_review_sha256"] = pre["baseline_provenance"]["score_review_sha256"]
    run_config = {"runtime": {"seed": seed}, "data": {
        "profile": "full_8", "score_review_policy": identity["score_review_policy"],
        "allow_train_validation_audio_overlap": train_val["allow_train_validation_audio_overlap"]},
        "training": base["training"], "validation": base["validation"], "inference": base["inference"],
        "logo": {**CLASSIFICATION, "fold_id": generator_id, "fold_membership_sha256": fold_digest,
                 "baseline_git_commit": pre["baseline_provenance"]["git_commit"]}}
    return identity, run_config


def run_fold(pre: dict, generator_id: str, output_root: Path, *, device_name: str,
             resume: bool, smoke_epochs: int | None = None, seed: int | None = None) -> dict:
    if generator_id not in pre["eligible"] or re.fullmatch(r"[A-Za-z0-9_-]+", generator_id) is None:
        raise ValueError(f"ineligible or unsafe generator ID: {generator_id}")
    folds = fold_indices(pre["rows"], generator_id)
    root = output_root / f"fold_{generator_id}"
    seed = pre["baseline"]["runtime"]["seed"] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    identity, run_config = _identity(pre, generator_id, folds, seed)
    if smoke_epochs is not None:
        run_config["logo"]["smoke_epochs"] = smoke_epochs
    signature = digest({"config": run_config, "identity": identity,
                        "metadata": digest(pre["rows"]), "baseline": pre["baseline_provenance"]["git_commit"]})
    completion_path = root / "completion.json"
    required = [root / "checkpoints" / name for name in ("best.pt", "last.pt")]
    required += [root / "predictions" / f"{name}.csv" for name in SPLIT_NAMES.values()]
    required += [root / "metrics.json", root / "training_history.csv", root / "fold_metadata.json"]
    if completion_path.exists():
        completed = json.loads(completion_path.read_text())
        if (completed.get("signature") != signature or not all(path.is_file() for path in required)
                or completed.get("output_sha256") != {str(path.relative_to(root)): file_sha256(path)
                                                    for path in required}):
            raise ValueError(f"completed fold {generator_id} is incompatible or incomplete")
        for role, name in SPLIT_NAMES.items():
            data = _read_csv(root / "predictions" / f"{name}.csv")
            if len(data) != len(folds[role]) or any(r["model_id"] != generator_id for r in data):
                raise ValueError(f"completed fold {generator_id} prediction coverage mismatch")
        return {"fold_id": generator_id, "status": "SKIPPED_VALIDATED", "root": str(root)}
    if root.exists() and any(path.name != "splits" for path in root.iterdir()) and not resume:
        raise FileExistsError(f"fold output already exists; use --resume after compatibility validation: {root}")
    root.mkdir(parents=True, exist_ok=True)
    existing_metadata = root / "fold_metadata.json"
    if existing_metadata.exists() and json.loads(existing_metadata.read_text()).get("signature") != signature:
        raise ValueError(f"existing fold metadata is incompatible: {generator_id}")
    metadata = {row["sample_id"]: row for split in pre["rows"] for row in pre["rows"][split]}
    for role, indices in folds.items():
        source = "train" if role in {"optimization", "internal_holdout"} else "validation" if role == "validation" else "test"
        atomic_csv(root / "splits" / f"{role}.csv",
                   [{"sample_id": pre["rows"][source][i]["sample_id"], "source_split": source,
                     "split": "train" if role == "optimization" else "validation", "fold_role": role,
                     "model_id": pre["rows"][source][i]["model_id"]} for i in indices],
                   ("sample_id", "source_split", "split", "fold_role", "model_id"))
    atomic_json(root / "fold_metadata.json", {**CLASSIFICATION, "fold_id": generator_id,
        "generator_name": metadata[pre["rows"]["train"][folds["internal_holdout"][0]]["sample_id"]]["model_name"],
        "source_commit": pre["baseline_provenance"]["git_commit"], "signature": signature,
        "fold_counts": {role: len(indices) for role, indices in folds.items()},
        "feature_store_identity": {split: store.identity for split, store in pre["stores"].items()},
        "score_contracts": pre["manifests"]["train"]["score_contracts"]})
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(device_name)
    gate = GatingNetwork(input_feature_dim=33032, num_experts=8, profile="full_8").to(device)
    training = pre["baseline"]["training"]
    trainer = MoeGateTrainer(gate, training["loss"], learning_rate=training["learning_rate"],
                            weight_decay=training["weight_decay"], diversity_weight=training["diversity_weight"])
    evaluator = MoeGateEvaluator(gate, training["loss"])
    train_view = LogoView(pre["stores"]["train"], pre["rows"]["train"], folds["optimization"], "optimization")
    hold_view = LogoView(pre["stores"]["train"], pre["rows"]["train"], folds["internal_holdout"], "internal_holdout")
    sampler = StatefulRandomSampler(train_view, seed=seed)
    train_loader = _loader(train_view, training["batch_size"], sampler=sampler)
    hold_loader = _loader(hold_view, pre["baseline"]["validation"]["batch_size"])
    manager = MoeCheckpointManager()
    checkpoint = root / "checkpoints" / "last.pt"
    best_path = root / "checkpoints" / "best.pt"
    if best_path.exists() and not checkpoint.exists():
        raise ValueError("best checkpoint exists without resumable last checkpoint")
    history_path = root / "training_history.csv"
    history = _read_csv(history_path) if history_path.is_file() else []
    start_epoch, best, patience, elapsed = 0, None, 0, 0.0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"existing fold checkpoint requires --resume: {checkpoint}")
        state = manager.load(checkpoint, gating_network=gate, optimizer=trainer.optimizer,
                             scheduler=trainer.scheduler, expected_config=run_config,
                             expected_feature_store_identity=identity,
                             expected_expert_checkpoint_provenance=pre["manifests"]["train"]["checkpoint_provenance"])
        start_epoch = int(state["epoch"])
        trainer.global_step = int(state["global_step"])
        sampler.load_state_dict(state["sampler_state"])
        if sampler.epoch != start_epoch or sampler.cursor != 0:
            raise ValueError("LOGO checkpoint is not an epoch-end resume point")
        best_epoch_rows = [r for r in history if r["is_best"] == "True"]
        best = (float(state["best_srcc"]), int(best_epoch_rows[-1]["epoch"])) if best_epoch_rows else None
        patience = int(state["patience"])
        elapsed = sum(float(row["epoch_seconds"]) for row in history)
    if not checkpoint.exists() and history:
        raise ValueError("training history exists without a compatible last checkpoint")
    max_epochs = smoke_epochs if smoke_epochs is not None else training["max_epochs"]
    for epoch in range(start_epoch, max_epochs):
        began = time.monotonic()
        sampler.set_epoch(epoch)
        loss_sum, batches = 0.0, 0
        for batch in train_loader:
            if any(metadata[sample_id]["model_id"] == generator_id for sample_id in batch["sample_ids"]):
                raise ValueError("held-out generator entered an optimization batch")
            result = trainer.train_step(batch)
            loss_sum += float(result["total_loss"])
            batches += 1
        if not batches:
            raise ValueError("empty optimization epoch")
        holdout = evaluator.evaluate(hold_loader)
        hold_srcc = holdout["metrics"]["srcc"]
        hold_loss = float(holdout["regression_loss_normalized"])
        trainer.scheduler_step(hold_loss)
        selected = select_best_epoch(best, hold_srcc, epoch)
        improved = selected != best
        if improved:
            best, patience = selected, 0
        elif hold_srcc is not None:
            patience += 1
        def save(path: Path) -> None:
            manager.save(path, gating_network=gate, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
                         epoch=epoch + 1, global_step=trainer.global_step, config=run_config,
                         feature_store_identity=identity,
                         expert_checkpoint_provenance=pre["manifests"]["train"]["checkpoint_provenance"],
                         sampler_state=sampler.state_dict(), best_srcc=best[0] if best else None,
                         patience=patience, seed=seed,
                         selection_policy="logo_train_holdout_srcc_earliest_tie_v1")
        sampler.set_epoch(epoch + 1)
        if improved:
            save(best_path)
        save(checkpoint)
        seconds = time.monotonic() - began
        elapsed += seconds
        history.append({"epoch": epoch, "global_step": trainer.global_step, "train_loss": loss_sum/batches,
                        "holdout_loss": hold_loss, "holdout_srcc": hold_srcc, "is_best": improved,
                        "learning_rate": trainer.optimizer.param_groups[0]["lr"], "epoch_seconds": seconds})
        atomic_csv(history_path, history, ("epoch", "global_step", "train_loss", "holdout_loss",
                                           "holdout_srcc", "is_best", "learning_rate", "epoch_seconds"))
        print(f"LOGO fold {generator_id} epoch {epoch+1}/{max_epochs} holdout SRCC={hold_srcc} best={best}", flush=True)
        if patience >= training["early_stop_patience"] > 0:
            break
    if best is None:
        atomic_json(root / "selection_failure.json", {"status": "NO_FINITE_HOLDOUT_SRCC", "fold_id": generator_id})
        return {"fold_id": generator_id, "status": "NO_FINITE_HOLDOUT_SRCC", "root": str(root)}
    manager.load(best_path, gating_network=gate, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
                 expected_config=run_config, expected_feature_store_identity=identity,
                 expected_expert_checkpoint_provenance=pre["manifests"]["train"]["checkpoint_provenance"],
                 restore_rng=False)
    fold_metrics = {}
    for role, name in SPLIT_NAMES.items():
        source = "train" if role == "internal_holdout" else "validation" if role == "validation" else "test"
        view = LogoView(pre["stores"][source], pre["rows"][source], folds[role], role)
        predictions = _predict(gate, _loader(view, pre["baseline"]["inference"]["batch_size"]), metadata,
                               generator_id) if view else []
        if len(predictions) != len(view):
            raise ValueError(f"incomplete {name} final inference for generator {generator_id}")
        atomic_csv(root / "predictions" / f"{name}.csv", predictions, PREDICTION_FIELDS)
        fold_metrics[name] = {**_prediction_metrics(predictions), "best_epoch": best[1],
                              "training_duration_seconds": elapsed}
    atomic_json(root / "metrics.json", fold_metrics)
    from training.logo_plots import plot_fold
    plot_fold(root)
    atomic_json(completion_path, {"status": "COMPLETE", "signature": signature, "fold_id": generator_id,
                                  "best_epoch": best[1], "best_srcc": best[0],
                                  "training_duration_seconds": elapsed,
                                  "output_sha256": {str(path.relative_to(root)): file_sha256(path) for path in required},
                                  **CLASSIFICATION})
    return {"fold_id": generator_id, "status": "COMPLETE", "root": str(root)}


def write_summary(pre: dict, output_root: Path) -> dict:
    summary = output_root / "summary"
    all_metrics, weight_rows = [], []
    complete, failures = [], []
    predictions_by_split = {name: [] for name in SPLIT_NAMES.values()}
    for generator_id in pre["eligible"]:
        root = output_root / f"fold_{generator_id}"
        completion = root / "completion.json"
        if completion.is_file():
            status = json.loads(completion.read_text()).get("status")
        else:
            status = "INCOMPLETE_OR_FAILED"
            failures.append(generator_id)
        for role, name in SPLIT_NAMES.items():
            source = "train" if role == "internal_holdout" else "validation" if role == "validation" else "test"
            expected = sum(row["model_id"] == generator_id for row in pre["rows"][source])
            rows = _read_csv(root / "predictions" / f"{name}.csv") if status == "COMPLETE" else []
            if status == "COMPLETE":
                ids = {row["sample_id"] for row in rows}
                if (len(rows) != expected or len(ids) != len(rows) or
                        any(row["model_id"] != generator_id or row["source_split"] != source for row in rows)):
                    raise ValueError(f"fold {generator_id} {name} prediction coverage mismatch")
                if generator_id not in complete:
                    complete.append(generator_id)
            numeric_rows = [{**row, "ground_truth": float(row["ground_truth"]),
                             "predicted_score": float(row["predicted_score"])} for row in rows]
            metrics = _prediction_metrics(numeric_rows)
            saved_metrics = json.loads((root / "metrics.json").read_text())[name] if status == "COMPLETE" else None
            if saved_metrics and any(metrics[key] != saved_metrics[key] for key in ("srcc", "lcc", "ktau", "mse")):
                raise ValueError(f"saved metrics differ from prediction CSV: {generator_id}/{name}")
            all_metrics.append({"fold_id": generator_id, "generator_name": next(
                r["model_name"] for r in pre["rows"]["train"] if r["model_id"] == generator_id),
                "split": name, "samples": len(rows), "expected_samples": expected, "status": status,
                "srcc": metrics["srcc"], "lcc": metrics["lcc"], "ktau": metrics["ktau"], "mse": metrics["mse"],
                "prediction_std": metrics.get("prediction_std"), "ground_truth_std": metrics.get("ground_truth_std"),
                "best_epoch": saved_metrics["best_epoch"] if saved_metrics else None})
            predictions_by_split[name].extend(rows)
            if rows:
                weight_rows.append({"fold_id": generator_id, "split": name,
                                    **{f"expert_{i}_weight": float(np.mean([float(r[f"expert_{i}_weight"])
                                                                            for r in rows])) for i in range(1, 9)}})
    atomic_csv(summary / "all_fold_metrics.csv", all_metrics,
               ("fold_id", "generator_name", "split", "samples", "expected_samples", "status", "srcc", "lcc",
                "ktau", "mse", "prediction_std", "ground_truth_std", "best_epoch"))
    atomic_csv(summary / "expert_weights.csv", weight_rows,
               ("fold_id", "split", *[f"expert_{i}_weight" for i in range(1, 9)]))
    aggregate = {}
    for name in SPLIT_NAMES.values():
        rows = predictions_by_split[name]
        atomic_csv(summary / f"{name}_predictions.csv", rows, PREDICTION_FIELDS)
        source = "train" if name == "train" else "validation" if name == "validation" else "test"
        total = sum(row["model_id"] in pre["eligible"] for row in pre["rows"][source])
        aggregate[name] = aggregate_predictions(rows, eligible_folds=len(pre["eligible"]), total_samples=total)
    atomic_json(summary / "aggregate_metrics.json", aggregate)
    report = ["# MOE8 LOGO-Lite research results", "", f"Status: {'COMPLETE' if len(complete)==len(pre['eligible']) else 'PARTIAL'}",
              f"Completed folds: {len(complete)}/{len(pre['eligible'])}; failed or incomplete: {', '.join(failures) or 'none'}", "",
              "Project-defined gate-only research metrics; not official XACLE metrics.", "",
              "| Fold | Generator | Split | Samples | SRCC | LCC | KTAU | MSE |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in all_metrics:
        report.append("| " + " | ".join(str(row[key]) if row[key] is not None else "—" for key in
                    ("fold_id", "generator_name", "split", "samples", "srcc", "lcc", "ktau", "mse")) + " |")
    report.extend(["", "## Aggregate LOGO metrics", "", "| Metric | Train holdout | Validation | Public test |",
                   "|---|---:|---:|---:|"])
    for key in ("logo_macro_srcc", "logo_weighted_srcc", "logo_pooled_srcc", "logo_worst_srcc",
                "logo_srcc_std", "macro_lcc", "pooled_lcc", "macro_ktau", "pooled_ktau", "macro_mse",
                "pooled_mse", "evaluated_samples", "valid_srcc_folds"):
        report.append("| " + key + " | " + " | ".join(str(aggregate[name].get(key))
                       if aggregate[name].get(key) is not None else "—" for name in SPLIT_NAMES.values()) + " |")
    report.extend(["", "Frozen experts may have seen held-out generators during prior fine-tuning. Official train holdout labels select checkpoints, so train-holdout metrics are selection-dependent. LOGO-Lite tests gate-level transfer, not strict whole-model domain generalization. Aggregated fold predictions are research summaries, not an ensemble submission."])
    (summary / "report.md").write_text("\n".join(report) + "\n")
    from training.logo_plots import plot_summary
    plot_summary(output_root)
    return {"completed_folds": complete, "incomplete_folds": failures, "aggregate": aggregate,
            "report": str(summary / "report.md")}


def run_experiment(config: dict, *, dataset_root: str | None = None,
                   artifacts_root: str | None = None, output_dir: str | None = None,
                   generator_ids: list[str] | None = None, dry_run: bool = False,
                   resume: bool = False, device: str | None = None,
                   smoke_epochs: int | None = None, seed: int | None = None) -> dict:
    output_root = Path(output_dir or config["output_dir"]).resolve()
    if smoke_epochs is not None:
        if smoke_epochs <= 0:
            raise ValueError("smoke_epochs must be positive")
        output_root = output_root / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pre = preflight(config, dataset_root=dataset_root, artifacts_root=artifacts_root)
    except Exception as exc:
        atomic_json(output_root / "preflight_report.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        raise
    selected = generator_ids or pre["eligible"]
    if any(generator_id not in pre["eligible"] for generator_id in selected):
        raise ValueError(f"selected generators must be eligible: {selected}")
    effective_seed = config.get("seed", pre["baseline"]["runtime"]["seed"]) if seed is None else seed
    if type(effective_seed) is not int or effective_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True).stdout.strip()
    identity = {split: store.identity for split, store in pre["stores"].items()}
    metadata = {**CLASSIFICATION, "base_commit": pre["baseline_provenance"]["git_commit"],
                "implementation_commit": commit, "feature_store_identity": identity,
                "seed": effective_seed,
                "baseline_checkpoint_sha256": pre["baseline_provenance"].get("gate_checkpoint_sha256"),
                "baseline_config_sha256": hashlib.sha256(pre["baseline_config_path"].read_bytes()).hexdigest(),
                "eligible_generators": pre["eligible"], "selected_generators": selected,
                "source_splits": {split: len(rows) for split, rows in pre["rows"].items()},
                "score_contracts": pre["manifests"]["train"]["score_contracts"],
                "effective_training_configuration": pre["baseline"], "smoke_epochs": smoke_epochs}
    atomic_json(output_root / "experiment_metadata.json", metadata)
    (output_root / "experiment_config.yaml").write_text(yaml.safe_dump(
        {**CLASSIFICATION, "configuration": config, "effective_baseline": pre["baseline"]}, sort_keys=False))
    atomic_csv(output_root / "generator_statistics.csv", pre["statistics"],
               ("model_id", "model_name", "role", "train_samples", "validation_samples", "test_samples"))
    atomic_json(output_root / "preflight_report.json", {"status": "PASS", "dataset_root": str(pre["metadata_root"]),
        "store_root": str(pre["store_root"]), "eligible_generators": pre["eligible"],
        "control_model_ids": pre["controls"], "statistics": pre["statistics"],
        "audio_overlap": pre["overlap"], "feature_store_identity": identity})
    for generator_id in pre["eligible"]:
        folds = fold_indices(pre["rows"], generator_id)
        for role, indices in folds.items():
            source = "train" if role in {"optimization", "internal_holdout"} else "validation" if role == "validation" else "test"
            path = output_root / f"fold_{generator_id}" / "splits" / f"{role}.csv"
            atomic_csv(path, [{"sample_id": pre["rows"][source][i]["sample_id"], "source_split": source,
                               "fold_role": role, "model_id": pre["rows"][source][i]["model_id"]} for i in indices],
                       ("sample_id", "source_split", "fold_role", "model_id"))
    if dry_run:
        return {"status": "PREFLIGHT_PASS", "eligible_generators": pre["eligible"],
                "statistics": pre["statistics"], "output_root": str(output_root)}
    results = []
    for generator_id in selected:
        results.append(run_fold(pre, generator_id, output_root, device_name=device or config["device"],
                                resume=resume, smoke_epochs=smoke_epochs,
                                seed=effective_seed))
    summary = write_summary(pre, output_root)
    return {"status": "COMPLETE" if len(summary["completed_folds"]) == len(pre["eligible"]) else "PARTIAL",
            "folds": results, "summary": summary, "output_root": str(output_root)}
