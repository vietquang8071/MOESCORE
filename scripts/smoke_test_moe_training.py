#!/usr/bin/env python3
"""Run a bounded gate-only smoke test on a validated feature store."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from model.moe_gating_core import GatingNetwork
from scripts.build_moe_feature_store import import_verified_outputs
from training.checkpoint_manager import MoeCheckpointManager
from training.moe_trainer import MoeGateTrainer
from training.target_transform import normalize_targets, score_interval_feasibility
from torch.utils.data import DataLoader


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform != "darwin" else 1))


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def run_smoke(
    feature_store: str | os.PathLike[str],
    *,
    artifact_dir: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    device: str = "cpu",
    batch_size: int = 2,
    max_steps: int = 3,
    seed: int = 2027,
) -> dict:
    if device != "cpu":
        raise ValueError("LOCAL-11 training smoke is restricted to CPU")
    if type(batch_size) is not int or batch_size <= 0 or batch_size > 4:
        raise ValueError("batch_size must be between 1 and 4")
    if type(max_steps) is not int or not 1 <= max_steps <= 3:
        raise ValueError("max_steps must be between 1 and 3")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    source_import = import_verified_outputs(
        artifact_dir=artifact_dir,
        output_root=feature_store,
        split="train",
        shard_size=2,
        dataset_root=dataset_root,
    )

    dataset = MoeTrainingDataset(feature_store, split="train", require_labels=True)
    if len(dataset) > 4:
        raise ValueError("LOCAL-11 real smoke accepts at most four verified training samples")
    if len(dataset) == 0:
        raise ValueError("feature store is empty")
    torch.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_moe_features,
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= max_steps:
            break
    if not batches:
        raise ValueError("training loader yielded no batches")

    manifest = dataset.manifest
    provenance = manifest["checkpoint_provenance"]
    feature_identity = {"train": dataset.identity}
    run_config = {
        "seed": seed,
        "device": "cpu",
        "loss": {
            "loss_weights": [1.0, -0.5],
            "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
            "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False},
        },
        "diversity_weight": 0.2,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
    }
    gate = GatingNetwork().cpu()
    trainer = MoeGateTrainer(
        gate,
        run_config["loss"],
        learning_rate=run_config["learning_rate"],
        weight_decay=run_config["weight_decay"],
        diversity_weight=run_config["diversity_weight"],
    )
    gate_before = [parameter.detach().clone() for parameter in gate.parameters()]
    input_copies = [
        (batch["scores"].clone(), batch["features"].clone(), batch["labels_raw"].clone())
        for batch in batches
    ]
    step_values: list[dict] = []
    for batch in batches:
        result = trainer.train_step(batch)
        step_values.append({
            "global_step": result["global_step"],
            "prediction_shape": list(result["prediction"].shape),
            "normalized_target_shape": list(result["targets_normalized"].shape),
            "clipped_mse": float(result["clipped_mse_loss"].item()),
            "contrastive": float(result["contrastive_loss"].item()),
            "regression": float(result["regression_loss"].item()),
            "diversity": float(result["diversity_loss"].item()),
            "total": float(result["total_loss"].item()),
            "gradient_norm": float(result["gradient_norm"].item()),
            "updated_parameter_count": result["updated_parameter_count"],
        })
    if trainer.global_step > max_steps:
        raise RuntimeError("smoke test exceeded the requested optimizer-step limit")
    updated_gate_parameters = sum(
        not torch.equal(before, after.detach())
        for before, after in zip(gate_before, gate.parameters())
    )
    inputs_unchanged = all(
        torch.equal(old_scores, batch["scores"])
        and torch.equal(old_features, batch["features"])
        and torch.equal(old_labels, batch["labels_raw"])
        for (old_scores, old_features, old_labels), batch in zip(input_copies, batches)
    )
    if not inputs_unchanged or updated_gate_parameters == 0:
        raise RuntimeError("smoke test failed input immutability or gating update verification")

    all_scores = torch.cat([batch["scores"] for batch in batches], dim=0)
    all_labels = torch.cat([batch["labels_raw"] for batch in batches], dim=0)
    normalized_labels = normalize_targets(all_labels)
    feasibility = score_interval_feasibility(all_scores, normalized_labels)
    row_lower = all_scores.min(dim=1).values
    row_upper = all_scores.max(dim=1).values
    row_distance = torch.maximum(row_lower - normalized_labels, normalized_labels - row_upper).clamp_min(0)
    selected_ids = [sample_id for batch in batches for sample_id in batch["sample_ids"]]
    feasibility["per_sample"] = [
        {
            "sample_id": sample_id,
            "label_raw": float(raw.item()),
            "target_normalized": float(target.item()),
            "expert_score_min": float(lower.item()),
            "expert_score_max": float(upper.item()),
            "outside_interval_distance": float(distance.item()),
        }
        for sample_id, raw, target, lower, upper, distance in zip(
            selected_ids, all_labels, normalized_labels, row_lower, row_upper, row_distance
        )
    ]
    checkpoint_manager = MoeCheckpointManager()
    gate.eval()
    with torch.inference_mode():
        reference_weights = gate(torch.cat([batch["features"] for batch in batches], dim=0))
    with tempfile.TemporaryDirectory(prefix="moescore-local11-smoke-") as temporary:
        checkpoint_path = Path(temporary) / "gate_smoke.pt"
        checkpoint_manager.save(
            checkpoint_path,
            gating_network=gate,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=0,
            global_step=trainer.global_step,
            config=run_config,
            feature_store_identity=feature_identity,
            expert_checkpoint_provenance=provenance,
            seed=seed,
        )
        restored_gate = GatingNetwork().cpu()
        restored_trainer = MoeGateTrainer(
            restored_gate,
            run_config["loss"],
            learning_rate=run_config["learning_rate"],
            weight_decay=run_config["weight_decay"],
            diversity_weight=run_config["diversity_weight"],
        )
        checkpoint = checkpoint_manager.load(
            checkpoint_path,
            gating_network=restored_gate,
            optimizer=restored_trainer.optimizer,
            scheduler=restored_trainer.scheduler,
            expected_config=run_config,
            expected_feature_store_identity=feature_identity,
            expected_expert_checkpoint_provenance=provenance,
            restore_rng=True,
        )
        restored_trainer.global_step = int(checkpoint["global_step"])
        restored_gate.eval()
        with torch.inference_mode():
            restored_weights = restored_gate(torch.cat([batch["features"] for batch in batches], dim=0))
        state_equal = all(
            torch.equal(value, restored_gate.state_dict()[name])
            for name, value in gate.state_dict().items()
        )
        optimizer_equal = _tree_equal(trainer.optimizer.state_dict(), restored_trainer.optimizer.state_dict())
        scheduler_equal = _tree_equal(trainer.scheduler.state_dict(), restored_trainer.scheduler.state_dict())
        if not state_equal or not optimizer_equal or not scheduler_equal or checkpoint["global_step"] != trainer.global_step:
            raise RuntimeError("gating checkpoint round-trip did not preserve the training state")
        torch.testing.assert_close(reference_weights, restored_weights, atol=0, rtol=0)

    return {
        "status": "PASS",
        "input_source": "REAL_VERIFIED_OUTPUTS",
        "feature_store_root": str(Path(feature_store).expanduser().resolve()),
        "feature_store_identity": dataset.identity,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_artifact_import": source_import,
        "checkpoint_provenance_evidence": "LOCAL-10.1 reports/artifact hashes plus checkpoint path/size/mtime; checkpoint files not reloaded",
        "sample_count": sum(len(batch["sample_ids"]) for batch in batches),
        "sample_ids": selected_ids,
        "batch_size": batch_size,
        "optimizer_steps": trainer.global_step,
        "target_normalization": "(raw_score - 5) / 5; original labels preserved in store",
        "step_results": step_values,
        "updated_gating_parameter_count": updated_gate_parameters,
        "expert_outputs_unchanged": inputs_unchanged,
        "score_interval_feasibility": feasibility,
        "checkpoint_save": "PASS_TEMPORARY_GATE_ONLY",
        "checkpoint_reload": "PASS_STRICT_STATE_OPTIMIZER_SCHEDULER_AND_STEP",
        "resume_status": "PASS_ROUND_TRIP; exact continuation separately verified by unit test",
        "restored_weights_exact": True,
        "real_expert_model_construction": False,
        "real_expert_checkpoint_deserialization": False,
        "cuda_usage": False,
        "device": "cpu",
        "process_peak_rss_bytes": _rss_bytes(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(
            args.feature_store,
            artifact_dir=args.artifact_dir,
            dataset_root=args.dataset_root,
            device=args.device,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "input_source": "UNVERIFIED",
            "error": f"{type(exc).__name__}: {exc}",
            "real_expert_model_construction": False,
            "real_expert_checkpoint_deserialization": False,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        output_path = Path(args.report_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
