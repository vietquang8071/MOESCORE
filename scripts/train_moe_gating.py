#!/usr/bin/env python3
"""Train only the LOCAL-10 gate from validated frozen-expert feature stores."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager
from training.config import load_moe_training_config
from training.moe_evaluator import MoeGateEvaluator
from training.moe_trainer import MoeGateTrainer
from training.sampler import StatefulRandomSampler


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device("cuda:0" if value == "cuda" else value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.pop("config_path", None)
    value.pop("output_dir", None)
    value.get("data", {}).pop("feature_store_root", None)
    value.get("runtime", {}).pop("device", None)
    return value


def _store_provenance(train: MoeTrainingDataset, validation: MoeTrainingDataset) -> tuple[dict[str, Any], dict[str, str]]:
    train_manifest, val_manifest = train.manifest, validation.manifest
    train_provenance = train_manifest["checkpoint_provenance"]
    if train_provenance != val_manifest["checkpoint_provenance"]:
        raise ValueError("train and validation feature stores use different expert checkpoint provenance")
    def contract(manifest: dict[str, Any]) -> Any:
        identity = manifest["preprocessing_identity"]
        return identity.get("contract") if isinstance(identity, dict) else identity
    train_contract, val_contract = contract(train_manifest), contract(val_manifest)
    if train_contract != val_contract:
        raise ValueError("train and validation preprocessing contracts differ")
    identity = {
        "train": train.identity,
        "validation": validation.identity,
        "preprocessing_contract": train_contract,
    }
    return train_provenance, identity


def run_training(config: dict[str, Any], *, resume: str | None = None, device_override: str | None = None) -> dict[str, Any]:
    device = _device(device_override or config["runtime"]["device"])
    torch.manual_seed(config["runtime"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["runtime"]["seed"])
    store_root = config["data"]["feature_store_root"]
    train_ds = MoeTrainingDataset(store_root, split="train", require_labels=True)
    validation_ds = MoeTrainingDataset(store_root, split="validation", require_labels=True)
    checkpoint_provenance, store_identity = _store_provenance(train_ds, validation_ds)
    sampler = StatefulRandomSampler(train_ds, seed=config["runtime"]["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_moe_features,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_moe_features,
    )
    gate = GatingNetwork().to(device)
    trainer = MoeGateTrainer(
        gate,
        config["training"]["loss"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        diversity_weight=config["training"]["diversity_weight"],
    )
    evaluator = MoeGateEvaluator(gate, config["training"]["loss"])
    manager = MoeCheckpointManager()
    run_config = _checkpoint_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_gate.pt"
    best_path = output_dir / "best_gate.pt"
    start_epoch = 0
    best_srcc: float | None = None
    patience = 0
    if resume:
        state = manager.load(
            resume,
            gating_network=gate,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            expected_config=run_config,
            expected_feature_store_identity=store_identity,
            expected_expert_checkpoint_provenance=checkpoint_provenance,
        )
        trainer.global_step = int(state["global_step"])
        best_srcc = state.get("best_srcc")
        patience = int(state.get("patience", 0))
        sampler_state = state.get("sampler_state")
        if sampler_state is not None:
            sampler.load_state_dict(sampler_state)
            if sampler.cursor == len(sampler.order):
                start_epoch = sampler.epoch + 1
                sampler.set_epoch(start_epoch)
            else:
                start_epoch = sampler.epoch
        else:
            start_epoch = int(state["epoch"]) + 1
            sampler.set_epoch(start_epoch)

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        if sampler.epoch != epoch:
            sampler.set_epoch(epoch)
        gate.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            result = trainer.train_step(batch)
            train_loss_sum += float(result["total_loss"].item())
            train_batches += 1
            interval = config["training"]["checkpoint_interval_steps"]
            if trainer.global_step % interval == 0:
                manager.save(
                    latest_path,
                    gating_network=gate,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    epoch=epoch,
                    global_step=trainer.global_step,
                    config=run_config,
                    feature_store_identity=store_identity,
                    expert_checkpoint_provenance=checkpoint_provenance,
                    sampler_state=sampler.state_dict(),
                    best_srcc=best_srcc,
                    patience=patience,
                    seed=config["runtime"]["seed"],
                )
        if train_batches == 0:
            raise ValueError("training loader produced no batches; reduce batch_size or check train store")
        validation = evaluator.evaluate(validation_loader)
        validation_loss = float(validation.get("regression_loss_normalized", validation["metrics"]["mse"]))
        trainer.scheduler_step(validation_loss)
        srcc = validation["metrics"]["srcc"]
        improved = srcc is not None and (best_srcc is None or srcc > best_srcc)
        if improved:
            best_srcc, patience = srcc, 0
            manager.save(
                best_path,
                gating_network=gate,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                global_step=trainer.global_step,
                config=run_config,
                feature_store_identity=store_identity,
                expert_checkpoint_provenance=checkpoint_provenance,
                sampler_state=sampler.state_dict(),
                best_srcc=best_srcc,
                patience=patience,
                seed=config["runtime"]["seed"],
            )
        elif srcc is not None:
            patience += 1
        history.append({
            "epoch": epoch,
            "global_step": trainer.global_step,
            "train_loss": train_loss_sum / train_batches,
            "validation_loss_normalized": validation_loss,
            "validation_metrics_raw": validation["metrics"],
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch + 1}: step={trainer.global_step}, train_loss={history[-1]['train_loss']:.6f}, "
              f"validation={validation['metrics']}, best_srcc={best_srcc}", file=sys.stderr, flush=True)
        sampler.set_epoch(epoch + 1)
        manager.save(
            latest_path,
            gating_network=gate,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=epoch + 1,
            global_step=trainer.global_step,
            config=run_config,
            feature_store_identity=store_identity,
            expert_checkpoint_provenance=checkpoint_provenance,
            sampler_state=sampler.state_dict(),
            best_srcc=best_srcc,
            patience=patience,
            seed=config["runtime"]["seed"],
        )
        if patience >= config["training"]["early_stop_patience"] > 0:
            break
    return {
        "status": "PASS",
        "device": str(device),
        "epochs_completed": len(history),
        "global_step": trainer.global_step,
        "best_srcc_raw_scale": best_srcc,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path),
        "history": history,
        "expert_model_construction": False,
        "expert_checkpoint_deserialization": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = run_training(load_moe_training_config(args.config), resume=args.resume, device_override=args.device)
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        target = Path(args.report_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
