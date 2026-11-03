#!/usr/bin/env python3
"""Evaluate a gating checkpoint on one labeled feature-store split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

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


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device("cuda:0" if value == "cuda" else value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def evaluate_checkpoint(config: dict, checkpoint: str, *, device_override: str | None = None) -> dict:
    device = _device(device_override or config["runtime"]["device"])
    dataset = MoeTrainingDataset(config["data"]["feature_store_root"], split="validation", require_labels=True)
    gate = GatingNetwork().to(device)
    payload = MoeCheckpointManager().load_for_inference(
        checkpoint,
        gating_network=gate,
        expected_expert_checkpoint_provenance=dataset.manifest["checkpoint_provenance"],
    )
    expected_contract = payload["feature_store_identity"].get("preprocessing_contract")
    observed_identity = dataset.manifest["preprocessing_identity"]
    observed_contract = observed_identity.get("contract") if isinstance(observed_identity, dict) else observed_identity
    if expected_contract != observed_contract:
        raise ValueError("validation preprocessing contract differs from training checkpoint")
    loader = DataLoader(dataset, batch_size=config["validation"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_moe_features)
    result = MoeGateEvaluator(gate, config["training"]["loss"]).evaluate(loader)
    return {
        "status": "PASS",
        "device": str(device),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": payload["epoch"],
        "sample_count": len(dataset),
        "metrics_raw_scale": result["metrics"],
        "regression_loss_normalized": result.get("regression_loss_normalized"),
        "optimizer_steps": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = evaluate_checkpoint(load_moe_training_config(args.config), args.checkpoint, device_override=args.device)
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
