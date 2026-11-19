#!/usr/bin/env python3
"""Run label-free gating inference and export the upstream CSV schema."""

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
from data.moe_feature_store import preprocessing_contract
from model.moe_gating_core import GatingNetwork
from model.expert_adapters.registry import profile_contract
from training.checkpoint_manager import MoeCheckpointManager
from training.config import load_moe_training_config
from training.moe_evaluator import MoeGateEvaluator, write_submission_csv


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device("cuda:0" if value == "cuda" else value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _reject_test_wav_overlap(checkpoint_identity: dict, test_manifest: dict) -> None:
    saved = checkpoint_identity.get("source_wav_sha256_by_split")
    if not isinstance(saved, dict) or set(saved) != {"train", "validation"}:
        raise ValueError("Full7/Full8 checkpoint lacks train/validation WAV fingerprint provenance")
    source_rows = test_manifest["assembly_identity"].get("ordered_source_fingerprints")
    if not isinstance(source_rows, list):
        raise ValueError("Full7/Full8 test store lacks source WAV fingerprints")
    earlier = {digest.lower() for split in ("train", "validation") for digest in saved[split]}
    overlap = earlier.intersection(row["wav_sha256"].lower() for row in source_rows)
    if overlap:
        raise ValueError(f"cross-split duplicate WAV SHA-256 in test: {sorted(overlap)[0]}")


def infer_checkpoint(config: dict, checkpoint: str, output: str, *, device_override: str | None = None) -> dict:
    device = _device(device_override or config["runtime"]["device"])
    dataset = MoeTrainingDataset(config["data"]["feature_store_root"], split="test", require_labels=False)
    profile = dataset.manifest.get("expert_profile", "top1_4")
    if config["data"].get("profile", profile) != profile:
        raise ValueError(f"configured expert profile differs from {profile} test feature store")
    order, _, width = profile_contract(profile)
    gate = GatingNetwork(input_feature_dim=width, num_experts=len(order), profile=profile).to(device)
    payload = MoeCheckpointManager().load_for_inference(
        checkpoint,
        gating_network=gate,
        expected_expert_checkpoint_provenance=dataset.manifest["checkpoint_provenance"],
    )
    expected_contract = payload["feature_store_identity"].get("preprocessing_contract")
    observed_contract = preprocessing_contract(dataset.manifest)
    if expected_contract != observed_contract:
        raise ValueError("test preprocessing contract differs from training checkpoint")
    if profile in {"full_7", "full_8"}:
        _reject_test_wav_overlap(payload["feature_store_identity"], dataset.manifest)
    loader = DataLoader(dataset, batch_size=config["inference"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_moe_features)
    rows = MoeGateEvaluator(gate).predict(loader)
    csv_path = write_submission_csv(rows, output,
                                    filename_column="wave_file_name" if profile in {"full_7", "full_8"} else "wav_file_name")
    return {
        "status": "PASS",
        "device": str(device),
        "sample_count": len(rows),
        "checkpoint_epoch": payload["epoch"],
        "output_csv": str(csv_path),
        "labels_required": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = infer_checkpoint(load_moe_training_config(args.config), args.checkpoint, args.output, device_override=args.device)
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
