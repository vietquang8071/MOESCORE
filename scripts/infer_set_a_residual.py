#!/usr/bin/env python3
"""Run E5 inference from a verified component cache and an E5 checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import SetAResidualHead
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from training.set_a_config import load_set_a_config
from training.set_a_trainer import _resolve_device, load_set_a_checkpoint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _move_components(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_components(item, device) for key, item in value.items()}
    return value


def main() -> int:
    args = _arguments()
    config = load_set_a_config(args.config)
    source_manifest = config["source_manifest_paths"].get(args.split)
    dataset = SetACachedDataset(
        config["feature_store_root"],
        split=args.split,
        source_manifest_path=source_manifest,
    )
    batch_size = args.batch_size or config["training"]["batch_size"]
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = _resolve_device(config["runtime"]["device"])
    model = SetAResidualAdapter(
        SetAResidualHead(
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
    ).to(device)
    checkpoint = load_set_a_checkpoint(
        args.checkpoint,
        model=model,
        expected_dataset=dataset,
        map_location=device,
    )
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "wave_file_name", "score_normalized", "score_raw"),
        )
        writer.writeheader()
        with torch.inference_mode():
            for batch in loader:
                component_outputs = _move_components(batch["component_outputs"], device)
                scores, _ = model({"component_outputs": component_outputs})
                for sample_id, wav_name, score in zip(
                    batch["sample_id"], batch["wave_file_name"], scores[:, 0].cpu().tolist()
                ):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "wave_file_name": wav_name,
                            "score_normalized": score,
                            "score_raw": score * 5.0 + 5.0,
                        }
                    )
                    count += 1
    print(
        f"E5 inference complete: split={args.split} rows={count} "
        f"checkpoint_epoch={checkpoint['epoch']} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
