#!/usr/bin/env python3
"""Predict E6 normalized and raw scores from a verified joined feature cache."""

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.set_b_cached_dataset import SetBCachedDataset
from model.composite_experts.set_b_residual import SetBResidualHead
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter
from training.set_b_config import load_set_b_config
from training.set_b_trainer import load_set_b_checkpoint, move_batch, resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_set_b_config(args.config)
    dataset = SetBCachedDataset(config["feature_store_root"], config["whisper_cache_root"], split=args.split,
        source_manifest_path=config["source_manifest_paths"].get(args.split),
        source_audio_dir=config["source_audio_dirs"].get(args.split))
    device = resolve_device(config["runtime"]["device"])
    model = SetBResidualAdapter(SetBResidualHead(**config["model"])).to(device)
    load_set_b_checkpoint(args.checkpoint, model=model, expected_dataset=dataset, map_location=device)
    model.eval()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "wave_file_name", "score_normalized", "score_raw"))
        writer.writeheader()
        with torch.inference_mode():
            for row in DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False):
                scores, _ = model(move_batch(row, device))
                for sample_id, name, score in zip(row["sample_id"], row["wave_file_name"], scores[:, 0].cpu().tolist()):
                    writer.writerow({"sample_id": sample_id, "wave_file_name": name,
                                     "score_normalized": score, "score_raw": 5.0 * score + 5.0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
