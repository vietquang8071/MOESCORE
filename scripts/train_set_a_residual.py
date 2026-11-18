#!/usr/bin/env python3
"""Train the E5 Set A residual head from an existing verified full9 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import SetAResidualHead
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter
from training.set_a_config import load_set_a_config
from training.set_a_trainer import SetAResidualTrainer, set_deterministic_seed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config = load_set_a_config(args.config)
    training = config["training"]
    source_manifest = config["source_manifest_paths"].get("train")
    dataset = SetACachedDataset(
        config["feature_store_root"],
        split="train",
        source_manifest_path=source_manifest,
    )
    set_deterministic_seed(training["seed"])
    model = SetAResidualAdapter(
        SetAResidualHead(
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
    )
    trainer = SetAResidualTrainer(
        model,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        gradient_clip=training["gradient_clip"],
        seed=training["seed"],
        device=config["runtime"]["device"],
        holdout_fraction=training["holdout_fraction"],
        early_stopping_patience=training["early_stopping_patience"],
        min_delta=training["min_delta"],
    )
    resume = args.resume or config["resume_from"]
    history = trainer.fit(
        dataset,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        output_dir=config["output_dir"],
        resume_from=resume,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "epochs_completed": trainer.completed_epochs,
                "global_step": trainer.global_step,
                "latest_checkpoint": str(config["output_dir"] / "expert5_latest.pt"),
                "best_checkpoint": str(config["output_dir"] / "expert5_best.pt"),
                "final_checkpoint": str(config["output_dir"] / "expert5_final.pt"),
                "final_checkpoint_role": "best holdout-MSE model; latest is the resumable last epoch",
                "history": history,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
