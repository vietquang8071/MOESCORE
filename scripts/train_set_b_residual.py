#!/usr/bin/env python3
"""Train E6 for exactly 30 epochs using official train and validation caches."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.set_b_cached_dataset import SetBCachedDataset
from model.composite_experts.set_b_residual import SetBResidualHead
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter
from training.set_b_config import load_set_b_config
from training.set_b_trainer import SetBResidualTrainer, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config = load_set_b_config(args.config)
    def dataset(split):
        return SetBCachedDataset(config["feature_store_root"], config["whisper_cache_root"], split=split,
            source_manifest_path=config["source_manifest_paths"].get(split),
            source_audio_dir=config["source_audio_dirs"].get(split))
    train = dataset("train")
    validation = dataset("validation")
    seed_everything(config["training"]["seed"])
    model = SetBResidualAdapter(SetBResidualHead(**config["model"]))
    t = config["training"]
    trainer = SetBResidualTrainer(model, learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
        gradient_clip=t["gradient_clip"], seed=t["seed"], device=config["runtime"]["device"])
    history = trainer.fit(train, validation, output_dir=config["output_dir"], epochs=t["epochs"],
        batch_size=t["batch_size"], num_workers=t["num_workers"], resume_from=args.resume,
        allow_train_validation_audio_overlap=t["allow_train_validation_audio_overlap"])
    report = {"completed_epochs": trainer.completed_epochs, "global_step": trainer.global_step,
              "train_sample_count": len(train), "validation_sample_count": len(validation),
              "best_epoch": trainer.best_epoch, "best_validation_mse": trainer.best_validation_mse,
              "audio_overlap_diagnostic": trainer.audio_overlap_diagnostic, "history": history}
    if trainer.audio_overlap_diagnostic["overlapping_sha256_count"]:
        report["validation_metrics_warning"] = trainer.audio_overlap_diagnostic["warning"]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
