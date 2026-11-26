#!/usr/bin/env python3
"""Preflight, train, resume, and summarize Full8 LOGO-Lite folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.logo_runner import preflight, run_experiment, write_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generator-id", action="append", dest="generator_ids")
    parser.add_argument("--all-generators", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    args = parser.parse_args(argv)
    if args.all_generators and args.generator_ids:
        parser.error("choose --all-generators or --generator-id, not both")
    config = yaml.safe_load(Path(args.config).read_text())
    required = {"output_dir", "device", "control_model_names", "allow_train_validation_audio_overlap",
                "allow_public_split_audio_overlap"}
    if not isinstance(config, dict) or not required <= set(config):
        parser.error(f"LOGO configuration lacks {sorted(required-set(config or {}))}")
    if args.aggregate_only:
        selected = preflight(config, dataset_root=args.dataset_root, artifacts_root=args.artifacts_root)
        result = write_summary(selected, Path(args.output_dir or config["output_dir"]))
    else:
        result = run_experiment(config, dataset_root=args.dataset_root, artifacts_root=args.artifacts_root,
                                output_dir=args.output_dir, generator_ids=args.generator_ids,
                                dry_run=args.dry_run, resume=args.resume, device=args.device,
                                smoke_epochs=args.smoke_epochs, seed=args.seed)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
