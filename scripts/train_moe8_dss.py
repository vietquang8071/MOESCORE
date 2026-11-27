#!/usr/bin/env python3
"""Train a fresh DSS gate from verified Full8 cache and official metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dss_runner import run_dss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    args = parser.parse_args()
    run_dss(args.dataset_root, args.artifacts_root, args.output_dir,
            run_mode=args.run_mode, device_name=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
