#!/usr/bin/env python3
"""Run one real-checkpoint MGA extraction with the original Top 2 16 kHz input."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import tempfile

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.expert_adapters.set_a_mga_source_adapter import SetAMGASourceAdapter
from scripts.smoke_test_top2_real_inference import _load_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "expert2_finetune" / "MGA-CLAP")
    parser.add_argument(
        "--architecture-yaml",
        type=Path,
        default=PROJECT_ROOT / "expert2_finetune" / "MGA-CLAP" / "settings" / "inference_example.yaml",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    config = {
        "top2_mga": {
            "pretrained_checkpoint": str(args.checkpoint.expanduser().resolve()),
            "source_dir": str(args.source_dir.expanduser().resolve()),
            "architecture_yaml": str(args.architecture_yaml.expanduser().resolve()),
        },
        "runtime": {"device": device, "offline": True},
    }
    batch, audio_path, caption = _load_batch(
        args.dataset_root.expanduser().resolve(), args.split, args.sample_index
    )
    previous_numba_cache = os.environ.get("NUMBA_CACHE_DIR")
    with tempfile.TemporaryDirectory(
        prefix=".set-a-mga-numba-cache-", dir=PROJECT_ROOT
    ) as numba_cache:
        os.environ["NUMBA_CACHE_DIR"] = numba_cache
        adapter = None
        try:
            adapter = SetAMGASourceAdapter.from_config(config, device=device)
            score, features = adapter(batch)
            if tuple(score.shape) != (1, 1) or tuple(features.shape) != (1, 4097):
                raise AssertionError(
                    f"unexpected source-compatible MGA output shapes: {score.shape}, {features.shape}"
                )
            if score.dtype != torch.float32 or features.dtype != torch.float32:
                raise AssertionError(f"expected float32 outputs, got {score.dtype}, {features.dtype}")
            if not torch.isfinite(score).all().item() or not torch.isfinite(features).all().item():
                raise AssertionError("source-compatible MGA output contains NaN or infinity")
            if adapter.model.training or any(parameter.requires_grad for parameter in adapter.model.parameters()):
                raise AssertionError("source-compatible MGA model is not frozen in eval mode")
            print(
                f"Set A source-compatible MGA PASS sample={args.split}:{args.sample_index} "
                f"audio={audio_path} sample_rate_hz=16000 samples=160000 "
                f"caption={caption!r} score={float(score.item()):.8f} "
                f"score_shape={tuple(score.shape)} feature_shape={tuple(features.shape)} "
                f"device={score.device} frozen=True"
            )
        finally:
            if adapter is not None:
                del adapter
            gc.collect()
            if previous_numba_cache is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = previous_numba_cache
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
