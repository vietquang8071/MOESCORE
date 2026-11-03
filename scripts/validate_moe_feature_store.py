#!/usr/bin/env python3
"""Validate one split of a verified frozen-expert feature store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_feature_store import validate_feature_store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        manifest = validate_feature_store(args.root, split=args.split, require_labels=args.require_labels)
        report = {
            "status": "PASS",
            "split": args.split,
            "sample_count": sum(shard["count"] for shard in manifest["shards"]),
            "shard_count": len(manifest["shards"]),
            "scores_shape": [sum(shard["count"] for shard in manifest["shards"]), 4],
            "features_shape": [sum(shard["count"] for shard in manifest["shards"]), 7936],
            "store_identity_sha256": manifest["store_identity_sha256"],
            "labels_present": manifest["labels_present"],
            "expert_order": manifest["expert_order"],
        }
    except Exception as exc:
        report = {"status": "FAIL", "split": args.split, "error": f"{type(exc).__name__}: {exc}"}
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
