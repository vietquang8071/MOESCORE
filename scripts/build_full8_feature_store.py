#!/usr/bin/env python3
"""Build a verified Full8 store by appending E8 cache shards to Full7 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.full8_assembler import build_full8_stores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full7-root", required=True)
    parser.add_argument("--expert8-cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expert8-cache-id")
    parser.add_argument("--split", action="append", choices=("train", "validation", "test"))
    parser.add_argument("--allow-train-validation-audio-overlap", action="store_true",
                        help="Allow known train/validation WAV duplicates for this research run only")
    parser.add_argument("--allow-public-split-audio-overlap", action="store_true",
                        help="D01 diagnostic-only: retain and allow identical WAV hashes across official public splits")
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    try:
        report = build_full8_stores(
            args.full7_root, args.expert8_cache_root, args.output_root,
            allow_train_validation_audio_overlap=args.allow_train_validation_audio_overlap,
            allow_public_split_audio_overlap=args.allow_public_split_audio_overlap,
            splits=tuple(dict.fromkeys(args.split or ("train", "validation", "test"))),
            expert8_cache_id=args.expert8_cache_id,
        )
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        target = Path(args.report_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
