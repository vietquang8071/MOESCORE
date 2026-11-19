#!/usr/bin/env python3
"""Build verified Full7 stores from existing Full9/Whisper caches and trusted heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.full7_assembler import build_full7_split
from data.set_a_cached_dataset import _read_source_manifest
from data.moe_feature_store import validate_feature_store


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full9-root", required=True)
    parser.add_argument("--whisper-root", required=True)
    parser.add_argument("--e5-checkpoint", required=True)
    parser.add_argument("--e6-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-audio-root", required=True,
                        help="Directory containing train/validation/test WAV subdirectories")
    parser.add_argument("--split", choices=("train", "validation", "test"), action="append", required=True)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    try:
        splits = list(dict.fromkeys(args.split))
        # Check source WAV overlap before producing any completed split.
        hashes = {}
        available_splits = [name for name in ("train", "validation", "test")
                            if (Path(args.full9_root) / name / "manifest.json").is_file()]
        for split in available_splits:
            manifest = validate_feature_store(args.full9_root, split=split)
            records = _read_source_manifest(Path(args.full9_root) / split / "source_manifest.json",
                                            split=split, expected_sha256=manifest["source_manifest_sha256"])
            for record in records:
                digest = record["wav_sha256"].lower()
                if digest in hashes and hashes[digest] != split:
                    raise ValueError(f"cross-split duplicate WAV SHA-256: {digest}")
                hashes[digest] = split
        reports = []
        for split in splits:
            audio_dir = Path(args.source_audio_root) / split
            if (audio_dir / "wav").is_dir():
                audio_dir = audio_dir / "wav"
            reports.append(build_full7_split(args.full9_root, args.whisper_root,
                           args.e5_checkpoint, args.e6_checkpoint, args.output_root,
                           split=split, source_audio_dir=audio_dir,
                           shard_size=args.shard_size))
        report = {"status": "PASS", "splits": reports}
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        path = Path(args.report_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
