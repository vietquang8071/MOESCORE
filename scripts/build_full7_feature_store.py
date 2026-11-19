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
from data.audio_overlap import (
    build_train_validation_audio_overlap_diagnostic,
    emit_audio_overlap_warning,
    raise_for_audio_overlap,
    validate_source_audio_records,
)


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
    parser.add_argument("--allow-train-validation-audio-overlap", action="store_true",
                        help="Allow identical WAV hashes between train and validation only (experimental)")
    args = parser.parse_args(argv)
    overlap_diagnostic = None
    test_overlap_diagnostic = {"overlapping_sha256_count": 0, "overlaps": []}
    try:
        splits = list(dict.fromkeys(args.split))
        # Check source WAV overlap before producing any completed split.
        records_by_split = {}
        available_splits = [name for name in ("train", "validation", "test")
                            if (Path(args.full9_root) / name / "manifest.json").is_file()]
        for split in available_splits:
            manifest = validate_feature_store(args.full9_root, split=split)
            records = _read_source_manifest(Path(args.full9_root) / split / "source_manifest.json",
                                            split=split, expected_sha256=manifest["source_manifest_sha256"])
            records_by_split[split] = validate_source_audio_records(records, split=split)
        overlap_diagnostic = build_train_validation_audio_overlap_diagnostic(
            records_by_split.get("train", []), records_by_split.get("validation", []),
            allow_train_validation_audio_overlap=args.allow_train_validation_audio_overlap,
        )
        if overlap_diagnostic["overlapping_sha256_count"]:
            if not args.allow_train_validation_audio_overlap:
                raise_for_audio_overlap(overlap_diagnostic, stage="Full7 builder")
            emit_audio_overlap_warning(overlap_diagnostic, stage="Full7 builder")
        test_records = records_by_split.get("test", [])
        if test_records:
            test_by_hash = {}
            for row in test_records:
                test_by_hash.setdefault(row["wav_sha256"], []).append(row["sample_id"])
            training_records = records_by_split.get("train", []) + records_by_split.get("validation", [])
            train_validation_by_hash = {}
            for row in training_records:
                train_validation_by_hash.setdefault(row["wav_sha256"], []).append(row["sample_id"])
            shared_test = sorted(set(test_by_hash) & set(train_validation_by_hash))
            if shared_test:
                test_overlap_diagnostic = {
                    "overlapping_sha256_count": len(shared_test),
                    "overlaps": [{
                        "wav_sha256": digest,
                        "train_or_validation_sample_ids": sorted(train_validation_by_hash[digest]),
                        "test_sample_ids": sorted(test_by_hash[digest]),
                    } for digest in shared_test],
                }
                digest = shared_test[0]
                raise ValueError(
                    "cross-split duplicate WAV SHA-256 involving test: "
                    f"wav_sha256={digest}, train_or_validation={train_validation_by_hash[digest]}, "
                    f"test={test_by_hash[digest]}"
                )
        reports = []
        for split in splits:
            audio_dir = Path(args.source_audio_root) / split
            if (audio_dir / "wav").is_dir():
                audio_dir = audio_dir / "wav"
            reports.append(build_full7_split(args.full9_root, args.whisper_root,
                           args.e5_checkpoint, args.e6_checkpoint, args.output_root,
                           split=split, source_audio_dir=audio_dir,
                           shard_size=args.shard_size))
        report = {"status": "PASS", "splits": reports,
                  "audio_overlap_diagnostic": overlap_diagnostic,
                  "test_overlap_diagnostic": test_overlap_diagnostic}
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        if overlap_diagnostic is not None:
            report["audio_overlap_diagnostic"] = overlap_diagnostic
        if test_overlap_diagnostic["overlapping_sha256_count"]:
            report["test_overlap_diagnostic"] = test_overlap_diagnostic
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        path = Path(args.report_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
