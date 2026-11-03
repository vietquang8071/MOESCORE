#!/usr/bin/env python3
"""Lightweight, model-free preflight for the unified XACLE data pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

import numpy as np
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.moe_config import load_moe_config, resolve_moe_paths
from data.unified_collate import collate_unified
from data.unified_dataset import UnifiedXACLEDataset, inspect_metadata_csv


def _expected_header(headers: list[str], *, split: str) -> dict[str, Any]:
    normalized = {name.strip().lower() for name in headers}
    filename = [name for name in ("wav_file_name", "wave_file_name") if name in normalized]
    score = [
        name
        for name in (
            "average_score",
            "average_semantic_alignement_score",
            "average_semantic_alignment_score",
        )
        if name in normalized
    ]
    missing = []
    if len(filename) != 1:
        missing.append("one wave filename column")
    if "text" not in normalized:
        missing.append("text")
    if split in {"train", "validation"} and len(score) != 1:
        missing.append("one average score column")
    if len(score) > 1:
        missing.append("unambiguous average score column")
    return {
        "required_columns_status": "PASS" if not missing else "FAIL",
        "filename_column": filename[0] if len(filename) == 1 else None,
        "score_column": score[0] if len(score) == 1 else None,
        "optional_model_metadata": [
            header for header in headers if header.strip().lower() in {"model_id", "model_name"}
        ],
        "missing_or_ambiguous": missing,
    }


def _synthetic_checks() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moescore-local09-") as temporary:
        root = Path(temporary)
        audio_dir = root / "wav" / "validation"
        audio_dir.mkdir(parents=True)
        manifest = root / "validation_average.csv"
        columns = ["wave_file_name", "text", "average_semantic_alignement_score", "model_ID", "model_name"]
        rows = []
        for index in range(4):
            name = f"fixture-{index}.wav"
            samples = (((np.arange(8_000, dtype=np.int32) + index * 37) % 1000) - 500).astype(np.int16)
            wavfile.write(audio_dir / name, 16_000, samples)
            rows.append(
                {
                    "wave_file_name": name,
                    "text": f"  synthetic caption {index}  ",
                    "average_semantic_alignement_score": str(5 + index),
                    "model_ID": f"fixture-model-{index}",
                    "model_name": f"Fixture Model {index}",
                }
            )
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        dataset = UnifiedXACLEDataset(manifest, audio_dir, split="validation", seed=2027)
        results: dict[str, Any] = {}
        for batch_size in (1, 2, 4):
            batch = collate_unified([dataset[index] for index in range(batch_size)])
            expected_shapes = {
                "expert1_wavs": [batch_size, 1, 480_000],
                "expert2_wavs": [batch_size, 1, 320_000],
                "expert3_wavs": [batch_size, 1, 160_000],
                "expert4_wavs": [batch_size, 1, 160_000],
                "expert4_wav_lens": [batch_size],
                "labels": [batch_size],
            }
            for key, shape in expected_shapes.items():
                if list(batch[key].shape) != shape:
                    raise AssertionError(f"B={batch_size}: {key} shape {list(batch[key].shape)} != {shape}")
                if batch[key].device.type != "cpu":
                    raise AssertionError(f"B={batch_size}: {key} unexpectedly left CPU")
            if batch["expert4_wav_lens"].dtype != __import__("torch").int64:
                raise AssertionError("Expert 4 valid lengths must collate as torch.int64")
            if batch["captions_raw"] != [f"  synthetic caption {i}  " for i in range(batch_size)]:
                raise AssertionError("Raw caption order/content changed during Dataset/Collate")
            if batch["wave_file_names"] != [f"fixture-{i}.wav" for i in range(batch_size)]:
                raise AssertionError("Wave sample order changed during Dataset/Collate")
            if batch["expert4_wav_lens"].tolist() != [8_000] * batch_size:
                raise AssertionError("Expert 4 lengths were replaced by padded waveform width")
            results[str(batch_size)] = {
                "status": "PASS",
                "shapes": {key: shape for key, shape in expected_shapes.items()},
                "wave_file_names": batch["wave_file_names"],
                "expert4_valid_lengths": batch["expert4_wav_lens"].tolist(),
                "labels_raw_0_to_10": batch["labels"].tolist(),
            }
        return results


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "PASS",
        "mode": "lightweight",
        "dataset_schema": {},
        "split_configuration": {},
        "expert_input_contracts": {
            "expert1": {"sample_rate_hz": 48_000, "shape": "[B,1,480000]", "policy": "source-compatible LAION-CLAP"},
            "expert2": {"sample_rate_hz": 32_000, "shape": "[B,1,320000]", "policy": "first crop, right zero-pad"},
            "expert3": {"sample_rate_hz": 16_000, "shape": "[B,1,160000]", "policy": "first crop, right zero-pad"},
            "expert4": {"sample_rate_hz": 16_000, "shape": "[B,1,160000]", "lengths": "int64 [B], pre-padding valid samples"},
        },
        "synthetic_tests": {},
        "collate_contract": {"batches": [1, 2, 4], "device": "cpu"},
        "model_construction": False,
        "checkpoint_deserialization": False,
        "real_waveform_parity": "NOT_VERIFIED_LOCAL_09_1",
        "real_caption_parity": "NOT_VERIFIED_LOCAL_09_1",
        "network_access": False,
        "errors": [],
    }
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    config = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        config = load_moe_config(config_path)
        resolved = resolve_moe_paths(config, config_path)
        data = resolved["data"]
        csv_path = Path(args.metadata_csv).expanduser().resolve() if args.metadata_csv else data.get(
            "validation_csv" if args.split == "validation" else f"{args.split}_csv"
        )
        audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else (
            Path(data["audio_root"]) / args.split
        )
        report["split_configuration"] = {
            "split": args.split,
            "metadata_csv": None if csv_path is None else str(csv_path),
            "audio_directory": str(audio_dir),
            "paths_exist": bool(csv_path is not None and Path(csv_path).is_file() and audio_dir.is_dir()),
            "audio_decoding_performed": False,
        }
        if not audio_dir.is_dir():
            report["status"] = "BLOCKED_MISSING_DATA_RESOURCE"
            report["errors"].append({"stage": "audio_directory", "detail": f"directory not found: {audio_dir}"})
        if csv_path is not None and Path(csv_path).is_file():
            schema = inspect_metadata_csv(csv_path)
            schema.update(_expected_header(schema["headers"], split=args.split))
            report["dataset_schema"] = schema
            if schema["required_columns_status"] != "PASS":
                report["status"] = "FAIL_DATASET_SCHEMA"
                report["errors"].append({"stage": "metadata_header", "detail": schema["missing_or_ambiguous"]})
        else:
            report["status"] = "BLOCKED_MISSING_DATA_RESOURCE"
            detail = "No metadata CSV configured for this split" if csv_path is None else f"CSV not found: {csv_path}"
            report["errors"].append({"stage": "metadata_header", "detail": detail})
    elif args.metadata_csv or args.audio_dir:
        if not (args.metadata_csv and args.audio_dir):
            raise ValueError("--metadata-csv and --audio-dir must be provided together without --config")
        csv_path = Path(args.metadata_csv).expanduser().resolve()
        audio_dir = Path(args.audio_dir).expanduser().resolve()
        report["split_configuration"] = {
            "split": args.split,
            "metadata_csv": str(csv_path),
            "audio_directory": str(audio_dir),
            "paths_exist": csv_path.is_file() and audio_dir.is_dir(),
            "audio_decoding_performed": False,
        }
        if csv_path.is_file():
            schema = inspect_metadata_csv(csv_path)
            schema.update(_expected_header(schema["headers"], split=args.split))
            report["dataset_schema"] = schema
            if schema["required_columns_status"] != "PASS":
                report["status"] = "FAIL_DATASET_SCHEMA"
                report["errors"].append({"stage": "metadata_header", "detail": schema["missing_or_ambiguous"]})
        else:
            report["status"] = "BLOCKED_MISSING_DATA_RESOURCE"
            report["errors"].append({"stage": "metadata_header", "detail": f"CSV not found: {csv_path}"})
        if not audio_dir.is_dir():
            report["status"] = "BLOCKED_MISSING_DATA_RESOURCE"
            report["errors"].append({"stage": "audio_directory", "detail": f"directory not found: {audio_dir}"})
    report["synthetic_tests"] = _synthetic_checks()
    report["split_configuration"].setdefault("status", "PASS" if report["dataset_schema"] else "NOT_CONFIGURED_OR_UNAVAILABLE")
    report["source_contract"] = "data.unified_dataset + data.unified_collate"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--mode", choices=("lightweight",), default="lightweight")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "status": "FAIL_PREFLIGHT",
            "mode": "lightweight",
            "dataset_schema": {},
            "split_configuration": {},
            "expert_input_contracts": {},
            "synthetic_tests": {},
            "collate_contract": {},
            "model_construction": False,
            "checkpoint_deserialization": False,
            "real_waveform_parity": "NOT_VERIFIED_LOCAL_09_1",
            "real_caption_parity": "NOT_VERIFIED_LOCAL_09_1",
            "network_access": False,
            "errors": [{"stage": "preflight", "type": type(exc).__name__, "detail": str(exc), "traceback": traceback.format_exc()}],
        }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report_json:
        destination = args.report_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
