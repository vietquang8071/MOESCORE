#!/usr/bin/env python3
"""Verify native XACLE preprocessing against the CPU-only unified dataset.

This script decodes only a deterministic, small set of real WAV files. It scans
metadata and WAV headers for all selected splits, but never constructs an expert
model or deserializes a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import traceback
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_ROWS = {"train": 7500, "validation": 3000, "test": 3000}
SPLITS = ("train", "validation", "test")
SCORE_ALIASES = (
    "average_score",
    "average_semantic_alignement_score",
    "average_semantic_alignment_score",
)
FILENAME_ALIASES = ("wav_file_name", "wave_file_name")
TARGET_SAMPLES = {"expert1": 480_000, "expert2": 320_000, "expert3": 160_000, "expert4": 160_000}


def _column(headers: Iterable[str], aliases: Iterable[str], *, required: bool, field: str) -> str | None:
    by_lower = {name.strip().lower(): name for name in headers}
    matches = [by_lower[alias.lower()] for alias in aliases if alias.lower() in by_lower]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {field} columns: {matches}")
    if not matches and required:
        raise ValueError(f"Missing required {field} column; accepted names: {tuple(aliases)}")
    return matches[0] if matches else None


def classify_failure(error: BaseException) -> str:
    """Map common verifier exceptions to an actionable failure category."""
    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        return "MISSING_AUDIO_FILE" if "audio" in message or "wav" in message else "MISSING_METADATA_CSV"
    if isinstance(error, (csv.Error, UnicodeError)):
        return "METADATA_SCHEMA_MISMATCH"
    if isinstance(error, AssertionError):
        return "WAVEFORM_NUMERICAL_PARITY_FAILURE"
    if "caption" in message or "text" in message:
        return "CAPTION_PARITY_FAILURE"
    if "label" in message or "score" in message:
        return "LABEL_PARITY_FAILURE"
    if "shape" in message or "collate" in message or "length" in message:
        return "COLLATE_CONTRACT_FAILURE"
    return "NATIVE_DATASET_ERROR"


def compare_tensors(native: torch.Tensor, unified: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    """Return exact and tolerance-based comparison diagnostics without coercion."""
    if not isinstance(native, torch.Tensor) or not isinstance(unified, torch.Tensor):
        raise TypeError("waveform comparison expects torch.Tensor inputs")
    result: dict[str, Any] = {
        "native_shape": list(native.shape),
        "unified_shape": list(unified.shape),
        "native_dtype": str(native.dtype),
        "unified_dtype": str(unified.dtype),
        "shape_match": tuple(native.shape) == tuple(unified.shape),
        "dtype_match": native.dtype == unified.dtype,
        "device_match": native.device == unified.device,
        "finite": bool(torch.isfinite(native).all() and torch.isfinite(unified).all()),
        "exact_equal": False,
        "allclose": None,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": None,
        "mean_absolute_error": None,
        "max_relative_error": None,
        "mismatch_count": None,
        "first_mismatch_index": None,
    }
    if not result["shape_match"] or not result["dtype_match"] or not result["device_match"]:
        return result
    if not result["finite"]:
        result["mismatch_count"] = int((native != unified).sum().item())
        return result
    result["exact_equal"] = bool(torch.equal(native, unified))
    result["allclose"] = bool(torch.allclose(native, unified, atol=atol, rtol=rtol))
    difference = (native - unified).abs()
    relative = difference / native.abs().clamp_min(torch.finfo(native.dtype).tiny)
    result["max_absolute_error"] = float(difference.max().item()) if difference.numel() else 0.0
    result["mean_absolute_error"] = float(difference.mean().item()) if difference.numel() else 0.0
    result["max_relative_error"] = float(relative.max().item()) if relative.numel() else 0.0
    unequal = native != unified
    result["mismatch_count"] = int(unequal.sum().item())
    mismatch = torch.nonzero(unequal, as_tuple=False)
    if mismatch.numel():
        result["first_mismatch_index"] = [int(value) for value in mismatch[0].tolist()]
    return result


def compare_e4_representation(
    native_waveform: torch.Tensor,
    unified_waveform: torch.Tensor,
    *,
    valid_length: int,
) -> dict[str, Any]:
    """Compare native unpadded Expert 4 samples to Unified fixed-width audio."""
    native = native_waveform.reshape(-1)
    unified = unified_waveform.reshape(-1)
    length_valid = isinstance(valid_length, int) and 0 < valid_length <= unified.numel()
    native_length_matches = length_valid and native.numel() == valid_length
    if not length_valid:
        return {
            "valid_prefix": None,
            "padding_is_zero": False,
            "native_length": int(native.numel()),
            "unified_length": int(unified.numel()),
            "valid_length": valid_length,
            "length_matches_native": False,
        }
    prefix = compare_tensors(native[:valid_length], unified[:valid_length], atol=0.0, rtol=0.0)
    suffix = unified[valid_length:]
    return {
        "valid_prefix": prefix,
        "padding_is_zero": bool(torch.count_nonzero(suffix).item() == 0),
        "native_length": int(native.numel()),
        "unified_length": int(unified.numel()),
        "valid_length": int(valid_length),
        "length_matches_native": bool(native_length_matches),
        "padding_sample_count": int(suffix.numel()),
    }


def e1_crop_start(*, seed: int, epoch: int, index: int, last_start: int) -> int:
    """Mirror the Unified Dataset's isolated per-item uniform crop draw."""
    if type(seed) is not int or type(epoch) is not int or type(index) is not int:
        raise TypeError("seed, epoch and index must be integers")
    if epoch < 0 or index < 0 or last_start < 0:
        raise ValueError("epoch, index and last_start must be non-negative")
    return random.Random(f"{seed}:{epoch}:{index}").randint(0, last_start)


def write_json_report(report: dict[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def scan_split_metadata(
    metadata_csv: str | os.PathLike[str],
    audio_dir: str | os.PathLike[str],
    split: str,
    *,
    inspect_wav_headers: bool = True,
) -> dict[str, Any]:
    """Scan CSV records and check file existence/header metadata, without PCM decoding."""
    csv_path = Path(metadata_csv).expanduser().resolve()
    wav_root = Path(audio_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV is missing: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate metadata headers in {csv_path}: {headers}")
        filename_column = _column(headers, FILENAME_ALIASES, required=True, field="wave filename")
        caption_column = _column(headers, ("text",), required=True, field="caption")
        label_column = _column(headers, SCORE_ALIASES, required=False, field="average score")
        lower_headers = {header.lower(): header for header in headers}
        model_id_column = lower_headers.get("model_id")
        model_name_column = lower_headers.get("model_name")
        rows: list[dict[str, Any]] = []
        names: list[str] = []
        pairs: list[tuple[str, str]] = []
        captions: list[str] = []
        missing_audio: list[str] = []
        unsafe_names: list[str] = []
        unreadable_headers: list[dict[str, str]] = []
        empty_captions = 0
        missing_filenames = 0
        malformed_csv_rows = 0
        missing_labels = 0
        invalid_labels = 0
        nonfinite_labels = 0
        out_of_range_labels = 0
        label_values: list[float] = []
        model_ids: Counter[str] = Counter()
        model_names: Counter[str] = Counter()
        header_categories: Counter[str] = Counter()
        sample_rates: Counter[str] = Counter()
        channels: Counter[str] = Counter()
        containers: Counter[str] = Counter()
        subtypes: Counter[str] = Counter()
        durations: Counter[str] = Counter()
        for row_index, raw in enumerate(reader):
            row_number = row_index + 2
            if None in raw or any(value is None for value in raw.values()):
                malformed_csv_rows += 1
            filename = (raw.get(filename_column) or "").strip()
            caption_value = raw.get(caption_column)
            caption = caption_value if isinstance(caption_value, str) else ""
            if not caption.strip():
                empty_captions += 1
            if not filename:
                missing_filenames += 1
            path = (wav_root / filename).resolve(strict=False) if filename else wav_root
            safe = bool(filename) and Path(filename).name == filename and "/" not in filename and "\\" not in filename
            if filename and not safe:
                unsafe_names.append(filename)
            exists = safe and path.is_relative_to(wav_root) and path.is_file() and os.access(path, os.R_OK)
            if filename:
                names.append(filename)
                pairs.append((filename, caption))
                captions.append(caption)
            if not exists and filename:
                missing_audio.append(filename)
            label_value: float | None = None
            if label_column is not None:
                label_text = raw.get(label_column)
                if label_text is None or not label_text.strip():
                    missing_labels += 1
                else:
                    try:
                        label_value = float(label_text)
                    except (TypeError, ValueError):
                        invalid_labels += 1
                    else:
                        if not math.isfinite(label_value):
                            nonfinite_labels += 1
                        elif not 0.0 <= label_value <= 10.0:
                            out_of_range_labels += 1
                        else:
                            label_values.append(label_value)
            else:
                missing_labels += 1
            model_id = (raw.get(model_id_column) or "").strip() if model_id_column else ""
            model_name = (raw.get(model_name_column) or "").strip() if model_name_column else ""
            if model_id:
                model_ids[model_id] += 1
            if model_name:
                model_names[model_name] += 1
            info_dict: dict[str, Any] | None = None
            if exists and inspect_wav_headers:
                try:
                    info = sf.info(str(path))
                    rate = int(info.samplerate)
                    channel_count = int(info.channels)
                    frame_count = int(info.frames)
                    duration = frame_count / rate if rate else 0.0
                    duration_category = "short" if duration < 10.0 - 1e-6 else "long" if duration > 10.0 + 1e-6 else "exact_10_seconds"
                    categories = [duration_category]
                    if rate != 16_000:
                        categories.append("nonstandard_sample_rate")
                    if channel_count != 1:
                        categories.append("stereo_or_multichannel")
                    if caption != caption.strip():
                        categories.append("caption_whitespace")
                    for category in categories:
                        header_categories[category] += 1
                    sample_rates[str(rate)] += 1
                    channels[str(channel_count)] += 1
                    containers[str(info.format)] += 1
                    subtypes[str(info.subtype)] += 1
                    durations[duration_category] += 1
                    info_dict = {
                        "sample_rate": rate,
                        "channels": channel_count,
                        "frames": frame_count,
                        "container_format": str(info.format),
                        "subtype": str(info.subtype),
                        "duration_seconds": duration,
                        "duration_category": duration_category,
                        "requires_e1_resampling": rate != 48_000,
                        "requires_e2_resampling": rate != 32_000,
                        "requires_e3_resampling": rate != 16_000,
                        "requires_e4_resampling": rate != 16_000,
                    }
                except Exception as exc:
                    unreadable_headers.append({"filename": filename, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "csv_row_number": row_number,
                    "wave_file_name": filename,
                    "caption": caption,
                    "label_value": label_value,
                    "model_id": model_id,
                    "model_name": model_name,
                    "audio_path": str(path),
                    "audio_exists": bool(exists),
                    "audio_info": info_dict,
                }
            )
    filename_counts = Counter(names)
    pair_counts = Counter(pairs)
    caption_counts = Counter(captions)
    unexpected_formats = sorted({name for name in names if Path(name).suffix.lower() != ".wav"})
    return {
        "split": split,
        "csv_path": str(csv_path),
        "audio_dir": str(wav_root),
        "headers": headers,
        "columns": {
            "filename": filename_column,
            "caption": caption_column,
            "label": label_column,
            "model_id": model_id_column,
            "model_name": model_name_column,
        },
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROWS.get(split),
        "row_count_matches_documented": len(rows) == EXPECTED_ROWS.get(split),
        "unique_wave_filenames": len(filename_counts),
        "unique_filename_caption_pairs": len(pair_counts),
        "duplicate_wave_filenames": sorted(name for name, count in filename_counts.items() if count > 1),
        "duplicate_pair_identities": [list(pair) for pair, count in pair_counts.items() if count > 1],
        "malformed_csv_rows": malformed_csv_rows,
        "missing_filenames": missing_filenames,
        "empty_captions": empty_captions,
        "captions_with_outer_whitespace": sum(1 for caption in captions if caption != caption.strip()),
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "nonfinite_labels": nonfinite_labels,
        "out_of_range_labels": out_of_range_labels,
        "label_min": min(label_values) if label_values else None,
        "label_max": max(label_values) if label_values else None,
        "label_mean": float(np.mean(label_values)) if label_values else None,
        "model_id_counts": dict(sorted(model_ids.items())),
        "model_name_counts": dict(sorted(model_names.items())),
        "missing_audio_files": sorted(set(missing_audio)),
        "unsafe_filenames": sorted(set(unsafe_names)),
        "unexpected_filename_formats": unexpected_formats,
        "caption_whitespace_count": sum(1 for caption in captions if caption != caption.strip()),
        "unreadable_wav_headers": unreadable_headers,
        "wav_header_categories": dict(sorted(header_categories.items())),
        "sample_rate_counts": dict(sorted(sample_rates.items())),
        "channel_counts": dict(sorted(channels.items())),
        "container_format_counts": dict(sorted(containers.items())),
        "subtype_counts": dict(sorted(subtypes.items())),
        "duration_category_counts": dict(sorted(durations.items())),
        "rows": rows,
    }


def scan_auxiliary_csv(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Inspect listener-level/source CSV schema without decoding its audio."""
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Missing CSV header in {csv_path}")
        row_count = 0
        malformed_rows = 0
        blank_by_column = Counter()
        for row in reader:
            row_count += 1
            if None in row:
                malformed_rows += 1
            for field in headers:
                if not (row.get(field) or "").strip():
                    blank_by_column[field] += 1
    return {
        "path": str(csv_path),
        "headers": headers,
        "row_count": row_count,
        "rows_with_extra_fields": malformed_rows,
        "blank_field_counts": dict(blank_by_column),
    }


def audit_overlaps(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    overlaps: dict[str, Any] = {}
    splits = [split for split in SPLITS if split in split_reports]
    for left_pos, left in enumerate(splits):
        for right in splits[left_pos + 1 :]:
            a, b = split_reports[left], split_reports[right]
            key = f"{left}:{right}"
            left_names = {row["wave_file_name"] for row in a["rows"] if row["wave_file_name"]}
            right_names = {row["wave_file_name"] for row in b["rows"] if row["wave_file_name"]}
            left_pairs = {(row["wave_file_name"], row["caption"]) for row in a["rows"]}
            right_pairs = {(row["wave_file_name"], row["caption"]) for row in b["rows"]}
            left_captions = {row["caption"] for row in a["rows"] if row["caption"]}
            right_captions = {row["caption"] for row in b["rows"] if row["caption"]}
            left_models = {row["model_id"] for row in a["rows"] if row["model_id"]}
            right_models = {row["model_id"] for row in b["rows"] if row["model_id"]}
            overlaps[key] = {
                "wave_filename_overlap": sorted(left_names & right_names),
                "exact_filename_caption_overlap": [list(pair) for pair in sorted(left_pairs & right_pairs)],
                "caption_overlap_diagnostic_only": len(left_captions & right_captions),
                "shared_model_ids_diagnostic_only": sorted(left_models & right_models),
            }
    return overlaps


def select_real_samples(split_report: dict[str, Any], *, seed: int, samples_per_split: int) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Choose stable baseline records and available edge-case records."""
    rows = [row for row in split_report["rows"] if row["audio_exists"] and row["audio_info"] is not None and row["caption"].strip()]
    ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['split']}:{row['wave_file_name']}".encode()).hexdigest())
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = {}
    for row in ranked[:samples_per_split]:
        name = row["wave_file_name"]
        selected[name] = row
        reasons.setdefault(name, set()).add("deterministic_hash_sample")
    categories = (
        ("short_audio", lambda row: row["audio_info"]["duration_category"] == "short"),
        ("exact_10_seconds", lambda row: row["audio_info"]["duration_category"] == "exact_10_seconds"),
        ("long_audio", lambda row: row["audio_info"]["duration_category"] == "long"),
        ("nonstandard_source_rate", lambda row: row["audio_info"]["sample_rate"] != 16_000),
        ("stereo_or_multichannel", lambda row: row["audio_info"]["channels"] != 1),
        ("caption_outer_whitespace", lambda row: row["caption"] != row["caption"].strip()),
    )
    availability: dict[str, bool] = {}
    for category, predicate in categories:
        candidates = [row for row in rows if predicate(row)]
        availability[category] = bool(candidates)
        if candidates:
            row = min(candidates, key=lambda item: hashlib.sha256(f"{seed}:{item['split']}:{item['wave_file_name']}:{category}".encode()).hexdigest())
            name = row["wave_file_name"]
            selected[name] = row
            reasons.setdefault(name, set()).add(category)
    result = []
    for row in sorted(selected.values(), key=lambda item: item["row_index"]):
        result.append({**row, "selection_reasons": sorted(reasons[row["wave_file_name"]]), "caption_excerpt": row["caption"][:100]})
    return result, availability


def _tensor_scalar(value: Any) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).reshape(()) if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=torch.float32)


def _load_native_classes():
    from expert1_finetune.Train_Expert1 import Expert1Dataset
    from expert2_finetune.expert2_data import Expert2Dataset
    from expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import MoescoreExpert3Dataset
    from expert4_finetune.dataset import Expert4Dataset, collate_expert4

    return Expert1Dataset, Expert2Dataset, MoescoreExpert3Dataset, Expert4Dataset, collate_expert4


def _make_native_datasets(split_report: dict[str, Any]):
    import importlib

    e1_cls, e2_cls, e3_cls, e4_cls, e4_collate = _load_native_classes()
    csv_path = Path(split_report["csv_path"])
    audio_dir = Path(split_report["audio_dir"])
    columns = split_report["columns"]
    if columns["label"] is None:
        raise ValueError(f"Native score-based datasets cannot be compared for unlabeled split {split_report['split']}")
    if columns["label"].lower() != "average_score":
        # The original E1 source has a fixed source column; don't rewrite it for parity.
        if columns["label"].lower() not in {"average_semantic_alignement_score", "average_semantic_alignment_score"}:
            raise ValueError(f"Unsupported native average-score column: {columns['label']}")
    e1 = e1_cls(csv_path, audio_dir, training=False)
    e1_training = e1_cls(csv_path, audio_dir, training=True) if split_report["split"] == "train" else None
    e2 = e2_cls(csv_path, audio_dir)
    e3 = e3_cls(csv_path, audio_dir, filename_column=columns["filename"], caption_column=columns["caption"], score_column=columns["label"], require_scores=True)
    e4 = e4_cls(csv_path, audio_dir, require_score=True)
    return {
        "expert1": e1,
        "expert1_training": e1_training,
        "expert1_module": importlib.import_module("expert1_finetune.Train_Expert1"),
        "expert2": e2,
        "expert3": e3,
        "expert4": e4,
        "expert4_collate": e4_collate,
    }


def _e1_full_resampled(path: str | os.PathLike[str]) -> torch.Tensor:
    """Read the full E1 pre-crop waveform using its original decode/resampler."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0)
    if sample_rate != 48_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48_000)
    return waveform


def _native_e1_item_for_row(native: dict[str, Any], unified_dataset: Any, row: dict[str, Any], *, seed: int):
    """Use a controlled matching crop for train and native center crop for eval."""
    index = row["row_index"]
    if row["split"] != "train":
        return native["expert1"][index]
    training_dataset = native["expert1_training"]
    full = _e1_full_resampled(row["audio_path"])
    last_start = max(0, full.numel() - 480_000)
    if last_start == 0:
        return training_dataset[index]
    from unittest import mock

    crop_offset = e1_crop_start(seed=seed, epoch=unified_dataset.epoch, index=index, last_start=last_start)
    with mock.patch.object(native["expert1_module"].random, "randint", return_value=crop_offset):
        return training_dataset[index]


def _compare_record(native: dict[str, Any], unified: dict[str, Any], row: dict[str, Any], *, atol: float, rtol: float) -> dict[str, Any]:
    results: dict[str, Any] = {
        "split": row["split"],
        "csv_row_number": row["csv_row_number"],
        "filename": row["wave_file_name"],
        "native_sources": {
            "expert1": {
                "class": "Expert1Dataset",
                "source": "expert1_finetune/Train_Expert1.py",
                "mode": "controlled_train_random_crop" if row["split"] == "train" else "evaluation_center_crop",
            },
            "expert2": {"class": "Expert2Dataset", "source": "expert2_finetune/expert2_data.py"},
            "expert3": {"class": "MoescoreExpert3Dataset", "source": "expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py"},
            "expert4": {"class": "Expert4Dataset", "source": "expert4_finetune/dataset.py"},
        },
    }
    n1 = _native_e1_item_for_row(native, unified, row, seed=unified.seed)
    n2 = native["expert2"][row["row_index"]]
    n3 = native["expert3"][row["row_index"]]
    n4 = native["expert4"][row["row_index"]]
    uni = unified[row["row_index"]]
    source_audio = {"expert1": n1["clap_wav"], "expert2": n2["audio"].unsqueeze(0), "expert3": n3["wav"], "expert4": n4["wav"]}
    unified_audio = {key: uni[f"{key}_wavs"] for key in source_audio}
    results["audio"] = {}
    for expert, native_wave in source_audio.items():
        if expert == "expert4":
            e4 = compare_e4_representation(native_wave, unified_audio[expert], valid_length=int(uni["expert4_wav_lens"]))
            e4["native_shape"] = list(native_wave.shape)
            e4["unified_shape"] = list(unified_audio[expert].shape)
            e4["native_dtype"] = str(native_wave.dtype)
            e4["unified_dtype"] = str(unified_audio[expert].dtype)
            e4["allclose"] = bool(e4["valid_prefix"]["allclose"] and e4["padding_is_zero"] and e4["length_matches_native"])
            e4["status"] = "PASS" if e4["allclose"] else "FAIL"
            if not e4["allclose"]:
                e4["failure_category"] = "WAVEFORM_NUMERICAL_PARITY_FAILURE"
            results["audio"][expert] = e4
        else:
            result = compare_tensors(native_wave, unified_audio[expert], atol=atol, rtol=rtol)
            result["allclose"] = bool(result["allclose"])
            result["status"] = "PASS" if result["allclose"] else "FAIL"
            if not result["allclose"]:
                result["failure_category"] = "WAVEFORM_NUMERICAL_PARITY_FAILURE"
            results["audio"][expert] = result
    raw_caption = row["caption"]
    results["caption"] = {
        "raw_csv_equals_unified": raw_caption == uni["captions_raw"],
        "expert1_native_equals_raw": n1["caption"] == raw_caption,
        "expert2_native_equals_stripped": n2["caption"] == raw_caption.strip(),
        "expert3_native_equals_stripped": n3["caption"] == raw_caption.strip(),
        "expert4_native_equals_stripped": n4["caption"] == raw_caption.strip(),
        "raw_caption_repr": repr(raw_caption[:100]),
        "native_normalized": {
            "expert1": repr(n1["caption"][:100]),
            "expert2": repr(n2["caption"][:100]),
            "expert3": repr(n3["caption"][:100]),
            "expert4": repr(n4["caption"][:100]),
        },
    }
    raw_values = {
        "expert1": n1["score_raw"],
        "expert2": n2["score_raw"],
        "expert3": n3.get("score_raw", None),
        "expert4": n4.get("score_raw", None),
    }
    label_match = {}
    for expert, native_label in raw_values.items():
        if native_label is None:
            label_match[expert] = False
        else:
            native_tensor = _tensor_scalar(native_label)
            unified_tensor = uni["label"].detach().cpu().reshape(())
            label_match[expert] = bool(torch.equal(native_tensor, unified_tensor))
    results["labels"] = {
        "csv_value": row["label_value"],
        "unified_float32": float(uni["label"].item()),
        "matches_all_native_raw_labels": all(label_match.values()),
        "per_expert_raw_match": label_match,
        "scale_preserved_0_10": 0.0 <= float(uni["label"].item()) <= 10.0,
    }
    native_names = {
        "expert1": str(native["expert1"].rows.iloc[row["row_index"]]["wav_file_name"]),
        "expert2": n2["wav_file_name"],
        "expert3": n3["wav_file_name"],
        "expert4": n4["sample_id"],
    }
    results["sample_identity"] = {
        "unified_sample_id": uni["sample_id"],
        "unified_filename": uni["wave_file_name"],
        "native_filenames": native_names,
        "filename_matches_all_native": all(name == row["wave_file_name"] for name in native_names.values()),
        "split_preserved": uni["split"] == row["split"],
    }
    results["all_audio_pass"] = all(value["allclose"] for value in results["audio"].values())
    results["all_caption_pass"] = (
        results["caption"]["raw_csv_equals_unified"]
        and all(value for key, value in results["caption"].items() if key.endswith("equals_raw") or key.endswith("equals_stripped"))
    )
    results["labels_pass"] = results["labels"]["matches_all_native_raw_labels"] and results["labels"]["scale_preserved_0_10"]
    results["identity_pass"] = results["sample_identity"]["filename_matches_all_native"] and results["sample_identity"]["split_preserved"]
    results["status"] = "PASS" if results["all_audio_pass"] and results["all_caption_pass"] and results["labels_pass"] and results["identity_pass"] else "FAIL"
    return results


def _verify_e1_train_crop(dataset: Any, native_train: Any, row: dict[str, Any], *, seed: int) -> dict[str, Any]:
    import importlib
    from unittest import mock

    import torchaudio

    path = row["audio_path"]
    full_wave, rate = torchaudio.load(path)
    full_wave = full_wave.mean(dim=0)
    if rate != 48_000:
        full_wave = torchaudio.functional.resample(full_wave, rate, 48_000)
    maximum = 480_000
    last_start = max(0, int(full_wave.numel()) - maximum)
    if last_start <= 0:
        return {
            "status": "NOT_PRESENT_IN_REAL_DATA" if full_wave.numel() <= maximum else "NOT_APPLICABLE",
            "reason": "Selected real training WAV has no nontrivial random crop window",
            "resampled_samples": int(full_wave.numel()),
            "maximum_samples": maximum,
            "allowed_crop_start_range": [0, last_start],
        }
    epoch = 0
    expected_start = e1_crop_start(seed=seed, epoch=epoch, index=row["row_index"], last_start=last_start)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    dataset.set_epoch(epoch)
    unified_item = dataset[row["row_index"]]
    python_unchanged = random.getstate() == python_state
    numpy_unchanged = all(
        (a == b).all() if isinstance(a, np.ndarray) else a == b
        for a, b in zip(np.random.get_state(), numpy_state)
    )
    torch_unchanged = torch.equal(torch.get_rng_state(), torch_state)
    expert1_module = importlib.import_module("expert1_finetune.Train_Expert1")
    with mock.patch.object(expert1_module.random, "randint", return_value=expected_start) as randint:
        native_item = native_train[row["row_index"]]
    randint.assert_called_once_with(0, last_start)
    comparison = compare_tensors(native_item["clap_wav"], unified_item["expert1_wavs"], atol=0.0, rtol=0.0)
    epoch_starts = [e1_crop_start(seed=seed, epoch=e, index=row["row_index"], last_start=last_start) for e in range(16)]
    return {
        "status": "PASS" if comparison["exact_equal"] and python_unchanged and numpy_unchanged and torch_unchanged else "FAIL",
        "seed": seed,
        "epoch": epoch,
        "index": row["row_index"],
        "allowed_crop_start_range": [0, last_start],
        "controlled_crop_start": expected_start,
        "epoch_zero_crop_reproducible": expected_start == e1_crop_start(seed=seed, epoch=0, index=row["row_index"], last_start=last_start),
        "distinct_offsets_across_16_epochs": len(set(epoch_starts)) > 1,
        "uniform_integer_policy": "random.Random(...).randint(0,last_start), inclusive endpoints; native uses random.randint over same range",
        "same_controlled_offset_waveform_parity": comparison,
        "python_global_rng_unchanged": python_unchanged,
        "numpy_global_rng_unchanged": numpy_unchanged,
        "torch_global_rng_unchanged": torch_unchanged,
    }


def _verify_native_e4_collate(native_ds: Any, unified_ds: Any, selected_rows: list[dict[str, Any]], native_collate: Any) -> dict[str, Any]:
    from data.unified_collate import collate_unified

    results = []
    for count in (1, 2):
        if count == 1 and selected_rows:
            rows = [min(selected_rows, key=lambda row: min(160_000, int(round(row["audio_info"]["duration_seconds"] * 16_000))))]
        elif len(selected_rows) >= 2:
            lengths = {
                row["wave_file_name"]: min(160_000, int(round(row["audio_info"]["duration_seconds"] * 16_000)))
                for row in selected_rows
            }
            if len(set(lengths.values())) > 1:
                shortest = min(selected_rows, key=lambda row: lengths[row["wave_file_name"]])
                longest = max(selected_rows, key=lambda row: lengths[row["wave_file_name"]])
                rows = sorted((shortest, longest), key=lambda row: row["row_index"])
            else:
                rows = selected_rows[:2]
        else:
            rows = selected_rows[:count]
        if len(rows) < count:
            results.append({"batch_size": count, "status": "NOT_VERIFIED", "reason": "Not enough selected real records"})
            continue
        indices = [row["row_index"] for row in rows]
        native_items = [native_ds[index] for index in indices]
        unified_items = [unified_ds[index] for index in indices]
        native_batch = native_collate(native_items)
        unified_batch = collate_unified(unified_items)
        native_wavs = native_batch["wavs"]
        unified_wavs = unified_batch["expert4_wavs"]
        native_width = native_wavs.shape[-1]
        value_comparison = compare_tensors(native_wavs, unified_wavs[..., :native_width], atol=0.0, rtol=0.0)
        length_equal = torch.equal(native_batch["wav_lens"], unified_batch["expert4_wav_lens"])
        unified_tail_zero = bool(torch.count_nonzero(unified_wavs[..., native_width:]).item() == 0)
        native_lengths = native_batch["wav_lens"]
        native_padding_zero = all(
            torch.count_nonzero(native_wavs[i, :, int(length):]).item() == 0
            for i, length in enumerate(native_lengths.tolist())
        )
        result = {
            "batch_size": count,
            "status": "PASS" if value_comparison["exact_equal"] and length_equal and unified_tail_zero and native_padding_zero else "FAIL",
            "native_shape": list(native_wavs.shape),
            "unified_shape": list(unified_wavs.shape),
            "native_dynamic_width": int(native_width),
            "native_lengths": native_lengths.tolist(),
            "unified_lengths": unified_batch["expert4_wav_lens"].tolist(),
            "native_values_equal_unified_prefix_to_dynamic_width": value_comparison,
            "lengths_equal": length_equal,
            "unified_suffix_after_native_dynamic_width_is_zero": unified_tail_zero,
            "native_per_sample_padding_is_zero": native_padding_zero,
            "filenames": native_batch["sample_ids"],
        }
        results.append(result)
    return {"batches": results, "status": "PASS" if all(item.get("status") == "PASS" for item in results) else "PARTIAL"}


def _verify_real_dataloader(dataset: Any, split_report: dict[str, Any], selected_rows: list[dict[str, Any]], batch_sizes: list[int]) -> dict[str, Any]:
    from torch.utils.data import DataLoader, Subset
    from data.unified_collate import collate_unified

    result: dict[str, Any] = {}
    for batch_size in batch_sizes:
        if len(selected_rows) < batch_size:
            result[str(batch_size)] = {"status": "NOT_VERIFIED", "reason": "Not enough selected records"}
            continue
        if batch_size == 1:
            chosen = [min(selected_rows, key=lambda row: min(160_000, int(round(row["audio_info"]["duration_seconds"] * 16_000))))]
        elif batch_size == 2:
            lengths = {
                row["wave_file_name"]: min(160_000, int(round(row["audio_info"]["duration_seconds"] * 16_000)))
                for row in selected_rows
            }
            if len(set(lengths.values())) > 1:
                chosen = [
                    min(selected_rows, key=lambda row: lengths[row["wave_file_name"]]),
                    max(selected_rows, key=lambda row: lengths[row["wave_file_name"]]),
                ]
                chosen = sorted({row["row_index"]: row for row in chosen}.values(), key=lambda row: row["row_index"])
            else:
                chosen = sorted(selected_rows, key=lambda row: row["row_index"])[:2]
        else:
            chosen = sorted(selected_rows, key=lambda row: row["row_index"])[:batch_size]
        indices = [row["row_index"] for row in chosen]
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_unified)
        batch = next(iter(loader))
        expected_rows = chosen
        expected_names = [row["wave_file_name"] for row in expected_rows]
        expected_ids = [f"{split_report['split']}:{name}" for name in expected_names]
        expected_captions = [row["caption"] for row in expected_rows]
        expected_labels = torch.tensor([row["label_value"] for row in expected_rows], dtype=torch.float32) if all(row["label_value"] is not None for row in expected_rows) else None
        dimensions = {
            "expert1_wavs": [batch_size, 1, 480_000],
            "expert2_wavs": [batch_size, 1, 320_000],
            "expert3_wavs": [batch_size, 1, 160_000],
            "expert4_wavs": [batch_size, 1, 160_000],
            "expert4_wav_lens": [batch_size],
        }
        shapes = {key: list(batch[key].shape) for key in dimensions}
        shape_pass = all(shapes[key] == expected for key, expected in dimensions.items())
        cpu_pass = all(batch[key].device.type == "cpu" for key in dimensions)
        order_pass = batch["wave_file_names"] == expected_names and batch["sample_ids"] == expected_ids and batch["captions_raw"] == expected_captions
        label_pass = expected_labels is None and "labels" not in batch or expected_labels is not None and torch.equal(batch["labels"], expected_labels)
        length_pass = batch["expert4_wav_lens"].dtype == torch.int64 and bool(torch.all(batch["expert4_wav_lens"] > 0))
        result[str(batch_size)] = {
            "status": "PASS" if shape_pass and cpu_pass and order_pass and label_pass and length_pass else "FAIL",
            "batch_size": batch_size,
            "shapes": shapes,
            "dtypes": {key: str(batch[key].dtype) for key in dimensions},
            "all_waveforms_cpu": cpu_pass,
            "sample_caption_label_order_preserved": order_pass and label_pass,
            "labels": batch.get("labels").tolist() if "labels" in batch else None,
            "captions_raw": batch["captions_raw"],
            "wave_file_names": batch["wave_file_names"],
            "sample_ids": batch["sample_ids"],
            "expert4_wav_lens": batch["expert4_wav_lens"].tolist(),
        }
    return result


def _environment() -> dict[str, Any]:
    import scipy

    try:
        import torchaudio
        torchaudio_version = torchaudio.__version__
    except Exception as exc:
        torchaudio_version = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchaudio": torchaudio_version,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "numpy": np.__version__,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "cuda_inference": False,
        "offline_env": {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }


def run_verification(
    dataset_root: str | os.PathLike[str],
    *,
    splits: Iterable[str] = SPLITS,
    seed: int = 2027,
    samples_per_split: int = 2,
    parity_splits: Iterable[str] = ("train", "validation"),
    batch_sizes: Iterable[int] = (1, 2),
    atol: float = 1e-6,
    rtol: float = 1e-5,
    inspect_wav_headers: bool = True,
) -> dict[str, Any]:
    from data.unified_dataset import UnifiedXACLEDataset

    root = Path(dataset_root).expanduser().resolve()
    requested = list(splits)
    if not requested or any(split not in SPLITS for split in requested) or len(requested) != len(set(requested)):
        raise ValueError(f"splits must be unique names from {SPLITS}")
    parity_requested = list(parity_splits)
    if any(split not in requested for split in parity_requested) or len(parity_requested) != len(set(parity_requested)):
        raise ValueError("parity_splits must be unique and included in splits")
    if samples_per_split < 1:
        raise ValueError("samples_per_split must be positive")
    report: dict[str, Any] = {
        "status": "RUNNING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "dataset_root": str(root),
        "available_splits": [],
        "real_parity_splits": parity_requested,
        "metadata_schema": {},
        "split_row_counts": {},
        "duplicate_summary": {},
        "overlap_summary": {},
        "selected_samples": [],
        "expert1_parity": [],
        "expert1_eval_parity": [],
        "expert2_parity": [],
        "expert3_parity": [],
        "expert4_parity": [],
        "expert1_training_crop_policy": {},
        "expert4_native_collate_comparison": {},
        "caption_parity": [],
        "label_parity": [],
        "sample_alignment": [],
        "dataloader_b1": {},
        "dataloader_b2": {},
        "numerical_tolerances": {"atol": atol, "rtol": rtol, "exact_expected_for_same_source_calls": True},
        "numerical_differences": {},
        "missing_real_edge_cases": [],
        "failures": [],
        "native_preprocessing_status": {},
        "native_preprocessing_sources": {
            "expert1": "expert1_finetune/Train_Expert1.py::Expert1Dataset.__getitem__",
            "expert2": "expert2_finetune/expert2_data.py::Expert2Dataset.__getitem__",
            "expert3": "expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py::MoescoreExpert3Dataset.__getitem__",
            "expert4": "expert4_finetune/dataset.py::Expert4Dataset.__getitem__, collate_expert4",
        },
        "limitations": [],
        "model_construction": False,
        "checkpoint_deserialization": False,
        "gpu_inference": False,
    }
    split_reports: dict[str, dict[str, Any]] = {}
    selections: dict[str, list[dict[str, Any]]] = {}
    availability: dict[str, dict[str, bool]] = {}
    expected_edge_categories = ("short_audio", "exact_10_seconds", "long_audio", "nonstandard_source_rate", "stereo_or_multichannel", "caption_outer_whitespace")
    for split in requested:
        csv_path = root / "meta_data" / f"{split}_average.csv"
        audio_dir = root / "wav" / split
        if not csv_path.is_file() or not audio_dir.is_dir():
            report["failures"].append({"stage": "dataset_paths", "split": split, "category": "MISSING_DATASET_ROOT", "csv": str(csv_path), "audio_dir": str(audio_dir)})
            continue
        try:
            scanned = scan_split_metadata(csv_path, audio_dir, split, inspect_wav_headers=inspect_wav_headers)
            split_reports[split] = scanned
            if split in parity_requested:
                selected, available_categories = select_real_samples(scanned, seed=seed, samples_per_split=samples_per_split)
            else:
                selected = []
                available_categories = {
                    "short_audio": scanned["duration_category_counts"].get("short", 0) > 0,
                    "exact_10_seconds": scanned["duration_category_counts"].get("exact_10_seconds", 0) > 0,
                    "long_audio": scanned["duration_category_counts"].get("long", 0) > 0,
                    "nonstandard_source_rate": any(key != "16000" for key in scanned["sample_rate_counts"]),
                    "stereo_or_multichannel": any(key != "1" for key in scanned["channel_counts"]),
                    "caption_outer_whitespace": scanned["caption_whitespace_count"] > 0,
                }
            selections[split] = selected
            availability[split] = available_categories
            report["available_splits"].append(split)
            report["metadata_schema"][split] = {"headers": scanned["headers"], "columns": scanned["columns"], "csv_path": str(csv_path)}
            listener_csv = root / "meta_data" / f"{split}.csv"
            if listener_csv.is_file():
                report["metadata_schema"][split]["listener_level_csv"] = scan_auxiliary_csv(listener_csv)
            report["split_row_counts"][split] = {
                "observed": scanned["row_count"],
                "documented_expected": scanned["expected_row_count"],
                "matches_documented": scanned["row_count_matches_documented"],
            }
            report["duplicate_summary"][split] = {
                "unique_filenames": scanned["unique_wave_filenames"],
                "duplicate_filenames": scanned["duplicate_wave_filenames"],
                "duplicate_pair_identities": scanned["duplicate_pair_identities"],
                "missing_audio_count": len(scanned["missing_audio_files"]),
                "missing_audio_files": scanned["missing_audio_files"][:100],
                "invalid_rows": {
                    "empty_captions": scanned["empty_captions"],
                    "missing_filenames": scanned["missing_filenames"],
                    "malformed_csv_rows": scanned["malformed_csv_rows"],
                    "missing_labels": scanned["missing_labels"],
                    "invalid_labels": scanned["invalid_labels"],
                    "nonfinite_labels": scanned["nonfinite_labels"],
                    "out_of_range_labels": scanned["out_of_range_labels"],
                    "unsafe_filenames": scanned["unsafe_filenames"],
                    "unexpected_filename_formats": scanned["unexpected_filename_formats"],
                    "unreadable_wav_headers": scanned["unreadable_wav_headers"][:100],
                },
                "label_statistics_0_10": {"min": scanned["label_min"], "max": scanned["label_max"], "mean": scanned["label_mean"]},
                "unique_filename_caption_pairs": scanned["unique_filename_caption_pairs"],
                "captions_with_outer_whitespace": scanned["caption_whitespace_count"],
                "model_id_counts": scanned["model_id_counts"],
                "model_name_counts": scanned["model_name_counts"],
                "sample_rate_counts": scanned["sample_rate_counts"],
                "channel_counts": scanned["channel_counts"],
                "container_format_counts": scanned["container_format_counts"],
                "subtype_counts": scanned["subtype_counts"],
                "duration_category_counts": scanned["duration_category_counts"],
            }
            for category, present in available_categories.items():
                if not present:
                    report["missing_real_edge_cases"].append({"split": split, "category": category, "status": "NOT_PRESENT_IN_REAL_DATA"})
            for sample in selected:
                report["selected_samples"].append({key: sample[key] for key in ("split", "csv_row_number", "wave_file_name", "caption_excerpt", "label_value", "model_id", "model_name", "audio_info", "selection_reasons")})
        except Exception as exc:
            report["failures"].append({"stage": "metadata_scan", "split": split, "category": classify_failure(exc), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    report["overlap_summary"] = audit_overlaps(split_reports)
    if not split_reports:
        report["status"] = "FAIL"
        report["real_waveform_parity"] = "NOT_VERIFIED"
        report["real_caption_parity"] = "NOT_VERIFIED"
        return report
    all_rows = [row for scanned in split_reports.values() for row in scanned["rows"]]
    global_categories: Counter[str] = Counter()
    for scanned in split_reports.values():
        global_categories.update(scanned["wav_header_categories"])
    report["wav_inventory"] = {
        "sample_rates": dict(Counter({k: sum(int(s["sample_rate_counts"].get(k, 0)) for s in split_reports.values()) for k in set().union(*(set(s["sample_rate_counts"]) for s in split_reports.values()))})),
        "channels": dict(Counter({k: sum(int(s["channel_counts"].get(k, 0)) for s in split_reports.values()) for k in set().union(*(set(s["channel_counts"]) for s in split_reports.values()))})),
        "container_formats": dict(Counter({k: sum(int(s["container_format_counts"].get(k, 0)) for s in split_reports.values()) for k in set().union(*(set(s["container_format_counts"]) for s in split_reports.values()))})),
        "subtypes": dict(Counter({k: sum(int(s["subtype_counts"].get(k, 0)) for s in split_reports.values()) for k in set().union(*(set(s["subtype_counts"]) for s in split_reports.values()))})),
        "duration_categories": dict(global_categories),
    }
    all_available = {category: any(availability.get(split, {}).get(category, False) for split in availability) for category in expected_edge_categories}
    for category, present in all_available.items():
        if not present and not any(item["category"] == category and item["status"] == "NOT_PRESENT_IN_REAL_DATA" for item in report["missing_real_edge_cases"]):
            report["missing_real_edge_cases"].append({"split": "all_available", "category": category, "status": "NOT_PRESENT_IN_REAL_DATA"})
    for split, scanned in split_reports.items():
        try:
            if split not in parity_requested:
                report["native_preprocessing_status"][split] = {
                    "status": "METADATA_ONLY_NOT_USED_FOR_PREPROCESSING",
                    "pcm_decoded": False,
                    "reason": "The split is included in metadata/WAV-header integrity checks but excluded from native-vs-Unified preprocessing comparisons.",
                }
                continue
            has_labels = scanned["columns"]["label"] is not None
            unified = UnifiedXACLEDataset(scanned["csv_path"], scanned["audio_dir"], split=split, seed=seed, require_labels=has_labels)
            if not has_labels:
                report["native_preprocessing_status"][split] = {
                    "status": "NOT_VERIFIED_NO_AVERAGE_LABELS",
                    "reason": "Native E1/E2 score datasets require the average-score field; Unified Dataset preserves unlabeled records without inventing labels.",
                }
                report["limitations"].append(f"Native-vs-Unified expert preprocessing for {split} was not run because its CSV has no average-score column.")
                if split == "validation":
                    report["real_dataloader_validation"] = _verify_real_dataloader(unified, scanned, selections.get(split, []), list(batch_sizes))
                    report["dataloader_b1"] = report["real_dataloader_validation"].get("1", {})
                    report["dataloader_b2"] = report["real_dataloader_validation"].get("2", {})
                continue
            report["native_preprocessing_status"][split] = {"status": "RUNNING"}
            native = _make_native_datasets(scanned)
            if len(native["expert1"]) != len(unified) or len(native["expert2"]) != len(unified) or len(native["expert3"]) != len(unified) or len(native["expert4"]) != len(unified):
                raise ValueError(f"Native/Unified row counts differ for {split}")
            sample_results = []
            for row in selections.get(split, []):
                try:
                    item_result = _compare_record(native, unified, row, atol=atol, rtol=rtol)
                    item_result["selection_reasons"] = row["selection_reasons"]
                    sample_results.append(item_result)
                    for expert in ("expert1", "expert2", "expert3", "expert4"):
                        report[f"{expert}_parity"].append({"split": split, "filename": row["wave_file_name"], **item_result["audio"][expert]})
                        if expert == "expert1" and split != "train":
                            report["expert1_eval_parity"].append({"split": split, "filename": row["wave_file_name"], **item_result["audio"][expert]})
                    report["caption_parity"].append({"split": split, "filename": row["wave_file_name"], **item_result["caption"]})
                    report["label_parity"].append({"split": split, "filename": row["wave_file_name"], **item_result["labels"]})
                    report["sample_alignment"].append({"split": split, "filename": row["wave_file_name"], **item_result["sample_identity"]})
                except Exception as exc:
                    failure = {"stage": "native_vs_unified_item", "split": split, "filename": row["wave_file_name"], "category": classify_failure(exc), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    report["failures"].append(failure)
                    sample_results.append({"split": split, "filename": row["wave_file_name"], "status": "FAIL", "failure": failure})
            report.setdefault("sample_comparisons", {})[split] = sample_results
            report["native_preprocessing_status"][split] = {"status": "PASS", "sampled_records": len(sample_results)}
            report["expert4_native_collate_comparison"][split] = _verify_native_e4_collate(native["expert4"], unified, selections.get(split, []), native["expert4_collate"])
            if split == "train":
                e1_training = type(native["expert1"])(Path(scanned["csv_path"]), Path(scanned["audio_dir"]), training=True)
                long_rows = [row for row in selections.get(split, []) if row["audio_info"] and row["audio_info"]["duration_seconds"] > 10.0]
                if long_rows:
                    report["expert1_training_crop_policy"] = _verify_e1_train_crop(unified, e1_training, long_rows[0], seed=seed)
                else:
                    report["expert1_training_crop_policy"] = {
                        "status": "NOT_PRESENT_IN_REAL_DATA" if not any(row["audio_info"] and row["audio_info"]["duration_seconds"] > 10.0 for row in scanned["rows"]) else "NOT_SELECTED",
                        "source_policy": "uniform inclusive random crop [0, resampled_length-480000], repeat-floor then right-zero-pad for short inputs",
                        "real_long_training_wav_count": sum(1 for row in scanned["rows"] if row["audio_info"] and row["audio_info"]["duration_seconds"] > 10.0),
                        "note": "Eval parity is tested separately; no uncontrolled native training crop was compared.",
                    }
            if split == "validation":
                report["real_dataloader_validation"] = _verify_real_dataloader(unified, scanned, selections.get(split, []), list(batch_sizes))
                report["dataloader_b1"] = report["real_dataloader_validation"].get("1", {})
                report["dataloader_b2"] = report["real_dataloader_validation"].get("2", {})
        except Exception as exc:
            report["failures"].append({"stage": "dataset_or_native_construction", "split": split, "category": classify_failure(exc), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    if "real_dataloader_validation" not in report:
        report["limitations"].append("No labeled validation split was available for the real CPU DataLoader B=1/B=2 checks.")
    successful_items = [item for values in report.get("sample_comparisons", {}).values() for item in values]
    parity_pass = bool(successful_items) and all(item.get("all_audio_pass") and item.get("all_caption_pass") and item.get("labels_pass") and item.get("identity_pass") for item in successful_items)
    collate_pass = all(value.get("status") == "PASS" for value in report["expert4_native_collate_comparison"].values())
    loader = report.get("real_dataloader_validation", {})
    loader_pass = all(loader.get(str(size), {}).get("status") == "PASS" for size in batch_sizes)
    data_errors = any(
        result["invalid_rows"][key]
        for split, result in report["duplicate_summary"].items()
        for key in ("empty_captions", "missing_filenames", "malformed_csv_rows", "invalid_labels", "nonfinite_labels", "out_of_range_labels", "unsafe_filenames", "unexpected_filename_formats", "unreadable_wav_headers")
    ) or any(
        result["invalid_rows"]["missing_labels"]
        for split, result in report["duplicate_summary"].items()
        if report["metadata_schema"][split]["columns"]["label"] is not None
    ) or any(result["duplicate_filenames"] or result["duplicate_pair_identities"] or result["missing_audio_files"] for result in report["duplicate_summary"].values())
    row_count_mismatch = any(not value["matches_documented"] for value in report["split_row_counts"].values())
    report["real_waveform_parity"] = "PASS" if parity_pass and collate_pass else "FAIL" if successful_items else "NOT_VERIFIED"
    report["real_caption_parity"] = "PASS" if parity_pass else "FAIL" if successful_items else "NOT_VERIFIED"
    report["label_parity_status"] = "PASS" if parity_pass else "FAIL" if successful_items else "NOT_VERIFIED"
    report["sample_alignment_status"] = "PASS" if parity_pass else "FAIL" if successful_items else "NOT_VERIFIED"
    report["dataloader_status"] = "PASS" if loader_pass else "FAIL" if loader else "NOT_VERIFIED"
    report["metadata_integrity_status"] = "FAIL" if data_errors or row_count_mismatch else "PASS"
    report["status"] = (
        "PASS"
        if parity_pass and collate_pass and loader_pass and not data_errors and not row_count_mismatch and not report["failures"]
        else "FAIL"
        if report["failures"] or data_errors or (successful_items and not parity_pass) or (loader and not loader_pass)
        else "PARTIAL"
    )
    if any(value.get("status") == "NOT_VERIFIED_NO_AVERAGE_LABELS" for value in report["native_preprocessing_status"].values()) and report["status"] == "PASS":
        report["status"] = "PARTIAL"
    report["numerical_differences"] = {
        expert: {
            "comparisons": len(values),
            "max_absolute_error": max((item.get("max_absolute_error") or item.get("valid_prefix", {}).get("max_absolute_error") or 0.0 for item in values), default=None),
            "mean_absolute_error_mean": float(np.mean([item.get("mean_absolute_error") or item.get("valid_prefix", {}).get("mean_absolute_error") or 0.0 for item in values])) if values else None,
            "exact_equal_count": sum(bool(item.get("exact_equal", item.get("valid_prefix", {}).get("exact_equal", False))) for item in values),
            "allclose_count": sum(bool(item.get("allclose")) for item in values),
        }
        for expert in ("expert1", "expert2", "expert3", "expert4")
        for values in [[item for item in report[f"{expert}_parity"]]]
    }
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--parity-splits", default="train,validation", help="Splits whose real WAV PCM may be decoded for native-vs-Unified checks")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.offline:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
    try:
        splits = [value.strip() for value in args.splits.split(",") if value.strip()]
        parity_splits = [value.strip() for value in args.parity_splits.split(",") if value.strip()]
        batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
        if not batch_sizes or any(value not in (1, 2, 4) for value in batch_sizes):
            raise ValueError("batch sizes must be selected from 1,2,4")
        report = run_verification(
            args.dataset_root,
            splits=splits,
            seed=args.seed,
            samples_per_split=args.samples_per_split,
            parity_splits=parity_splits,
            batch_sizes=batch_sizes,
            atol=args.atol,
            rtol=args.rtol,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"category": classify_failure(exc), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            "model_construction": False,
            "checkpoint_deserialization": False,
            "gpu_inference": False,
        }
    write_json_report(report, args.report_json)
    summary = {
        "status": report["status"],
        "available_splits": report.get("available_splits", []),
        "real_parity_splits": report.get("real_parity_splits", []),
        "split_row_counts": report.get("split_row_counts", {}),
        "real_waveform_parity": report.get("real_waveform_parity", "NOT_VERIFIED"),
        "real_caption_parity": report.get("real_caption_parity", "NOT_VERIFIED"),
        "dataloader_status": report.get("dataloader_status", "NOT_VERIFIED"),
        "failures": len(report.get("failures", [])),
        "report_json": str(args.report_json.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
