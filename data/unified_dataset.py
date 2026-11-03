"""Lazy, source-compatible XACLE dataset for all four MOESCORE experts.

This module prepares CPU tensors only. It does not import expert models,
tokenizers, checkpoints, or GPU runtimes. Each expert retains its original
decoder and waveform policy at this boundary.
"""

from __future__ import annotations

import csv
import math
import os
import random
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = ("train", "validation", "test")
TARGET_SAMPLES = {"expert1": 480_000, "expert2": 320_000, "expert3": 160_000, "expert4": 160_000}
TARGET_RATES = {"expert1": 48_000, "expert2": 32_000, "expert3": 16_000, "expert4": 16_000}
_FILENAME_ALIASES = ("wav_file_name", "wave_file_name")
_SCORE_ALIASES = (
    "average_score",
    "average_semantic_alignement_score",
    "average_semantic_alignment_score",
)


def _choose_column(headers: list[str], aliases: tuple[str, ...], *, required: bool, label: str) -> Optional[str]:
    normalized = {header.strip().lower(): header for header in headers}
    matches = [normalized[name] for name in aliases if name in normalized]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {label} columns in CSV: {matches}")
    if not matches and required:
        raise ValueError(f"CSV is missing required {label} column; accepted names: {aliases}")
    return matches[0] if matches else None


def inspect_metadata_csv(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read only the CSV header; do not scan rows or open/decode audio."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file() or not os.access(csv_path, os.R_OK):
        raise FileNotFoundError(f"Metadata CSV is missing or unreadable: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if not headers or any(not name for name in headers) or len(set(headers)) != len(headers):
            raise ValueError(f"Invalid or duplicate CSV header fields in {csv_path}: {headers}")
    return {"path": str(csv_path), "headers": headers, "row_count": None, "rows_scanned": False}


class UnifiedXACLEDataset(Dataset):
    """One metadata row mapped to four expert-specific, fixed-duration waveforms.

    The CSV row order is retained. Audio is decoded lazily in ``__getitem__``.
    ``split`` is mandatory so train/validation/test data cannot be joined by
    accident. Raw labels remain on the challenge's 0--10 scale.
    """

    def __init__(
        self,
        metadata_csv: str | os.PathLike[str],
        audio_dir: str | os.PathLike[str],
        *,
        split: str,
        seed: int = 2027,
        require_labels: Optional[bool] = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.metadata_csv = Path(metadata_csv).expanduser().resolve()
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        if not self.metadata_csv.is_file() or not os.access(self.metadata_csv, os.R_OK):
            raise FileNotFoundError(f"Metadata CSV is missing or unreadable: {self.metadata_csv}")
        if not self.audio_dir.is_dir() or not os.access(self.audio_dir, os.R_OK):
            raise FileNotFoundError(f"Audio directory is missing or unreadable: {self.audio_dir}")
        if require_labels is None:
            require_labels = split in {"train", "validation"}
        if type(require_labels) is not bool:
            raise TypeError("require_labels must be bool or None")
        self.require_labels = require_labels

        with self.metadata_csv.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            self.headers = list(reader.fieldnames or [])
            if not self.headers or any(not name for name in self.headers):
                raise ValueError(f"Invalid or missing CSV header in {self.metadata_csv}")
            if len(set(self.headers)) != len(self.headers):
                raise ValueError(f"Duplicate CSV header fields in {self.metadata_csv}")
            self.filename_column = _choose_column(
                self.headers, _FILENAME_ALIASES, required=True, label="wave filename"
            )
            self.caption_column = _choose_column(
                self.headers, ("text",), required=True, label="caption"
            )
            self.label_column = _choose_column(
                self.headers, _SCORE_ALIASES, required=require_labels, label="average score"
            )
            self.model_id_column = _choose_column(
                self.headers, ("model_id",), required=False, label="model ID"
            )
            self.model_name_column = _choose_column(
                self.headers, ("model_name",), required=False, label="model name"
            )
            self.rows: list[dict[str, Any]] = []
            seen_names: set[str] = set()
            for row_index, raw_row in enumerate(reader, start=2):
                name = (raw_row.get(self.filename_column) or "").strip()
                caption = raw_row.get(self.caption_column)
                if not name or Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError(
                        f"Invalid wave filename at CSV row {row_index}; expected a basename, got {name!r}"
                    )
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Missing or empty caption at CSV row {row_index} ({name})")
                if name in seen_names:
                    raise ValueError(f"Duplicate sample/audio identity {name!r} at CSV row {row_index}")
                audio_path = (self.audio_dir / name).resolve(strict=False)
                if not audio_path.is_relative_to(self.audio_dir):
                    raise ValueError(f"Audio path escapes split directory at CSV row {row_index}: {name!r}")
                if not audio_path.is_file() or not os.access(audio_path, os.R_OK):
                    raise FileNotFoundError(
                        f"Audio file missing or unreadable for CSV row {row_index}: {audio_path}"
                    )

                label: Optional[float] = None
                if self.label_column is not None:
                    try:
                        label = float(raw_row[self.label_column])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid average score at CSV row {row_index} ({name})") from exc
                    if not math.isfinite(label) or not 0.0 <= label <= 10.0:
                        raise ValueError(
                            f"Average score at CSV row {row_index} ({name}) must be finite and in [0,10]"
                        )
                elif require_labels:
                    raise ValueError(f"Missing average score at CSV row {row_index} ({name})")

                record: dict[str, Any] = {
                    "wave_file_name": name,
                    "audio_path": audio_path,
                    "captions_raw": caption,
                    "label_value": label,
                    "sample_id": f"{split}:{name}",
                    "split": split,
                }
                if self.model_id_column is not None:
                    record["model_id"] = raw_row.get(self.model_id_column, "")
                if self.model_name_column is not None:
                    record["model_name"] = raw_row.get(self.model_name_column, "")
                self.rows.append(record)
                seen_names.add(name)
        if not self.rows:
            raise ValueError(f"Metadata CSV has no sample rows: {self.metadata_csv}")

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | os.PathLike[str],
        *,
        split: str,
        seed: int = 2027,
        require_labels: Optional[bool] = None,
    ) -> "UnifiedXACLEDataset":
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        root = Path(dataset_root).expanduser().resolve()
        return cls(
            root / "meta_data" / f"{split}_average.csv",
            root / "wav" / split,
            split=split,
            seed=seed,
            require_labels=require_labels,
        )

    @classmethod
    def from_config(
        cls,
        resolved_config: Mapping[str, Any],
        *,
        split: str,
        seed: int = 2027,
        require_labels: Optional[bool] = None,
    ) -> "UnifiedXACLEDataset":
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        try:
            data = resolved_config["data"]
            audio_root = Path(data["audio_root"])
            csv_value = data["validation_csv" if split == "validation" else f"{split}_csv"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Resolved configuration is missing data paths for split {split!r}") from exc
        if csv_value is None:
            raise ValueError(f"Resolved configuration has no data.{split}_csv for split {split!r}")
        if not audio_root.is_absolute() or not isinstance(csv_value, (str, os.PathLike)):
            raise ValueError("from_config requires paths returned by resolve_moe_paths")
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            raise ValueError("from_config requires an absolute split CSV path")
        return cls(
            csv_path,
            audio_root / split,
            split=split,
            seed=seed,
            require_labels=require_labels,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic train-crop epoch without changing global RNG."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _mono_float(array: np.ndarray, *, unsigned_center: bool = False) -> np.ndarray:
        if array.size == 0:
            raise ValueError("empty waveform")
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            if unsigned_center and info.min == 0:
                array = (array.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
            else:
                array = array.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            array = array.astype(np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"expected non-empty mono waveform after channel mean; got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("waveform contains non-finite values")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _expert1_audio(self, path: Path, index: int) -> torch.Tensor:
        try:
            import torchaudio
            import torch.nn.functional as torch_f
        except Exception as exc:
            raise RuntimeError(f"Expert 1 requires its existing torchaudio decoder: {exc}") from exc
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0)
        if waveform.numel() == 0:
            raise ValueError("empty waveform")
        if sample_rate != TARGET_RATES["expert1"]:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_RATES["expert1"])
        if not torch.isfinite(waveform).all():
            raise ValueError("waveform contains non-finite values")
        maximum = TARGET_SAMPLES["expert1"]
        if waveform.numel() > maximum:
            last_start = waveform.numel() - maximum
            if self.split == "train":
                rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
                start = rng.randint(0, last_start)
            else:
                start = last_start // 2
            waveform = waveform[start : start + maximum]
        elif waveform.numel() < maximum:
            repeats = maximum // waveform.numel()
            waveform = waveform.repeat(repeats)
            waveform = torch_f.pad(waveform, (0, maximum - waveform.numel()))
        return waveform.to(dtype=torch.float32).contiguous().unsqueeze(0)

    def _expert2_audio(self, path: Path) -> torch.Tensor:
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        sample_rate, raw = wavfile.read(path)
        if sample_rate <= 0:
            raise ValueError(f"invalid sample rate {sample_rate}")
        audio = self._mono_float(raw)
        if sample_rate != TARGET_RATES["expert2"]:
            common = math.gcd(int(sample_rate), TARGET_RATES["expert2"])
            audio = resample_poly(
                audio, TARGET_RATES["expert2"] // common, int(sample_rate) // common
            ).astype(np.float32)
        audio = audio[: TARGET_SAMPLES["expert2"]]
        if audio.size < TARGET_SAMPLES["expert2"]:
            audio = np.pad(audio, (0, TARGET_SAMPLES["expert2"] - audio.size))
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

    def _expert3_audio(self, path: Path) -> torch.Tensor:
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate <= 0 or audio.size == 0:
            raise ValueError(f"invalid sample rate or empty waveform ({sample_rate})")
        audio = audio.mean(axis=1, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise ValueError("waveform contains non-finite values")
        if sample_rate != TARGET_RATES["expert3"]:
            common = math.gcd(int(sample_rate), TARGET_RATES["expert3"])
            audio = resample_poly(
                audio, TARGET_RATES["expert3"] // common, int(sample_rate) // common
            ).astype(np.float32)
        audio = audio[: TARGET_SAMPLES["expert3"]]
        if audio.size < TARGET_SAMPLES["expert3"]:
            audio = np.pad(audio, (0, TARGET_SAMPLES["expert3"] - audio.size))
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

    def _expert4_audio(self, path: Path) -> tuple[torch.Tensor, int]:
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        sample_rate, raw = wavfile.read(path)
        if sample_rate <= 0:
            raise ValueError(f"invalid sample rate {sample_rate}")
        audio = self._mono_float(raw, unsigned_center=True)
        if sample_rate != TARGET_RATES["expert4"]:
            ratio = Fraction(TARGET_RATES["expert4"], int(sample_rate))
            audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
        audio = audio[: TARGET_SAMPLES["expert4"]]
        valid_length = int(audio.size)
        if valid_length <= 0:
            raise ValueError("empty waveform after resampling/cropping")
        if valid_length < TARGET_SAMPLES["expert4"]:
            audio = np.pad(audio, (0, TARGET_SAMPLES["expert4"] - valid_length))
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0), valid_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        row = self.rows[index]
        path = row["audio_path"]
        try:
            expert1 = self._expert1_audio(path, index)
            expert2 = self._expert2_audio(path)
            expert3 = self._expert3_audio(path)
            expert4, expert4_length = self._expert4_audio(path)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError(f"Failed to prepare {row['wave_file_name']} for {self.split}: {exc}") from exc
        item: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "wave_file_name": row["wave_file_name"],
            "split": self.split,
            "captions_raw": row["captions_raw"],
            "expert1_wavs": expert1,
            "expert2_wavs": expert2,
            "expert3_wavs": expert3,
            "expert4_wavs": expert4,
            "expert4_wav_lens": expert4_length,
        }
        if row["label_value"] is not None:
            item["label"] = torch.tensor(row["label_value"], dtype=torch.float32)
        if "model_id" in row:
            item["model_id"] = row["model_id"]
        if "model_name" in row:
            item["model_name"] = row["model_name"]
        return item


def audit_split_overlaps(metadata_csvs: Mapping[str, str | os.PathLike[str]]) -> dict[str, Any]:
    """Compare filename and exact audio-caption identities using CSV metadata only."""
    if not isinstance(metadata_csvs, Mapping):
        raise TypeError("metadata_csvs must map split names to CSV paths")
    if any(split not in SPLITS for split in metadata_csvs):
        raise ValueError(f"split names must be drawn from {SPLITS}")
    if len(metadata_csvs) < 2 or any(not Path(path).is_file() for path in metadata_csvs.values()):
        return {
            "status": "NOT_VERIFIED",
            "reason": "At least two readable split CSVs are required",
            "wave_file_name_overlaps": {},
            "audio_text_pair_overlaps": {},
            "model_id_used_as_input": False,
        }
    identities: dict[str, tuple[set[str], set[tuple[str, str]]]] = {}
    for split, path in metadata_csvs.items():
        with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            headers = list(reader.fieldnames or [])
            filename_col = _choose_column(headers, _FILENAME_ALIASES, required=True, label="wave filename")
            caption_col = _choose_column(headers, ("text",), required=True, label="caption")
            names: set[str] = set()
            pairs: set[tuple[str, str]] = set()
            for row in reader:
                name = (row.get(filename_col) or "").strip()
                caption = row.get(caption_col) or ""
                if not name or not caption.strip():
                    raise ValueError(f"Invalid filename/caption in split metadata {path}")
                names.add(name)
                pairs.add((name, caption))
            identities[split] = (names, pairs)
    names_overlap: dict[str, list[str]] = {}
    pair_overlap: dict[str, list[list[str]]] = {}
    splits = list(identities)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            key = f"{left}:{right}"
            names_overlap[key] = sorted(identities[left][0] & identities[right][0])
            pair_overlap[key] = [list(pair) for pair in sorted(identities[left][1] & identities[right][1])]
    return {
        "status": "PASS",
        "wave_file_name_overlaps": names_overlap,
        "audio_text_pair_overlaps": pair_overlap,
        "model_id_used_as_input": False,
    }


__all__ = [
    "SPLITS",
    "TARGET_RATES",
    "TARGET_SAMPLES",
    "UnifiedXACLEDataset",
    "audit_split_overlaps",
    "inspect_metadata_csv",
]
