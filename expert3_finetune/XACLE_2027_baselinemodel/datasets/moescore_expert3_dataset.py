"""XACLE average-score pairs prepared for the vendored 16 kHz M2D-CLAP."""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


class MoescoreExpert3Dataset(Dataset):
    def __init__(self, csv_path, wav_dir, sample_rate=16000, max_len=10,
                 filename_column="wav_file_name", caption_column="text", score_column="average_score",
                 require_scores=True):
        self.csv_path = Path(csv_path)
        self.wav_dir = Path(wav_dir)
        self.sample_rate = int(sample_rate)
        self.max_samples = int(sample_rate * max_len)
        self.filename_column = filename_column
        self.caption_column = caption_column
        self.score_column = score_column
        with self.csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            self.has_scores = score_column in fields
            required = {filename_column, caption_column}
            if require_scores:
                required.add(score_column)
            missing = required - fields
            if missing:
                raise ValueError(f"{self.csv_path}: missing CSV columns {sorted(missing)}; found {sorted(fields)}")
            if require_scores and not self.has_scores:
                raise ValueError(f"{self.csv_path}: missing score column {score_column!r}")
            self.rows = list(reader)
        minimum_rows = 2 if require_scores else 1
        if len(self.rows) < minimum_rows:
            raise ValueError(f"{self.csv_path}: expected at least {minimum_rows} pair(s)")

        names = set()
        class_counts = [0] * 10
        self.class_ids = []
        for row in self.rows:
            filename = row[filename_column].strip()
            caption = row[caption_column].strip()
            if self.has_scores:
                try:
                    score = float(row[score_column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid score for {filename!r} in {self.csv_path}") from exc
            else:
                score = 0.0
            if not filename or Path(filename).name != filename or not caption:
                raise ValueError(f"Invalid WAV filename or caption in {self.csv_path}: {row}")
            if filename in names:
                raise ValueError(f"Duplicate WAV pair {filename!r} in {self.csv_path}")
            if self.has_scores and (not math.isfinite(score) or not 0.0 <= score <= 10.0):
                raise ValueError(f"Score for {filename!r} must be finite and in [0,10], got {score}")
            if not (self.wav_dir / filename).is_file():
                raise FileNotFoundError(f"Missing WAV for {self.csv_path}: {self.wav_dir / filename}")
            names.add(filename)
            # Matches the baseline's right-closed score bins; CBL is disabled,
            # but the counts remain available to the shared loss interface.
            class_id = min(bisect.bisect_left(range(1, 11), score), 9) if self.has_scores else 0
            self.class_ids.append(class_id)
            class_counts[class_id] += 1
        self.num_class = [class_counts[class_id] for class_id in self.class_ids] if self.has_scores else [1] * len(self.rows)
        self.wav_names = names

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        filename = row[self.filename_column].strip()
        audio, source_rate = sf.read(self.wav_dir / filename, dtype="float32", always_2d=True)
        if audio.size == 0:
            raise ValueError(f"Empty waveform: {self.wav_dir / filename}")
        audio = audio.mean(axis=1, dtype=np.float32)
        if source_rate != self.sample_rate:
            gcd = math.gcd(int(source_rate), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // gcd, int(source_rate) // gcd).astype(np.float32)
        audio = audio[:self.max_samples]
        if audio.shape[0] < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - audio.shape[0]))
        item = {
            "wav": torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0),
            "caption": row[self.caption_column].strip(),
            "num_class": torch.tensor(self.num_class[index], dtype=torch.long),
            "wav_path": str(self.wav_dir / filename),
            "wav_file_name": filename,
        }
        if self.has_scores:
            score = float(row[self.score_column])
            item["score"] = torch.tensor((score - 5.0) / 5.0, dtype=torch.float32)
            item["score_raw"] = torch.tensor(score, dtype=torch.float32)
        return item

    @staticmethod
    def collate_fn(items):
        batch = {
            "wavs": torch.stack([item["wav"] for item in items]),
            "captions": [item["caption"] for item in items],
            "wav_paths": [item["wav_path"] for item in items],
            "wav_file_names": [item["wav_file_name"] for item in items],
        }
        if all("score" in item for item in items):
            batch["scores"] = torch.stack([item["score"] for item in items]).reshape(-1)
            batch["scores_raw"] = torch.stack([item["score_raw"] for item in items]).reshape(-1)
            batch["num_class"] = torch.stack([item["num_class"] for item in items]).reshape(-1)
        return batch
