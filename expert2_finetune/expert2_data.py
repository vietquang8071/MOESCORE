"""XACLE average-score pairs prepared for MGA-CLAP's 32 kHz frontend."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


SAMPLE_RATE = 32000
MAX_SAMPLES = SAMPLE_RATE * 10


class Expert2Dataset(Dataset):
    def __init__(self, csv_path, wav_dir):
        self.wav_dir = Path(wav_dir)
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames or []
            self.file_column = "wav_file_name" if "wav_file_name" in names else "wave_file_name"
            self.score_column = (
                "average_score" if "average_score" in names else "average_semantic_alignement_score"
            )
            if not {self.file_column, "text", self.score_column} <= set(names):
                raise ValueError(f"{csv_path}: expected average_score, text and wav_file_name columns")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"{csv_path}: empty average-score CSV")
        seen = set()
        for row in self.rows:
            name = row[self.file_column]
            text = row["text"]
            score = float(row[self.score_column])
            if not name or Path(name).name != name or not text.strip():
                raise ValueError(f"Invalid filename or caption: {row}")
            if name in seen:
                raise ValueError(f"Duplicate audio pair: {name}")
            seen.add(name)
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError(f"Invalid score for {name}: {score}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = row[self.file_column]
        sample_rate, audio = wavfile.read(self.wav_dir / name)
        if audio.size == 0:
            raise ValueError(f"Empty waveform: {name}")
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            common = math.gcd(sample_rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // common, sample_rate // common)
        audio = audio[:MAX_SAMPLES]
        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        score = float(row[self.score_column])
        return {
            "audio": torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
            "caption": row["text"].strip(),
            "score": (score - 5.0) / 5.0,
            "score_raw": score,
            "wav_file_name": name,
        }


def collate_expert2(items):
    return {
        "mga_wavs": torch.stack([x["audio"] for x in items]).unsqueeze(1),
        "caption_tokens": [x["caption"] for x in items],
        "scores": torch.tensor([x["score"] for x in items], dtype=torch.float32),
        "scores_raw": torch.tensor([x["score_raw"] for x in items], dtype=torch.float32),
        "wav_file_name": [x["wav_file_name"] for x in items],
    }
