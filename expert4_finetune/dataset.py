"""Average-score XACLE examples for independent Expert 4 training."""
import csv
import math
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.nn import functional as F
from torch.utils.data import Dataset


def _column(headers, choices, required=True):
    matches = [name for name in headers if name.lower() in choices]
    if len(matches) > 1 or (not matches and required):
        raise ValueError(f"Expected one column from {choices}; found {matches}, headers={headers}")
    return matches[0] if matches else None


class Expert4Dataset(Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate=16000, max_seconds=10, require_score=True):
        self.csv_path, self.audio_dir = Path(csv_path), Path(audio_dir)
        self.sample_rate, self.max_samples = sample_rate, int(sample_rate * max_seconds)
        if not self.csv_path.is_file() or not self.audio_dir.is_dir():
            raise FileNotFoundError(f"CSV or audio directory missing: {self.csv_path}, {self.audio_dir}")
        with self.csv_path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            filename = _column(headers, {'wav_file_name', 'wave_file_name'})
            caption = _column(headers, {'text'})
            # Ignore test labels even when a score column happens to be present.
            score = _column(headers, {'average_score', 'average_semantic_alignement_score', 'average_semantic_alignment_score'}) if require_score else None
            model_id = _column(headers, {'model_id'}, False)
            model_name = _column(headers, {'model_name'}, False)
            self.rows = []
            seen = set()
            model_names_by_id = {}
            for index, row in enumerate(reader, start=2):
                name, text = (row[filename] or '').strip(), (row[caption] or '').strip()
                try:
                    raw = float(row[score]) if score is not None else None
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid score at CSV row {index}") from exc
                if not name or not text or (raw is not None and (not math.isfinite(raw) or not 0 <= raw <= 10)):
                    raise ValueError(f"Invalid filename, caption, or score at CSV row {index}")
                path = (self.audio_dir / name.lstrip('/')).resolve()
                if not path.is_relative_to(self.audio_dir.resolve()) or not path.is_file():
                    raise FileNotFoundError(f"Audio missing or outside audio directory at CSV row {index}: {path}")
                if (name, text) in seen:
                    raise ValueError(f"Duplicate audio-text pair at CSV row {index}: {name}")
                seen.add((name, text))
                model_id_value = (row.get(model_id) or '').strip() if model_id else ''
                model_name_value = (row.get(model_name) or '').strip() if model_name else ''
                if model_id and model_name:
                    if bool(model_id_value) != bool(model_name_value):
                        raise ValueError(f'Inconsistent model metadata at CSV row {index}')
                    previous = model_names_by_id.get(model_id_value)
                    if model_id_value and previous is not None and previous != model_name_value:
                        raise ValueError(f'Inconsistent model metadata for ID {model_id_value} at CSV row {index}')
                    if model_id_value:
                        model_names_by_id[model_id_value] = model_name_value
                self.rows.append({'path': path, 'sample_id': name, 'caption': text, 'score_raw': raw,
                                  'model_id': model_id_value, 'model_name': model_name_value})
        if not self.rows:
            raise ValueError(f"No examples in {self.csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            sr, wav = wavfile.read(row['path'])
        except Exception as exc:
            raise ValueError(f"Cannot read WAV {row['path']}: {exc}") from exc
        if sr <= 0 or wav.ndim not in (1, 2) or len(wav) == 0:
            raise ValueError(f"Invalid waveform or sample rate in {row['path']}")
        if np.issubdtype(wav.dtype, np.integer):
            info = np.iinfo(wav.dtype)
            if info.min == 0:
                wav = (wav.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
            else:
                wav = wav.astype(np.float32) / max(abs(info.min), info.max)
        else:
            wav = wav.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if not np.isfinite(wav).all():
            raise ValueError(f"Non-finite waveform in {row['path']}")
        if sr != self.sample_rate:
            warnings.warn(f"Resampling {row['path']} from {sr} Hz to {self.sample_rate} Hz", RuntimeWarning)
            ratio = Fraction(self.sample_rate, int(sr))
            wav = resample_poly(wav, ratio.numerator, ratio.denominator).astype(np.float32)
        wav = wav[:self.max_samples]
        return {**row, 'wav': torch.from_numpy(wav.copy()), 'max_samples': self.max_samples}


def collate_expert4(batch):
    if not batch or len({item['max_samples'] for item in batch}) != 1:
        raise ValueError('Batch must contain examples with consistent maximum duration')
    maximum = batch[0]['max_samples']
    lengths = torch.tensor([item['wav'].numel() for item in batch], dtype=torch.long)
    wavs = torch.stack([F.pad(item['wav'], (0, maximum - item['wav'].numel())) for item in batch]).unsqueeze(1)
    scores = [item['score_raw'] for item in batch]
    if any(value is None for value in scores) and not all(value is None for value in scores):
        raise ValueError('A batch cannot mix labeled and unlabeled examples')
    raw = None if scores[0] is None else torch.tensor(scores, dtype=torch.float32)
    return {'wavs': wavs, 'wav_lens': lengths, 'captions': [item['caption'] for item in batch],
            'scores': None if raw is None else (raw - 5.0) / 5.0, 'scores_raw': raw,
            'sample_ids': [item['sample_id'] for item in batch],
            'model_ids': [item['model_id'] for item in batch],
            'model_names': [item['model_name'] for item in batch]}
