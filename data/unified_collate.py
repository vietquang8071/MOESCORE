"""Order-preserving CPU collation for :mod:`data.unified_dataset`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_WAVEFORM_SHAPES = {
    "expert1_wavs": (1, 480_000),
    "expert2_wavs": (1, 320_000),
    "expert3_wavs": (1, 160_000),
    "expert4_wavs": (1, 160_000),
}
_REQUIRED = frozenset(
    {
        "sample_id",
        "wave_file_name",
        "split",
        "captions_raw",
        "expert1_wavs",
        "expert2_wavs",
        "expert3_wavs",
        "expert4_wavs",
        "expert4_wav_lens",
    }
)


def _valid_length(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype != torch.int64 or value.device.type != "cpu":
            raise ValueError("expert4_wav_lens must be a CPU int64 scalar per sample")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 160_000:
        raise ValueError("expert4_wav_lens must be a positive integer no greater than 160000")
    return value


def collate_unified(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size expert waveforms and preserve all row-level ordering.

    Collation is intentionally a shape/value validation boundary: it performs
    no resampling, padding, caption normalization, shuffling or device transfer.
    """
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        raise ValueError("Unified batch cannot be empty")
    batch_size = len(samples)
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"Sample {index} must be a mapping")
        missing = sorted(_REQUIRED - set(sample))
        if missing:
            raise ValueError(f"Sample {index} is missing required fields: {', '.join(missing)}")
        if sample["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Sample {index} has invalid split {sample['split']!r}")
        if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
            raise ValueError(f"Sample {index} has invalid sample_id")
        if not isinstance(sample["wave_file_name"], str) or not sample["wave_file_name"]:
            raise ValueError(f"Sample {index} has invalid wave_file_name")
        caption = sample["captions_raw"]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Sample {index} has invalid captions_raw")
        _valid_length(sample["expert4_wav_lens"])
        for key, expected_shape in _WAVEFORM_SHAPES.items():
            waveform = sample[key]
            if not isinstance(waveform, torch.Tensor):
                raise TypeError(f"Sample {index} field {key} must be a torch.Tensor")
            if tuple(waveform.shape) != expected_shape:
                raise ValueError(f"Sample {index} field {key} must have shape {expected_shape}; got {tuple(waveform.shape)}")
            if waveform.dtype != torch.float32 or waveform.device.type != "cpu":
                raise ValueError(f"Sample {index} field {key} must be CPU torch.float32")
            if not torch.isfinite(waveform).all():
                raise ValueError(f"Sample {index} field {key} contains non-finite values")

        if "label" in sample:
            label = sample["label"]
            if not isinstance(label, torch.Tensor) or label.ndim != 0 or label.dtype != torch.float32 or label.device.type != "cpu":
                raise ValueError(f"Sample {index} label must be a CPU float32 scalar")
            value = float(label.item())
            if not math.isfinite(value) or not 0.0 <= value <= 10.0:
                raise ValueError(f"Sample {index} label must be finite and in [0,10]")

    splits = {sample["split"] for sample in samples}
    if len(splits) != 1:
        raise ValueError("A unified batch cannot mix dataset splits")
    labels_present = ["label" in sample for sample in samples]
    if any(labels_present) and not all(labels_present):
        raise ValueError("A unified batch cannot mix labeled and unlabeled samples")
    for optional in ("model_id", "model_name"):
        presence = [optional in sample for sample in samples]
        if any(presence) and not all(presence):
            raise ValueError(f"A unified batch cannot mix samples with and without {optional} metadata")
        if all(presence) and any(not isinstance(sample[optional], str) for sample in samples):
            raise ValueError(f"{optional} metadata must contain strings")

    result: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in _WAVEFORM_SHAPES
    }
    result["expert4_wav_lens"] = torch.tensor(
        [_valid_length(sample["expert4_wav_lens"]) for sample in samples], dtype=torch.int64
    )
    result["captions_raw"] = [sample["captions_raw"] for sample in samples]
    result["labels"] = (
        torch.stack([sample["label"] for sample in samples], dim=0)
        if all(labels_present)
        else None
    )
    if result["labels"] is None:
        result.pop("labels")
    result["wave_file_names"] = [sample["wave_file_name"] for sample in samples]
    result["sample_ids"] = [sample["sample_id"] for sample in samples]
    result["split"] = next(iter(splits))
    if all("model_id" in sample for sample in samples):
        result["model_ids"] = [sample["model_id"] for sample in samples]
    if all("model_name" in sample for sample in samples):
        result["model_names"] = [sample["model_name"] for sample in samples]
    return result


__all__ = ["collate_unified"]
