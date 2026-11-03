"""Training/evaluation batches over verified expert-output feature stores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from data.moe_feature_store import MoeFeatureStoreDataset
from model.expert_adapters.registry import EXPECTED_FEATURE_WIDTH


class MoeTrainingDataset(Dataset):
    """Split-explicit facade over cached scores/features; never reads WAVs."""

    def __init__(self, root: str, *, split: str, require_labels: bool | None = None):
        self.store = MoeFeatureStoreDataset(root, split=split, require_labels=require_labels)
        self.split = split
        self.identity = self.store.identity
        self.manifest = self.store.manifest

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.store[index]


def collate_moe_features(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise ValueError("cannot collate an empty feature batch")
    splits = {sample.get("split") for sample in samples}
    if len(splits) != 1 or None in splits:
        raise ValueError("a feature batch must contain records from exactly one split")
    sample_ids = [sample.get("sample_id") for sample in samples]
    filenames = [sample.get("wave_file_name") for sample in samples]
    if any(not isinstance(value, str) or not value for value in sample_ids + filenames):
        raise ValueError("feature batch requires non-empty sample IDs and filenames")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("feature batch contains duplicate sample IDs")
    if len(filenames) != len(set(filenames)):
        raise ValueError("feature batch contains duplicate filenames")

    score_tensors = [sample.get("scores") for sample in samples]
    feature_tensors = [sample.get("features") for sample in samples]
    if any(not isinstance(value, torch.Tensor) or value.shape != (4,) or value.dtype != torch.float32 for value in score_tensors):
        raise ValueError("each sample score must be CPU float32 [4]")
    if any(not isinstance(value, torch.Tensor) or value.shape != (EXPECTED_FEATURE_WIDTH,) or value.dtype != torch.float32 for value in feature_tensors):
        raise ValueError("each sample feature must be CPU float32 [7936]")
    if any(value.device.type != "cpu" for value in score_tensors + feature_tensors):
        raise ValueError("feature-store batches must remain on CPU")
    if any(not torch.isfinite(value).all().item() for value in score_tensors + feature_tensors):
        raise ValueError("feature batch contains non-finite scores or features")

    label_present = ["label_raw" in sample for sample in samples]
    if any(label_present) and not all(label_present):
        raise ValueError("feature batch cannot mix labeled and unlabeled records")
    class_present = ["num_class" in sample for sample in samples]
    if any(class_present) and not all(class_present):
        raise ValueError("feature batch cannot mix records with and without num_class")

    batch: dict[str, Any] = {
        "sample_ids": sample_ids,
        "wave_file_names": filenames,
        "split": next(iter(splits)),
        "scores": torch.stack(score_tensors, dim=0),
        "features": torch.stack(feature_tensors, dim=0),
    }
    if all(label_present):
        labels = [sample["label_raw"] for sample in samples]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 0 or value.dtype != torch.float32 for value in labels):
            raise ValueError("label_raw must be a float32 scalar per sample")
        batch["labels_raw"] = torch.stack(labels, dim=0)
        batch["labels_scale"] = "raw_0_to_10"
    if all(class_present):
        classes = [sample["num_class"] for sample in samples]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 0 or value.dtype != torch.int64 for value in classes):
            raise ValueError("num_class must be an int64 scalar per sample")
        batch["num_class"] = torch.stack(classes, dim=0)
    return batch


__all__ = ["MoeTrainingDataset", "collate_moe_features"]
