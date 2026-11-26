import torch
import pytest

from training.logo_runner import LogoView, select_best_epoch
from data.moe_training_dataset import collate_moe_features


class TinyStore:
    def __init__(self, split):
        self.split = split

    def __getitem__(self, index):
        return {"sample_id": f"{self.split}:{index}.wav", "wave_file_name": f"{index}.wav",
                "split": self.split, "expert_profile": "full_8",
                "scores": torch.zeros(8), "features": torch.zeros(33032)}


def test_view_preserves_source_split_and_attaches_test_label():
    rows = [{"sample_id": "test:0.wav", "source_split": "test", "wave_file_name": "0.wav",
             "ground_truth": 7.0, "model_id": "a", "model_name": "A"}]
    view = LogoView(TinyStore("test"), rows, [0], "public_test")
    sample = view[0]
    assert sample["source_split"] == "test"
    assert sample["split"] == "validation"
    assert sample["fold_role"] == "public_test"
    assert collate_moe_features([sample])["labels_raw"].tolist() == [7.0]


def test_best_epoch_uses_finite_srcc_and_earliest_tie():
    assert select_best_epoch(None, 0.4, 1) == (0.4, 1)
    assert select_best_epoch((0.4, 1), 0.4, 2) == (0.4, 1)
    assert select_best_epoch((0.4, 1), None, 2) == (0.4, 1)
    assert select_best_epoch((0.4, 1), float("nan"), 2) == (0.4, 1)
    assert select_best_epoch(None, None, 1) is None
    assert select_best_epoch((0.4, 1), 0.5, 3) == (0.5, 3)
