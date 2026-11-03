"""Resumable deterministic order for feature-store training batches."""

from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class StatefulRandomSampler(Sampler[int]):
    """Epoch permutation with a serializable cursor for interruption-safe resume."""

    def __init__(self, data_source: Sized, *, seed: int = 2027, epoch: int = 0) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.data_source = data_source
        self.seed = seed
        self.epoch = epoch
        self.order: list[int] = []
        self.cursor = 0
        self._make_order()

    def _make_order(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(self.data_source), generator=generator).tolist()
        self.cursor = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if epoch != self.epoch:
            self.epoch = epoch
            self._make_order()

    def __iter__(self) -> Iterator[int]:
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self) -> int:
        return len(self.order) - self.cursor

    def state_dict(self) -> dict[str, int | list[int]]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "size": len(self.data_source),
            "order": list(self.order),
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("sampler state must be a mapping")
        expected_keys = {"seed", "epoch", "size", "order", "cursor"}
        if set(state) != expected_keys:
            raise ValueError("sampler state schema mismatch")
        size = len(self.data_source)
        order = state["order"]
        cursor = state["cursor"]
        if state["seed"] != self.seed or state["size"] != size:
            raise ValueError("sampler seed or dataset size differs from checkpoint")
        if type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("sampler epoch is invalid")
        if not isinstance(order, list) or sorted(order) != list(range(size)):
            raise ValueError("sampler order is not a permutation of the dataset")
        if type(cursor) is not int or not 0 <= cursor <= size:
            raise ValueError("sampler cursor is outside the saved order")
        self.epoch = state["epoch"]
        self.order = list(order)
        self.cursor = cursor


__all__ = ["StatefulRandomSampler"]
