"""Deterministic four-domain, 32-sample queues for DSS training."""

from __future__ import annotations

from collections import Counter
import random

from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, model_ids, *, seed: int = 2027, steps_per_epoch: int = 59):
        self.indices = {domain: [i for i, value in enumerate(model_ids) if value == domain]
                        for domain in sorted(set(model_ids))}
        if len(self.indices) != 9 or any(len(rows) < 32 for rows in self.indices.values()):
            raise ValueError("DSS needs nine domains with at least 32 distinct training samples each")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.rng = random.Random(seed)
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0
        self.queues = {domain: self.rng.sample(rows, len(rows)) for domain, rows in self.indices.items()}
        self.cursors = dict.fromkeys(self.indices, 0)
        self.cumulative_seen = set()
        self.last_history = None

    def __len__(self):
        return self.steps_per_epoch

    @property
    def cumulative_unique_count(self):
        return len(self.cumulative_seen)

    def _draw(self, domain: int):
        chosen = []
        while len(chosen) < 32:
            if self.cursors[domain] == len(self.queues[domain]):
                queue = self.rng.sample(self.indices[domain], len(self.indices[domain]))
                self.queues[domain] = [i for i in queue if i not in chosen] + [i for i in queue if i in chosen]
                self.cursors[domain] = 0
            candidate = self.queues[domain][self.cursors[domain]]
            self.cursors[domain] += 1
            if candidate not in chosen:
                chosen.append(candidate)
        return chosen

    def __iter__(self):
        domain_order = self.rng.sample(list(self.indices), len(self.indices))
        appearances = Counter()
        drawn = []
        for step in range(self.steps_per_epoch):
            domains = [domain_order[(step * 4 + offset) % 9] for offset in range(4)]
            batch = [index for domain in domains for index in self._draw(domain)]
            appearances.update(domains)
            drawn.extend(batch)
            self.cumulative_seen.update(batch)
            yield batch
        self.epoch += 1
        self.last_history = {
            "epoch": self.epoch, "domain_appearances": dict(appearances),
            "total_draws": len(drawn), "unique_samples": len(set(drawn)),
            "repeated_samples": len(drawn) - len(set(drawn)),
            "cumulative_unique_samples": len(self.cumulative_seen),
            "cumulative_per_domain_unique_samples": {
                domain: len(self.cumulative_seen.intersection(rows)) for domain, rows in self.indices.items()},
            "per_domain_draws": {domain: 32 * appearances[domain] for domain in self.indices},
        }

    def state_dict(self):
        return {"indices": {key: list(value) for key, value in self.indices.items()},
                "rng_state": self.rng.getstate(),
                "queues": {key: list(value) for key, value in self.queues.items()},
                "cursors": dict(self.cursors), "epoch": self.epoch,
                "cumulative_seen": sorted(self.cumulative_seen)}

    def load_state_dict(self, state):
        if state["indices"] != self.indices:
            raise ValueError("sampler state belongs to a different training split")
        self.rng.setstate(state["rng_state"])
        self.queues = state["queues"]
        self.cursors = state["cursors"]
        self.epoch = state["epoch"]
        self.cumulative_seen = set(state["cumulative_seen"])
