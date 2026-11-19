"""Auditable score semantics and train-only diagnostics for Full7."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from model.expert_adapters.registry import FULL7_EXPERT_ORDER


_FIELDS = {"source", "semantics", "target_scale", "output_scale", "conversion", "identity", "audit_status"}
_SOURCES = ("expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "m2d_lora")
_SEMANTICS = ("clap_similarity", "mga_similarity", "fine_tuned_normalized_regression",
              "tanh_regression", "set_a_normalized_residual", "set_b_normalized_residual",
              "m2d_lora_coral_regression_normalized")


def validate_score_contracts(contracts: Mapping, *, synthetic: bool) -> None:
    if not isinstance(contracts, Mapping) or set(contracts) != set(FULL7_EXPERT_ORDER):
        raise ValueError("Full7 requires seven explicit score contracts")
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        entry = contracts[name]
        if not isinstance(entry, Mapping) or not _FIELDS.issubset(entry):
            raise ValueError(f"{name} score contract is incomplete")
        if any(not isinstance(entry[key], str) or not entry[key] for key in _FIELDS):
            raise ValueError(f"{name} score contract contains an empty field")
        if entry["source"] != _SOURCES[index]:
            raise ValueError(f"{name} score source contradicts Full7 mapping")
        if not synthetic and entry["semantics"] != _SEMANTICS[index]:
            raise ValueError(f"{name} score semantics contradict frozen source semantics")
        expected_scale = "unitless_similarity_proxy" if index < 2 else "normalized_minus1_to1"
        if not synthetic and entry["output_scale"] != expected_scale:
            raise ValueError(f"{name} score output scale contradicts source semantics")
        if entry["audit_status"] == "UNVERIFIED":
            raise ValueError(f"{name} score semantics are UNVERIFIED")
        if synthetic and entry["audit_status"] != "SYNTHETIC_VERIFIED":
            raise ValueError(f"{name} synthetic score provenance mismatch")
        if not synthetic and entry["audit_status"] != "CODE_VERIFIED":
            raise ValueError(f"{name} real score contract has not been verified")
        if entry["target_scale"] != "normalized_minus1_to1" or entry["conversion"] != "identity" or entry["output_scale"] not in {
            "normalized_minus1_to1", "unitless_similarity_proxy"
        }:
            raise ValueError(f"{name} score conversion/scale contradicts the frozen gate contract")
        if len(entry["identity"]) != 64 or any(c not in "0123456789abcdef" for c in entry["identity"].lower()):
            raise ValueError(f"{name} score identity must be SHA-256")


def audit_train_scores(scores: np.ndarray, labels_raw: np.ndarray) -> dict:
    scores = np.asarray(scores)
    labels = np.asarray(labels_raw)
    if scores.ndim != 2 or scores.shape[1] != 7 or scores.shape[0] == 0 or scores.dtype != np.float32:
        raise ValueError("train scores must be float32 [B,7]")
    if labels.shape != (scores.shape[0],) or labels.dtype != np.float32:
        raise ValueError("train labels must be float32 [B]")
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 10)):
        raise ValueError("train labels must be finite raw 0–10 values")

    def stats(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("score audit has no finite values")
        return {"min": float(finite.min()), "max": float(finite.max()),
                "mean": float(finite.mean()), "std": float(finite.std()),
                "quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
                "non_finite_count": int(values.size - finite.size),
                "outside_minus1_to1_fraction": float(np.mean((finite < -1) | (finite > 1)))}

    experts = {}
    for index, name in enumerate(FULL7_EXPERT_ORDER):
        values = scores[:, index]
        entry = stats(values)
        valid = np.isfinite(values)
        raw_proxy = 5.0 * values[valid].astype(np.float64) + 5.0
        targets = labels[valid].astype(np.float64)
        correlation = (float(np.corrcoef(raw_proxy, targets)[0, 1])
                       if raw_proxy.size > 1 and np.std(raw_proxy) > 0 and np.std(targets) > 0 else None)
        if correlation is not None and not math.isfinite(correlation):
            correlation = None
        entry["train_label_diagnostics"] = {
            "raw_proxy_mse": float(np.mean((raw_proxy - targets) ** 2)),
            "raw_proxy_mae": float(np.mean(np.abs(raw_proxy - targets))),
            "pearson": correlation,
        }
        experts[name] = entry
    result = {"sample_count": int(scores.shape[0]), "experts": experts,
              "labels_raw": stats(labels)}
    return result
