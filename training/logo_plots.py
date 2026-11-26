"""Diagnostics from saved LOGO CSVs only; never invokes a model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_fold(root: Path) -> None:
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "public_test"):
        rows = _rows(root / "predictions" / f"{split}.csv")
        if not rows:
            continue
        truth = np.asarray([float(r["ground_truth"]) for r in rows])
        pred = np.asarray([float(r["predicted_score"]) for r in rows])
        weights = np.asarray([[float(r[f"expert_{i}_weight"]) for i in range(1, 9)] for r in rows])
        entropy = np.asarray([float(r["gate_entropy"]) for r in rows])
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].hist(truth, bins=np.linspace(0, 10, 21), alpha=.6, label="truth")
        axes[0, 0].hist(pred, bins=np.linspace(0, 10, 21), alpha=.6, label="prediction")
        axes[0, 0].legend(); axes[0, 0].set_title("Scores, fixed 0–10 bins")
        axes[0, 1].scatter(truth, pred, s=8, alpha=.5)
        axes[0, 1].plot([0, 10], [0, 10], color="black", lw=1)
        axes[0, 1].set(xlim=(0, 10), ylim=(0, 10), title="Truth vs prediction")
        axes[0, 2].bar(range(1, 9), weights.mean(axis=0)); axes[0, 2].set_title("Mean gate weights")
        axes[1, 0].hist(entropy, bins=np.linspace(0, 1, 21)); axes[1, 0].set_title("Normalized gate entropy")
        axes[1, 1].hist(pred-truth, bins=30); axes[1, 1].set_title("Prediction bias")
        axes[1, 2].boxplot([truth, pred], tick_labels=["truth", "prediction"])
        axes[1, 2].set(ylim=(0, 10), title="Score distribution")
        fig.suptitle(f"{root.name}: {split}")
        fig.tight_layout(); fig.savefig(plots / f"{split}_diagnostics.png", dpi=130); plt.close(fig)
    history = _rows(root / "training_history.csv")
    if history:
        epoch = [int(r["epoch"]) + 1 for r in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epoch, [float(r["train_loss"]) for r in history], label="optimization")
        axes[0].plot(epoch, [float(r["holdout_loss"]) for r in history], label="internal holdout")
        axes[0].legend(); axes[0].set_title("Loss")
        axes[1].plot(epoch, [float(r["holdout_srcc"]) if r["holdout_srcc"] not in ("", "None")
                              else np.nan for r in history])
        best = [int(r["epoch"]) + 1 for r in history if r["is_best"] == "True"]
        if best:
            axes[1].axvline(best[-1], color="red", linestyle="--", label="selected epoch")
            axes[1].legend()
        axes[1].set_title("Internal-holdout SRCC")
        fig.tight_layout(); fig.savefig(plots / "training.png", dpi=130); plt.close(fig)


def plot_summary(root: Path) -> None:
    summary = root / "summary"
    rows = _rows(summary / "all_fold_metrics.csv")
    plots = summary / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    folds = sorted({r["fold_id"] for r in rows})
    splits = ("train", "validation", "public_test")
    for metric in ("srcc", "lcc", "ktau", "mse"):
        fig, ax = plt.subplots(figsize=(max(8, len(folds)*1.3), 4))
        for j, split in enumerate(splits):
            values = {r["fold_id"]: r[metric] for r in rows if r["split"] == split}
            ax.bar(np.arange(len(folds)) + (j-1)*.25,
                   [float(values[f]) if values.get(f) not in (None, "", "None") else np.nan for f in folds],
                   width=.24, label=split)
        ax.set_xticks(range(len(folds)), folds); ax.set_title(f"Per-generator {metric.upper()}")
        ax.legend(); fig.tight_layout(); fig.savefig(plots / f"{metric}_by_generator.png", dpi=130); plt.close(fig)
    weight_rows = _rows(summary / "expert_weights.csv")
    if weight_rows:
        values = np.asarray([[float(r[f"expert_{i}_weight"]) for i in range(1, 9)] for r in weight_rows])
        fig, ax = plt.subplots(figsize=(9, max(4, len(values)*.4)))
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(values)), [f"{r['fold_id']} {r['split']}" for r in weight_rows])
        ax.set_xticks(range(8), [f"E{i}" for i in range(1, 9)])
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plots / "gate_weight_heatmap.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(folds)*1.3), 4))
    for j, metric in enumerate(("ground_truth_std", "prediction_std")):
        values = {r["fold_id"]: r[metric] for r in rows if r["split"] == "public_test"}
        ax.bar(np.arange(len(folds)) + (j-.5)*.35,
               [float(values[f]) if values.get(f) not in (None, "", "None") else np.nan for f in folds],
               width=.34, label=metric)
    ax.set_xticks(range(len(folds)), folds); ax.legend(); ax.set_title("Public-test score standard deviation")
    fig.tight_layout(); fig.savefig(plots / "score_std.png", dpi=130); plt.close(fig)
    aggregate = json.loads((summary / "aggregate_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, split in enumerate(splits):
        values = aggregate[split]
        ax.bar(np.arange(3) + (j-1)*.25,
               [values.get(key) if values.get(key) is not None else np.nan
                for key in ("logo_macro_srcc", "logo_weighted_srcc", "logo_pooled_srcc")],
               width=.24, label=split)
    ax.set_xticks(range(3), ["Macro", "Weighted", "Pooled"]); ax.legend()
    ax.set_title("LOGO SRCC research summaries")
    fig.tight_layout(); fig.savefig(plots / "aggregate_logo_srcc.png", dpi=130); plt.close(fig)
