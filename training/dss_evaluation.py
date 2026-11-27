"""Complete, non-replacement DSS inference and split/domain reporting."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dss_dataset import collate_dss
from training.moe_evaluator import compute_metrics
from training.target_transform import denormalize_predictions


def predict_rows(gate, dataset, device, *, batch_size=64, checkpoint_sha256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_dss)
    gate.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            weights = gate(batch["features"].to(device))
            scores = batch["scores"].to(device)
            prediction = (weights * scores).sum(dim=1)
            if not torch.isfinite(prediction).all() or not torch.isfinite(weights).all():
                raise FloatingPointError("nonfinite DSS inference scores or gate weights")
            if not torch.allclose(weights.sum(dim=1), torch.ones_like(prediction), atol=1e-5):
                raise ValueError("DSS gate weights do not sum to one")
            pred_raw = denormalize_predictions(prediction).cpu().tolist()
            pred_norm = prediction.cpu().tolist()
            for index, sample_id in enumerate(batch["sample_ids"]):
                row = {
                    "sample_id": sample_id,
                    "source_split": dataset.split,
                    "wav_file_name": batch["wave_file_names"][index],
                    "model_id": int(batch["model_ids"][index]),
                    "model_name": batch["model_names"][index],
                    "ground_truth_raw": float(batch["labels_raw"][index]),
                    "ground_truth_normalized": (float(batch["labels_raw"][index]) - 5) / 5,
                    "prediction_raw": pred_raw[index],
                    "prediction_normalized": pred_norm[index],
                    "checkpoint_sha256": checkpoint_sha256,
                }
                row.update({f"expert_score_{i+1}": float(value) for i, value in enumerate(batch["scores"][index])})
                row.update({f"gate_weight_{i+1}": float(value) for i, value in enumerate(weights[index].cpu())})
                rows.append(row)
    if [row["sample_id"] for row in rows] != dataset.store.sample_ids:
        raise ValueError(f"{dataset.split}: inference did not cover every official cache sample once")
    return rows


def _group_diagnostics(rows, split, domain):
    weights = np.asarray([[float(row[f"gate_weight_{i+1}"]) for i in range(8)] for row in rows])
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1, atol=1e-5):
        raise ValueError("nonfinite or non-simplex DSS gate weights")
    target = np.asarray([float(row["ground_truth_raw"]) for row in rows])
    prediction = np.asarray([float(row["prediction_raw"]) for row in rows])
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / math.log(8)
    top = np.bincount(weights.argmax(axis=1), minlength=8) / len(rows)
    result = {
        "split": split, "domain": domain, "sample_count": len(rows),
        "normalized_entropy": float(entropy.mean()),
        "average_max_weight": float(weights.max(axis=1).mean()),
        "ground_truth_mean": float(target.mean()), "ground_truth_std": float(target.std()),
        "prediction_mean": float(prediction.mean()), "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()),
        "prediction_bias": float((prediction - target).mean()),
        "prediction_to_target_std_ratio": (float(prediction.std() / target.std()) if target.std() > 0 else None),
    }
    for index in range(8):
        result[f"expert_{index+1}_mean_weight"] = float(weights[:, index].mean())
        result[f"expert_{index+1}_std_weight"] = float(weights[:, index].std())
        result[f"expert_{index+1}_top1_fraction"] = float(top[index])
    return result


def summarize_predictions(rows, split, *, expected_count):
    if len(rows) != expected_count:
        raise ValueError(f"{split}: prediction count {len(rows)} differs from {expected_count}")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split}: duplicate prediction sample identities")
    if any(not np.isfinite(float(row[field])) for row in rows for field in ("prediction_raw", "ground_truth_raw")):
        raise ValueError(f"{split}: nonfinite prediction or target")
    global_metrics = compute_metrics(
        [row["prediction_raw"] for row in rows], [row["ground_truth_raw"] for row in rows]
    )
    global_metrics["split"] = split
    domains = sorted({int(row["model_id"]) for row in rows})
    per_domain, diagnostics = [], [_group_diagnostics(rows, split, "global")]
    for domain in domains:
        group = [row for row in rows if int(row["model_id"]) == domain]
        metrics = compute_metrics(
            [row["prediction_raw"] for row in group], [row["ground_truth_raw"] for row in group]
        )
        per_domain.append({"split": split, "model_id": domain, "model_name": group[0]["model_name"], **metrics})
        diagnostics.append(_group_diagnostics(group, split, group[0]["model_name"]))
    valid = [row for row in per_domain if row["srcc"] is not None]
    srccs = np.asarray([row["srcc"] for row in valid])
    aggregate = {
        "split": split, "sample_coverage": len(rows), "valid_domain_count": len(valid),
        "global_pooled_srcc": global_metrics["srcc"],
        "macro_srcc": float(srccs.mean()) if len(srccs) else None,
        "macro_synthetic_srcc": (float(np.mean([row["srcc"] for row in valid if row["model_id"] != 0]))
                                 if any(row["model_id"] != 0 for row in valid) else None),
        "weighted_srcc": (float(np.average(srccs, weights=[row["sample_count"] for row in valid]))
                          if len(valid) else None),
        "worst_domain_srcc": float(srccs.min()) if len(srccs) else None,
        "per_domain_srcc_population_std": float(srccs.std()) if len(srccs) else None,
        "macro_lcc": _defined_mean(per_domain, "lcc"),
        "macro_ktau": _defined_mean(per_domain, "ktau"),
        "macro_mse": _defined_mean(per_domain, "mse"),
    }
    aggregate["pooled_minus_macro_srcc"] = (
        global_metrics["srcc"] - aggregate["macro_srcc"]
        if global_metrics["srcc"] is not None and aggregate["macro_srcc"] is not None else None
    )
    return global_metrics, per_domain, aggregate, diagnostics


def _defined_mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_final_evaluation(output_dir, predictions, training_history):
    """Write required metrics, diagnostics, prediction CSVs and plots."""
    root = Path(output_dir)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    expected = {"train": 7500, "validation": 3000, "test": 3000}
    global_metrics, per_domain, aggregates, diagnostics = {}, [], {}, []
    for split, rows in predictions.items():
        write_csv(root / "predictions" / f"{'public_test' if split == 'test' else split}.csv", rows)
        g, p, a, d = summarize_predictions(rows, split, expected_count=expected[split])
        global_metrics[split] = g
        per_domain.extend(p)
        aggregates[split] = a
        diagnostics.extend(d)
    if set(predictions) != set(expected):
        raise ValueError("final DSS evaluation requires all three official splits")
    if len({r["sample_id"] for rows in predictions.values() for r in rows}) != 13500:
        raise ValueError("final DSS prediction identities do not total 13,500")
    (root / "metrics" / "global_metrics.json").write_text(json.dumps(global_metrics, indent=2))
    (root / "metrics" / "aggregate_metrics.json").write_text(json.dumps(aggregates, indent=2))
    write_csv(root / "metrics" / "per_generator_metrics.csv", per_domain)
    write_csv(root / "metrics" / "gate_diagnostics.csv", diagnostics)
    _plot_results(root / "plots", predictions, training_history, per_domain, aggregates, diagnostics)
    return global_metrics, per_domain, aggregates


def _plot_results(directory, predictions, history, per_domain, aggregates, diagnostics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for axis, field, title in zip(axes, ("global_loss", "domain_loss", "total_loss"),
                                  ("Global Soft Spearman", "Domain-wise Soft Spearman", "Hybrid loss")):
        axis.plot(epochs, [row[field] for row in history])
        axis.set(ylabel="Loss", title=title)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(directory / "training_loss.png"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(epochs, [row["validation_srcc"] for row in history], marker="o")
    best = max((row for row in history if row["validation_srcc"] is not None), key=lambda x: x["validation_srcc"])
    axis.scatter([best["epoch"]], [best["validation_srcc"]], color="red", label=f"Selected epoch {best['epoch']}")
    axis.set(xlabel="Epoch", ylabel="Full validation hard SRCC", title="DSS checkpoint selection")
    axis.legend(); fig.tight_layout(); fig.savefig(directory / "validation_srcc.png"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5))
    names = [row["model_name"] for row in per_domain if row["split"] == "train"]
    x = np.arange(len(names))
    for offset, split in zip((-.25, 0, .25), ("train", "validation", "test")):
        scores = [row["srcc"] if row["srcc"] is not None else np.nan for row in per_domain if row["split"] == split]
        axis.bar(x + offset, scores, width=.24, label=split)
    axis.set(xticks=x, xticklabels=names, ylabel="SRCC", title="Per-domain DSS ranking")
    axis.tick_params(axis="x", rotation=30); axis.legend(); fig.tight_layout()
    fig.savefig(directory / "per_generator_srcc.png"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for split in ("train", "validation", "test"):
        axis.scatter([split], [aggregates[split]["global_pooled_srcc"]], label="Pooled" if split == "train" else None, color="C0")
        axis.scatter([split], [aggregates[split]["macro_srcc"]], label="Macro" if split == "train" else None, color="C1")
    axis.set(ylabel="SRCC", title="Pooled versus macro SRCC"); axis.legend(); fig.tight_layout()
    fig.savefig(directory / "global_vs_macro_srcc.png"); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for row_index, split in enumerate(("train", "validation", "test")):
        rows = predictions[split]
        target = [row["ground_truth_raw"] for row in rows]
        pred = [row["prediction_raw"] for row in rows]
        axes[row_index, 0].hist(target, bins=np.linspace(0, 10, 41), alpha=.6, label="Ground truth")
        axes[row_index, 0].hist(pred, bins=np.linspace(0, 10, 41), alpha=.6, label="Prediction")
        axes[row_index, 0].set(title=f"{split} score distributions", xlabel="Raw score", ylabel="Count")
        axes[row_index, 0].legend()
        axes[row_index, 1].scatter(target, pred, s=2, alpha=.25)
        axes[row_index, 1].set(title=f"{split}: prediction vs ground truth", xlabel="Ground truth raw", ylabel="Prediction raw")
    fig.tight_layout(); fig.savefig(directory / "score_distributions.png"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    global_rows = [row for row in diagnostics if row["domain"] == "global"]
    x = np.arange(8)
    for offset, row in zip((-.25, 0, .25), global_rows):
        axis.bar(x + offset, [row[f"expert_{i+1}_mean_weight"] for i in range(8)], width=.24, label=row["split"])
    axis.set(xticks=x, xticklabels=[f"E{i+1}" for i in range(8)], ylabel="Mean gate weight", title="DSS expert routing")
    axis.legend(); fig.tight_layout(); fig.savefig(directory / "expert_weights.png"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar([row["split"] for row in global_rows], [row["normalized_entropy"] for row in global_rows])
    axis.set(ylabel="Normalized entropy", title="DSS routing entropy", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(directory / "routing_entropy.png"); plt.close(fig)
