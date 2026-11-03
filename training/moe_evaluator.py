"""No-gradient validation and label-free inference over feature-store batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
import torch

from loss.loss_function import get_loss_function
from model.moe_gating_core import GatingNetwork
from model.expert_adapters.registry import EXPECTED_FEATURE_WIDTH
from training.target_transform import denormalize_predictions, normalize_targets


def compute_metrics(prediction_raw: Any, target_raw: Any) -> dict[str, Any]:
    prediction = np.asarray(
        prediction_raw.detach().cpu().numpy() if isinstance(prediction_raw, torch.Tensor) else prediction_raw,
        dtype=np.float64,
    ).reshape(-1)
    target = np.asarray(
        target_raw.detach().cpu().numpy() if isinstance(target_raw, torch.Tensor) else target_raw,
        dtype=np.float64,
    ).reshape(-1)
    if prediction.size == 0 or prediction.shape != target.shape:
        raise ValueError("prediction_raw and target_raw must have the same non-empty shape")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("metric inputs must be finite")
    constant_input = np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0
    srcc: Optional[float]
    lcc: Optional[float]
    ktau: Optional[float]
    if constant_input:
        srcc = None
        srcc_status = "UNDEFINED_CONSTANT_INPUT"
        lcc = None
        lcc_status = "UNDEFINED_CONSTANT_INPUT"
        ktau = None
        ktau_status = "UNDEFINED_CONSTANT_INPUT"
    else:
        from scipy.stats import kendalltau, pearsonr, spearmanr

        def defined_correlation(value: Any) -> tuple[Optional[float], str]:
            correlation = float(value)
            if not math.isfinite(correlation):
                return None, "UNDEFINED_NUMERICAL_RESULT"
            return correlation, "DEFINED"

        srcc, srcc_status = defined_correlation(spearmanr(target, prediction).statistic)
        lcc, lcc_status = defined_correlation(pearsonr(target, prediction).statistic)
        ktau, ktau_status = defined_correlation(kendalltau(target, prediction, variant="b").statistic)
    errors = prediction - target
    return {
        "score_scale": "raw_0_to_10",
        "sample_count": int(prediction.size),
        "srcc": srcc,
        "srcc_status": srcc_status,
        "lcc": lcc,
        "lcc_status": lcc_status,
        "ktau": ktau,
        "ktau_status": ktau_status,
        "mse": float(np.mean(errors * errors)),
        "mae": float(np.mean(np.abs(errors))),
    }


class MoeGateEvaluator:
    """Run the actual sample-dependent gate in eval mode, without optimizer use."""

    def __init__(self, gating_network: GatingNetwork, loss_config: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(gating_network, GatingNetwork):
            raise TypeError("gating_network must be the verified GatingNetwork")
        self.gating_network = gating_network
        self.loss_function = get_loss_function(dict(loss_config)) if loss_config is not None else None
        self.uses_class_balancing = bool(
            loss_config is not None
            and (
                loss_config["clipped_mse"].get("cbl", False)
                or loss_config["contrastive"].get("cbl", False)
            )
        )

    def _batch_outputs(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("feature batch must be a mapping")
        required = {"scores", "features", "sample_ids", "wave_file_names"}
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"feature batch is missing fields: {missing}")
        scores, features = batch["scores"], batch["features"]
        ids, filenames = batch["sample_ids"], batch["wave_file_names"]
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[1] != 4:
            raise ValueError("scores must have shape [B,4]")
        batch_size = scores.shape[0]
        if not isinstance(features, torch.Tensor) or features.shape != (batch_size, EXPECTED_FEATURE_WIDTH):
            raise ValueError("features must have shape [B,7936]")
        if scores.dtype != torch.float32 or features.dtype != torch.float32:
            raise TypeError("feature-store inputs must be float32")
        if not torch.isfinite(scores).all().item() or not torch.isfinite(features).all().item():
            raise ValueError("scores and features must be finite")
        if not isinstance(ids, list) or not isinstance(filenames, list) or len(ids) != batch_size or len(filenames) != batch_size:
            raise ValueError("sample_ids and wave_file_names must be ordered lists of length B")
        if any(not isinstance(value, str) or not value for value in ids + filenames):
            raise ValueError("sample IDs and filenames must be non-empty strings")
        parameter = next(self.gating_network.parameters())
        scores = scores.to(device=parameter.device, dtype=parameter.dtype)
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        weights = self.gating_network(features)
        prediction = (
            scores[:, 0:1] * weights[:, 0:1]
            + scores[:, 1:2] * weights[:, 1:2]
            + scores[:, 2:3] * weights[:, 2:3]
            + scores[:, 3:4] * weights[:, 3:4]
        )
        if not torch.isfinite(prediction).all().item():
            raise ValueError("gating prediction contains non-finite values")
        return prediction.reshape(-1), weights, scores, features

    def evaluate(self, batches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Iterable):
            raise TypeError("validation batches must be iterable")
        self.gating_network.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        sample_ids: list[str] = []
        wave_names: list[str] = []
        regression_total = 0.0
        loss_sample_count = 0
        batch_count = 0
        with torch.no_grad():
            for batch in batches:
                batch_count += 1
                if batch.get("split") != "validation":
                    raise ValueError("evaluation requires validation split data")
                if "labels_raw" not in batch:
                    raise ValueError("validation metrics require raw labels")
                if batch.get("labels_scale") != "raw_0_to_10":
                    raise ValueError("validation labels must be explicitly marked as raw_0_to_10")
                prediction_norm, _, _, _ = self._batch_outputs(batch)
                labels = batch["labels_raw"]
                if not isinstance(labels, torch.Tensor) or labels.shape != prediction_norm.shape or labels.dtype != torch.float32:
                    raise ValueError("validation labels must be float32 [B]")
                labels = labels.to(device=prediction_norm.device)
                target_norm = normalize_targets(labels)
                if self.loss_function is not None:
                    classes = batch.get("num_class") if self.uses_class_balancing else None
                    if self.uses_class_balancing and (
                        not isinstance(classes, torch.Tensor)
                        or classes.shape != labels.shape
                        or classes.dtype != torch.int64
                        or torch.any(classes <= 0).item()
                    ):
                        raise ValueError("validation class balancing requires explicit num_class metadata")
                    if self.uses_class_balancing:
                        raise ValueError(
                            "the current original ClippedMSELoss class-balancing path broadcasts [B,1] weights against [B]; "
                            "validation loss is rejected until that source behavior is reviewed"
                        )
                    loss = self.loss_function(prediction_norm.reshape(-1), target_norm, classes)
                    regression_total += float(loss.item()) * len(labels)
                    loss_sample_count += len(labels)
                predictions.append(prediction_norm.detach().cpu())
                targets.append(labels.detach().cpu())
                sample_ids.extend(batch["sample_ids"])
                wave_names.extend(batch["wave_file_names"])
        if batch_count == 0:
            raise ValueError("validation requires at least one batch")
        if len(sample_ids) != len(set(sample_ids)) or len(wave_names) != len(set(wave_names)):
            raise ValueError("validation batches contain duplicate sample IDs or filenames")
        prediction_normalized = torch.cat(predictions)
        target_raw = torch.cat(targets)
        prediction_raw = denormalize_predictions(prediction_normalized)
        result = {
            "sample_ids": sample_ids,
            "wave_file_names": wave_names,
            "prediction_normalized": prediction_normalized,
            "prediction_raw": prediction_raw,
            "target_raw": target_raw,
            "metrics": compute_metrics(prediction_raw, target_raw),
        }
        if loss_sample_count:
            result["regression_loss_normalized"] = regression_total / loss_sample_count
        return result

    def predict(self, batches: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Iterable):
            raise TypeError("inference batches must be iterable")
        self.gating_network.eval()
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_files: set[str] = set()
        batch_count = 0
        with torch.no_grad():
            for batch in batches:
                batch_count += 1
                if batch.get("split") != "test":
                    raise ValueError("label-free prediction requires test split data")
                normalized, _, _, _ = self._batch_outputs(batch)
                raw = denormalize_predictions(normalized).cpu()
                for sample_id, filename, score in zip(batch["sample_ids"], batch["wave_file_names"], raw.tolist()):
                    if sample_id in seen_ids or filename in seen_files:
                        raise ValueError("test inference contains duplicate sample IDs or filenames")
                    if not math.isfinite(score):
                        raise ValueError("test prediction must be finite")
                    seen_ids.add(sample_id)
                    seen_files.add(filename)
                    rows.append({
                        "sample_id": sample_id,
                        "wave_file_name": filename,
                        "pred_score": round(float(score), 2),
                    })
        if batch_count == 0:
            raise ValueError("inference requires at least one batch")
        return rows


def write_submission_csv(rows: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]) -> Path:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("submission rows must be a non-empty sequence")
    output: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    filenames: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each submission row must be a mapping")
        sample_id, filename, score = row.get("sample_id"), row.get("wave_file_name"), row.get("pred_score")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(filename, str) or not filename:
            raise ValueError("submission rows require sample_id and wave_file_name")
        if sample_id in sample_ids or filename in filenames:
            raise ValueError("duplicate sample ID or filename in submission rows")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("submission predictions must be finite numeric values")
        sample_ids.add(sample_id)
        filenames.add(filename)
        output.append({"wav_file_name": filename, "pred_score": float(score)})
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["wav_file_name", "pred_score"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(output)
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


__all__ = ["MoeGateEvaluator", "compute_metrics", "write_submission_csv"]
