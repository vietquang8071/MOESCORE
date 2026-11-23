"""Read-only MOE8 gate diagnostics for XACLE 2027 Experiment D02."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
import torch

from model.expert_adapters.registry import profile_contract


ENTROPY_EPSILON = 1e-12
ENTROPY_REGULARIZATION_WEIGHT = 0.2
SCORE_BIN_LABELS = tuple(f"[{i},{i + 1})" for i in range(9)) + ("[9,10]",)
PREDICTION_COLUMNS = {
    "learned": "predicted_score",
    "uniform": "uniform_prediction_score",
    "hard_top1": "hard_top1_prediction_score",
}


def validate_full8_contract(
    feature_width: int, score_width: int, expert_order: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, int], int]:
    order, dimensions, width = profile_contract("full_8")
    if int(feature_width) != width:
        raise ValueError(f"Full8 feature width must be {width}; got {feature_width}")
    if int(score_width) != len(order):
        raise ValueError(f"Full8 score width must be {len(order)}; got {score_width}")
    if tuple(expert_order) != tuple(order):
        raise ValueError(f"Full8 expert order mismatch: expected {tuple(order)}")
    return tuple(order), dict(dimensions), int(width)


def validate_full8_arrays(
    features: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    expert_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order, _, width = profile_contract("full_8")
    if expert_order is not None and tuple(expert_order) != tuple(order):
        raise ValueError(f"Full8 expert order mismatch: expected {tuple(order)}")
    features = _finite_array(features, "features")
    scores = _finite_array(scores, "expert scores")
    weights = _finite_array(weights, "gate weights")
    if features.ndim != 2 or features.shape[1] != width:
        raise ValueError(f"Full8 features must have shape [N,{width}] (feature width)")
    count = features.shape[0]
    if count == 0 or scores.shape != (count, len(order)):
        raise ValueError(f"Full8 expert scores must have shape [N,{len(order)}]")
    if weights.shape != scores.shape:
        raise ValueError(f"Full8 gate weights must have shape {scores.shape}")
    _validate_weights(weights)
    return features, scores, weights


def validate_checkpoint_header(payload: Mapping) -> None:
    """Reject incompatible profiles and gate architectures before cache work."""
    if not isinstance(payload, Mapping):
        raise ValueError("MOE8 checkpoint payload must be a mapping")
    order, _, width = profile_contract("full_8")
    if payload.get("expert_profile") != "full_8":
        raise ValueError("checkpoint is not a Full8 checkpoint")
    validate_full8_contract(
        payload.get("gate_input_dim", -1),
        payload.get("gate_expert_count", -1),
        payload.get("expert_order", ()),
    )
    from training.checkpoint_manager import FULL8_GATE_ARCHITECTURE

    if payload.get("gate_architecture") != FULL8_GATE_ARCHITECTURE:
        raise ValueError("Full8 checkpoint gate architecture is incompatible")
    if payload.get("gate_input_dim") != width or payload.get("expert_order") != list(order):
        raise ValueError("Full8 checkpoint gate contract is incompatible")


def validate_checkpoint_cache_provenance(
    payload: Mapping, expected_expert_checkpoint_provenance: Mapping
) -> None:
    order, _, _ = profile_contract("full_8")
    actual = payload.get("expert_checkpoint_provenance") if isinstance(payload, Mapping) else None
    if not isinstance(expected_expert_checkpoint_provenance, Mapping):
        raise ValueError("Full8 feature-store checkpoint provenance is missing")
    if set(expected_expert_checkpoint_provenance) != set(order) or actual != expected_expert_checkpoint_provenance:
        raise ValueError("gating checkpoint provenance mismatch with the Full8 feature store")


def entropy_statistics(weights: np.ndarray) -> pd.DataFrame:
    weights = _finite_array(weights, "gate weights")
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] != len(profile_contract("full_8")[0]):
        raise ValueError("Full8 gate weights must have shape [N,8]")
    _validate_weights(weights)
    entropy = -(weights * np.log(weights + ENTROPY_EPSILON)).sum(axis=1)
    normalized = entropy / np.log(8.0)
    sorted_weights = np.sort(weights, axis=1)
    top1_index = weights.argmax(axis=1)
    return pd.DataFrame(
        {
            "shannon_entropy": entropy,
            "normalized_entropy": normalized,
            "effective_experts": np.exp(entropy),
            "max_gate_weight": weights.max(axis=1),
            "min_gate_weight": weights.min(axis=1),
            "gate_weight_std": weights.std(axis=1),
            "top1_expert_index": top1_index + 1,
            "top1_expert": [f"expert{i + 1}" for i in top1_index],
            "top1_weight": weights[np.arange(len(weights)), top1_index],
            "top2_cumulative_weight": sorted_weights[:, -2:].sum(axis=1),
            "entropy_regularization_value": 1.0 - normalized,
            "entropy_regularization_contribution": ENTROPY_REGULARIZATION_WEIGHT * (1.0 - normalized),
        }
    )


def infer_gate_weights(
    gate: torch.nn.Module,
    features: torch.Tensor | np.ndarray,
    *,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Run the original Full8 gate in eval/inference mode without optimizer state."""
    order, _, width = profile_contract("full_8")
    if getattr(gate, "profile", None) != "full_8":
        raise ValueError("inference gate must use profile full_8")
    if getattr(gate, "input_feature_dim", None) != width or getattr(gate, "num_experts", None) != len(order):
        raise ValueError("inference gate does not match the Full8 contract")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(features, torch.Tensor):
        x = features
        if x.ndim != 2 or x.shape[1] != width or x.shape[0] == 0:
            raise ValueError(f"Full8 features must have shape [N,{width}]")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all().item():
            raise ValueError("Full8 features must be finite floating-point values")
    else:
        x = _finite_array(features, "features")
        if x.ndim != 2 or x.shape[1] != width or x.shape[0] == 0:
            raise ValueError(f"Full8 features must have shape [N,{width}]")

    parameter = next(gate.parameters())
    gate_device = parameter.device
    target_device = gate_device if device is None else torch.device(device)
    if target_device != gate_device:
        raise ValueError(f"gate is on {gate_device}; requested inference device is {target_device}")
    gate.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            if not isinstance(batch, torch.Tensor):
                batch = torch.as_tensor(np.array(batch, copy=True), dtype=parameter.dtype)
            if batch.dtype != parameter.dtype:
                raise TypeError(f"feature dtype {batch.dtype} does not match gate dtype {parameter.dtype}")
            weights = gate(batch.to(gate_device))
            if weights.shape != (len(batch), len(order)) or not torch.isfinite(weights).all().item():
                raise ValueError("Full8 gate returned invalid weights")
            if (weights < 0).any().item() or not torch.allclose(
                weights.sum(dim=1),
                torch.ones(len(batch), device=gate_device, dtype=weights.dtype),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError("Full8 gate weights must be nonnegative and sum to one")
            outputs.append(weights.cpu())
    return torch.cat(outputs, dim=0)


def build_per_sample_frame(
    sample_ids: Sequence[str],
    wave_file_names: Sequence[str],
    split: str,
    expert_scores_normalized: np.ndarray,
    gate_weights: np.ndarray | torch.Tensor,
    *,
    labels_raw: Sequence[float] | np.ndarray | None = None,
    model_ids: Sequence[object] | None = None,
    model_names: Sequence[object] | None = None,
) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    order = tuple(profile_contract("full_8")[0])
    scores = _finite_array(expert_scores_normalized, "expert scores").astype(np.float64, copy=False)
    weights = _finite_array(
        gate_weights.detach().cpu().numpy() if isinstance(gate_weights, torch.Tensor) else gate_weights,
        "gate weights",
    ).astype(np.float64, copy=False)
    if scores.ndim != 2 or scores.shape[1] != len(order) or scores.shape[0] == 0:
        raise ValueError("expert scores must have shape [N,8]")
    if weights.shape != scores.shape:
        raise ValueError(f"gate weights must have shape {scores.shape}")
    _validate_weights(weights)
    ids, filenames = list(sample_ids), list(wave_file_names)
    count = len(scores)
    if len(ids) != count or len(filenames) != count:
        raise ValueError("sample IDs and wave filenames must match the score rows")
    if any(not isinstance(value, str) or not value for value in ids + filenames):
        raise ValueError("sample IDs and wave filenames must be non-empty strings")
    if any(not value.startswith(f"{split}:") for value in ids):
        raise ValueError("sample IDs do not match the requested split")
    labels = None if labels_raw is None else _finite_array(labels_raw, "raw labels").reshape(-1)
    if labels is not None and (labels.shape != (count,) or np.any((labels < 0) | (labels > 10))):
        raise ValueError("raw labels must match the rows and lie in [0,10]")
    for values, field in ((model_ids, "model_ids"), (model_names, "model_names")):
        if values is not None and len(values) != count:
            raise ValueError(f"{field} length must match sample IDs")

    predicted_normalized = np.einsum("ij,ij->i", scores, weights)
    predicted_raw = predicted_normalized * 5.0 + 5.0
    raw_expert_scores = scores * 5.0 + 5.0
    top1 = weights.argmax(axis=1)
    rows: dict[str, object] = {
        "sample_id": ids,
        "wave_file_name": filenames,
        "split": split,
        "ground_truth_score": labels if labels is not None else np.full(count, np.nan),
        "predicted_score": predicted_raw,
        "predicted_score_normalized": predicted_normalized,
        "uniform_prediction_score": scores.mean(axis=1) * 5.0 + 5.0,
        "hard_top1_prediction_score": raw_expert_scores[np.arange(count), top1],
        "top1_expert_index": top1 + 1,
        "top1_expert": [f"expert{i + 1}" for i in top1],
    }
    if model_ids is not None:
        rows["model_id"] = list(model_ids)
    if model_names is not None:
        rows["model_name"] = list(model_names)
    for index in range(len(order)):
        rows[f"expert_{index + 1}_score"] = raw_expert_scores[:, index]
        rows[f"expert_{index + 1}_score_normalized"] = scores[:, index]
        rows[f"gate_weight_{index + 1}"] = weights[:, index]
    result = pd.DataFrame(rows)
    entropy = entropy_statistics(weights)
    for column in entropy.columns:
        result[column] = entropy[column].to_numpy()
    if labels is not None:
        best_expert = np.abs(raw_expert_scores - labels[:, None]).argmin(axis=1)
        result["oracle_diagnostic_only_score"] = raw_expert_scores[np.arange(count), best_expert]
        result["oracle_diagnostic_only_expert"] = [f"expert{i + 1}" for i in best_expert]
        result["prediction_error"] = predicted_raw - labels
        result["absolute_prediction_error"] = np.abs(predicted_raw - labels)
        result["ground_truth_score_bin"] = _score_bin_labels(labels)
    else:
        result["prediction_error"] = np.nan
        result["absolute_prediction_error"] = np.nan
    return result


def join_metadata_by_filename(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    filename_column: str | None = None,
) -> pd.DataFrame:
    if "wave_file_name" not in predictions:
        raise ValueError("predictions require wave_file_name for metadata alignment")
    candidates = (filename_column,) if filename_column else (
        "wave_file_name", "wav_file_name", "filename", "file_name"
    )
    key = next((name for name in candidates if name and name in metadata.columns), None)
    if key is None:
        raise ValueError("metadata has no supported wave filename column")
    pred = predictions.copy()
    meta = metadata.copy()
    pred["_join_filename"] = pred["wave_file_name"].map(_basename)
    meta["_join_filename"] = meta[key].map(_basename)
    if pred["_join_filename"].isna().any() or meta["_join_filename"].isna().any():
        raise ValueError("metadata join contains missing wave filenames")
    missing = set(pred["_join_filename"]) - set(meta["_join_filename"])
    extra = set(meta["_join_filename"]) - set(pred["_join_filename"])
    if missing or extra:
        raise ValueError(f"metadata join missing/extra filenames: missing={len(missing)}, extra={len(extra)}")
    overlap = (set(pred.columns) & set(meta.columns)) - {"_join_filename", key, "wave_file_name"}
    if overlap:
        raise ValueError(f"metadata join would overwrite prediction columns: {sorted(overlap)}")
    joined = pred.merge(meta.drop(columns=[key]), on="_join_filename", how="left", sort=False)
    if "model_id" not in joined or joined["model_id"].isna().any() or (joined["model_id"].astype(str).str.strip() == "").any():
        raise ValueError("metadata join has missing model_id values")
    return joined.drop(columns="_join_filename")


def score_bin_summary(rows: pd.DataFrame) -> pd.DataFrame:
    _require_labels(rows)
    frame = rows.copy()
    if "ground_truth_score_bin" not in frame:
        frame["ground_truth_score_bin"] = _score_bin_labels(frame["ground_truth_score"].to_numpy())
    output = []
    for label in SCORE_BIN_LABELS:
        group = frame.loc[frame["ground_truth_score_bin"] == label]
        labels = group["ground_truth_score"].to_numpy(dtype=float)
        predictions = group["predicted_score"].to_numpy(dtype=float)
        errors = predictions - labels
        count = len(group)
        output.append(
            {
                "ground_truth_bin": label,
                "sample_count": count,
                "mean_entropy": _mean(group, "shannon_entropy"),
                "median_entropy": _median(group, "shannon_entropy"),
                "entropy_std": _std(group, "shannon_entropy"),
                "mean_normalized_entropy": _mean(group, "normalized_entropy"),
                "mean_max_gate_weight": _mean(group, "max_gate_weight"),
                "mean_effective_experts": _mean(group, "effective_experts"),
                "mean_prediction": float(predictions.mean()) if count else None,
                "mean_ground_truth": float(labels.mean()) if count else None,
                "mean_prediction_bias": float(errors.mean()) if count else None,
                "mae": float(np.abs(errors).mean()) if count else None,
                "mse": float(np.square(errors).mean()) if count else None,
            }
        )
    return pd.DataFrame(output)


def gate_weights_by_expert(rows: pd.DataFrame) -> pd.DataFrame:
    order = tuple(profile_contract("full_8")[0])
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", rows)]
    if "ground_truth_score" in rows and rows["ground_truth_score"].notna().all():
        frame = rows if "ground_truth_score_bin" in rows else rows.assign(
            ground_truth_score_bin=_score_bin_labels(rows["ground_truth_score"].to_numpy())
        )
        groups.extend(("score_bin", str(key), group) for key, group in frame.groupby("ground_truth_score_bin", sort=False, observed=True))
    if "model_id" in rows:
        groups.extend(("generator", str(key), group) for key, group in rows.groupby("model_id", sort=True, dropna=True))
    output = []
    for group_type, group_value, group in groups:
        if group.empty:
            continue
        top1 = group["top1_expert_index"].to_numpy(dtype=int)
        for index, expert in enumerate(order, 1):
            values = group[f"gate_weight_{index}"].to_numpy(dtype=float)
            output.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "sample_count": len(group),
                    "expert": expert,
                    "mean_weight": float(values.mean()),
                    "median_weight": float(np.median(values)),
                    "std_weight": float(values.std()),
                    "max_observed_weight": float(values.max()),
                    "top1_frequency": float(np.mean(top1 == index)),
                    "uniform_reference": 1.0 / 8.0,
                }
            )
    return pd.DataFrame(output)


def generator_summary(rows: pd.DataFrame) -> pd.DataFrame:
    _require_labels(rows)
    if "model_id" not in rows or rows["model_id"].isna().any():
        raise ValueError("generator summary requires joined model_id values")
    output = []
    for model_id, group in rows.groupby("model_id", sort=True):
        labels = group["ground_truth_score"].to_numpy(dtype=float)
        prediction = group["predicted_score"].to_numpy(dtype=float)
        srcc, srcc_status = _spearman(labels, prediction)
        error = prediction - labels
        high = labels >= 9.0
        row = {
            "model_id": str(model_id),
            "model_name": _first_nonempty(group.get("model_name")),
            "sample_count": len(group),
            "mean_entropy": float(group["shannon_entropy"].mean()),
            "median_entropy": float(group["shannon_entropy"].median()),
            "mean_normalized_entropy": float(group["normalized_entropy"].mean()),
            "mean_max_gate_weight": float(group["max_gate_weight"].mean()),
            "mean_effective_experts": float(group["effective_experts"].mean()),
            "srcc": srcc,
            "srcc_status": srcc_status,
            "mse": float(np.square(error).mean()),
            "mae": float(np.abs(error).mean()),
            "mean_prediction_bias": float(error.mean()),
            "high_score_underprediction_rate": float(np.mean(prediction[high] < 9.0)) if high.any() else None,
        }
        for index in range(1, 9):
            row[f"mean_gate_weight_{index}"] = float(group[f"gate_weight_{index}"].mean())
            row[f"top1_frequency_{index}"] = float(np.mean(group["top1_expert_index"].to_numpy() == index))
        output.append(row)
    return pd.DataFrame(output)


def pairwise_inversion_rates(
    labels: Sequence[float],
    predictions: Sequence[float],
    *,
    min_gaps: Sequence[float] = (0.5, 1.0, 2.0),
    block_size: int = 128,
) -> dict[str, float | int | None]:
    y = _finite_array(labels, "labels").reshape(-1)
    p = _finite_array(predictions, "predictions").reshape(-1)
    if y.shape != p.shape or y.size < 2:
        raise ValueError("pairwise inversion inputs require at least two matched labels/predictions")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    gaps = tuple(float(gap) for gap in min_gaps)
    if any(gap < 0 for gap in gaps):
        raise ValueError("minimum label gaps must be nonnegative")
    counts = {gap: 0 for gap in gaps}
    inversions = {gap: 0 for gap in gaps}
    columns = np.arange(len(y))
    for start in range(0, len(y) - 1, block_size):
        rows = np.arange(start, min(start + block_size, len(y) - 1))
        true_delta = y[None, :] - y[rows, None]
        pred_delta = p[None, :] - p[rows, None]
        later = columns[None, :] > rows[:, None]
        for gap in gaps:
            selected = later & (np.abs(true_delta) >= gap)
            counts[gap] += int(selected.sum())
            inversions[gap] += int((selected & (true_delta * pred_delta < 0)).sum())
    result: dict[str, float | int | None] = {}
    for gap in gaps:
        key = f"gap_{gap:g}"
        result[f"pair_count_{key}"] = counts[gap]
        result[f"inversion_rate_{key}"] = inversions[gap] / counts[gap] if counts[gap] else None
    return result


def compare_predictions(
    rows: pd.DataFrame,
    *,
    bootstrap_samples: int = 500,
    seed: int = 2027,
) -> pd.DataFrame:
    _require_labels(rows)
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    labels = rows["ground_truth_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    methods = dict(PREDICTION_COLUMNS)
    if "oracle_diagnostic_only_score" in rows:
        methods["ORACLE_DIAGNOSTIC_ONLY"] = "oracle_diagnostic_only_score"
    output = []
    for method, column in methods.items():
        prediction = rows[column].to_numpy(dtype=float)
        rho, rho_status = _spearman(labels, prediction)
        lcc, _ = _correlation(labels, prediction, pearsonr)
        tau, _ = _correlation(labels, prediction, kendalltau)
        error = prediction - labels
        row = {
            "prediction_method": method,
            "diagnostic_only": method == "ORACLE_DIAGNOSTIC_ONLY",
            "sample_count": len(rows),
            "srcc": rho,
            "srcc_status": rho_status,
            "lcc": lcc,
            "kendall_tau_b": tau,
            "mse": float(np.square(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "mae": float(np.abs(error).mean()),
            "predicted_mean": float(prediction.mean()),
            "predicted_std": float(prediction.std()),
            "predicted_min": float(prediction.min()),
            "predicted_max": float(prediction.max()),
            "fraction_predicted_9_10": float(np.mean((prediction >= 9.0) & (prediction <= 10.0))),
            "fraction_predicted_0_2": float(np.mean((prediction >= 0.0) & (prediction < 2.0))),
            "fraction_predicted_7_8": float(np.mean((prediction >= 7.0) & (prediction < 8.0))),
            "fraction_predicted_8_9": float(np.mean((prediction >= 8.0) & (prediction < 9.0))),
            "mean_prediction_bias": float(error.mean()),
            "high_score_underprediction_rate": float(np.mean(prediction[labels >= 9.0] < 9.0)) if np.any(labels >= 9.0) else None,
            "low_score_overprediction_rate": float(np.mean(prediction[labels < 2.0] > 2.0)) if np.any(labels < 2.0) else None,
        }
        row.update(pairwise_inversion_rates(labels, prediction))
        srcc_ci, mae_ci = _bootstrap_intervals(labels, prediction, bootstrap_samples, rng)
        row["srcc_ci_95_low"], row["srcc_ci_95_high"] = srcc_ci
        row["mae_ci_95_low"], row["mae_ci_95_high"] = mae_ci
        output.append(row)
    return pd.DataFrame(output)


def _bootstrap_intervals(
    labels: np.ndarray, predictions: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
    if samples == 0:
        return (None, None), (None, None)
    rho_values, mae_values = [], []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        y, p = labels[indices], predictions[indices]
        rho, status = _spearman(y, p)
        if rho is not None:
            rho_values.append(rho)
        mae_values.append(float(np.abs(p - y).mean()))
    rho_ci = tuple(np.quantile(rho_values, [0.025, 0.975]).tolist()) if rho_values else (None, None)
    mae_ci = tuple(np.quantile(mae_values, [0.025, 0.975]).tolist()) if mae_values else (None, None)
    return rho_ci, mae_ci


def _finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_weights(weights: np.ndarray) -> None:
    if np.any(weights < 0):
        raise ValueError("gate weights must be nonnegative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("gate weights must sum to one per sample")


def _score_bin_labels(labels: Sequence[float] | np.ndarray) -> np.ndarray:
    values = _finite_array(labels, "raw labels").reshape(-1)
    if np.any((values < 0) | (values > 10)):
        raise ValueError("raw labels must lie in [0,10]")
    indices = np.minimum(values.astype(np.int64, copy=False), 9)
    return np.asarray([SCORE_BIN_LABELS[index] for index in indices], dtype=object)


def _require_labels(rows: pd.DataFrame) -> None:
    if "ground_truth_score" not in rows or rows["ground_truth_score"].isna().any():
        raise ValueError("this diagnostic requires evaluation labels for every matched row")


def _spearman(labels: np.ndarray, predictions: np.ndarray) -> tuple[float | None, str]:
    if len(labels) < 2:
        return None, "undefined_insufficient_samples"
    if np.unique(labels).size < 2:
        return None, "undefined_constant_scores"
    if np.unique(predictions).size < 2:
        return None, "undefined_constant_predictions"
    value = float(spearmanr(labels, predictions).statistic)
    return (value, "defined") if np.isfinite(value) else (None, "undefined_numeric")


def _correlation(labels: np.ndarray, predictions: np.ndarray, function) -> tuple[float | None, str]:
    if len(labels) < 2 or np.unique(labels).size < 2 or np.unique(predictions).size < 2:
        return None, "undefined_constant_or_insufficient"
    value = float(function(labels, predictions).statistic)
    return (value, "defined") if np.isfinite(value) else (None, "undefined_numeric")


def _basename(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return Path(str(value).replace("\\", "/")).name


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    return float(frame[column].mean()) if len(frame) else None


def _median(frame: pd.DataFrame, column: str) -> float | None:
    return float(frame[column].median()) if len(frame) else None


def _std(frame: pd.DataFrame, column: str) -> float | None:
    return float(frame[column].std(ddof=0)) if len(frame) else None


def _first_nonempty(values: pd.Series | None) -> str:
    if values is None:
        return ""
    present = values.dropna().astype(str).str.strip()
    present = present[present != ""]
    return present.iloc[0] if len(present) else ""


__all__ = [
    "ENTROPY_EPSILON",
    "ENTROPY_REGULARIZATION_WEIGHT",
    "SCORE_BIN_LABELS",
    "build_per_sample_frame",
    "compare_predictions",
    "entropy_statistics",
    "gate_weights_by_expert",
    "generator_summary",
    "infer_gate_weights",
    "join_metadata_by_filename",
    "pairwise_inversion_rates",
    "score_bin_summary",
    "validate_checkpoint_cache_provenance",
    "validate_checkpoint_header",
    "validate_full8_arrays",
    "validate_full8_contract",
]
