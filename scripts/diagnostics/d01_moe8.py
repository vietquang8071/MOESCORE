"""Frozen MOE8 convex-score-envelope diagnostics; all core inputs are normalized."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RAW_SCORE_MIN = 0.0
RAW_SCORE_MAX = 10.0
NORMALIZED_SCORE_SCALE = "normalized_minus1_to1"
RAW_SCORE_SCALE = "raw_0_to_10"
DEFAULT_ACCURACY_THRESHOLD_RAW = 0.5
DEFAULT_SENSITIVITY_THRESHOLDS_RAW = (0.25, 0.5, 1.0)
SCORE_BINS = tuple(
    (float(i), float(i + 1), "[9,10]" if i == 9 else f"[{i},{i + 1})")
    for i in range(10)
)
TAIL_BINS = (
    ("low_tail", 0.0, 3.0, "[0,3)"),
    ("middle", 3.0, 7.0, "[3,7)"),
    ("high", 7.0, 9.0, "[7,9)"),
    ("extreme_high", 9.0, 10.0, "[9,10]"),
)
FULL8_INFERENCE_COMPAT_FORMAT_VERSION = 7
FULL8_INFERENCE_COMPAT_SOURCE_FILES = (
    "model/moe_gating_core.py", "training/moe_evaluator.py",
    "training/target_transform.py", "data/moe_feature_store.py",
    "data/full8_contract.py", "data/full7_assembler.py",
    "model/expert_adapters/registry.py",
    "model/expert_adapters/expert8_m2d_clap_adapter.py",
    "model/expert_adapters/expert8_m2d_clap_runtime.py",
    "model/expert_adapters/m2d_lora_runtime.py",
    "scripts/extract_expert8_features.py",
    "expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py",
    "expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py",
)
FULL8_GATE_ARCHITECTURE = {
    "input_width": 33032, "hidden_width": 512, "normalization": "LayerNorm",
    "activation": "ReLU", "dropout": 0.1, "output_width": 8,
    "output": "Softmax(dim=-1)",
}


def prepare_format7_gate_payload(
    payload: Mapping[str, Any], *, current_source_hashes: Mapping[str, str],
    checkpoint_compat_source_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Adapt the released Full8-v7 inference metadata to the pinned v5 validator."""
    if not isinstance(payload, Mapping) or payload.get("format_version") != FULL8_INFERENCE_COMPAT_FORMAT_VERSION:
        raise ValueError("expected a Full8 format-7 checkpoint payload")
    if (payload.get("expert_profile") != "full_8"
            or payload.get("expert_order") != [f"expert{i}" for i in range(1, 9)]
            or payload.get("expert_feature_dimensions") != {
                "expert1": 2049, "expert2": 4097, "expert3": 3840, "expert4": 1024,
                "expert5": 11267, "expert6": 4609, "expert7": 3073, "expert8": 3073,
            }
            or payload.get("gate_input_dim") != 33032 or payload.get("gate_expert_count") != 8
            or payload.get("gate_architecture") != FULL8_GATE_ARCHITECTURE):
        raise ValueError("Full8 format-7 checkpoint architecture or expert contract is incompatible")
    recorded_source = payload.get("source_code_sha256")
    if not isinstance(recorded_source, Mapping) or not recorded_source:
        raise ValueError("Full8 format-7 checkpoint source-code provenance is missing")
    for name, digest in recorded_source.items():
        if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Full8 format-7 checkpoint source-code provenance is malformed")
    compatibility_source = (current_source_hashes if checkpoint_compat_source_hashes is None
                            else checkpoint_compat_source_hashes)
    missing = [name for name in FULL8_INFERENCE_COMPAT_SOURCE_FILES if name not in recorded_source]
    changed = [name for name in FULL8_INFERENCE_COMPAT_SOURCE_FILES
               if name in recorded_source and recorded_source[name] != compatibility_source.get(name)]
    if missing or changed:
        raise ValueError(f"Full8 format-7 inference source code is incompatible: missing={missing}, changed={changed}")

    identity = payload.get("feature_store_identity")
    config = payload.get("config")
    if not isinstance(identity, Mapping) or not isinstance(config, Mapping):
        raise ValueError("Full8 format-7 checkpoint identity/configuration is missing")
    hashes, sample_ids = identity.get("source_wav_sha256_by_split"), identity.get("source_sample_ids_by_split")
    if (not isinstance(hashes, Mapping) or set(hashes) != {"train", "validation"}
            or not isinstance(sample_ids, Mapping) or set(sample_ids) != {"train", "validation"}):
        raise ValueError("Full8 format-7 source audio membership is incomplete")
    source_rows = {}
    for split in ("train", "validation"):
        split_hashes, split_ids = hashes[split], sample_ids[split]
        if (not isinstance(split_hashes, list) or not isinstance(split_ids, list)
                or len(split_hashes) != len(split_ids)):
            raise ValueError(f"Full8 format-7 {split} source audio membership is invalid")
        source_rows[split] = [
            {"sample_id": sample_id, "wav_sha256": digest}
            for sample_id, digest in zip(split_ids, split_hashes)
        ]
    from data.audio_overlap import build_train_validation_audio_overlap_diagnostic
    strict = build_train_validation_audio_overlap_diagnostic(
        source_rows["train"], source_rows["validation"],
        allow_train_validation_audio_overlap=False,
    )
    allow_overlap = bool(strict["overlapping_sha256_count"])
    diagnostic = build_train_validation_audio_overlap_diagnostic(
        source_rows["train"], source_rows["validation"],
        allow_train_validation_audio_overlap=allow_overlap,
    )
    data_config = config.get("data")
    if not isinstance(data_config, Mapping) or data_config.get("profile") != "full_8":
        raise ValueError("Full8 format-7 checkpoint configuration is invalid")
    if ("allow_train_validation_audio_overlap" in data_config
            and (type(data_config["allow_train_validation_audio_overlap"]) is not bool
                 or data_config["allow_train_validation_audio_overlap"] != allow_overlap)):
        raise ValueError("Full8 format-7 overlap policy differs from source membership")
    if "audio_overlap_diagnostic" in identity:
        if identity["audio_overlap_diagnostic"] != diagnostic:
            raise ValueError("Full8 format-7 overlap diagnostic differs from source membership")

    adapted_identity = dict(identity)
    adapted_identity["audio_overlap_diagnostic"] = diagnostic
    adapted_config = dict(config)
    adapted_data_config = dict(data_config)
    adapted_data_config["allow_train_validation_audio_overlap"] = allow_overlap
    adapted_config["data"] = adapted_data_config
    adapted = dict(payload)
    adapted.update({
        "format_version": 5,
        "feature_store_identity": adapted_identity,
        "config": adapted_config,
        "source_code_sha256": dict(current_source_hashes),
    })
    changed_files = sorted(name for name, digest in recorded_source.items()
                           if current_source_hashes.get(name) != digest)
    audit = {
        "format_version": FULL8_INFERENCE_COMPAT_FORMAT_VERSION,
        "verified_runtime_files": list(FULL8_INFERENCE_COMPAT_SOURCE_FILES),
        "different_or_unavailable_files": changed_files,
        "checkpoint_source_file_count": len(recorded_source),
        "checkpoint_source_file_missing_from_pinned_tree": sorted(
            name for name in recorded_source if name not in current_source_hashes),
        "runtime_files_unrecorded_by_checkpoint": sorted(
            name for name in FULL8_INFERENCE_COMPAT_SOURCE_FILES if name not in recorded_source),
    }
    origin = "reconstructed_from_checkpoint_source_hashes_format7"
    return adapted, diagnostic, audit, origin


def load_gate_checkpoint_for_d01(
    checkpoint_path: str | Path, *, gate: Any,
    expected_expert_checkpoint_provenance: Mapping[str, Any],
    checkpoint_compat_source_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Load the frozen gate using pinned validation plus the released v7 adapter."""
    import torch
    from training import checkpoint_manager as checkpoint_module

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported gating checkpoint schema")
    expected_order = [f"expert{i}" for i in range(1, 9)]
    if (payload.get("expert_profile") != "full_8" or payload.get("expert_order") != expected_order
            or payload.get("gate_input_dim") != 33032 or payload.get("gate_expert_count") != 8):
        raise ValueError("attached gate is not the expected eight-expert Full8 checkpoint")
    manager = checkpoint_module.MoeCheckpointManager()
    if payload.get("format_version") == checkpoint_module.FULL8_CHECKPOINT_FORMAT_VERSION:
        loaded = manager.load_for_inference(
            checkpoint_path, gating_network=gate,
            expected_expert_checkpoint_provenance=expected_expert_checkpoint_provenance,
        )
        current_source = checkpoint_module.full8_source_code_sha256()
        recorded_source = loaded.get("source_code_sha256", {})
        audit = {
            "format_version": loaded.get("format_version"),
            "verified_runtime_files": list(FULL8_INFERENCE_COMPAT_SOURCE_FILES),
            "different_or_unavailable_files": sorted(
                name for name, digest in recorded_source.items() if current_source.get(name) != digest),
            "checkpoint_source_file_count": len(recorded_source),
            "checkpoint_source_file_missing_from_pinned_tree": sorted(
                name for name in recorded_source if name not in current_source),
            "runtime_files_unrecorded_by_checkpoint": sorted(
                name for name in FULL8_INFERENCE_COMPAT_SOURCE_FILES if name not in recorded_source),
        }
        return loaded, loaded["feature_store_identity"]["audio_overlap_diagnostic"], audit, "checkpoint"
    if payload.get("format_version") != FULL8_INFERENCE_COMPAT_FORMAT_VERSION:
        raise ValueError(f"unsupported Full8 gate checkpoint format: {payload.get('format_version')!r}")

    current_source = checkpoint_module.full8_source_code_sha256()
    adapted, diagnostic, audit, origin = prepare_format7_gate_payload(
        payload, current_source_hashes=current_source,
        checkpoint_compat_source_hashes=checkpoint_compat_source_hashes,
    )
    checkpoint_module._validate_gate_payload(adapted, gate)
    expected_provenance = dict(expected_expert_checkpoint_provenance)
    if payload.get("expert_checkpoint_provenance") != expected_provenance:
        raise ValueError("gating checkpoint provenance mismatch with the inference feature store")
    for field, value in (("config", payload.get("config")),
                         ("feature_store_identity", payload.get("feature_store_identity")),
                         ("expert_checkpoint_provenance", payload.get("expert_checkpoint_provenance"))):
        if payload.get(f"{field}_sha256") != checkpoint_module._fingerprint(value, field):
            raise ValueError(f"checkpoint {field} fingerprint mismatch")
    state_dict = payload.get("gating_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no gating state dictionary")
    try:
        gate.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"gating state dictionary is incompatible: {exc}") from exc
    return payload, diagnostic, audit, origin


def _numeric(value: Any, field: str) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{field} must contain real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must contain real numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def normalize_raw_scores(scores_raw: Any) -> np.ndarray:
    """Convert human labels from the official raw [0,10] scale to [-1,1]."""
    raw = _numeric(scores_raw, "scores_raw")
    if np.any((raw < RAW_SCORE_MIN) | (raw > RAW_SCORE_MAX)):
        raise ValueError("scores_raw must remain on the original [0,10] scale")
    return (raw - 5.0) / 5.0


def denormalize_scores(scores_normalized: Any) -> np.ndarray:
    """Convert normalized values to raw-score units without clamping."""
    return _numeric(scores_normalized, "scores_normalized") * 5.0 + 5.0


def _validated_inputs(
    expert_scores_normalized: Any,
    gate_weights: Any,
    moe_prediction_normalized: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = _numeric(expert_scores_normalized, "expert_scores_normalized")
    weights = _numeric(gate_weights, "gate_weights")
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("expert_scores_normalized must have shape [N,E] with N,E > 0")
    if weights.shape != scores.shape:
        raise ValueError(f"gate_weights must have shape {scores.shape}; got {weights.shape}")
    if np.any(weights < -1e-7):
        raise ValueError("gate_weights must be nonnegative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("gate_weights must sum to one for every sample")

    recomputed = np.einsum("ij,ij->i", scores, weights)
    if moe_prediction_normalized is None:
        prediction = recomputed
    else:
        prediction = _numeric(moe_prediction_normalized, "moe_prediction_normalized")
        if prediction.shape != (scores.shape[0],):
            raise ValueError(f"moe_prediction_normalized must have shape [{scores.shape[0]}]")
        if not np.allclose(prediction, recomputed, atol=1e-5, rtol=1e-5):
            raise ValueError("provided MOE predictions differ from weighted expert scores")
    lower = scores.min(axis=1)
    upper = scores.max(axis=1)
    tolerance = 1e-5 * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    if np.any(prediction < lower - tolerance) or np.any(prediction > upper + tolerance):
        raise ValueError("convex MOE prediction falls outside the expert score interval")
    return scores, weights, prediction


def compute_envelope_diagnostics(
    expert_scores_normalized: Any,
    gate_weights: Any,
    moe_prediction_normalized: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute each sample's frozen-gate prediction and expert score interval."""
    scores, weights, prediction = _validated_inputs(
        expert_scores_normalized, gate_weights, moe_prediction_normalized
    )
    lower = scores.min(axis=1)
    upper = scores.max(axis=1)
    result = {
        "expert_scores_normalized": scores,
        "gate_weights": weights,
        "expert_min_normalized": lower,
        "expert_max_normalized": upper,
        "envelope_width_normalized": upper - lower,
        "moe_prediction_normalized": prediction,
        "expert_min_raw": denormalize_scores(lower),
        "expert_max_raw": denormalize_scores(upper),
        "envelope_width_raw": (upper - lower) * 5.0,
        "moe_prediction_raw": denormalize_scores(prediction),
    }
    return result


def _empty_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "above_expert_max_count": 0,
        "above_expert_max_fraction": None,
        "below_expert_min_count": 0,
        "below_expert_min_fraction": None,
        "outside_envelope_count": 0,
        "outside_envelope_fraction": None,
        "mean_envelope_width": None,
        "median_envelope_width": None,
        "mean_envelope_width_raw": None,
        "median_envelope_width_raw": None,
        "mean_envelope_width_normalized": None,
        "median_envelope_width_normalized": None,
        "mean_minimum_absolute_error": None,
        "median_minimum_absolute_error": None,
        "mean_minimum_absolute_error_raw": None,
        "median_minimum_absolute_error_raw": None,
        "mean_minimum_absolute_error_normalized": None,
        "median_minimum_absolute_error_normalized": None,
        "convex_mse_lower_bound": None,
        "convex_mse_lower_bound_raw": None,
        "convex_mse_lower_bound_normalized": None,
        "actual_moe_mse": None,
        "actual_moe_mse_raw": None,
        "actual_moe_mse_normalized": None,
        "actual_moe_mse_minus_lower_bound": None,
        "actual_moe_mse_minus_lower_bound_raw": None,
        "actual_moe_mse_minus_lower_bound_normalized": None,
        "mean_signed_error": None,
        "mean_absolute_error": None,
        "mean_signed_error_raw": None,
        "mean_absolute_error_raw": None,
    }


def _summary_for_indices(envelope: Mapping[str, np.ndarray], labels_norm: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    count = int(indices.size)
    summary = _empty_summary()
    summary["sample_count"] = count
    if count == 0:
        return summary

    lower = envelope["expert_min_normalized"][indices]
    upper = envelope["expert_max_normalized"][indices]
    prediction = envelope["moe_prediction_normalized"][indices]
    target = labels_norm[indices]
    below = target < lower
    above = target > upper
    outside = below | above
    distance = np.maximum.reduce((lower - target, target - upper, np.zeros(count, dtype=np.float64)))
    actual_error = prediction - target
    oracle_mse_norm = float(np.mean(distance**2))
    actual_mse_norm = float(np.mean(actual_error**2))
    if actual_mse_norm + max(1e-12, abs(oracle_mse_norm) * 1e-10) < oracle_mse_norm:
        raise RuntimeError("actual MOE MSE is below the convex-envelope lower bound")

    width = upper - lower
    summary.update({
        "above_expert_max_count": int(above.sum()),
        "above_expert_max_fraction": float(above.mean()),
        "below_expert_min_count": int(below.sum()),
        "below_expert_min_fraction": float(below.mean()),
        "outside_envelope_count": int(outside.sum()),
        "outside_envelope_fraction": float(outside.mean()),
        "mean_envelope_width": float(width.mean() * 5.0),
        "median_envelope_width": float(np.median(width) * 5.0),
        "mean_envelope_width_raw": float(width.mean() * 5.0),
        "median_envelope_width_raw": float(np.median(width) * 5.0),
        "mean_envelope_width_normalized": float(width.mean()),
        "median_envelope_width_normalized": float(np.median(width)),
        "mean_minimum_absolute_error": float(distance.mean() * 5.0),
        "median_minimum_absolute_error": float(np.median(distance) * 5.0),
        "mean_minimum_absolute_error_raw": float(distance.mean() * 5.0),
        "median_minimum_absolute_error_raw": float(np.median(distance) * 5.0),
        "mean_minimum_absolute_error_normalized": float(distance.mean()),
        "median_minimum_absolute_error_normalized": float(np.median(distance)),
        "convex_mse_lower_bound": oracle_mse_norm * 25.0,
        "convex_mse_lower_bound_raw": oracle_mse_norm * 25.0,
        "convex_mse_lower_bound_normalized": oracle_mse_norm,
        "actual_moe_mse": actual_mse_norm * 25.0,
        "actual_moe_mse_raw": actual_mse_norm * 25.0,
        "actual_moe_mse_normalized": actual_mse_norm,
        "actual_moe_mse_minus_lower_bound": (actual_mse_norm - oracle_mse_norm) * 25.0,
        "actual_moe_mse_minus_lower_bound_raw": (actual_mse_norm - oracle_mse_norm) * 25.0,
        "actual_moe_mse_minus_lower_bound_normalized": actual_mse_norm - oracle_mse_norm,
        "mean_signed_error": float(actual_error.mean() * 5.0),
        "mean_absolute_error": float(np.abs(actual_error).mean() * 5.0),
        "mean_signed_error_raw": float(actual_error.mean() * 5.0),
        "mean_absolute_error_raw": float(np.abs(actual_error).mean() * 5.0),
    })
    return summary


def _score_bin_index(labels_raw: np.ndarray) -> np.ndarray:
    # side="right" makes each integer boundary start the following bin; score 10 maps to bin 9.
    return np.searchsorted(np.arange(1.0, 10.0), labels_raw, side="right")


def _named_metrics(summary: Mapping[str, Any], *, label_mean: float | None = None,
                   moe_mean: float | None = None, lower_mean: float | None = None,
                   upper_mean: float | None = None) -> dict[str, Any]:
    return {
        **dict(summary),
        "mean_ground_truth": label_mean,
        "mean_moe_prediction": moe_mean,
        "mean_expert_min": lower_mean,
        "mean_expert_max": upper_mean,
    }


def analyze_split(
    expert_scores_normalized: Any,
    gate_weights: Any,
    moe_prediction_normalized: Any | None = None,
    *,
    labels_raw: Any | None = None,
    accuracy_threshold_raw: float = DEFAULT_ACCURACY_THRESHOLD_RAW,
    sensitivity_thresholds_raw: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS_RAW,
) -> dict[str, Any]:
    """Return row arrays and split, bin, tail, decomposition, and class summaries.

    ``labels_raw=None`` is label-free: no ground-truth, oracle, or error metrics
    are returned. Classification threshold is fixed in raw 0–10 score units.
    """
    envelope = compute_envelope_diagnostics(
        expert_scores_normalized, gate_weights, moe_prediction_normalized
    )
    sample_count = envelope["expert_scores_normalized"].shape[0]
    result: dict[str, Any] = {
        "score_scale": NORMALIZED_SCORE_SCALE,
        "raw_interpretation_scale": RAW_SCORE_SCALE,
        "sample_count": sample_count,
        "labels_present": labels_raw is not None,
        "envelope": envelope,
        "global_metrics": None,
        "per_bin_metrics": [],
        "tail_metrics": [],
        "structural_vs_gating": None,
        "classification": None,
    }
    if labels_raw is None:
        return result

    labels = _numeric(labels_raw, "labels_raw")
    if labels.shape != (sample_count,):
        raise ValueError(f"labels_raw must have shape [{sample_count}]")
    labels_norm = normalize_raw_scores(labels)
    if not np.isfinite(accuracy_threshold_raw) or accuracy_threshold_raw < 0:
        raise ValueError("accuracy_threshold_raw must be finite and nonnegative")
    thresholds = _numeric(sensitivity_thresholds_raw, "sensitivity_thresholds_raw").reshape(-1)
    if thresholds.size == 0 or np.any(thresholds < 0):
        raise ValueError("sensitivity_thresholds_raw must contain nonnegative values")

    above = labels_norm > envelope["expert_max_normalized"]
    below = labels_norm < envelope["expert_min_normalized"]
    outside = above | below
    distance = np.maximum.reduce((
        envelope["expert_min_normalized"] - labels_norm,
        labels_norm - envelope["expert_max_normalized"],
        np.zeros(sample_count, dtype=np.float64),
    ))
    actual_error_norm = envelope["moe_prediction_normalized"] - labels_norm
    actual_sq_norm = actual_error_norm**2
    oracle_sq_norm = distance**2
    difference_norm = actual_sq_norm - oracle_sq_norm
    if np.any(difference_norm < -1e-10):
        raise RuntimeError("sample-wise actual squared error is below its convex oracle lower bound")
    envelope.update({
        "label_raw": labels,
        "label_normalized": labels_norm,
        "oracle_projection_normalized": np.clip(
            labels_norm, envelope["expert_min_normalized"], envelope["expert_max_normalized"]
        ),
        "oracle_projection_raw": denormalize_scores(np.clip(
            labels_norm, envelope["expert_min_normalized"], envelope["expert_max_normalized"]
        )),
        "above_expert_max": above,
        "below_expert_min": below,
        "outside_envelope": outside,
        "minimum_absolute_error_normalized": distance,
        "minimum_absolute_error_raw": distance * 5.0,
        "actual_gate_squared_error_normalized": actual_sq_norm,
        "oracle_convex_squared_error_normalized": oracle_sq_norm,
        "actual_minus_oracle_squared_error_normalized": difference_norm,
        "actual_gate_squared_error_raw": actual_sq_norm * 25.0,
        "oracle_convex_squared_error_raw": oracle_sq_norm * 25.0,
        "actual_minus_oracle_squared_error_raw": difference_norm * 25.0,
    })

    all_indices = np.arange(sample_count)
    global_summary = _summary_for_indices(envelope, labels_norm, all_indices)
    result["global_metrics"] = global_summary
    bin_indices = _score_bin_index(labels)
    per_bin = []
    for index, (start, end, name) in enumerate(SCORE_BINS):
        selected = all_indices[bin_indices == index]
        summary = _summary_for_indices(envelope, labels_norm, selected)
        mean = lambda field: float(np.mean(envelope[field][selected])) if selected.size else None
        per_bin.append({
            "ground_truth_bin": name,
            **_named_metrics(
                summary,
                label_mean=float(np.mean(labels[selected])) if selected.size else None,
                moe_mean=mean("moe_prediction_raw"),
                lower_mean=mean("expert_min_raw"),
                upper_mean=mean("expert_max_raw"),
            ),
            "class_A_count": int(np.sum((selected.size > 0) & above[selected])) if selected.size else 0,
            "class_B_count": int(np.sum((selected.size > 0) & below[selected])) if selected.size else 0,
        })
    result["per_bin_metrics"] = per_bin

    tails = []
    for name, start, end, label in TAIL_BINS:
        selected = all_indices[(labels >= start) & ((labels <= end) if end == 10.0 else (labels < end))]
        summary = _summary_for_indices(envelope, labels_norm, selected)
        tails.append({
            "tail": name, "ground_truth_range": label,
            **_named_metrics(
                summary,
                label_mean=float(np.mean(labels[selected])) if selected.size else None,
                moe_mean=float(np.mean(envelope["moe_prediction_raw"][selected])) if selected.size else None,
                lower_mean=float(np.mean(envelope["expert_min_raw"][selected])) if selected.size else None,
                upper_mean=float(np.mean(envelope["expert_max_raw"][selected])) if selected.size else None,
            ),
        })
    result["tail_metrics"] = tails
    result["structural_vs_gating"] = {
        "score_scale_normalized": NORMALIZED_SCORE_SCALE,
        "raw_interpretation_scale": RAW_SCORE_SCALE,
        "actual_gate_squared_error_mean_normalized": float(actual_sq_norm.mean()),
        "oracle_convex_squared_error_mean_normalized": float(oracle_sq_norm.mean()),
        "difference_mean_normalized": float(difference_norm.mean()),
        "actual_gate_squared_error_mean_raw": float(actual_sq_norm.mean() * 25.0),
        "oracle_convex_squared_error_mean_raw": float(oracle_sq_norm.mean() * 25.0),
        "difference_mean_raw": float(difference_norm.mean() * 25.0),
        "by_score_bin": [
            {
                "ground_truth_bin": row["ground_truth_bin"],
                "sample_count": row["sample_count"],
                "actual_gate_squared_error_mean_raw": row["actual_moe_mse"],
                "oracle_convex_squared_error_mean_raw": row["convex_mse_lower_bound"],
                "difference_mean_raw": row["actual_moe_mse_minus_lower_bound"],
                "actual_gate_squared_error_mean_normalized": row["actual_moe_mse_normalized"],
                "oracle_convex_squared_error_mean_normalized": row["convex_mse_lower_bound_normalized"],
                "difference_mean_normalized": row["actual_moe_mse_minus_lower_bound_normalized"],
            }
            for row in per_bin
        ],
    }

    signed_error_raw = actual_error_norm * 5.0
    classifications = np.full(sample_count, "D", dtype="U1")
    inside = ~outside
    classifications[inside & (np.abs(signed_error_raw) > accuracy_threshold_raw)] = "C"
    classifications[above] = "A"
    classifications[below] = "B"
    class_summary = {
        label: {
            "sample_count": int(np.sum(classifications == label)),
            "fraction": float(np.mean(classifications == label)),
        }
        for label in ("A", "B", "C", "D")
    }
    sensitivity = []
    for threshold in sorted(set(float(value) for value in thresholds) | {float(accuracy_threshold_raw)}):
        classes = np.full(sample_count, "D", dtype="U1")
        classes[inside & (np.abs(signed_error_raw) > threshold)] = "C"
        classes[above] = "A"
        classes[below] = "B"
        sensitivity.append({
            "accuracy_threshold_raw": threshold,
            "classification_counts": {label: int(np.sum(classes == label)) for label in ("A", "B", "C", "D")},
        })
    result["classification"] = {
        "accuracy_threshold_raw": float(accuracy_threshold_raw),
        "class_summary": class_summary,
        "sensitivity_check": sensitivity,
        "per_sample": classifications.tolist(),
    }
    return result


def summarize_by_generator(
    analysis: Mapping[str, Any],
    *,
    generator_ids: Sequence[Any] | None,
    generator_names: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Summarize label-based envelope metrics by official generator metadata."""
    if not analysis.get("labels_present"):
        return [{"status": "LABELS_UNAVAILABLE"}]
    if generator_ids is None and generator_names is None:
        return [{"status": "GENERATOR_METADATA_UNAVAILABLE"}]
    ids = generator_ids if generator_ids is not None else generator_names
    if len(ids) != analysis["sample_count"]:
        raise ValueError("generator metadata length must match feature-store order")
    if any(value is None or not str(value).strip() for value in ids):
        raise ValueError("generator IDs/names must be present for every joined sample")
    names = generator_names
    if names is not None and len(names) != len(ids):
        raise ValueError("generator_names length must match generator_ids")
    envelope = analysis["envelope"]
    labels_norm = envelope["label_normalized"]
    groups: dict[str, list[int]] = {}
    display_names: dict[str, str] = {}
    for index, value in enumerate(ids):
        key = str(value).strip()
        groups.setdefault(key, []).append(index)
        if names is not None:
            display_names.setdefault(key, str(names[index]).strip())
    rows = []
    for key in sorted(groups):
        selected = np.asarray(groups[key], dtype=np.int64)
        summary = _summary_for_indices(envelope, labels_norm, selected)
        rows.append({
            "status": "AVAILABLE",
            "generator_id": key,
            "generator_name": display_names.get(key, ""),
            "sample_count": summary["sample_count"],
            "outside_envelope_fraction": summary["outside_envelope_fraction"],
            "above_max_fraction": summary["above_expert_max_fraction"],
            "below_min_fraction": summary["below_expert_min_fraction"],
            "convex_mse_lower_bound": summary["convex_mse_lower_bound"],
            "convex_mse_lower_bound_normalized": summary["convex_mse_lower_bound_normalized"],
            "actual_moe_mse": summary["actual_moe_mse"],
            "actual_moe_mse_normalized": summary["actual_moe_mse_normalized"],
            "mean_envelope_width": summary["mean_envelope_width"],
            "mean_envelope_width_normalized": summary["mean_envelope_width_normalized"],
        })
    return rows


def build_prediction_rows(
    sample_ids: Sequence[str],
    split: str,
    expert_scores_normalized: Any,
    gate_weights: Any,
    analysis: Mapping[str, Any],
    *,
    labels_raw: Any | None = None,
    model_ids: Sequence[Any] | None = None,
    model_names: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build ordered, CSV-ready rows without changing or calibrating scores."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    ids = list(sample_ids)
    scores, weights, prediction = _validated_inputs(
        expert_scores_normalized, gate_weights, analysis["envelope"]["moe_prediction_normalized"]
    )
    count = scores.shape[0]
    if len(ids) != count or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("sample_ids must be non-empty strings in feature-store order")
    if len(ids) != len(set(ids)):
        raise ValueError("sample_ids must be unique")
    if any(not sample_id.startswith(split + ":") for sample_id in ids):
        raise ValueError("sample_ids do not match the requested split")
    labels = None if labels_raw is None else _numeric(labels_raw, "labels_raw")
    if labels is not None and (labels.shape != (count,) or np.any((labels < 0) | (labels > 10))):
        raise ValueError("labels_raw must be finite [N] on the original [0,10] scale")
    for values, name in ((model_ids, "model_ids"), (model_names, "model_names")):
        if values is not None and len(values) != count:
            raise ValueError(f"{name} length must match sample_ids")
    if labels is None and analysis.get("labels_present"):
        raise ValueError("analysis includes labels but prediction rows were asked to omit them")
    if labels is not None and not analysis.get("labels_present"):
        raise ValueError("prediction labels were provided but analysis did not include them")
    classes = analysis.get("classification", {}).get("per_sample") if labels is not None else None
    envelope = analysis["envelope"]
    rows: list[dict[str, Any]] = []
    for i, sample_id in enumerate(ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": split,
            "label_raw": "" if labels is None else float(labels[i]),
        }
        if model_ids is not None:
            row["model_id"] = str(model_ids[i])
        if model_names is not None:
            row["model_name"] = str(model_names[i])
        for expert_index in range(scores.shape[1]):
            expert_name = f"expert{expert_index + 1}"
            row[f"{expert_name}_score_normalized"] = float(scores[i, expert_index])
            row[f"{expert_name}_score_raw"] = float(denormalize_scores(scores[i, expert_index]))
            row[f"{expert_name}_gate_weight"] = float(weights[i, expert_index])
        row.update({
            "moe_prediction_normalized": float(prediction[i]),
            "moe_prediction_raw": float(envelope["moe_prediction_raw"][i]),
            "expert_min_normalized": float(envelope["expert_min_normalized"][i]),
            "expert_max_normalized": float(envelope["expert_max_normalized"][i]),
            "expert_min_raw": float(envelope["expert_min_raw"][i]),
            "expert_max_raw": float(envelope["expert_max_raw"][i]),
            "envelope_width_normalized": float(envelope["envelope_width_normalized"][i]),
            "envelope_width_raw": float(envelope["envelope_width_raw"][i]),
        })
        if labels is not None:
            for key in (
                "above_expert_max", "below_expert_min", "outside_envelope",
                "actual_gate_squared_error_normalized", "oracle_convex_squared_error_normalized",
                "actual_minus_oracle_squared_error_normalized", "actual_gate_squared_error_raw",
                "oracle_convex_squared_error_raw", "actual_minus_oracle_squared_error_raw",
            ):
                row[key] = (bool(envelope[key][i]) if key in {"above_expert_max", "below_expert_min", "outside_envelope"}
                            else float(envelope[key][i]))
            row["convex_class"] = classes[i]
            row["oracle_projection_normalized"] = float(envelope["oracle_projection_normalized"][i])
            row["oracle_projection_raw"] = float(envelope["oracle_projection_raw"][i])
        rows.append(row)
    return rows


def write_prediction_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write prediction rows in the supplied sample order."""
    if not rows:
        raise ValueError("cannot write an empty prediction file")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("prediction rows must have identical ordered columns")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_d01_feature_store(
    root: str | Path,
    profile: str,
    expected_counts: Mapping[str, int],
    *,
    source_manifest_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Validate each public split against its own canonical source manifest."""
    import hashlib

    from data.moe_feature_store import preprocessing_contract, validate_feature_store
    from data.set_a_cached_dataset import _read_source_manifest
    from model.expert_adapters.registry import profile_contract

    if profile not in {"full_7", "full_8"}:
        raise ValueError(f"D01 profile must be full_7 or full_8, got {profile!r}")
    splits = ("train", "validation", "test")
    if set(expected_counts) != set(splits) or any(
        type(expected_counts[split]) is not int or expected_counts[split] <= 0 for split in splits
    ):
        raise ValueError("D01 expected_counts must give positive integer counts for train/validation/test")

    order, dimensions, width = profile_contract(profile)
    source_root = Path(source_manifest_root)
    result: dict[str, dict[str, Any]] = {}
    reference_provenance = None
    reference_preprocessing = None
    for split in splits:
        context = {"root": str(root), "split": split, "profile": profile}
        try:
            manifest = validate_feature_store(root, split=split, require_labels=split != "test")
            if manifest.get("split") != split:
                raise ValueError("manifest split identity mismatch")
            if manifest.get("expert_profile") != profile:
                raise ValueError(f"expert profile mismatch: {manifest.get('expert_profile')!r}")
            if (manifest.get("expert_order") != list(order)
                    or manifest.get("feature_dimensions") != dimensions
                    or manifest.get("feature_width") != width
                    or manifest.get("score_width") != len(order)):
                raise ValueError("expert order/count or feature dimensions do not match the profile contract")
            rows = manifest["shards"]
            sample_ids = [sample_id for shard in rows for sample_id in shard["sample_ids"]]
            filenames = [name for shard in rows for name in shard["wave_file_names"]]
            sample_count = sum(shard["count"] for shard in rows)
            if sample_count != expected_counts[split]:
                raise ValueError(f"sample count {sample_count} != expected {expected_counts[split]}")
            if len(sample_ids) != len(set(sample_ids)) or len(filenames) != len(set(filenames)):
                raise ValueError("sample IDs or source filenames are not unique")
            if manifest.get("labels_present") is not (split != "test"):
                raise ValueError("label presence does not match the public split contract")

            source_digest = manifest.get("source_manifest_sha256")
            if (not isinstance(source_digest, str) or len(source_digest) != 64
                    or any(character not in "0123456789abcdef" for character in source_digest.lower())):
                raise ValueError("source manifest digest is not a 64-character hexadecimal SHA-256")
            source_records = _read_source_manifest(
                source_root / split / "source_manifest.json",
                split=split,
                expected_sha256=source_digest,
            )
            if len(source_records) != expected_counts[split]:
                raise ValueError("source manifest row count differs from the expected split count")
            if ([record["sample_id"] for record in source_records] != sample_ids
                    or [record["wave_file_name"] for record in source_records] != filenames):
                raise ValueError("source manifest sample IDs/filenames differ from feature-store order")
            fingerprints = manifest["assembly_identity"]["ordered_source_fingerprints"]
            observed_fingerprints = [
                {
                    "sample_id": record["sample_id"],
                    "wav_sha256": record["wav_sha256"].lower(),
                    "caption_sha256": hashlib.sha256(record["raw_caption"].encode("utf-8")).hexdigest(),
                }
                for record in source_records
            ]
            if fingerprints != observed_fingerprints:
                raise ValueError("source manifest WAV/caption fingerprints differ from feature-store assembly")

            provenance = manifest["checkpoint_provenance"]
            preprocessing = preprocessing_contract(manifest)
            if reference_provenance is None:
                reference_provenance, reference_preprocessing = provenance, preprocessing
            elif provenance != reference_provenance:
                raise ValueError("expert checkpoint provenance differs across public splits")
            elif preprocessing != reference_preprocessing:
                raise ValueError("compatible preprocessing contracts differ across public splits")
            result[split] = manifest
        except Exception as exc:
            raise ValueError(f"{split}: D01 feature-store validation failed ({context}): {exc}") from exc
    return result


__all__ = [
    "DEFAULT_ACCURACY_THRESHOLD_RAW",
    "DEFAULT_SENSITIVITY_THRESHOLDS_RAW",
    "FULL8_INFERENCE_COMPAT_FORMAT_VERSION",
    "FULL8_INFERENCE_COMPAT_SOURCE_FILES",
    "FULL8_GATE_ARCHITECTURE",
    "NORMALIZED_SCORE_SCALE",
    "RAW_SCORE_SCALE",
    "SCORE_BINS",
    "TAIL_BINS",
    "analyze_split",
    "build_prediction_rows",
    "compute_envelope_diagnostics",
    "denormalize_scores",
    "load_gate_checkpoint_for_d01",
    "normalize_raw_scores",
    "prepare_format7_gate_payload",
    "summarize_by_generator",
    "validate_d01_feature_store",
    "write_prediction_csv",
]
