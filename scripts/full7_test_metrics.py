"""Compute explicitly post-inference research metrics for frozen Full7 predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def _read_rows(path: Path, *, required_columns: set[str], description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{description} must contain columns: {', '.join(sorted(required_columns))}"
            )
        rows = list(reader)
    return rows


def _validated_names(rows: list[dict[str, str]], *, column: str, description: str) -> list[str]:
    names = [row.get(column, "") for row in rows]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{description} contains an empty filename")
    if len(names) != len(set(names)):
        raise ValueError(f"{description} contains duplicate filenames")
    return names


def _scores(rows: list[dict[str, str]], *, column: str, description: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{description} contains a nonnumeric score") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{description} contains nonfinite scores")
    if ((values < 0.0) | (values > 10.0)).any():
        raise ValueError(f"{description} scores must be in [0,10]")
    return values


def _correlation_value(result: Any) -> float | None:
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def evaluate_post_inference_metrics(
    prediction_csv: str | Path,
    label_csv: str | Path,
    output_json: str | Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    """Match frozen predictions to test labels one-to-one and save research metrics.

    Call this only after the label-free inference process has written its final CSV.
    """
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    predictions_path = Path(prediction_csv).expanduser().resolve()
    labels_path = Path(label_csv).expanduser().resolve()
    prediction_rows = _read_rows(
        predictions_path,
        required_columns={"wave_file_name", "pred_score"},
        description="prediction",
    )
    label_rows = _read_rows(
        labels_path,
        required_columns={"wav_file_name", "text", "average_score"},
        description="official test label",
    )
    prediction_names = _validated_names(
        prediction_rows, column="wave_file_name", description="prediction CSV",
    )
    label_names = _validated_names(label_rows, column="wav_file_name", description="test label CSV")
    if len(prediction_rows) != expected_count or len(label_rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} test rows, got {len(prediction_rows)} predictions "
            f"and {len(label_rows)} labels"
        )
    if set(prediction_names) != set(label_names):
        missing_predictions = sorted(set(label_names) - set(prediction_names))[:5]
        missing_labels = sorted(set(prediction_names) - set(label_names))[:5]
        raise ValueError(
            "prediction and label filenames do not match one-to-one "
            f"(missing predictions={missing_predictions}, missing labels={missing_labels})"
        )
    predictions = _scores(prediction_rows, column="pred_score", description="prediction CSV")
    label_by_name = {name: row for name, row in zip(label_names, label_rows)}
    aligned_label_rows = [label_by_name[name] for name in prediction_names]
    labels = _scores(aligned_label_rows, column="average_score", description="test label CSV")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "srcc": _correlation_value(spearmanr(predictions, labels)),
            "lcc": _correlation_value(pearsonr(predictions, labels)),
            "kendall_tau": _correlation_value(kendalltau(predictions, labels)),
            "mse": float(np.mean(np.square(predictions - labels))),
            "mae": float(np.mean(np.abs(predictions - labels))),
        }
    prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    report = {
        "status": "POST_INFERENCE_RESEARCH_ONLY",
        "run_purpose": "research_only",
        "label_access_stage": "after_test_predictions_frozen",
        "sample_count": expected_count,
        "prediction_sha256": prediction_sha256,
        "prediction_file": predictions_path.name,
        "label_file": labels_path.name,
        "score_scale": "raw_0_to_10",
        "metrics": metrics,
        "warning": (
            "These post-inference test metrics are research-only and do not establish "
            "independent validation or competition compliance."
        ),
    }
    target = Path(output_json).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(target)
    return report

