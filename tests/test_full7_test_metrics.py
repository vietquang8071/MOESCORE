import csv
import json

import pytest

from scripts.full7_test_metrics import evaluate_post_inference_metrics


def _write_csv(path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_post_inference_metrics_require_exact_filenames_and_report_raw_scale(tmp_path):
    predictions = tmp_path / "submission.csv"
    labels = tmp_path / "test_average.csv"
    output = tmp_path / "test_metrics.json"
    _write_csv(predictions, [
        {"wave_file_name": "a.wav", "pred_score": "1"},
        {"wave_file_name": "b.wav", "pred_score": "3"},
        {"wave_file_name": "c.wav", "pred_score": "5"},
        {"wave_file_name": "d.wav", "pred_score": "7"},
    ], ["wave_file_name", "pred_score"])
    _write_csv(labels, [
        {"wav_file_name": "a.wav", "text": "a", "average_score": "0"},
        {"wav_file_name": "b.wav", "text": "b", "average_score": "4"},
        {"wav_file_name": "c.wav", "text": "c", "average_score": "4"},
        {"wav_file_name": "d.wav", "text": "d", "average_score": "8"},
    ], ["wav_file_name", "text", "average_score"])

    result = evaluate_post_inference_metrics(predictions, labels, output, expected_count=4)

    assert result["status"] == "POST_INFERENCE_RESEARCH_ONLY"
    assert result["sample_count"] == 4
    assert result["score_scale"] == "raw_0_to_10"
    assert set(result["metrics"]) == {"srcc", "lcc", "kendall_tau", "mse", "mae"}
    assert result["metrics"]["mse"] == pytest.approx(1.0)
    assert json.loads(output.read_text())["prediction_sha256"] == result["prediction_sha256"]


@pytest.mark.parametrize("prediction_names,label_names", [
    (["a.wav", "b.wav"], ["a.wav", "c.wav"]),
    (["a.wav", "a.wav"], ["a.wav", "b.wav"]),
])
def test_post_inference_metrics_reject_misaligned_or_duplicate_filenames(
    tmp_path, prediction_names, label_names,
):
    predictions, labels = tmp_path / "pred.csv", tmp_path / "labels.csv"
    _write_csv(predictions, [{"wave_file_name": n, "pred_score": "5"} for n in prediction_names],
               ["wave_file_name", "pred_score"])
    _write_csv(labels, [{"wav_file_name": n, "text": "x", "average_score": "5"} for n in label_names],
               ["wav_file_name", "text", "average_score"])
    with pytest.raises(ValueError, match="duplicate|one-to-one"):
        evaluate_post_inference_metrics(predictions, labels, tmp_path / "out.json", expected_count=2)


@pytest.mark.parametrize("prediction,label", [("nan", "5"), ("11", "5"), ("5", "-1")])
def test_post_inference_metrics_reject_nonfinite_or_out_of_scale_values(tmp_path, prediction, label):
    predictions, labels = tmp_path / "pred.csv", tmp_path / "labels.csv"
    _write_csv(predictions, [{"wave_file_name": "a.wav", "pred_score": prediction}],
               ["wave_file_name", "pred_score"])
    _write_csv(labels, [{"wav_file_name": "a.wav", "text": "x", "average_score": label}],
               ["wav_file_name", "text", "average_score"])
    with pytest.raises(ValueError, match=r"finite|\[0,10\]"):
        evaluate_post_inference_metrics(predictions, labels, tmp_path / "out.json", expected_count=1)
