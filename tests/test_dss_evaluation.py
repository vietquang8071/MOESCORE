import math

import pytest

from training.dss_evaluation import summarize_predictions, write_final_evaluation


def test_raw_metrics_and_domain_aggregation():
    rows = []
    for domain in range(2):
        for index in range(4):
            y = float(index)
            rows.append({"sample_id": f"train:{domain}:{index}", "model_id": domain,
                         "model_name": str(domain), "ground_truth_raw": y,
                         "prediction_raw": y + 1 if domain == 0 else 4 - y,
                         **{f"gate_weight_{i+1}": 0.125 for i in range(8)}})
    global_metrics, per_domain, aggregate, diagnostics = summarize_predictions(rows, "train", expected_count=8)
    assert global_metrics["sample_count"] == 8
    assert global_metrics["mse"] == pytest.approx(3.5)
    assert [row["srcc"] for row in per_domain] == pytest.approx([1, -1])
    assert aggregate["macro_srcc"] == pytest.approx(0)
    assert aggregate["weighted_srcc"] == pytest.approx(0)
    assert aggregate["worst_domain_srcc"] == pytest.approx(-1)
    assert aggregate["per_domain_srcc_population_std"] == pytest.approx(1)
    assert diagnostics[0]["normalized_entropy"] == pytest.approx(1)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_predictions(rows + [rows[0]], "train", expected_count=9)


def test_final_exports_cover_all_splits_and_required_plots(tmp_path):
    from data.dss_dataset import DOMAIN_NAMES, EXPECTED_COUNTS

    predictions = {}
    for split, counts in EXPECTED_COUNTS.items():
        rows = []
        for domain, count in counts.items():
            for index in range(count):
                target = float(index % 10)
                rows.append({"sample_id": f"{split}:{domain}_{index}.wav", "source_split": split,
                             "wav_file_name": f"{domain}_{index}.wav", "model_id": domain,
                             "model_name": DOMAIN_NAMES[domain], "ground_truth_raw": target,
                             "prediction_raw": target + .1,
                             **{f"expert_score_{i+1}": 0.1 for i in range(8)},
                             **{f"gate_weight_{i+1}": .125 for i in range(8)},
                             "checkpoint_sha256": "a" * 64})
        predictions[split] = rows
    history = [{"epoch": 1, "global_loss": 1., "domain_loss": 1., "total_loss": 1.,
                "validation_srcc": .4},
               {"epoch": 2, "global_loss": .8, "domain_loss": .7, "total_loss": .75,
                "validation_srcc": .5}]
    global_metrics, per_domain, aggregates = write_final_evaluation(tmp_path, predictions, history)
    assert sum(row["sample_count"] for row in global_metrics.values()) == 13500
    assert len(per_domain) == 27 and len(aggregates) == 3
    assert global_metrics["test"]["mse"] == pytest.approx(.01)
    assert (tmp_path / "predictions/public_test.csv").is_file()
    for name in ("training_loss", "validation_srcc", "per_generator_srcc", "score_distributions", "expert_weights"):
        assert (tmp_path / "plots" / f"{name}.png").is_file()
