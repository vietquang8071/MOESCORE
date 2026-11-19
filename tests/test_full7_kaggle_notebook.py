import json
import re
from pathlib import Path


def test_full7_kaggle_notebook_propagates_experimental_flags_and_metrics_stage():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    code_cells = [source for cell, source in zip(notebook["cells"], cells)
                  if cell.get("cell_type") == "code"]
    for index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    combined = "\n".join(code_cells)
    assert re.search(r"^ALLOW_TRAIN_VALIDATION_AUDIO_OVERLAP\s*=\s*True\s*$", combined, re.MULTILINE)
    assert 'e6_cfg["training"]["allow_train_validation_audio_overlap"] = ALLOW_TRAIN_VALIDATION_AUDIO_OVERLAP' in combined
    assert '"--allow-train-validation-audio-overlap"' in combined
    assert 'gate_cfg["data"]["allow_train_validation_audio_overlap"] = ALLOW_TRAIN_VALIDATION_AUDIO_OVERLAP' in combined
    assert re.search(r"^SKIP_SCORE_REVIEW_EXPERIMENTAL\s*=\s*True\s*$", combined, re.MULTILINE)
    assert re.search(r"^RUN_TEST_METRICS\s*=\s*True\s*$", combined, re.MULTILINE)
    assert re.search(r"^RUN_SCORE_AUDIT\s*=\s*True\s*$", combined, re.MULTILINE)
    assert re.search(r"^HUMAN_SCORE_REVIEW_CONFIRMED\s*=\s*False\s*$", combined, re.MULTILINE)
    assert "gate_env[\"MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW\"] = \"1\"" in combined
    assert "gate_env.pop(\"MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW\", None)" in combined
    assert '"EXPERIMENTAL_SKIPPED" if SKIP_SCORE_REVIEW_EXPERIMENTAL else "HUMAN_REVIEWED"' in combined
    assert "current_source_sha256 = full7_source_code_sha256()" in combined
    assert "identity == expected_feature_store_identity" in combined
    assert 'payload.get("selection_policy") == "official_validation_srcc_strict_v1_provisional"' in combined
    assert "test_metrics_report[\"prediction_sha256\"] == test_metrics_report[\"submission_sha256\"]" in combined
    inference_index = next(
        i for i, (cell, source) in enumerate(zip(notebook["cells"], cells))
        if cell.get("cell_type") == "code" and 'run_logged("full7_test_inference"' in source
    )
    metrics_index = inference_index + 1
    assert notebook["cells"][metrics_index]["cell_type"] == "code"
    assert "evaluate_post_inference_metrics" in cells[metrics_index]
    assert 'DATA_ROOT / "meta_data" / "test_average.csv"' in cells[metrics_index]
    assert "POST_INFERENCE_RESEARCH_ONLY" in Path(root / "scripts/full7_test_metrics.py").read_text()
    assert all(
        "test_average.csv" not in "".join(cell.get("source", []))
        for index, cell in enumerate(notebook["cells"][:inference_index + 1])
        if cell.get("cell_type") == "code"
    )
    review_cell = next(source for source in cells if "No reviewer, evidence, or score_review.json is being fabricated." in source)
    experimental_branch = review_cell.split("else:", 1)[0]
    assert "REVIEWED_FOR_GATE_PROXY" not in experimental_branch
    assert "SCORE_REVIEW_PATH.write_text" not in experimental_branch
    pinned = re.search(r'^SOURCE_COMMIT\s*=\s*"([0-9a-f]{40})"', combined, re.MULTILINE)
    assert pinned and pinned.group(1) == "fe7de437842e4907421d645cfe71d0015bce895c"
