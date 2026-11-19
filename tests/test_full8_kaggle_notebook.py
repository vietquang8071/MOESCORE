from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "MOESCORE_Full8_T4x2_Kaggle.ipynb"


def _cells():
    body = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return body, ["".join(cell.get("source", [])) for cell in body["cells"]]


def test_full8_kaggle_notebook_json_code_cells_and_source_pin():
    notebook, cells = _cells()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"Full8 notebook cell {index}", "exec")
    config = next(cell for cell in cells if "SOURCE_COMMIT =" in cell)
    match = re.search(r'SOURCE_COMMIT\s*=\s*["\']([0-9a-f]{40})["\']', config)
    assert match, "Full8 notebook must pin a complete implementation commit SHA"
    subprocess.run(["git", "cat-file", "-e", f"{match.group(1)}^{{commit}}"], check=True)


def test_full8_notebook_uses_exact_e8_checkpoint_and_preflights_before_expensive_stages():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert '"m2d-lora-Nam.pt"' in combined
    assert "E8_LORA_CHECKPOINT_OVERRIDE = None" in combined
    assert 'input_root.rglob(filename)' in combined
    assert "Ambiguous" in combined
    assert "load_expert8_checkpoint(E8_LORA_CHECKPOINT)" in combined
    assert "validate_base_m2d_checkpoint(E8_BASE_CHECKPOINT)" in combined
    assert "EXPECTED_LORA_SHAPES" in combined and "== 24" in combined
    all_source = "\n".join(cells)
    preflight_index = all_source.index("validate_base_m2d_checkpoint(E8_BASE_CHECKPOINT)")
    e5_index = all_source.index("E5_CONFIG = CONFIG_ROOT")
    whisper_index = all_source.index("def whisper_command(split")
    assert preflight_index < e5_index and preflight_index < whisper_index


def test_full8_notebook_has_cache_reuse_and_full_pipeline_in_order():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "verify_full7_candidate" in combined
    assert "NEED_FULL7_BUILD" in combined
    assert "Full7 Gate is never trained" in combined
    ordered = (
        "from scripts.kaggle_full9_artifacts import restore_feature_cache, validate_full9_store",
        "E5_CONFIG = CONFIG_ROOT", "def whisper_command(split", "E6_CONFIG = CONFIG_ROOT",
        "FULL7_BUILD_REPORT =", "E8_CONFIG=CONFIG_ROOT", "FULL8_BUILD_REPORT=",
        "full8_train_only_score_audit", 'run_logged("full8_gate_train"',
        'if RUN_VALIDATION:', 'if RUN_TEST_INFERENCE:',
        "FROZEN_AFTER_LABEL_FREE_INFERENCE", "evaluate_post_inference_metrics",
    )
    positions = [combined.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert "E8_BATCH_SIZE = 1" in combined
    assert "E8_SHARD_SIZE = 8" in combined
    assert "--allow-train-validation-audio-overlap" in combined


def test_full8_notebook_propagates_experimental_review_and_never_fabricates_review():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "SKIP_SCORE_REVIEW_EXPERIMENTAL = True" in combined
    assert 'gate_env["MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW"]="1"' in combined
    assert "EXPERIMENTAL_SKIPPED" in combined
    assert "RUN_SCORE_AUDIT = True" in combined
    assert "_real_score_review" in combined
    assert "score_review_file" in combined
    assert "REVIEWED_FOR_GATE_PROXY" not in combined
    assert "reviewer\"] =" not in combined
    assert "evidence\"] =" not in combined


def test_test_ground_truth_is_accessed_only_after_inference_freeze():
    _, cells = _cells()
    notebook, _ = _cells()
    accesses = [i for i, cell in enumerate(notebook["cells"])
                if cell.get("cell_type") == "code" and "test_average.csv" in "".join(cell.get("source", []))]
    assert len(accesses) == 1
    inference_cell = next(i for i, text in enumerate(cells) if 'run_logged("full8_test_inference"' in text)
    freeze_cell = next(i for i, text in enumerate(cells) if "FROZEN_AFTER_LABEL_FREE_INFERENCE" in text)
    metric_cell = next(i for i, text in enumerate(cells) if "evaluate_post_inference_metrics" in text)
    assert inference_cell == freeze_cell < accesses[0] == metric_cell
    assert cells[inference_cell].index('run_logged("full8_test_inference"') < cells[inference_cell].index("FROZEN_AFTER_LABEL_FREE_INFERENCE")
    assert '["wave_file_name","pred_score"]' in cells[inference_cell]
    assert "ALLOW_POST_INFERENCE_TEST_LABEL_ACCESS" in cells[metric_cell]


def test_full7_kaggle_notebook_remains_unmodified():
    root = Path(__file__).parents[1]
    actual = (root / "notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb").read_bytes()
    expected = subprocess.run(
        ["git", "show", "HEAD:notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb"],
        check=True, capture_output=True,
    ).stdout
    assert actual == expected
