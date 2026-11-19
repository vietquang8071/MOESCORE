from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MOESCORE_Full10_Kaggle.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook=None):
    return [
        "".join(cell.get("source", []))
        for cell in (notebook or _notebook())["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_full10_notebook_code_cells_compile_and_have_no_stale_outputs():
    notebook = _notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None, f"cell {index} has stale execution count"
        assert cell.get("outputs", []) == [], f"cell {index} has stale execution output"
        compile("".join(cell.get("source", [])), f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_full10_notebook_has_real_inputs_and_save_run_all_defaults():
    config = _code_cells()[0]
    for expected in (
        "/kaggle/input/datasets/qnv8071/xacle-2027-data/",
        "XACLE_Challenge_2027_development_dataset",
        "/kaggle/input/datasets/qnv8071/xacle-weights",
        'EXPERT_PROFILE = "full_10"',
        'REPO_REF = "feature/moe-10-m2d-lora-coral"',
        'KAGGLE_USERNAME = "qnv8071"',
        'FEATURE_DATASET_SLUG = "moescore-full10-feature-cache"',
        "SMOKE_ONLY = False",
        "RUN_GPU_SMOKE = True",
        "RUN_FULL_EXTRACTION = True",
        "RUN_GATE_TRAINING = True",
        "RUN_EVALUATION = True",
        "RUN_TEST_INFERENCE = True",
        "RUN_TEST_EVALUATION = False",
        "UPLOAD_FEATURE_CACHE = False",
        "REBUILD_HF_CACHE = False",
        "UPLOAD_REPAIRED_HF_CACHE = False",
    ):
        assert expected in config


def test_full10_notebook_checkpoints_hf_cache_and_gpu_contract_are_configured():
    config = _code_cells()[0]
    assert '"top2_msclap.checkpoint"' in config
    assert '"m2d_lora.finetuned_checkpoint"' in config
    assert "EXPERT10_CHECKPOINT" in config
    assert "d6edbec6b724be48a18f618c98a2e25d044f06108e8715d86b58c59e97e6cbdb" in config
    assert "moescore-hf-text-cache-v2" in config
    assert "m2d_lora_coral" in config
    assert "1 if name == \"m2d_lora_coral\" else 4" in config
    assert "GATE_DEVICE = \"cuda:0\"" in config
    assert "TRAIN_BATCH_SIZE = 8" in config
    assert "TRAIN_EPOCHS = 50" in config
    assert "TRAIN_LEARNING_RATE = 1e-5" in config
    assert "SEED = 2027" in config


def test_full10_notebook_restores_optional_cache_before_full_extraction():
    cells = _code_cells()
    restore_cell = next(i for i, source in enumerate(cells) if "RESTORE_EXISTING_FEATURE_CACHE" in source)
    extraction_cell = next(i for i, source in enumerate(cells) if "RUN_FULL_EXTRACTION" in source and "kaggle_full9_dual_gpu.py" in source)
    assert restore_cell < extraction_cell
    assert "EXISTING_FEATURE_CACHE is None" in cells[restore_cell]
    assert "profile=\"full_10\"" in cells[extraction_cell]
    assert "completed_stores_match" in cells[extraction_cell]


def test_full10_notebook_has_all_ordered_automated_stages_and_run_summary():
    all_code = "\n".join(_code_cells())
    required = (
        "source_checkout",
        "dependency_setup",
        "hf_cache_restore",
        "config_generation",
        "offline_preflight",
        "gpu_smoke",
        "smoke_gate",
        "existing_feature_cache_restore",
        "full_extraction",
        "feature_store_validation",
        "gate_training",
        "validation_evaluation",
        "test_inference",
        "test_evaluation",
        "final_artifact_verification",
    )
    positions = [all_code.index(name) for name in required]
    assert positions == sorted(positions)
    assert "run_summary.json" in all_code
    assert "REPO_COMMIT_SHA" in all_code
    assert "profile_contract(\"full_10\")" in all_code
    assert "facebook/bart-base" in all_code
    assert "BartTokenizer.from_pretrained" in all_code
    assert "labels_required" in all_code or "require_labels=False" in all_code
