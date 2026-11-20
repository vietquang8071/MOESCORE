import ast
import json
import re
import subprocess
from pathlib import Path


def test_full7_kaggle_notebook_has_no_cross_split_wav_policy_and_keeps_metrics_stage():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    code_cells = [source for cell, source in zip(notebook["cells"], cells)
                  if cell.get("cell_type") == "code"]
    for index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    combined = "\n".join(code_cells)
    assert "ALLOW_TRAIN_VALIDATION_AUDIO_OVERLAP" not in combined
    assert "allow_train_validation_audio_overlap" not in combined
    assert "audio_overlap_diagnostic" not in combined
    assert "--allow-train-validation-audio-overlap" not in combined
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
    assert pinned
    subprocess.run(["git", "cat-file", "-e", f"{pinned.group(1)}^{{commit}}"], check=True)


def _load_gate_decision_helpers(notebook):
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "def select_gate_training_action" in "".join(cell.get("source", []))
    )
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "select_gate_training_action", "gate_resume_args", "isolated_gate_output_root",
        }
    ]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 "notebook-gate-decision-helpers", "exec"), namespace)
    return (namespace["select_gate_training_action"], namespace["gate_resume_args"],
            namespace["isolated_gate_output_root"], source)


def test_full7_gate_resume_requires_both_checkpoints_and_isolates_incompatible_artifacts(tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    choose_action, resume_args, isolated_output, cell_source = _load_gate_decision_helpers(notebook)

    # A: both checkpoints are valid, but the run is incomplete, so resume latest.
    latest = tmp_path / "safe-resume" / "latest_gate.pt"
    best = tmp_path / "safe-resume" / "best_gate.pt"
    latest.parent.mkdir()
    latest.write_bytes(b"latest-good")
    best.write_bytes(b"best-good")
    decision = choose_action(latest, best, True, True, False)
    assert decision == {
        "action": "resume",
        "resume_checkpoint": latest,
        "isolate_output": False,
    }
    assert resume_args(decision["resume_checkpoint"]) == ["--resume", latest]

    # B: a compatible latest is insufficient when best is missing.
    missing_best_root = tmp_path / "missing-best"
    missing_latest = missing_best_root / "latest_gate.pt"
    missing_latest.parent.mkdir()
    missing_latest.write_bytes(b"preserve-me")
    missing_best = missing_best_root / "best_gate.pt"
    decision = choose_action(missing_latest, missing_best, True, False, False)
    fresh_command = ["train_moe_gating.py", *resume_args(decision["resume_checkpoint"])]
    assert decision["action"] == "fresh_isolated"
    assert decision["isolate_output"] is True
    assert decision["resume_checkpoint"] is None
    assert "--resume" not in fresh_command
    fresh_root = isolated_output(missing_best_root, tmp_path / "gate-runs", 101)
    assert fresh_root == tmp_path / "gate-runs" / "missing-best_fresh_101"
    assert fresh_root != missing_best_root and not fresh_root.exists()
    assert missing_latest.read_bytes() == b"preserve-me"
    assert not missing_best.exists()

    # C: a present but corrupted/incompatible best checkpoint also forces fresh training.
    bad_best_root = tmp_path / "bad-best"
    bad_best_root.mkdir()
    bad_latest, bad_best = bad_best_root / "latest_gate.pt", bad_best_root / "best_gate.pt"
    bad_latest.write_bytes(b"latest-good")
    bad_best.write_bytes(b"corrupt-best")
    decision = choose_action(bad_latest, bad_best, True, False, False)
    assert decision["action"] == "fresh_isolated"
    assert decision["resume_checkpoint"] is None
    assert isolated_output(bad_best_root, tmp_path / "gate-runs", 102) != bad_best_root
    assert bad_latest.read_bytes() == b"latest-good"
    assert bad_best.read_bytes() == b"corrupt-best"
    decision = choose_action(bad_latest, bad_best, False, False, False)
    assert decision["action"] == "fresh_isolated"
    assert decision["resume_checkpoint"] is None
    assert bad_latest.read_bytes() == b"latest-good"
    assert bad_best.read_bytes() == b"corrupt-best"

    # D: a missing or incompatible latest forces isolation even if best is compatible.
    missing_latest_root = tmp_path / "missing-latest"
    missing_latest_root.mkdir()
    missing_latest, valid_best = missing_latest_root / "latest_gate.pt", missing_latest_root / "best_gate.pt"
    valid_best.write_bytes(b"best-good")
    decision = choose_action(missing_latest, valid_best, False, True, False)
    assert decision["action"] == "fresh_isolated"
    assert decision["resume_checkpoint"] is None
    assert valid_best.read_bytes() == b"best-good"

    incompatible_latest = missing_latest_root / "latest_gate.pt"
    incompatible_latest.write_bytes(b"old-latest")
    decision = choose_action(incompatible_latest, valid_best, False, True, False)
    assert decision["action"] == "fresh_isolated"
    assert decision["resume_checkpoint"] is None
    assert incompatible_latest.read_bytes() == b"old-latest"

    # E: completed runs are reused only when both checkpoint compatibility checks pass.
    decision = choose_action(latest, best, True, True, True)
    assert decision["action"] == "reuse_completed"
    assert decision["resume_checkpoint"] is None
    assert decision["isolate_output"] is False
    assert "--resume" not in resume_args(decision["resume_checkpoint"])
    decision = choose_action(latest, best, False, True, True)
    assert decision["action"] == "fresh_isolated"
    assert decision["resume_checkpoint"] is None

    # With no existing checkpoints, start fresh in the selected directory.
    empty_root = tmp_path / "empty"
    decision = choose_action(empty_root / "latest_gate.pt", empty_root / "best_gate.pt",
                             False, False, False)
    assert decision["action"] == "fresh"
    assert decision["isolate_output"] is False
    assert decision["resume_checkpoint"] is None

    # The notebook must consume this tested decision, reset state during isolation,
    # and preserve every checkpoint identity predicate used to derive compatibility.
    assert 'gate_action = select_gate_training_action(' in cell_source
    assert 'resume_checkpoint = gate_action["resume_checkpoint"]' in cell_source
    assert 'if gate_action["isolate_output"]:' in cell_source
    assert 'GATE_ROOT = isolated_gate_output_root(previous_root, GATE_ROOT_BASE, time.time_ns())' in cell_source
    assert 'write_gate_config(GATE_ROOT)' in cell_source
    assert 'latest = GATE_ROOT / "latest_gate.pt"' in cell_source
    assert 'best = GATE_ROOT / "best_gate.pt"' in cell_source
    assert 'completed = GATE_ROOT / "run_manifest.json"' in cell_source
    assert 'resume_checkpoint = None' in cell_source
    assert 'cmd += gate_resume_args(resume_checkpoint)' in cell_source
    completed_check = cell_source.split("completed_compatible = (", 1)[1].split("\n)", 1)[0]
    assert "and latest_compatible" in completed_check
    for predicate in (
        'payload.get("format_version") == 6',
        'identity.get("score_review_policy") == SCORE_REVIEW_POLICY',
        'payload.get("config") == expected_gate_config',
        'payload.get("source_code_sha256") == current_source_sha256',
        'identity == expected_feature_store_identity',
        'payload.get("expert_checkpoint_provenance") == expected_expert_provenance',
        'payload.get("feature_store_identity_sha256") == canonical_sha256(identity)',
    ):
        assert predicate in cell_source
