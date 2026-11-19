from __future__ import annotations

import pytest

from training.full8_resume_policy import choose_full8_gate_run, gate_resume_args


def _paths(root):
    return root / "latest_gate.pt", root / "best_gate.pt", root / "run_manifest.json"


def test_resume_requires_both_compatible_checkpoints_and_uses_latest(tmp_path):
    root = tmp_path / "gate"
    root.mkdir()
    latest, best, manifest = _paths(root)
    latest.write_bytes(b"latest")
    best.write_bytes(b"best")
    selected = choose_full8_gate_run(
        root, latest_path=latest, best_path=best, manifest_path=manifest,
        latest_compatible=True, best_compatible=True, completed_compatible=False,
    )
    assert selected.action == "resume_latest"
    assert selected.output_dir == root.resolve()
    assert selected.resume_checkpoint == latest.resolve()
    assert gate_resume_args(selected) == ["--resume", str(latest.resolve())]


@pytest.mark.parametrize(
    "latest_exists,best_exists,latest_ok,best_ok,manifest_exists,completed_ok",
    [
        (True, False, True, False, False, False),  # missing best
        (True, True, True, False, False, False),   # corrupted/incompatible best
        (False, True, False, True, False, False),  # missing latest
        (True, True, False, True, False, False),   # incompatible latest
        (True, True, True, True, True, False),     # invalid completed run
        (True, True, False, False, False, False),  # both incompatible
    ],
)
def test_unsafe_artifacts_are_preserved_and_isolated_without_resume(
    tmp_path, latest_exists, best_exists, latest_ok, best_ok, manifest_exists, completed_ok,
):
    root = tmp_path / "old_gate"
    root.mkdir()
    latest, best, manifest = _paths(root)
    if latest_exists:
        latest.write_bytes(b"old-latest")
    if best_exists:
        best.write_bytes(b"old-best")
    if manifest_exists:
        manifest.write_text('{"old": true}')
    before = {path: path.read_bytes() for path in (latest, best, manifest) if path.is_file()}
    isolated = tmp_path / "old_gate_fresh_1"
    selected = choose_full8_gate_run(
        root, latest_path=latest, best_path=best, manifest_path=manifest,
        latest_compatible=latest_ok, best_compatible=best_ok,
        completed_compatible=completed_ok, isolated_output_dir=isolated,
    )
    assert selected.action == "fresh_isolated"
    assert selected.output_dir == isolated.resolve()
    assert selected.resume_checkpoint is None
    assert gate_resume_args(selected) == []
    assert {path: path.read_bytes() for path in before} == before


def test_fully_completed_compatible_run_reuses_best_without_training(tmp_path):
    root = tmp_path / "gate"
    root.mkdir()
    latest, best, manifest = _paths(root)
    latest.write_bytes(b"latest")
    best.write_bytes(b"best")
    manifest.write_text('{"complete": true}')
    selected = choose_full8_gate_run(
        root, latest_path=latest, best_path=best, manifest_path=manifest,
        latest_compatible=True, best_compatible=True, completed_compatible=True,
    )
    assert selected.action == "reuse_completed"
    assert selected.checkpoint_for_use == best.resolve()
    assert selected.resume_checkpoint is None
    assert gate_resume_args(selected) == []


def test_no_existing_artifacts_starts_fresh_in_selected_directory(tmp_path):
    root = tmp_path / "gate"
    root.mkdir()
    latest, best, manifest = _paths(root)
    selected = choose_full8_gate_run(
        root, latest_path=latest, best_path=best, manifest_path=manifest,
        latest_compatible=False, best_compatible=False, completed_compatible=False,
    )
    assert selected.action == "fresh"
    assert selected.output_dir == root.resolve()
    assert selected.resume_checkpoint is None
    assert gate_resume_args(selected) == []
