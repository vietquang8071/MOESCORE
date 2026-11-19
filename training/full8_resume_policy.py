"""Safe output-directory and resume selection for Full8 Kaggle Gate runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Full8GateRunSelection:
    action: str
    output_dir: Path
    resume_checkpoint: Path | None
    checkpoint_for_use: Path | None


def choose_full8_gate_run(
    output_dir: str | Path,
    *,
    latest_path: str | Path,
    best_path: str | Path,
    manifest_path: str | Path,
    latest_compatible: bool,
    best_compatible: bool,
    completed_compatible: bool,
    isolated_output_dir: str | Path | None = None,
) -> Full8GateRunSelection:
    """Resume only with a compatible latest+best pair; isolate unsafe artifacts."""
    root = Path(output_dir).expanduser().resolve()
    latest = Path(latest_path).expanduser().resolve()
    best = Path(best_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if any(path.parent != root for path in (latest, best, manifest)):
        raise ValueError("Full8 Gate checkpoint paths must belong to the selected output directory")
    for name, value in (("latest_compatible", latest_compatible),
                        ("best_compatible", best_compatible),
                        ("completed_compatible", completed_compatible)):
        if type(value) is not bool:
            raise TypeError(f"{name} must be boolean")
    if latest_compatible and not latest.is_file():
        raise ValueError("latest checkpoint cannot be marked compatible when it is missing")
    if best_compatible and not best.is_file():
        raise ValueError("best checkpoint cannot be marked compatible when it is missing")
    if completed_compatible:
        if not (latest_compatible and best_compatible and manifest.is_file()):
            raise ValueError("completed Full8 run requires compatible latest, best and run manifest")
        return Full8GateRunSelection("reuse_completed", root, None, best)
    if latest_compatible and best_compatible and not manifest.exists():
        return Full8GateRunSelection("resume_latest", root, latest, None)

    has_artifacts = latest.exists() or best.exists() or manifest.exists()
    if not has_artifacts:
        return Full8GateRunSelection("fresh", root, None, None)
    if isolated_output_dir is None:
        raise ValueError("incompatible Full8 artifacts require a fresh isolated output directory")
    isolated = Path(isolated_output_dir).expanduser().resolve()
    if isolated == root:
        raise ValueError("fresh Full8 output directory must be isolated from existing artifacts")
    if any((isolated / name).exists() for name in ("latest_gate.pt", "best_gate.pt", "run_manifest.json")):
        raise FileExistsError(f"isolated Full8 output directory already contains Gate artifacts: {isolated}")
    return Full8GateRunSelection("fresh_isolated", isolated, None, None)


def gate_resume_args(selection: Full8GateRunSelection) -> list[str]:
    """Build an empty CLI suffix for fresh runs, preventing stale resume paths."""
    if selection.resume_checkpoint is None:
        return []
    return ["--resume", str(selection.resume_checkpoint)]


__all__ = ["Full8GateRunSelection", "choose_full8_gate_run", "gate_resume_args"]
