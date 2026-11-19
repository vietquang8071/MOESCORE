import json
import re
from pathlib import Path


def test_full7_kaggle_notebook_propagates_experimental_overlap_flag():
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
    pinned = re.search(r'^SOURCE_COMMIT\s*=\s*"([0-9a-f]{40})"', combined, re.MULTILINE)
    assert pinned and pinned.group(1) == "ac227fab0aea91b2cfdfd6c01a95c465cbbeca90"
