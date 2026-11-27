import json
import re
from pathlib import Path


def test_dss_kernel_is_pinned_and_syncs_notebook_copies():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/MOESCORE_Full8_DSS_Kaggle.ipynb").read_text())
    deployed = json.loads((root / "kaggle/dss/MOESCORE_Full8_DSS_Kaggle.ipynb").read_text())
    metadata = json.loads((root / "kaggle/dss/kernel-metadata.json").read_text())
    assert notebook == deployed
    config = "".join(notebook["cells"][1]["source"])
    match = re.search(r'^SOURCE_COMMIT = "([0-9a-f]{40})"$', config, re.MULTILINE)
    assert match, "DSS notebook must checkout a reviewed exact commit"
    assert 'RUN_MODE = "smoke"' in config
    assert metadata["enable_gpu"] == "true"
    assert set(metadata["dataset_sources"]) == {
        "qnv8071/xacle-2027-data", "qnv8071/moescore-full8-artifacts"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "dss_notebook_cell", "exec")
