"""Verify the bundled BEATs source against the original training bytes.

Only the two reviewed package-relative imports are normalized. All other
bytes, including modules.py, must reproduce the checkpoint's pinned source.
The observed Git commit is never rewritten or presented as the original one.
"""
from pathlib import Path
import hashlib

ORIGINAL_COMMIT = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
ORIGINAL_SHA256 = "fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487"


def verify_bundled_import_equivalence(source_dir, expected):
    if expected != {"git_commit": ORIGINAL_COMMIT, "sha256": ORIGINAL_SHA256}:
        raise ValueError("UniLM source identity is not the reviewed original training revision")
    root = Path(source_dir)
    root = root if (root / "BEATs.py").is_file() else root / "beats"
    digest = hashlib.sha256()
    replacements = {"BEATs.py": (b"from .backbone import (", b"from backbone import ("),
                    "backbone.py": (b"from .modules import (", b"from modules import (")}
    for name in ("BEATs.py", "backbone.py", "modules.py"):
        data = (root / name).read_bytes()
        if name in replacements:
            relative, absolute = replacements[name]
            if data.count(relative) == 1:
                data = data.replace(relative, absolute, 1)
        digest.update(name.encode())
        digest.update(data)
    if digest.hexdigest() != ORIGINAL_SHA256:
        raise ValueError("UniLM BEATs model source differs beyond the two reviewed relative imports")
    return {"status": "VERIFIED_IMPORT_ONLY_EQUIVALENCE", "normalized_sha256": digest.hexdigest(),
            "checkpoint_source_identity": expected, "source_dir": str(root.resolve())}
