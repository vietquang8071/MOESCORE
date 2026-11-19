#!/usr/bin/env python3
"""Report train-only Full7 expert-score and raw-label diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.full7_score_contract import audit_train_scores
from data.moe_feature_store import MoeFeatureStoreDataset, _canonical_json


def audit_store(root: str | Path) -> dict:
    dataset = MoeFeatureStoreDataset(root, split="train", require_labels=True)
    if dataset.manifest.get("expert_profile") != "full_7":
        raise ValueError("score audit requires a verified full_7 train store")
    scores = np.stack([dataset[index]["scores"].numpy() for index in range(len(dataset))])
    labels = np.asarray([dataset[index]["label_raw"].item() for index in range(len(dataset))], np.float32)
    contracts = dataset.manifest["score_contracts"]
    diagnostics = audit_train_scores(scores, labels)
    return {"split": "train", "store_identity_sha256": dataset.identity,
            "score_contracts": contracts, "diagnostics": diagnostics,
            "score_contract_sha256": hashlib.sha256(_canonical_json(contracts)).hexdigest(),
            "score_audit_sha256": hashlib.sha256(_canonical_json(diagnostics)).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = {"status": "PASS", **audit_store(args.store_root)}
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.output_json:
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
