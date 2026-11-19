import numpy as np
import pytest

from data.full7_score_contract import validate_score_contracts, audit_train_scores


def test_missing_or_unverified_contract_fails_closed():
    contracts = {f"expert{i}": {"source": "m2d_lora" if i == 7 else f"expert{i}", "semantics": "synthetic_score",
        "target_scale": "normalized_minus1_to1", "output_scale": "normalized_minus1_to1", "conversion": "identity",
        "identity": "a" * 64, "audit_status": "SYNTHETIC_VERIFIED"} for i in range(1, 8)}
    validate_score_contracts(contracts, synthetic=True)
    contracts["expert3"]["audit_status"] = "UNVERIFIED"
    with pytest.raises(ValueError, match="UNVERIFIED"):
        validate_score_contracts(contracts, synthetic=True)
    del contracts["expert3"]["semantics"]
    with pytest.raises(ValueError):
        validate_score_contracts(contracts, synthetic=True)
    contracts["expert3"].update({"semantics": "synthetic_score", "audit_status": "SYNTHETIC_VERIFIED",
                                  "output_scale": "raw_0_to_10"})
    with pytest.raises(ValueError, match="conversion/scale"):
        validate_score_contracts(contracts, synthetic=True)


def test_train_score_audit_reports_range_and_labels():
    result = audit_train_scores(np.array([[0.0] * 7, [2.0] * 7], np.float32),
                                np.array([5.0, 6.0], np.float32))
    assert result["sample_count"] == 2
    assert result["experts"]["expert1"]["outside_minus1_to1_fraction"] == 0.5
    assert result["labels_raw"]["mean"] == 5.5


def test_real_score_contract_rejects_contradictory_source_and_semantics():
    semantics = ("clap_similarity", "mga_similarity", "fine_tuned_normalized_regression",
                 "tanh_regression", "set_a_normalized_residual", "set_b_normalized_residual",
                 "m2d_lora_coral_regression_normalized")
    contracts = {f"expert{i}": {"source": "m2d_lora" if i == 7 else f"expert{i}",
        "semantics": semantics[i - 1], "target_scale": "normalized_minus1_to1",
        "output_scale": "unitless_similarity_proxy" if i <= 2 else "normalized_minus1_to1",
        "conversion": "identity", "identity": "a" * 64, "audit_status": "CODE_VERIFIED"}
        for i in range(1, 8)}
    validate_score_contracts(contracts, synthetic=False)
    contracts["expert3"]["semantics"] = "raw_regression"
    with pytest.raises(ValueError, match="semantics"):
        validate_score_contracts(contracts, synthetic=False)
    contracts["expert3"]["semantics"] = semantics[2]
    contracts["expert7"]["source"] = "expert7"
    with pytest.raises(ValueError, match="source"):
        validate_score_contracts(contracts, synthetic=False)


def test_real_full9_score_evidence_requires_hashed_source_code():
    from data.full7_assembler import _score_contracts
    source_names = ("expert1", "expert2", "expert3", "expert4", "m2d_lora")
    manifest = {"per_expert_cache_identities": {
        name: {"cache_id": "a" * 64, "implementation_provenance": {"source": "real"},
               "checkpoint_provenance": {"sha256": "b" * 64},
               "score_contract": {"score_shape": [None, 1], "score_dtype": "float32",
                                  "feature_shape": [None, width], "feature_dtype": "float32"}}
        for name, width in zip(source_names, (2049, 4097, 3840, 1024, 3073))},
        "checkpoint_provenance": {name: {"sha256": "b" * 64} for name in source_names}}
    with pytest.raises(ValueError, match="source-code evidence"):
        _score_contracts(manifest, "c" * 64, "d" * 64)


def test_real_training_requires_bound_score_review(tmp_path):
    from scripts.train_moe_gating import _real_score_review
    import hashlib
    import json
    manifest = {"store_identity_sha256": "a" * 64,
                "score_contracts": {"expert1": {"semantics": "clap_similarity"}},
                "expert_order": ["expert1"]}
    audit = {"diagnostics": {"sample_count": 2, "outside_minus1_to1_fraction": 0.5}}
    with pytest.raises(ValueError, match="review"):
        _real_score_review(None, manifest, audit)
    path = tmp_path / "review.json"
    path.write_text('{"schema_version":1,"reviewer":"researcher","evidence":"audited source",'
                    '"train_store_identity_sha256":"wrong","score_contract_sha256":"wrong",'
                    '"score_audit_sha256":"wrong","experts":{}}')
    with pytest.raises(ValueError, match="review"):
        _real_score_review(path, manifest, audit)
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    accepted = {"schema_version": 1, "reviewer": "researcher", "evidence": "train audit and code",
                "train_store_identity_sha256": manifest["store_identity_sha256"],
                "score_contract_sha256": canonical(manifest["score_contracts"]),
                "score_audit_sha256": canonical(audit["diagnostics"]),
                "experts": {"expert1": "REVIEWED_FOR_GATE_PROXY"}}
    path.write_text(json.dumps(accepted))
    assert _real_score_review(path, manifest, audit) == hashlib.sha256(path.read_bytes()).hexdigest()
