import hashlib
import json

import pytest

from scripts import train_moe_gating


def _review_fixture():
    manifest = {
        "store_identity_sha256": "a" * 64,
        "score_contracts": {"expert1": {"scale": "raw_0_to_10"}},
        "expert_order": ["expert1", "expert2"],
        "assembly_identity": {"synthetic_provenance": False},
    }
    audit = {"diagnostics": {"sample_count": 3}}
    return manifest, audit


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def test_experimental_skip_runs_train_only_audit_without_fake_review(tmp_path, monkeypatch):
    manifest, audit = _review_fixture()
    calls = []
    monkeypatch.setattr(train_moe_gating, "audit_store", lambda root: calls.append(root) or audit)
    monkeypatch.setattr(train_moe_gating, "_real_score_review", lambda *args: pytest.fail("review must be skipped"))
    identity = {"train": "b" * 64, "validation": "c" * 64}

    result = train_moe_gating._resolve_full7_score_review(
        tmp_path, manifest, identity, None, skip_review=True,
    )

    assert calls == [tmp_path]
    assert result[0] is audit
    assert result[1:] == (None, "EXPERIMENTAL_SKIPPED", False)
    assert identity["score_review_policy"] == "EXPERIMENTAL_SKIPPED"
    assert "score_review_sha256" not in identity
    assert "reviewer" not in identity and "evidence" not in identity


def test_strict_policy_requires_verified_review_and_binds_its_sha256(tmp_path, monkeypatch):
    manifest, audit = _review_fixture()
    monkeypatch.setattr(train_moe_gating, "audit_store", lambda root: audit)
    identity = {}
    with pytest.raises(ValueError, match="requires a score review file"):
        train_moe_gating._resolve_full7_score_review(
            tmp_path, manifest, identity, None, skip_review=False,
        )
    assert "score_review_policy" not in identity

    review = {
        "schema_version": 1,
        "train_store_identity_sha256": manifest["store_identity_sha256"],
        "score_contract_sha256": _fingerprint(manifest["score_contracts"]),
        "score_audit_sha256": _fingerprint(audit["diagnostics"]),
        "reviewer": "Researcher",
        "evidence": "Reviewed train-only score audit",
        "experts": {name: "REVIEWED_FOR_GATE_PROXY" for name in manifest["expert_order"]},
    }
    review_path = tmp_path / "score_review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(train_moe_gating, "audit_store", lambda root: audit)
    resolved = train_moe_gating._resolve_full7_score_review(
        tmp_path, manifest, identity, review_path, skip_review=False,
    )
    assert resolved[1:] == (hashlib.sha256(review_path.read_bytes()).hexdigest(), "HUMAN_REVIEWED", True)
    assert identity["score_review_policy"] == "HUMAN_REVIEWED"
    assert identity["score_review_sha256"] == resolved[1]


def test_existing_gate_artifacts_cannot_be_reused_without_explicit_resume(tmp_path):
    (tmp_path / "best_gate.pt").write_bytes(b"old checkpoint")
    with pytest.raises(FileExistsError, match="fresh output directory or an explicit compatible resume"):
        train_moe_gating._reject_accidental_checkpoint_overwrite(tmp_path, resume=None)


def test_synthetic_full7_has_explicit_nonhuman_policy(tmp_path, monkeypatch):
    manifest, audit = _review_fixture()
    manifest["assembly_identity"]["synthetic_provenance"] = True
    monkeypatch.setattr(train_moe_gating, "audit_store", lambda root: audit)
    identity = {}
    result = train_moe_gating._resolve_full7_score_review(
        tmp_path, manifest, identity, None, skip_review=True,
    )
    assert result[1:] == (None, "SYNTHETIC_NOT_APPLICABLE", False)
    assert identity["score_review_policy"] == "SYNTHETIC_NOT_APPLICABLE"

