"""Source-bound score contracts for the independent Full8 profile."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np

from model.expert_adapters.registry import FULL7_EXPERT_ORDER, FULL8_EXPERT_ORDER


_E8_SCORE_SEMANTICS = "m2d_clap_coral_regression_normalized"
_E8_SCORE_CONVERSION = "(sigmoid(logits.float()).sum(-1)-5.0)/5.0"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def build_expert8_score_contract(checkpoint_sha256: str, *, synthetic: bool = False) -> dict[str, str]:
    """Bind E8's normalized CORAL score semantics to its trained list checkpoint."""
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("Expert8 LoRA checkpoint SHA-256 is invalid")
    identity_body = {
        "expert": "expert8",
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "semantics": _E8_SCORE_SEMANTICS,
        "conversion": _E8_SCORE_CONVERSION,
    }
    identity = hashlib.sha256(json.dumps(
        identity_body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return {
        "source": "expert8",
        "semantics": _E8_SCORE_SEMANTICS,
        "target_scale": "normalized_minus1_to1",
        "output_scale": "normalized_minus1_to1",
        "conversion": _E8_SCORE_CONVERSION,
        "identity": identity,
        "audit_status": "SYNTHETIC_VERIFIED" if synthetic else "CODE_VERIFIED",
    }


def validate_full8_score_contracts(
    contracts: Mapping,
    *,
    synthetic: bool,
    expert8_checkpoint_sha256: str,
) -> None:
    """Validate unchanged E1-E7 semantics and the checkpoint-bound E8 score."""
    if not isinstance(contracts, Mapping) or set(contracts) != set(FULL8_EXPERT_ORDER):
        raise ValueError("Full8 requires eight explicit score contracts")
    from data.full7_score_contract import validate_score_contracts

    validate_score_contracts(
        {name: contracts[name] for name in FULL7_EXPERT_ORDER}, synthetic=synthetic
    )
    expected = build_expert8_score_contract(expert8_checkpoint_sha256, synthetic=synthetic)
    if contracts["expert8"] != expected:
        raise ValueError("Expert8 score contract or checkpoint identity is incompatible")


def validate_full8_assembly_identity(
    assembly: Mapping,
    *,
    split: str,
    source_manifest_sha256: str,
    checkpoint_provenance: Mapping,
) -> None:
    """Validate the Full7+E8 source identities embedded in a Full8 store."""
    required = {
        "schema_version", "expert_profile", "split", "source_manifest_sha256",
        "full7_store_identity_sha256", "full7_assembly_contract_sha256",
        "expert8_cache_identity", "expert8_cache_manifest_sha256",
        "ordered_source_fingerprints", "score_contracts", "shared_source_contract",
        "head_membership", "synthetic_provenance", "core_assembly_contract",
    }
    if not isinstance(assembly, Mapping) or not required.issubset(assembly):
        raise ValueError("full_8 assembly identity is incomplete")
    if (
        assembly.get("schema_version") != 1
        or assembly.get("expert_profile") != "full_8"
        or assembly.get("split") != split
        or assembly.get("source_manifest_sha256") != source_manifest_sha256
        or assembly.get("core_assembly_contract") != "full8_v1"
    ):
        raise ValueError("full_8 assembly/source split identity mismatch")
    for field in (
        "source_manifest_sha256", "full7_store_identity_sha256",
        "full7_assembly_contract_sha256", "expert8_cache_manifest_sha256",
    ):
        if not _valid_sha256(assembly.get(field)):
            raise ValueError(f"full_8 {field} is not a valid SHA-256")
    cache_identity = assembly.get("expert8_cache_identity")
    from data.moe_expert_cache import validate_expert_cache_identity
    try:
        cache_id = validate_expert_cache_identity(cache_identity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"full_8 Expert8 cache identity is invalid: {exc}") from exc
    if cache_identity.get("expert_name") != "expert8":
        raise ValueError("full_8 Expert8 cache identity names a different expert")
    cache_provenance = cache_identity.get("checkpoint_provenance")
    if not isinstance(checkpoint_provenance, Mapping) or set(checkpoint_provenance) != set(FULL8_EXPERT_ORDER):
        raise ValueError("full_8 checkpoint provenance must include all eight experts")
    e8_provenance = checkpoint_provenance["expert8"]
    if cache_provenance != e8_provenance:
        raise ValueError("full_8 Expert8 checkpoint provenance differs from its cache identity")
    base_hash = e8_provenance.get("base_checkpoint", {}).get("sha256") if isinstance(e8_provenance, Mapping) else None
    lora_hash = e8_provenance.get("finetuned_checkpoint", {}).get("sha256") if isinstance(e8_provenance, Mapping) else None
    from model.expert_adapters.expert8_m2d_clap_runtime import EXPECTED_BASE_M2D_SHA256
    if base_hash != EXPECTED_BASE_M2D_SHA256 or not _valid_sha256(lora_hash):
        raise ValueError("full_8 Expert8 M2D base or LoRA checkpoint provenance is incompatible")
    if assembly.get("expert8_cache_id") not in {None, cache_id}:
        raise ValueError("full_8 Expert8 cache ID differs from its cache identity")
    rows = assembly.get("ordered_source_fingerprints")
    if not isinstance(rows, list) or not rows:
        raise ValueError("full_8 ordered source fingerprints are missing")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("full_8 ordered source fingerprint is malformed")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.startswith(split + ":") or sample_id in seen:
            raise ValueError("full_8 source sample membership is duplicate or cross-split")
        if not _valid_sha256(row.get("wav_sha256")) or not _valid_sha256(row.get("caption_sha256")):
            raise ValueError("full_8 source WAV/caption fingerprints are invalid")
        seen.add(sample_id)
    synthetic = assembly.get("synthetic_provenance")
    if type(synthetic) is not bool:
        raise ValueError("full_8 synthetic_provenance must be boolean")
    validate_full8_score_contracts(
        assembly.get("score_contracts"), synthetic=synthetic,
        expert8_checkpoint_sha256=lora_hash,
    )
    if not isinstance(assembly.get("shared_source_contract"), Mapping) or not assembly["shared_source_contract"]:
        raise ValueError("full_8 shared source contract is missing")
    membership = assembly.get("head_membership")
    if not isinstance(membership, Mapping) or set(membership) != {"expert5", "expert6"}:
        raise ValueError("full_8 E5/E6 row membership provenance is incomplete")
    for name in ("expert5", "expert6"):
        if not isinstance(membership[name], Mapping) or len(membership[name].get("per_row", [])) != len(rows):
            raise ValueError("full_8 E5/E6 row membership provenance is incomplete")


def audit_train_scores_full8(scores: np.ndarray, labels_raw: np.ndarray) -> dict:
    """Return train-only score and raw-label diagnostics for all eight experts."""
    scores = np.asarray(scores)
    labels = np.asarray(labels_raw)
    if scores.ndim != 2 or scores.shape[1] != 8 or scores.shape[0] == 0 or scores.dtype != np.float32:
        raise ValueError("train scores must be float32 [B,8]")
    if labels.shape != (scores.shape[0],) or labels.dtype != np.float32:
        raise ValueError("train labels must be float32 [B]")
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 10)):
        raise ValueError("train labels must be finite raw 0–10 values")

    def stats(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("score audit has no finite values")
        return {
            "min": float(finite.min()), "max": float(finite.max()),
            "mean": float(finite.mean()), "std": float(finite.std()),
            "quantiles": {str(q): float(np.quantile(finite, q))
                          for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
            "non_finite_count": int(values.size - finite.size),
            "outside_minus1_to1_fraction": float(np.mean((finite < -1) | (finite > 1))),
        }

    experts = {}
    for index, name in enumerate(FULL8_EXPERT_ORDER):
        values = scores[:, index]
        entry = stats(values)
        valid = np.isfinite(values)
        raw_proxy = 5.0 * values[valid].astype(np.float64) + 5.0
        targets = labels[valid].astype(np.float64)
        correlation = (float(np.corrcoef(raw_proxy, targets)[0, 1])
                       if raw_proxy.size > 1 and np.std(raw_proxy) > 0 and np.std(targets) > 0 else None)
        if correlation is not None and not math.isfinite(correlation):
            correlation = None
        entry["train_label_diagnostics"] = {
            "raw_proxy_mse": float(np.mean((raw_proxy - targets) ** 2)),
            "raw_proxy_mae": float(np.mean(np.abs(raw_proxy - targets))),
            "pearson": correlation,
        }
        experts[name] = entry
    return {"sample_count": int(scores.shape[0]), "experts": experts,
            "labels_raw": stats(labels)}


__all__ = [
    "audit_train_scores_full8", "build_expert8_score_contract", "validate_full8_assembly_identity",
    "validate_full8_score_contracts",
]
