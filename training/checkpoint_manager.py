"""Atomic, gating-only checkpoints with strict provenance and safe loading."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import torch

from model.expert_adapters.registry import EXPECTED_FEATURE_WIDTH
from model.moe_gating_core import GatingNetwork


CHECKPOINT_FORMAT_VERSION = 1


def _canonical(value: Any, field: str) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must contain only JSON-safe finite values") from exc


def _fingerprint(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field)).hexdigest()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class MoeCheckpointManager:
    """Persist gate and training state without serializing expert modules."""

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        config: Mapping[str, Any],
        feature_store_identity: Mapping[str, str],
        expert_checkpoint_provenance: Mapping[str, Any],
        sampler_state: Optional[Mapping[str, Any]] = None,
        best_srcc: Optional[float] = None,
        patience: int = 0,
        seed: int = 2027,
    ) -> Path:
        if not isinstance(gating_network, GatingNetwork) or gating_network.input_feature_dim != EXPECTED_FEATURE_WIDTH:
            raise ValueError("checkpoint must contain the verified 7936-dimensional gating network")
        if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
            raise ValueError("epoch and global_step must be non-negative integers")
        if type(patience) is not int or patience < 0 or type(seed) is not int or seed < 0:
            raise ValueError("patience and seed must be non-negative integers")
        if best_srcc is not None and (not isinstance(best_srcc, (int, float)) or not torch.isfinite(torch.tensor(float(best_srcc))).item()):
            raise ValueError("best_srcc must be finite or None")
        config_body = json.loads(_canonical(dict(config), "config").decode("utf-8"))
        identity_body = json.loads(_canonical(dict(feature_store_identity), "feature_store_identity").decode("utf-8"))
        provenance_body = json.loads(_canonical(dict(expert_checkpoint_provenance), "expert_checkpoint_provenance").decode("utf-8"))
        if not identity_body or not provenance_body:
            raise ValueError("feature-store identity and expert checkpoint provenance are required")
        if sampler_state is not None:
            sampler_body = json.loads(_canonical(dict(sampler_state), "sampler_state").decode("utf-8"))
        else:
            sampler_body = None
        gate_state = {key: value.detach().cpu().clone() for key, value in gating_network.state_dict().items()}
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "gate_input_dim": gating_network.input_feature_dim,
            "gate_expert_count": gating_network.num_experts,
            "gating_state_dict": gate_state,
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "scheduler_state_dict": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config_body,
            "config_sha256": _fingerprint(config_body, "config"),
            "feature_store_identity": identity_body,
            "feature_store_identity_sha256": _fingerprint(identity_body, "feature_store_identity"),
            "expert_checkpoint_provenance": provenance_body,
            "expert_checkpoint_provenance_sha256": _fingerprint(provenance_body, "expert_checkpoint_provenance"),
            "sampler_state": sampler_body,
            "best_srcc": float(best_srcc) if best_srcc is not None else None,
            "patience": patience,
            "seed": seed,
            "torch_rng_state": torch.get_rng_state().cpu().clone(),
            "cuda_rng_states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
        }
        target = Path(path).expanduser().resolve()
        _atomic_torch_save(payload, target)
        return target

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        expected_config: Mapping[str, Any],
        expected_feature_store_identity: Mapping[str, str],
        expected_expert_checkpoint_provenance: Mapping[str, Any],
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"gating checkpoint is missing: {target}")
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"gating checkpoint cannot be safely loaded: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported gating checkpoint schema")
        if payload.get("gate_input_dim") != EXPECTED_FEATURE_WIDTH:
            raise ValueError("checkpoint feature width is incompatible; legacy 7168-dimensional gates are rejected")
        if payload.get("gate_expert_count") != 4:
            raise ValueError("checkpoint expert count is incompatible")
        expected_values = (
            ("config", dict(expected_config), "configuration"),
            ("feature_store_identity", dict(expected_feature_store_identity), "feature store identity"),
            ("expert_checkpoint_provenance", dict(expected_expert_checkpoint_provenance), "checkpoint provenance"),
        )
        for field, expected, label in expected_values:
            _canonical(expected, label)
            if payload.get(field) != expected:
                raise ValueError(f"gating checkpoint {label} mismatch")
        if payload.get("config_sha256") != _fingerprint(payload["config"], "config"):
            raise ValueError("checkpoint configuration fingerprint mismatch")
        if payload.get("feature_store_identity_sha256") != _fingerprint(payload["feature_store_identity"], "feature_store_identity"):
            raise ValueError("checkpoint feature-store fingerprint mismatch")
        if payload.get("expert_checkpoint_provenance_sha256") != _fingerprint(payload["expert_checkpoint_provenance"], "expert_checkpoint_provenance"):
            raise ValueError("checkpoint expert provenance fingerprint mismatch")
        state_dict = payload.get("gating_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no gating state dictionary")
        try:
            gating_network.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"gating state dictionary is incompatible: {exc}") from exc
        if not isinstance(payload.get("optimizer_state_dict"), dict):
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is None:
            if payload.get("scheduler_state_dict") is not None:
                raise ValueError("checkpoint contains a scheduler but none was configured")
        elif not isinstance(payload.get("scheduler_state_dict"), dict):
            raise ValueError("checkpoint is missing the configured scheduler state")
        else:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if restore_rng:
            rng_state = payload.get("torch_rng_state")
            if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8:
                raise ValueError("checkpoint has invalid PyTorch RNG state")
            torch.set_rng_state(rng_state.cpu())
            cuda_states = payload.get("cuda_rng_states", [])
            if cuda_states and torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.set_rng_state_all(cuda_states)
        return payload

    def load_for_inference(
        self,
        path: str | os.PathLike[str],
        *,
        gating_network: GatingNetwork,
        expected_expert_checkpoint_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load only gate weights after validating checkpoint/source provenance."""
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"gating checkpoint is missing: {target}")
        try:
            payload = torch.load(target, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"gating checkpoint cannot be safely loaded: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported gating checkpoint schema")
        if payload.get("gate_input_dim") != EXPECTED_FEATURE_WIDTH:
            raise ValueError("checkpoint feature width is incompatible; legacy 7168-dimensional gates are rejected")
        if payload.get("gate_expert_count") != 4:
            raise ValueError("checkpoint expert count is incompatible")
        expected_provenance = dict(expected_expert_checkpoint_provenance)
        _canonical(expected_provenance, "expected_expert_checkpoint_provenance")
        if payload.get("expert_checkpoint_provenance") != expected_provenance:
            raise ValueError("gating checkpoint provenance mismatch with the inference feature store")
        if payload.get("expert_checkpoint_provenance_sha256") != _fingerprint(payload["expert_checkpoint_provenance"], "expert_checkpoint_provenance"):
            raise ValueError("checkpoint expert provenance fingerprint mismatch")
        if payload.get("config_sha256") != _fingerprint(payload.get("config"), "config"):
            raise ValueError("checkpoint configuration fingerprint mismatch")
        if payload.get("feature_store_identity_sha256") != _fingerprint(payload.get("feature_store_identity"), "feature_store_identity"):
            raise ValueError("checkpoint training feature-store fingerprint mismatch")
        state_dict = payload.get("gating_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no gating state dictionary")
        try:
            gating_network.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"gating state dictionary is incompatible: {exc}") from exc
        return payload


__all__ = ["CHECKPOINT_FORMAT_VERSION", "MoeCheckpointManager"]
