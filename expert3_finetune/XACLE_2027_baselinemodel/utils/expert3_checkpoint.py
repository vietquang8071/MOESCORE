"""Atomic, versioned checkpoints for the partially fine-tuned Expert 3."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 1


def _trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _base_identity(model):
    return dict(getattr(model, "base_checkpoint_identity", {}))


def _write_atomic(payload, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_payload(model, checkpoint_kind, epoch, best_srcc, metrics, config):
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "model_trainable": _trainable_state(model),
        "model_config": dict(model.model_config),
        "base_checkpoint_identity": _base_identity(model),
        "epoch": int(epoch),
        "best_srcc": float(best_srcc),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
    }


def save_best_checkpoint(model, path, epoch, metrics, config=None):
    srcc = float(metrics["SRCC"])
    payload = _make_payload(model, "best", epoch, srcc, metrics, config)
    _write_atomic(payload, path)


def _load(path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(Path(path), map_location="cpu")


def _restore_trainable_state(model, payload):
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported Expert 3 checkpoint version: {payload.get('format_version')}")
    expected_base = _base_identity(model)
    saved_base = payload.get("base_checkpoint_identity", {})
    if expected_base.get("sha256") != saved_base.get("sha256"):
        raise ValueError("Fine-tuned checkpoint was created from a different M2D pretrained checkpoint")
    if dict(model.model_config) != payload.get("model_config"):
        raise ValueError("Fine-tuned checkpoint model configuration does not match this model")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = payload.get("model_trainable", {})
    if set(saved) != expected:
        missing = sorted(expected - set(saved))
        unexpected = sorted(set(saved) - expected)
        raise ValueError(f"Trainable checkpoint keys mismatch; missing={missing}, unexpected={unexpected}")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in saved.items():
            parameters[name].copy_(tensor.to(device=parameters[name].device, dtype=parameters[name].dtype))
    return payload


def load_best_checkpoint(model, path):
    payload = _load(path)
    if payload.get("checkpoint_kind") != "best":
        raise ValueError(f"Expected a best-model checkpoint, got {payload.get('checkpoint_kind')}")
    return _restore_trainable_state(model, payload)


def _rng_state(loader_generator=None):
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.copy()),
            "position": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def _restore_rng(state, loader_generator=None):
    random.setstate(tuple(state["python"]))
    np_state = state["numpy"]
    np.random.set_state((
        np_state["name"], np_state["keys"].cpu().numpy().astype(np.uint32),
        int(np_state["position"]), int(np_state["has_gauss"]), float(np_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"].cpu())


def save_resume_checkpoint(
    model, path, optimizer, scheduler, scaler, epoch, best_srcc, stale_epochs,
    config=None, metrics=None, loader_generator=None,
):
    payload = _make_payload(model, "resume", epoch, best_srcc, metrics, config)
    payload.update({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "stale_epochs": int(stale_epochs),
        "rng": _rng_state(loader_generator),
    })
    _write_atomic(payload, path)


def load_resume_checkpoint(
    model, path, optimizer, scheduler, scaler, loader_generator=None,
):
    payload = _load(path)
    if payload.get("checkpoint_kind") != "resume":
        raise ValueError(f"Expected a resume checkpoint, got {payload.get('checkpoint_kind')}")
    _restore_trainable_state(model, payload)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    _restore_rng(payload["rng"], loader_generator)
    return payload
