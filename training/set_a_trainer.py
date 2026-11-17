"""CPU/CUDA training and checkpoint handling for the Set A residual head."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_DIM,
    feature_order_contract,
)
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter


CHECKPOINT_FORMAT_VERSION = 1
SCORE_NORMALIZATION = {
    "target_scale": "normalized_0_to_1_about_5",
    "formula": "(target_raw - 5.0) / 5.0",
    "raw_min": 0.0,
    "raw_max": 10.0,
}


def set_deterministic_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(value: str | torch.device) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable")
    return device


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def _component_cache_ids(dataset: SetACachedDataset) -> dict[str, str]:
    return {
        name: dataset.component_cache_identities[name]["cache_id"]
        for name in SET_A_COMPONENT_ORDER
    }


def _checkpoint_payload(
    *,
    model: SetAResidualAdapter,
    optimizer: torch.optim.Optimizer,
    dataset: SetACachedDataset,
    epoch: int,
    global_step: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture_config": model.head.architecture_config,
        "feature_contract": feature_order_contract(),
        "score_normalization": dict(SCORE_NORMALIZATION),
        "epoch": epoch,
        "global_step": global_step,
        "component_model_identities": dataset.component_model_identities,
        "component_cache_identities": _component_cache_ids(dataset),
        "data_provenance": dataset.provenance,
        "seed": seed,
    }


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary_name)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def load_set_a_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: SetAResidualAdapter,
    optimizer: Optional[torch.optim.Optimizer] = None,
    expected_dataset: Optional[SetACachedDataset] = None,
    expected_data_provenance: Optional[Mapping[str, Any]] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore an E5 head after checking architecture, contract and provenance."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Set A checkpoint does not exist: {checkpoint_path}")
    if not isinstance(model, SetAResidualAdapter):
        raise TypeError("model must be a SetAResidualAdapter")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Set A checkpoint cannot be read: {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or incomplete Set A checkpoint format")
    contract = checkpoint.get("feature_contract")
    expected_contract = feature_order_contract()
    if not isinstance(contract, Mapping) or contract.get("feature_width") != SET_A_FEATURE_DIM:
        raise ValueError(f"Set A checkpoint feature width must be {SET_A_FEATURE_DIM}")
    if dict(contract) != expected_contract:
        raise ValueError("Set A checkpoint feature ordering/representation mismatch")
    architecture = checkpoint.get("architecture_config")
    if not isinstance(architecture, Mapping) or dict(architecture) != model.head.architecture_config:
        raise ValueError("Set A checkpoint architecture configuration mismatch")
    if checkpoint.get("score_normalization") != SCORE_NORMALIZATION:
        raise ValueError("Set A checkpoint score normalization mismatch")

    saved_model_identities = checkpoint.get("component_model_identities")
    saved_cache_identities = checkpoint.get("component_cache_identities")
    if (
        not isinstance(saved_model_identities, Mapping)
        or set(saved_model_identities) != set(SET_A_COMPONENT_ORDER)
        or not isinstance(saved_cache_identities, Mapping)
        or set(saved_cache_identities) != set(SET_A_COMPONENT_ORDER)
    ):
        raise ValueError("Set A checkpoint component provenance is missing or incomplete")
    if expected_dataset is not None:
        expected_models = expected_dataset.component_model_identities
        expected_caches = _component_cache_ids(expected_dataset)
        if dict(saved_model_identities) != expected_models:
            raise ValueError("Set A checkpoint component model provenance mismatch")
        if dict(saved_cache_identities) != expected_caches:
            raise ValueError("Set A checkpoint component cache provenance mismatch")
    if expected_data_provenance is not None and checkpoint.get("data_provenance") != dict(expected_data_provenance):
        raise ValueError("Set A checkpoint training data provenance mismatch")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Set A checkpoint model state dictionary is missing")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"Set A checkpoint architecture/state mismatch: {exc}") from exc
    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("Set A checkpoint optimizer state is missing for resume")
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            raise ValueError(f"Set A checkpoint optimizer state mismatch: {exc}") from exc
    return dict(checkpoint)


class SetAResidualTrainer:
    """AdamW/MSE trainer that accepts only the official train feature split."""

    def __init__(
        self,
        model: SetAResidualAdapter,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
        seed: int = 42,
        device: str | torch.device = "auto",
    ) -> None:
        if not isinstance(model, SetAResidualAdapter):
            raise TypeError("model must be a SetAResidualAdapter")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if gradient_clip is not None and (
            isinstance(gradient_clip, bool)
            or not isinstance(gradient_clip, (int, float))
            or gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be positive or None")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        self.seed = seed
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("Set A residual model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.global_step = 0
        self.completed_epochs = 0

    def fit(
        self,
        train_dataset: SetACachedDataset,
        *,
        epochs: int,
        batch_size: int,
        output_dir: str | os.PathLike[str],
        num_workers: int = 0,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> list[dict[str, float]]:
        if not isinstance(train_dataset, SetACachedDataset) or train_dataset.split != "train":
            raise ValueError("SetAResidualTrainer accepts only a SetA train split")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not len(train_dataset):
            raise ValueError("Set A training dataset is empty")

        set_deterministic_seed(self.seed)
        start_epoch = 0
        if resume_from is not None:
            checkpoint = load_set_a_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                expected_dataset=train_dataset,
                expected_data_provenance=train_dataset.provenance,
                map_location=self.device,
            )
            start_epoch = int(checkpoint["epoch"])
            self.completed_epochs = start_epoch
            self.global_step = int(checkpoint["global_step"])
            checkpoint_seed = checkpoint.get("seed")
            if checkpoint_seed != self.seed:
                raise ValueError(
                    f"Set A checkpoint seed mismatch: {checkpoint_seed} != configured {self.seed}"
                )
        if start_epoch >= epochs:
            raise ValueError(
                f"resume checkpoint already completed {start_epoch} epochs; requested total is {epochs}"
            )

        target_dir = Path(output_dir).expanduser().resolve()
        latest_path = target_dir / "expert5_latest.pt"
        final_path = target_dir / "expert5_final.pt"
        history: list[dict[str, float]] = []
        for epoch_index in range(start_epoch, epochs):
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch_index)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                generator=generator,
            )
            self.model.train()
            total_squared_error = 0.0
            total_examples = 0
            for raw_batch in loader:
                batch = _move_tensors(raw_batch, self.device)
                target = batch.get("target_normalized")
                if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
                    raise ValueError("Set A train batch target_normalized must be a float32 tensor")
                if target.ndim != 1 or not torch.isfinite(target).all().item():
                    raise ValueError("Set A train target must be finite with shape [B]")
                prediction, _ = self.model({"component_outputs": batch["component_outputs"]})
                target = target.unsqueeze(1)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss).item():
                    raise ValueError("Set A training MSE became non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                count = int(target.shape[0])
                total_squared_error += float(loss.detach()) * count
                total_examples += count
                self.global_step += 1

            if total_examples == 0:
                raise ValueError("Set A train loader produced no examples")
            self.completed_epochs = epoch_index + 1
            epoch_mse = total_squared_error / total_examples
            history.append({"epoch": float(self.completed_epochs), "mse": epoch_mse})
            payload = _checkpoint_payload(
                model=self.model,
                optimizer=self.optimizer,
                dataset=train_dataset,
                epoch=self.completed_epochs,
                global_step=self.global_step,
                seed=self.seed,
            )
            _atomic_torch_save(payload, latest_path)

        _atomic_torch_save(payload, final_path)
        return history


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "SCORE_NORMALIZATION",
    "SetAResidualTrainer",
    "load_set_a_checkpoint",
    "set_deterministic_seed",
]
