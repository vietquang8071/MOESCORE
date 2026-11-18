"""Train-only holdout training and checkpoint handling for the Set A head."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_DIM,
    feature_order_contract,
)
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter


CHECKPOINT_FORMAT_VERSION = 2
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
    """Resolve auto before constructing a device and reject unsupported devices."""
    if isinstance(value, str) and value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid runtime device {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported runtime device {device}; expected cpu, cuda, or auto")
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


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_train_holdout_split(
    dataset: SetACachedDataset,
    *,
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a repeatable train-only split, preferring complete source groups.

    Full9 source manifests currently do not provide group IDs, so the standard
    fallback orders sample IDs by SHA256(seed, sample_id). If every row has a
    verified group_id, whole groups are assigned with the same stable hash.
    """
    if not isinstance(dataset, SetACachedDataset) or dataset.split != "train":
        raise ValueError("internal holdout splitting accepts only a SetA train split")
    if isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, (int, float)) or not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    sample_ids = list(dataset.sample_ids)
    if len(sample_ids) < 2:
        raise ValueError("Set A train split needs at least two samples for a non-empty holdout")
    group_ids = getattr(dataset, "group_ids", None)
    if group_ids is not None and len(group_ids) != len(sample_ids):
        raise ValueError("Set A source group IDs do not align with sample IDs")

    if group_ids is None:
        strategy = "sample_id_sha256_v1"
        ordered = sorted(
            range(len(sample_ids)),
            key=lambda index: hashlib.sha256(f"{seed}\0{sample_ids[index]}".encode()).hexdigest(),
        )
        holdout_count = min(len(sample_ids) - 1, max(1, round(len(sample_ids) * holdout_fraction)))
        holdout_indices = set(ordered[:holdout_count])
    else:
        strategy = "group_id_sha256_v1"
        groups: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            groups.setdefault(group_id, []).append(index)
        if len(groups) < 2:
            raise ValueError("group-aware holdout needs at least two distinct group IDs")
        ordered_groups = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(f"{seed}\0{group_id}".encode()).hexdigest(),
        )
        holdout_group_count = min(
            len(ordered_groups) - 1,
            max(1, round(len(ordered_groups) * holdout_fraction)),
        )
        holdout_groups = set(ordered_groups[:holdout_group_count])
        holdout_indices = {
            index for group_id in holdout_groups for index in groups[group_id]
        }

    fit_indices = [index for index in range(len(sample_ids)) if index not in holdout_indices]
    holdout_indices_ordered = [index for index in range(len(sample_ids)) if index in holdout_indices]
    if not fit_indices or not holdout_indices_ordered:
        raise ValueError("Set A holdout split must have non-empty fit and holdout subsets")
    fit_sample_ids = [sample_ids[index] for index in fit_indices]
    holdout_sample_ids = [sample_ids[index] for index in holdout_indices_ordered]
    split_body = {
        "strategy": strategy,
        "seed": seed,
        "holdout_fraction": float(holdout_fraction),
        "fit_sample_ids": fit_sample_ids,
        "holdout_sample_ids": holdout_sample_ids,
        "group_ids_sha256": None if group_ids is None else _canonical_hash(group_ids),
    }
    return {
        **split_body,
        "fit_indices": fit_indices,
        "holdout_indices": holdout_indices_ordered,
        "fingerprint": _canonical_hash(split_body),
    }


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
    role: str,
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": role,
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
        "training_state": dict(training_state),
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("unsupported or incomplete Set A checkpoint format; holdout metadata is required")
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
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, Mapping) or not isinstance(training_state.get("holdout_split"), Mapping):
        raise ValueError("Set A checkpoint training/holdout metadata is missing")

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
    """AdamW/MSE trainer whose fit and holdout rows both come from train only."""

    def __init__(
        self,
        model: SetAResidualAdapter,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
        seed: int = 42,
        device: str | torch.device = "auto",
        holdout_fraction: float = 0.15,
        early_stopping_patience: int | None = 5,
        min_delta: float = 0.0,
    ) -> None:
        if not isinstance(model, SetAResidualAdapter):
            raise TypeError("model must be a SetAResidualAdapter")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if gradient_clip is not None and (
            isinstance(gradient_clip, bool) or not isinstance(gradient_clip, (int, float)) or gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be positive or None")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(holdout_fraction, bool)
            or not isinstance(holdout_fraction, (int, float))
            or not math.isfinite(holdout_fraction)
            or not 0 < holdout_fraction < 1
        ):
            raise ValueError("holdout_fraction must be strictly between 0 and 1")
        if early_stopping_patience is not None and (type(early_stopping_patience) is not int or early_stopping_patience <= 0):
            raise ValueError("early_stopping_patience must be a positive integer or None")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        self.seed = seed
        self.holdout_fraction = float(holdout_fraction)
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = float(min_delta)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("Set A residual model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.global_step = 0
        self.completed_epochs = 0
        self.best_holdout_mse = math.inf
        self.best_epoch = 0
        self.epochs_since_improvement = 0

    def _evaluate_mse(self, dataset: Subset, *, batch_size: int, num_workers: int) -> float:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model.eval()
        squared_error = 0.0
        count = 0
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move_tensors(raw_batch, self.device)
                target = batch.get("target_normalized")
                if not isinstance(target, torch.Tensor) or target.dtype != torch.float32 or target.ndim != 1:
                    raise ValueError("Set A evaluation target_normalized must be float32 with shape [B]")
                if not torch.isfinite(target).all().item():
                    raise ValueError("Set A evaluation target contains NaN or infinity")
                prediction, _ = self.model({"component_outputs": batch["component_outputs"]})
                target = target.unsqueeze(1)
                loss = torch.nn.functional.mse_loss(prediction, target, reduction="sum")
                if not torch.isfinite(loss).item():
                    raise ValueError("Set A evaluation MSE became non-finite")
                squared_error += float(loss)
                count += int(target.shape[0])
        if count == 0:
            raise ValueError("Set A evaluation subset is empty")
        return squared_error / count

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
        split = create_train_holdout_split(
            train_dataset, holdout_fraction=self.holdout_fraction, seed=self.seed
        )
        fit_dataset = Subset(train_dataset, split["fit_indices"])
        holdout_dataset = Subset(train_dataset, split["holdout_indices"])
        if set(split["fit_sample_ids"]) & set(split["holdout_sample_ids"]):
            raise ValueError("Set A fit and holdout sample IDs overlap")

        set_deterministic_seed(self.seed)
        start_epoch = 0
        stale = 0
        best = math.inf
        best_epoch = 0
        latest_path: Path | None = None
        best_path: Path | None = None
        resume_checkpoint: Mapping[str, Any] | None = None
        training_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "holdout_fraction": self.holdout_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
        }
        if resume_from is not None:
            resume_path = Path(resume_from).expanduser().resolve()
            resume_checkpoint = load_set_a_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                expected_dataset=train_dataset,
                expected_data_provenance=train_dataset.provenance,
                map_location=self.device,
            )
            if resume_checkpoint.get("checkpoint_role") != "latest":
                raise ValueError("resume requires expert5_latest.pt with optimizer and progress state")
            checkpoint_seed = resume_checkpoint.get("seed")
            if checkpoint_seed != self.seed:
                raise ValueError(f"Set A checkpoint seed mismatch: {checkpoint_seed} != configured {self.seed}")
            saved_training = resume_checkpoint["training_state"]
            if saved_training.get("training_config") != training_config:
                raise ValueError("Set A checkpoint split configuration mismatch")
            saved_split = saved_training.get("holdout_split")
            if not isinstance(saved_split, Mapping) or saved_split.get("fingerprint") != split["fingerprint"]:
                raise ValueError("Set A checkpoint internal split fingerprint mismatch")
            start_epoch = int(resume_checkpoint["epoch"])
            self.completed_epochs = start_epoch
            self.global_step = int(resume_checkpoint["global_step"])
            best = float(saved_training["best_holdout_mse"])
            best_epoch = int(saved_training["best_epoch"])
            stale = int(saved_training["epochs_since_improvement"])
            best_hash = saved_training.get("best_checkpoint_sha256")
            best_path = resume_path.parent / "expert5_best.pt"
            if not isinstance(best_hash, str) or not best_path.is_file() or _file_sha256(best_path) != best_hash:
                raise ValueError("Set A best checkpoint missing or changed; resume would lose model selection state")
            latest_path = resume_path
            if resume_path.parent != Path(output_dir).expanduser().resolve():
                raise ValueError("resume checkpoint must be in output_dir so its best checkpoint can be verified")
        if start_epoch >= epochs:
            raise ValueError(f"resume checkpoint already completed {start_epoch} epochs; requested total is {epochs}")

        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        latest_path = target_dir / "expert5_latest.pt"
        best_path = target_dir / "expert5_best.pt"
        final_path = target_dir / "expert5_final.pt"
        history: list[dict[str, float]] = []

        for epoch_index in range(start_epoch, epochs):
            epoch_seed = self.seed + epoch_index
            set_deterministic_seed(epoch_seed)
            generator = torch.Generator()
            generator.manual_seed(epoch_seed)
            loader = DataLoader(
                fit_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                generator=generator,
            )
            self.model.train()
            for raw_batch in loader:
                batch = _move_tensors(raw_batch, self.device)
                target = batch.get("target_normalized")
                if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
                    raise ValueError("Set A train batch target_normalized must be a float32 tensor")
                if target.ndim != 1 or not torch.isfinite(target).all().item():
                    raise ValueError("Set A train target must be finite with shape [B]")
                prediction, _ = self.model({"component_outputs": batch["component_outputs"]})
                loss = torch.nn.functional.mse_loss(prediction, target.unsqueeze(1))
                if not torch.isfinite(loss).item():
                    raise ValueError("Set A training MSE became non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.global_step += 1

            self.completed_epochs = epoch_index + 1
            train_mse = self._evaluate_mse(fit_dataset, batch_size=batch_size, num_workers=num_workers)
            holdout_mse = self._evaluate_mse(holdout_dataset, batch_size=batch_size, num_workers=num_workers)
            if not math.isfinite(train_mse) or not math.isfinite(holdout_mse):
                raise ValueError("Set A train or holdout MSE is non-finite")
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            improved = holdout_mse < best - self.min_delta
            if improved:
                best = holdout_mse
                best_epoch = self.completed_epochs
                stale = 0
            else:
                stale += 1
            metrics = {
                "epoch": float(self.completed_epochs),
                "train_mse": train_mse,
                "holdout_mse": holdout_mse,
                "best_holdout_mse": best,
                "learning_rate": learning_rate,
            }
            history.append(metrics)

            state = {
                "epoch": self.completed_epochs,
                "global_step": self.global_step,
                "seed": self.seed,
                "training_config": training_config,
                "holdout_split": {key: value for key, value in split.items() if key not in {"fit_indices", "holdout_indices"}},
                "fit_sample_ids": split["fit_sample_ids"],
                "holdout_sample_ids": split["holdout_sample_ids"],
                "best_holdout_mse": best,
                "best_epoch": best_epoch,
                "epochs_since_improvement": stale,
                "last_metrics": metrics,
            }
            payload = _checkpoint_payload(
                model=self.model,
                optimizer=self.optimizer,
                dataset=train_dataset,
                epoch=self.completed_epochs,
                global_step=self.global_step,
                seed=self.seed,
                role="best" if improved else "latest",
                training_state=state,
            )
            if improved:
                _atomic_torch_save(payload, best_path)
            state["best_checkpoint_sha256"] = _file_sha256(best_path)
            payload["training_state"] = state
            payload["checkpoint_role"] = "latest"
            _atomic_torch_save(payload, latest_path)
            self.best_holdout_mse = best
            self.best_epoch = best_epoch
            self.epochs_since_improvement = stale

            if self.early_stopping_patience is not None and stale >= self.early_stopping_patience:
                break

        if not best_path.is_file():
            raise ValueError("Set A training completed without producing a best checkpoint")
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_payload["checkpoint_role"] = "final_best"
        _atomic_torch_save(best_payload, final_path)
        return history


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "SCORE_NORMALIZATION",
    "SetAResidualTrainer",
    "create_train_holdout_split",
    "load_set_a_checkpoint",
    "set_deterministic_seed",
]
