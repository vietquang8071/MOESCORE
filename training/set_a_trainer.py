"""Official train/validation training and checkpoint handling for E5."""

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
from torch.utils.data import DataLoader

from data.set_a_cached_dataset import SetACachedDataset
from model.composite_experts.set_a_residual import (
    SET_A_COMPONENT_ORDER,
    SET_A_FEATURE_DIM,
    feature_order_contract,
)
from model.expert_adapters.set_a_residual_adapter import SetAResidualAdapter


CHECKPOINT_FORMAT_VERSION = 3
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


def _validate_split_provenance(
    value: Any,
    *,
    split: str,
    sample_ids: list[str],
    context: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("split") != split:
        raise ValueError(f"E5 {context} provenance split is missing or invalid")
    source_hash = value.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError(f"E5 {context} source-manifest identity is missing")
    try:
        int(source_hash, 16)
    except ValueError as exc:
        raise ValueError(f"E5 {context} source-manifest identity is invalid") from exc
    if value.get("sample_count") != len(sample_ids):
        raise ValueError(f"E5 {context} sample count differs from membership")
    if value.get("ordered_sample_ids_sha256") != _canonical_hash(sample_ids):
        raise ValueError(f"E5 {context} ordered membership fingerprint mismatch")
    if not isinstance(value.get("feature_store_identity"), str) or not value["feature_store_identity"]:
        raise ValueError(f"E5 {context} feature-store identity is missing")
    if "preprocessing_identity" not in value:
        raise ValueError(f"E5 {context} preprocessing provenance is missing")


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
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid E5 checkpoint")

    version = checkpoint.get("format_version")
    if version not in (2, 3):
        raise ValueError("Unsupported E5 checkpoint format")
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

    if not isinstance(training_state, Mapping):
        raise ValueError("E5 checkpoint training state is missing")

    if version == 2:
        if not isinstance(training_state.get("holdout_split"), Mapping):
            raise ValueError("Legacy E5 holdout metadata is missing")
    else:
        if training_state.get("selection_policy") != "official_validation_mse_v1":
            raise ValueError("E5 checkpoint selection policy mismatch")
        train_ids = training_state.get("train_sample_ids")
        validation_ids = training_state.get("validation_sample_ids")
        if (
            not isinstance(train_ids, list)
            or not train_ids
            or any(not isinstance(value, str) or not value.startswith("train:") for value in train_ids)
            or len(train_ids) != len(set(train_ids))
        ):
            raise ValueError("E5 official train membership is missing or invalid")
        if (
            not isinstance(validation_ids, list)
            or not validation_ids
            or any(not isinstance(value, str) or not value.startswith("validation:") for value in validation_ids)
            or len(validation_ids) != len(set(validation_ids))
        ):
            raise ValueError("E5 official validation membership is missing or invalid")
        if set(train_ids) & set(validation_ids):
            raise ValueError("E5 train and validation membership overlap")
        train_provenance = training_state.get("train_provenance")
        validation_provenance = training_state.get("validation_provenance")
        if checkpoint.get("data_provenance") != train_provenance:
            raise ValueError("E5 official train provenance is missing or inconsistent")
        _validate_split_provenance(
            train_provenance, split="train", sample_ids=train_ids, context="train"
        )
        _validate_split_provenance(
            validation_provenance, split="validation", sample_ids=validation_ids,
            context="validation",
        )
        best_mse = training_state.get("best_validation_mse")
        if (
            isinstance(best_mse, bool)
            or not isinstance(best_mse, (int, float))
            or not math.isfinite(best_mse)
            or best_mse < 0
        ):
            raise ValueError("E5 best official validation MSE is missing or invalid")
        epoch = checkpoint.get("epoch")
        global_step = checkpoint.get("global_step")
        best_epoch = training_state.get("best_epoch")
        if (
            type(epoch) is not int or epoch <= 0
            or type(global_step) is not int or global_step <= 0
            or training_state.get("epoch") != epoch
            or training_state.get("global_step") != global_step
            or type(best_epoch) is not int or not 1 <= best_epoch <= epoch
        ):
            raise ValueError("E5 checkpoint optimizer/training progress is invalid")
        metric_history = training_state.get("metric_history")
        metric_keys = {"epoch", "train_mse", "validation_mse", "best_validation_mse", "learning_rate"}
        if (
            not isinstance(metric_history, list)
            or len(metric_history) != epoch
            or any(not isinstance(row, Mapping) or set(row) != metric_keys for row in metric_history)
        ):
            raise ValueError("E5 checkpoint per-epoch metric history is missing or invalid")

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
    """AdamW/MSE trainer using all official train rows and official validation."""

    def __init__(
        self,
        model: SetAResidualAdapter,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: Optional[float] = 1.0,
        seed: int = 42,
        device: str | torch.device = "auto",
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
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = float(min_delta)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("Set A residual model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.global_step = 0
        self.completed_epochs = 0
        self.best_validation_mse = math.inf
        self.best_epoch = 0
        self.epochs_since_improvement = 0

    def _evaluate_mse(self, dataset: SetACachedDataset, *, batch_size: int, num_workers: int) -> float:
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
        validation_dataset: SetACachedDataset,
        epochs: int,
        batch_size: int,
        output_dir: str | os.PathLike[str],
        num_workers: int = 0,
        resume_from: str | os.PathLike[str] | None = None,
    ) -> list[dict[str, float]]:
        if not isinstance(train_dataset, SetACachedDataset) or train_dataset.split != "train":
            raise ValueError("E5 requires the official train split")
        if not isinstance(validation_dataset, SetACachedDataset) or validation_dataset.split != "validation":
            raise ValueError("E5 requires the official validation split")
        if not train_dataset.require_labels:
            raise ValueError("E5 requires labeled official train samples")
        if not validation_dataset.require_labels:
            raise ValueError("E5 requires labeled official validation samples")
        if not len(train_dataset) or not len(validation_dataset):
            raise ValueError("E5 train and validation datasets must be non-empty")
        if set(train_dataset.sample_ids) & set(validation_dataset.sample_ids):
            raise ValueError("E5 train/validation sample IDs overlap")
        if train_dataset.component_model_identities != validation_dataset.component_model_identities:
            raise ValueError("E5 component model identities differ across splits")
        if _component_cache_ids(train_dataset) != _component_cache_ids(validation_dataset):
            raise ValueError("E5 component cache identities differ across splits")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be positive")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        # The complete official train dataset is the only gradient source.
        fit_dataset = train_dataset
        evaluation_dataset = validation_dataset
        selection_policy = "official_validation_mse_v1"

        set_deterministic_seed(self.seed)
        start_epoch = 0
        stale = 0
        best = math.inf
        best_epoch = 0

        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        latest_path = target_dir / "expert5_latest.pt"
        best_path = target_dir / "expert5_best.pt"
        final_path = target_dir / "expert5_final.pt"

        training_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "selection_metric": "official_validation_mse",
            "selection_policy": selection_policy,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
        }

        if resume_from is not None:
            resume_path = Path(resume_from).expanduser().resolve()
            if resume_path != latest_path:
                raise ValueError("resume requires output_dir/expert5_latest.pt")

            checkpoint = load_set_a_checkpoint(
                resume_path,
                model=self.model,
                expected_dataset=train_dataset,
                expected_data_provenance=train_dataset.provenance,
                map_location=self.device,
            )

            if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("Cannot resume a legacy E5 holdout checkpoint")
            if checkpoint.get("checkpoint_role") != "latest":
                raise ValueError("E5 resume requires a latest checkpoint")
            if checkpoint.get("seed") != self.seed:
                raise ValueError("E5 resume seed mismatch")

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("E5 resume optimizer state is missing")
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as exc:
                raise ValueError(f"E5 resume optimizer state mismatch: {exc}") from exc

            saved = checkpoint["training_state"]
            if saved.get("training_config") != training_config:
                raise ValueError("E5 resume training configuration mismatch")
            if saved.get("validation_provenance") != validation_dataset.provenance:
                raise ValueError("E5 resume validation provenance mismatch")
            if saved.get("train_sample_ids") != list(train_dataset.sample_ids):
                raise ValueError("E5 resume train membership mismatch")
            if saved.get("validation_sample_ids") != list(validation_dataset.sample_ids):
                raise ValueError("E5 resume validation membership mismatch")

            best_hash = saved.get("best_checkpoint_sha256")
            if (
                not isinstance(best_hash, str)
                or not best_path.is_file()
                or _file_sha256(best_path) != best_hash
            ):
                raise ValueError("E5 best checkpoint missing or modified")

            start_epoch = int(checkpoint["epoch"])
            self.completed_epochs = start_epoch
            self.global_step = int(checkpoint["global_step"])
            best = float(saved["best_validation_mse"])
            best_epoch = int(saved["best_epoch"])
            stale = int(saved["epochs_since_improvement"])
            history = [dict(row) for row in saved["metric_history"]]
        else:
            history = []

        if start_epoch >= epochs:
            raise ValueError("E5 checkpoint already reached requested epochs")

        for epoch_index in range(start_epoch, epochs):
            epoch_seed = self.seed + epoch_index
            set_deterministic_seed(epoch_seed)

            generator = torch.Generator().manual_seed(epoch_seed)
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
                if (
                    not isinstance(target, torch.Tensor)
                    or target.dtype != torch.float32
                    or target.ndim != 1
                    or not torch.isfinite(target).all().item()
                ):
                    raise ValueError("Invalid E5 training target")

                prediction, _ = self.model({
                    "component_outputs": batch["component_outputs"]
                })
                loss = torch.nn.functional.mse_loss(
                    prediction, target.unsqueeze(1)
                )
                if not torch.isfinite(loss).item():
                    raise ValueError("E5 training MSE is non-finite")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()
                self.global_step += 1

            self.completed_epochs = epoch_index + 1
            train_mse = self._evaluate_mse(
                fit_dataset, batch_size=batch_size, num_workers=num_workers
            )
            validation_mse = self._evaluate_mse(
                evaluation_dataset, batch_size=batch_size, num_workers=num_workers
            )

            if not math.isfinite(train_mse) or not math.isfinite(validation_mse):
                raise ValueError("E5 train/validation MSE is non-finite")

            improved = validation_mse < best - self.min_delta
            if improved:
                best = validation_mse
                best_epoch = self.completed_epochs
                stale = 0
            else:
                stale += 1

            metrics = {
                "epoch": float(self.completed_epochs),
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "best_validation_mse": best,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(metrics)

            state = {
                "epoch": self.completed_epochs,
                "global_step": self.global_step,
                "seed": self.seed,
                "selection_policy": selection_policy,
                "training_config": training_config,
                "train_provenance": train_dataset.provenance,
                "validation_provenance": validation_dataset.provenance,
                "train_sample_ids": list(train_dataset.sample_ids),
                "validation_sample_ids": list(validation_dataset.sample_ids),
                "best_validation_mse": best,
                "best_epoch": best_epoch,
                "epochs_since_improvement": stale,
                "last_metrics": metrics,
                "metric_history": [*history],
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

            self.best_validation_mse = best
            self.best_epoch = best_epoch
            self.epochs_since_improvement = stale

            if (
                self.early_stopping_patience is not None
                and stale >= self.early_stopping_patience
            ):
                break

        if not best_path.is_file():
            raise ValueError("E5 training produced no best checkpoint")

        best_payload = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        best_payload["checkpoint_role"] = "final_best"
        _atomic_torch_save(best_payload, final_path)

        return history



__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "SCORE_NORMALIZATION",
    "SetAResidualTrainer",
    "load_set_a_checkpoint",
    "set_deterministic_seed",
]
