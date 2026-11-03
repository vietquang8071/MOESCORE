"""Training boundary for the trainable gate over detached expert outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

from loss.loss_function import get_loss_function
from model.moe_gating_core import GatingNetwork
from model.expert_adapters.registry import EXPECTED_FEATURE_WIDTH
from training.target_transform import normalize_targets


def diversity_loss(weights: torch.Tensor) -> torch.Tensor:
    """Source-equivalent entropy diversity penalty used by Train_MOE.py."""
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2 or weights.shape[1] != 4:
        raise ValueError("weights must have shape [B,4]")
    if not torch.isfinite(weights).all().item():
        raise ValueError("weights must be finite")
    num_experts = weights.shape[-1]
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
    normalizer = torch.log(torch.tensor(num_experts, dtype=torch.float, device=weights.device))
    return torch.mean(1 - entropy / normalizer)


def _uses_class_balancing(config: Mapping[str, Any]) -> bool:
    return bool(config["clipped_mse"].get("cbl", False) or config["contrastive"].get("cbl", False))


class MoeGateTrainer:
    """One-step trainer whose optimizer owns gating parameters only."""

    def __init__(
        self,
        gating_network: GatingNetwork,
        loss_config: Mapping[str, Any],
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        diversity_weight: float = 0.2,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        if not isinstance(gating_network, GatingNetwork):
            raise TypeError("gating_network must be the verified GatingNetwork")
        if gating_network.input_feature_dim != EXPECTED_FEATURE_WIDTH:
            raise ValueError("gating network feature width must match the expert registry")
        if not isinstance(loss_config, Mapping):
            raise TypeError("loss_config must be a mapping accepted by the original loss factory")
        weights = loss_config.get("loss_weights")
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError("original regression loss configuration requires two loss weights")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in weights):
            raise ValueError("loss weights must be numeric")
        for section in ("clipped_mse", "contrastive"):
            if section not in loss_config or not isinstance(loss_config[section], Mapping):
                raise ValueError(f"loss_config.{section} must be a mapping")
        self.gating_network = gating_network
        self.loss_config = dict(loss_config)
        self.loss_function = get_loss_function(self.loss_config)
        self.uses_class_balancing = _uses_class_balancing(self.loss_config)
        if not isinstance(diversity_weight, (int, float)) or not torch.isfinite(torch.tensor(float(diversity_weight))).item():
            raise ValueError("diversity_weight must be finite")
        self.diversity_weight = float(diversity_weight)
        parameters = [parameter for parameter in gating_network.parameters() if parameter.requires_grad]
        if not parameters or len(parameters) != sum(1 for _ in gating_network.parameters()):
            raise ValueError("all and only gating parameters must be trainable")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        self._validate_optimizer_parameters(parameters)
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.global_step = 0

    def _validate_optimizer_parameters(self, gate_parameters: list[nn.Parameter]) -> None:
        optimizer_parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(optimizer_parameters) != len(gate_parameters) or {
            id(parameter) for parameter in optimizer_parameters
        } != {id(parameter) for parameter in gate_parameters}:
            raise ValueError("optimizer must contain every gating parameter and no other parameters")

    def _batch_tensors(self, batch: Mapping[str, Any], *, expected_split: str) -> tuple[torch.Tensor, ...]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        if batch.get("split") != expected_split:
            split_label = "training" if expected_split == "train" else expected_split
            raise ValueError(f"this operation requires the {split_label} split")
        if batch.get("labels_scale") != "raw_0_to_10":
            raise ValueError("training targets must be explicitly marked as raw_0_to_10")
        required = {"scores", "features", "labels_raw", "labels_scale"}
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"batch is missing required field(s): {missing}")
        parameter = next(self.gating_network.parameters())
        scores = batch["scores"]
        features = batch["features"]
        labels = batch["labels_raw"]
        if not all(isinstance(value, torch.Tensor) for value in (scores, features, labels)):
            raise TypeError("scores, features, and labels_raw must be tensors")
        batch_size = scores.shape[0] if scores.ndim else 0
        if scores.shape != (batch_size, 4) or features.shape != (batch_size, EXPECTED_FEATURE_WIDTH) or labels.shape != (batch_size,):
            raise ValueError("batch tensors must have shapes [B,4], [B,7936], and [B]")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if scores.dtype != torch.float32 or features.dtype != torch.float32 or labels.dtype != torch.float32:
            raise TypeError("feature-store scores, features, and labels must be float32")
        if not torch.isfinite(scores).all().item() or not torch.isfinite(features).all().item():
            raise ValueError("expert scores and features must be finite")
        if self.uses_class_balancing:
            classes = batch.get("num_class")
            if not isinstance(classes, torch.Tensor) or classes.shape != (batch_size,) or classes.dtype != torch.int64:
                raise ValueError("class balancing requires explicit int64 num_class metadata for every training sample")
            if torch.any(classes <= 0).item():
                raise ValueError("num_class values must be positive")
        else:
            classes = None
        scores = scores.detach().to(device=parameter.device, dtype=parameter.dtype)
        features = features.detach().to(device=parameter.device, dtype=parameter.dtype)
        labels = labels.detach().to(device=parameter.device, dtype=torch.float32)
        classes = classes.detach().to(device=parameter.device) if classes is not None else None
        return scores, features, labels, classes

    def _losses(self, scores: torch.Tensor, features: torch.Tensor, labels_raw: torch.Tensor, classes: Optional[torch.Tensor]) -> dict[str, torch.Tensor]:
        targets = normalize_targets(labels_raw)
        weights = self.gating_network(features)
        prediction = (
            scores[:, 0:1] * weights[:, 0:1]
            + scores[:, 1:2] * weights[:, 1:2]
            + scores[:, 2:3] * weights[:, 2:3]
            + scores[:, 3:4] * weights[:, 3:4]
        )
        # Original ClippedMSE/ContrastiveLoss compare per-utterance vectors.
        # Flattening here prevents [B,1] x [B] broadcasting into [B,B].
        prediction_for_loss = prediction.reshape(-1)
        if self.uses_class_balancing:
            raise ValueError(
                "class balancing is unavailable with the current original loss implementation: "
                "ClippedMSELoss expands per-sample weights to [B,1] and broadcasts against [B], "
                "changing its reduction; leave cbl=false until that source behavior is reviewed"
            )
        loss_modules = self.loss_function.loss_instances
        clipped = loss_modules[0](prediction_for_loss, targets, classes)
        contrastive = loss_modules[1](prediction_for_loss, targets, classes)
        if clipped.ndim != 0 or contrastive.ndim != 0:
            raise ValueError(
                "the original configured loss component is not scalar; "
                "the source ClippedMSE class-balancing path broadcasts per-sample weights, "
                "so this configuration is rejected rather than silently changing its mathematics"
            )
        configured_weights = self.loss_function.loss_weights
        regression = configured_weights[0] * clipped + configured_weights[1] * contrastive
        diversity = diversity_loss(weights)
        total = regression + self.diversity_weight * diversity
        if total.ndim != 0 or not torch.isfinite(total).item():
            raise ValueError("combined MOESCORE objective must be a finite scalar")
        return {
            "prediction": prediction,
            "weights": weights,
            "targets_normalized": targets,
            "clipped_mse_loss": clipped,
            "contrastive_loss": contrastive,
            "regression_loss": regression,
            "diversity_loss": diversity,
            "total_loss": total,
        }

    @staticmethod
    def _detached(result: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in result.items()}

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        scores, features, labels, classes = self._batch_tensors(batch, expected_split="train")
        self.gating_network.train()
        before = [parameter.detach().clone() for parameter in self.gating_network.parameters()]
        self.optimizer.zero_grad(set_to_none=True)
        result = self._losses(scores, features, labels, classes)
        result["total_loss"].backward()
        gradients = [parameter.grad for parameter in self.gating_network.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("gating gradients must exist and be finite")
        gradient_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(gradient.detach()) for gradient in gradients]))
        self.optimizer.step()
        self.global_step += 1
        updated = sum(
            not torch.equal(old, parameter.detach())
            for old, parameter in zip(before, self.gating_network.parameters())
        )
        output = self._detached(result)
        output["gradient_norm"] = gradient_norm.detach().clone()
        output["updated_parameter_count"] = int(updated)
        output["global_step"] = self.global_step
        return output

    def validation_step(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        scores, features, labels, classes = self._batch_tensors(batch, expected_split="validation")
        self.gating_network.eval()
        with torch.no_grad():
            return self._detached(self._losses(scores, features, labels, classes))

    def scheduler_step(self, validation_loss: float) -> None:
        value = float(validation_loss)
        if not math_is_finite(value):
            raise ValueError("scheduler validation loss must be finite")
        self.scheduler.step(value)


def math_is_finite(value: float) -> bool:
    return bool(torch.isfinite(torch.tensor(value, dtype=torch.float64)).item())


__all__ = ["MoeGateTrainer", "diversity_loss"]
