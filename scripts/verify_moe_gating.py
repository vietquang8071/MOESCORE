#!/usr/bin/env python3
"""Run synthetic, CPU-only checks of the LOCAL-10 gating architecture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import (
    EXPECTED_FEATURE_WIDTH,
    EXPERT_FEATURE_DIMS,
    EXPERT_ORDER,
    validate_and_assemble_expert_outputs,
)
from model.moe_forward_integration import MoEForwardIntegration
from model.moe_gating_core import GATING_DROPOUT, GATING_HIDDEN_DIM, GatingNetwork


class _SyntheticExpert(ExpertAdapter):
    """Small deterministic adapter used only by this diagnostic process."""

    def __init__(self, name: str) -> None:
        super().__init__(EXPERT_FEATURE_DIMS[name])
        self.name = name
        self.audio_key = f"{name}_wavs"
        self.offset = float(EXPERT_ORDER.index(name)) / 10.0
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.5)
        self.forward_calls = 0
        self.grad_enabled = []

    def _forward(self, batch: dict[str, Any]):
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        signal = self.dropout(batch[self.audio_key].flatten(1).mean(dim=1))
        score = (signal * self.scale + self.offset).unsqueeze(1)
        feature_offsets = torch.arange(
            self.feature_dim, dtype=signal.dtype, device=signal.device
        ).unsqueeze(0)
        features = feature_offsets + signal.unsqueeze(1) * self.scale
        return score, features


def _controlled_sample_dependence(device: torch.device) -> dict[str, Any]:
    gate = GatingNetwork().to(device).eval()
    with torch.no_grad():
        first = gate.net[0]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        first.weight[1, 0] = -1.0
        gate.net[1].weight.fill_(1.0)
        gate.net[1].bias.zero_()
        final = gate.net[4]
        final.weight.zero_()
        final.bias.zero_()
        final.weight[0, 0] = 1.0
        final.weight[1, 1] = 1.0

    features = torch.zeros(2, EXPECTED_FEATURE_WIDTH, device=device)
    features[0, 0] = 1.0
    features[1, 0] = -1.0
    weights = gate(features)
    return {
        "status": "PASS" if not torch.allclose(weights[0], weights[1]) else "FAIL",
        "weights": weights.detach().cpu().tolist(),
        "distinct": not torch.allclose(weights[0], weights[1]),
    }


def _batch_independence(device: torch.device) -> dict[str, Any]:
    gate = GatingNetwork().to(device).eval()
    sample_a = torch.zeros(1, EXPECTED_FEATURE_WIDTH, device=device)
    sample_a[:, 0] = 1.0
    sample_b = torch.zeros(1, EXPECTED_FEATURE_WIDTH, device=device)
    sample_b[:, 0] = -1.0
    alone = gate(sample_a)
    together = gate(torch.cat((sample_a, sample_b), dim=0))[0:1]
    reordered = gate(torch.cat((sample_b, sample_a), dim=0))[1:2]
    passed = torch.allclose(alone, together, atol=1e-6, rtol=1e-5) and torch.allclose(
        alone, reordered, atol=1e-6, rtol=1e-5
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "alone": alone.detach().cpu().tolist(),
        "batched": together.detach().cpu().tolist(),
        "reordered": reordered.detach().cpu().tolist(),
    }


def _synthetic_forward(device: torch.device, batch_size: int) -> dict[str, Any]:
    experts = {name: _SyntheticExpert(name) for name in EXPERT_ORDER}
    model = MoEForwardIntegration(experts).to(device)
    model.eval()
    batch: dict[str, Any] = {
        f"{name}_wavs": torch.arange(
            batch_size, dtype=torch.float32, device=device
        ).reshape(batch_size, 1, 1)
        + EXPERT_ORDER.index(name)
        for name in EXPERT_ORDER
    }
    batch["captions_raw"] = [f"synthetic caption {i}" for i in range(batch_size)]
    batch["labels"] = torch.zeros(batch_size, device=device)
    batch["model_ids"] = ["diagnostic-only"] * batch_size

    scores_before: dict[str, torch.Tensor] = {}
    features_before: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in EXPERT_ORDER:
            scores_before[name], features_before[name] = experts[name](batch)
    # The extra calls above are solely an independent deterministic reference;
    # reset counters so the wrapper's one-call-per-expert contract is measured.
    for expert in experts.values():
        expert.forward_calls = 0
        expert.grad_enabled.clear()

    scores, features = validate_and_assemble_expert_outputs(
        {name: (scores_before[name], features_before[name]) for name in EXPERT_ORDER},
        batch_size=batch_size,
    )
    original_scores = scores.clone()
    original_features = features.clone()
    final_score, weights = model(batch)
    reference = (
        scores[:, 0:1] * weights[:, 0:1]
        + scores[:, 1:2] * weights[:, 1:2]
        + scores[:, 2:3] * weights[:, 2:3]
        + scores[:, 3:4] * weights[:, 3:4]
    )
    absolute_error = (final_score - reference).abs()
    parity = torch.allclose(final_score, reference, atol=1e-6, rtol=1e-5)
    convex = bool(
        ((final_score >= scores.min(dim=1, keepdim=True).values - 1e-6)
         & (final_score <= scores.max(dim=1, keepdim=True).values + 1e-6)).all().item()
    )
    weight_sum_error = (
        weights.sum(dim=1) - torch.ones(batch_size, device=device, dtype=weights.dtype)
    ).abs()
    weight_properties = bool(
        torch.isfinite(weights).all().item()
        and (weights >= 0).all().item()
        and weight_sum_error.max().item() <= 2e-7
    )
    immutable = torch.equal(scores, original_scores) and torch.equal(
        features, original_features
    )
    single_pass = all(expert.forward_calls == 1 for expert in experts.values())
    no_expert_grad = all(
        not parameter.requires_grad and parameter.grad is None
        for expert in experts.values()
        for parameter in expert.parameters()
    )
    result = {
        "status": "PASS" if all((parity, convex, immutable, single_pass, no_expert_grad)) else "FAIL",
        "batch_size": batch_size,
        "scores_shape": list(scores.shape),
        "features_shape": list(features.shape),
        "prediction_shape": list(final_score.shape),
        "weights_shape": list(weights.shape),
        "weights": weights.detach().cpu().tolist(),
        "predictions": final_score.detach().cpu().flatten().tolist(),
        "weighted_sum_parity": parity,
        "weighted_sum_max_abs_difference": float(absolute_error.max().item()),
        "weighted_sum_mean_abs_difference": float(absolute_error.mean().item()),
        "convex_combination": convex,
        "softmax_weight_properties": weight_properties,
        "weight_sum_max_abs_error": float(weight_sum_error.max().item()),
        "inputs_unchanged": immutable,
        "one_adapter_forward_each": single_pass,
        "experts_frozen_and_without_gradients": no_expert_grad,
    }

    model.train()
    result["train_mode_contract"] = {
        "gate_trainable": all(
            parameter.requires_grad for parameter in model.gating_network.parameters()
        ),
        "gate_training": model.gating_network.training,
        "gate_dropout_training": model.gating_network.net[3].training,
        "all_experts_eval": all(not expert.training for expert in experts.values()),
        "all_experts_frozen": all(
            not parameter.requires_grad
            for expert in experts.values()
            for parameter in expert.parameters()
        ),
    }
    optimizer = torch.optim.SGD(model.gating_network.parameters(), lr=0.01)
    optimizer.zero_grad()
    train_prediction, train_weights = model(batch)
    synthetic_loss = train_prediction.square().mean() + train_weights[:, 0].mean()
    synthetic_loss.backward()
    gate_gradients = [
        parameter.grad
        for parameter in model.gating_network.parameters()
        if parameter.grad is not None
    ]
    gradients_valid = bool(gate_gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gate_gradients
    )
    nonzero_gradient = any(
        torch.count_nonzero(gradient).item() > 0 for gradient in gate_gradients
    )
    experts_gradient_free = all(
        parameter.grad is None
        for expert in experts.values()
        for parameter in expert.parameters()
    )
    result["synthetic_backward"] = {
        "status": "PASS" if gradients_valid and nonzero_gradient and experts_gradient_free else "FAIL",
        "finite_gate_gradients": gradients_valid,
        "nonzero_gate_gradient": nonzero_gradient,
        "experts_have_no_gradients": experts_gradient_free,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()
    try:
        args.batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    except ValueError as exc:
        parser.error(f"--batch-sizes must be comma-separated positive integers: {exc}")
    if not args.batch_sizes or any(value <= 0 for value in args.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes cannot contain duplicates")
    return args


def main() -> int:
    args = _parse_args()
    if args.offline:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
    torch.manual_seed(args.seed)
    report: dict[str, Any] = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "synthetic_only",
        "seed": args.seed,
        "device": args.device,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "offline_requested": args.offline,
            "precision": "float32",
        },
        "expert_order": list(EXPERT_ORDER),
        "expert_feature_dims": dict(EXPERT_FEATURE_DIMS),
        "gate_input_dim": EXPECTED_FEATURE_WIDTH,
        "expert_count": len(EXPERT_ORDER),
        "gating_architecture": {
            "layers": ["Linear", "LayerNorm", "ReLU(inplace=True)", "Dropout", "Linear", "Softmax(dim=-1)"],
            "hidden_dim": GATING_HIDDEN_DIM,
            "dropout": GATING_DROPOUT,
            "state_dict_input_weight_shape": [GATING_HIDDEN_DIM, EXPECTED_FEATURE_WIDTH],
            "state_dict_output_weight_shape": [len(EXPERT_ORDER), GATING_HIDDEN_DIM],
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "errors": [],
        # No pretrained experts are constructed. The diagnostic does create
        # the gate and four tiny deterministic test adapters below.
        "model_construction": False,
        "pretrained_model_construction": False,
        "synthetic_adapter_stubs": True,
        "checkpoint_deserialization": False,
        "real_expert_inference": "NOT_VERIFIED_LOCAL_10_1",
        "real_score_distribution": "NOT_MEASURED",
        "full_moe_training": "NOT_IMPLEMENTED",
    }
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {device} is unavailable")
        report["forward_cases"] = [
            _synthetic_forward(device, batch_size)
            for batch_size in args.batch_sizes
        ]
        report["sample_dependence"] = _controlled_sample_dependence(device)
        report["batch_independence"] = _batch_independence(device)
        report["score_shape"] = {
            str(case["batch_size"]): case["scores_shape"]
            for case in report["forward_cases"]
        }
        report["feature_shape"] = {
            str(case["batch_size"]): case["features_shape"]
            for case in report["forward_cases"]
        }
        report["prediction_shape"] = {
            str(case["batch_size"]): case["prediction_shape"]
            for case in report["forward_cases"]
        }
        report["weight_shape"] = {
            str(case["batch_size"]): case["weights_shape"]
            for case in report["forward_cases"]
        }
        report["softmax_normalization"] = (
            "PASS"
            if all(case["softmax_weight_properties"] for case in report["forward_cases"])
            else "FAIL"
        )
        report["forward_numerical_parity"] = (
            "PASS"
            if all(case["weighted_sum_parity"] for case in report["forward_cases"])
            else "FAIL"
        )
        report["gradient_flow"] = (
            "PASS"
            if all(case["synthetic_backward"]["status"] == "PASS" for case in report["forward_cases"])
            else "FAIL"
        )
        report["frozen_expert_contract"] = (
            "PASS"
            if all(case["experts_frozen_and_without_gradients"] for case in report["forward_cases"])
            else "FAIL"
        )
        report["train_eval_behavior"] = {
            "dropout_probability": GATING_DROPOUT,
            "eval_repeat_deterministic": report["batch_independence"]["status"] == "PASS",
            "gate_trainable": all(
                case["train_mode_contract"]["gate_trainable"]
                for case in report["forward_cases"]
            ),
            "gate_enters_train_mode": all(
                case["train_mode_contract"]["gate_training"]
                for case in report["forward_cases"]
            ),
            "gate_dropout_enabled_in_train": all(
                case["train_mode_contract"]["gate_dropout_training"]
                for case in report["forward_cases"]
            ),
            "experts_frozen_by_integration": all(
                case["train_mode_contract"]["all_experts_eval"]
                and case["train_mode_contract"]["all_experts_frozen"]
                for case in report["forward_cases"]
            ),
        }
        state_source = GatingNetwork().to(device).eval()
        state_target = GatingNetwork().to(device).eval()
        state_target.load_state_dict(state_source.state_dict(), strict=True)
        probe = torch.linspace(
            -1, 1, EXPECTED_FEATURE_WIDTH, device=device
        ).reshape(1, EXPECTED_FEATURE_WIDTH)
        state_roundtrip = torch.equal(state_source(probe), state_target(probe))
        report["state_dict_roundtrip"] = "PASS" if state_roundtrip else "FAIL"
        report["score_scale_audit"] = {
            "expert1": "[-1,1] source embeddings are L2-normalized by local LAION-CLAP API; score is their dot product",
            "expert2": "[-1,1] dot product of L2-normalized MSC embeddings",
            "expert3": "unbounded linear regression-head output; no sigmoid/tanh/clamp",
            "expert4": "tanh-bounded source output, mathematically (-1,1)",
            "common_expert_score_normalization": "ABSENT_AND_NOT_ADDED",
            "weighted_sum_range": "convex hull of the four input scores per sample; not guaranteed [-1,1]",
        }
        report["label_scale_compatibility"] = {
            "unified_dataset_label_scale": "[0,10]",
            "original_loss_target_transform": "(score_raw - 5.0) / 5.0",
            "original_inference_inverse": "prediction * 5.0 + 5.0",
            "gating_core_receives_labels": False,
            "training_boundary": "DOCUMENTED_FOR_LOCAL_11; NOT_IMPLEMENTED_HERE",
        }
        report["loss_compatibility"] = {
            "clipped_mse_and_contrastive_source_unchanged": True,
            "entropy_diversity_coefficient": 0.2,
            "loss_expects_normalized_target": True,
        }
        required_checks = [
            *[case["status"] == "PASS" and case["synthetic_backward"]["status"] == "PASS" for case in report["forward_cases"]],
            report["softmax_normalization"] == "PASS",
            report["forward_numerical_parity"] == "PASS",
            report["gradient_flow"] == "PASS",
            report["frozen_expert_contract"] == "PASS",
            all(
                report["train_eval_behavior"][key]
                for key in (
                    "eval_repeat_deterministic",
                    "gate_trainable",
                    "gate_enters_train_mode",
                    "gate_dropout_enabled_in_train",
                    "experts_frozen_by_integration",
                )
            ),
            report["sample_dependence"]["status"] == "PASS",
            report["batch_independence"]["status"] == "PASS",
            report["state_dict_roundtrip"] == "PASS",
        ]
        if not all(required_checks):
            report["status"] = "FAIL"
    except Exception as exc:  # preserve a useful machine-readable failure result
        report["status"] = "FAIL"
        report["errors"].append(
            {"exception": type(exc).__name__, "message": str(exc)}
        )

    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json is not None:
        args.report_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report_json.expanduser().resolve().write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
