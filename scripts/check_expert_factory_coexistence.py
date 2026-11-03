#!/usr/bin/env python3
"""Run a lightweight, non-constructing preflight for expert integration.

This command imports the adapter wrappers in a selected order, validates the
LOCAL-03 configuration schema and tensor output contract, and inventories
configured paths. It never invokes ``from_config`` or constructs pretrained
models. Real factory coexistence and model memory are explicitly unverified.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_order(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    order = tuple(item.strip() for item in value.split(",") if item.strip())
    if not order or len(set(order)) != len(order):
        raise argparse.ArgumentTypeError("order must contain unique expert names")
    unknown = sorted(set(order) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown expert(s): {unknown}; expected {allowed}")
    return order


def _configured_resource_inventory(config: dict[str, Any], unilm_override: str | None) -> list[dict[str, Any]]:
    from configs.moe_config import resolve_moe_paths

    config_path = Path(config.pop("__config_path__"))
    resolved = resolve_moe_paths(
        config,
        config_path,
        project_root=REPO_ROOT,
        data_root=REPO_ROOT,
        resource_root=REPO_ROOT,
        environ={},
    )
    if unilm_override:
        source = Path(unilm_override).expanduser()
        if not source.is_absolute():
            source = (REPO_ROOT / source).resolve(strict=False)
        resolved["experts"]["expert4"]["unilm_source_dir"] = source.resolve(strict=False)

    fields = {
        "expert1": {
            "pretrained_checkpoint": "file",
            "finetuned_checkpoint": "file",
        },
        "expert2": {
            "source_dir": "directory",
            "architecture_yaml": "file",
            "training_config": "file",
            "pretrained_checkpoint": "file",
            "finetuned_checkpoint": "file",
        },
        "expert3": {
            "source_dir": "directory",
            "base_checkpoint": "file",
            "finetuned_checkpoint": "file",
            "model_config": "file",
        },
        "expert4": {
            "beats_checkpoint": "file",
            "clap_checkpoint": "file",
            "unilm_source_dir": "directory",
            "finetuned_checkpoint": "file",
        },
    }
    inventory = []
    for expert_name in ("expert1", "expert2", "expert3", "expert4"):
        section = resolved["experts"][expert_name]
        for field, resource_type in fields[expert_name].items():
            path = Path(section[field])
            placeholder = "REPLACE_WITH_" in str(path) or "/path/to/" in str(path)
            valid_type = path.is_dir() if resource_type == "directory" else path.is_file()
            readable = os.access(path, os.R_OK) if valid_type else False
            if placeholder:
                status = "NOT_CONFIGURED"
            elif valid_type and readable:
                status = "PASS"
            else:
                status = "BLOCKED_MISSING_RESOURCE"
            inventory.append(
                {
                    "expert": expert_name,
                    "field": field,
                    "expected_type": resource_type,
                    "path": str(path),
                    "exists": path.exists(),
                    "readable": readable,
                    "status": status,
                }
            )
    return inventory


def _source_identity_check(source_dir: str | None) -> dict[str, Any]:
    if source_dir is None:
        return {
            "status": "NOT_VERIFIED_SOURCE_IDENTITY",
            "reason": "Pass --unilm-source-dir to compare the source against Expert 4's verified checkpoint identity.",
        }
    source = Path(source_dir).expanduser()
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve(strict=False)
    source = source.resolve(strict=False)
    if not source.is_dir():
        return {"status": "BLOCKED_MISSING_RESOURCE", "path": str(source)}
    from expert4_finetune.trainer import _source_identity

    observed = _source_identity(source)
    expected = {
        "git_commit": "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd",
        "sha256": "fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487",
    }
    return {
        "status": "PASS" if observed == expected else "FAIL_CONFIGURATION",
        "path": str(source),
        "expected": expected,
        "observed": observed,
    }


def run_preflight(
    *,
    order: tuple[str, ...],
    config_path: Path,
    unilm_source_dir: str | None = None,
) -> dict[str, Any]:
    import socket
    import torch

    from configs.moe_config import load_moe_config
    from model.expert_adapters.registry import (
        EXPERT_FEATURE_DIMS,
        EXPERT_ORDER,
        EXPECTED_FEATURE_WIDTH,
        resolve_expert_adapter_class,
        validate_and_assemble_expert_outputs,
    )

    def _deny_network(*args, **kwargs):
        raise RuntimeError("network disabled by lightweight coexistence preflight")

    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    socket.create_connection = _deny_network
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"

    checks: dict[str, Any] = {
        "registry": {"status": "PASS", "order": list(EXPERT_ORDER), "feature_dims": dict(EXPERT_FEATURE_DIMS)},
        "configuration": {},
        "adapter_imports": [],
        "output_contract": {},
        "unilm_identity": {},
    }
    failures: list[str] = []
    config_file = config_path.expanduser()
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve(strict=False)
    try:
        config = load_moe_config(config_file)
        config_for_inventory = copy.deepcopy(config)
        config_for_inventory["__config_path__"] = str(config_file)
        checks["configuration"] = {
            "status": "PASS",
            "path": str(config_file),
            "schema_version": config["schema_version"],
            "configured_expert_order": list(config["gate"]["expert_order"]),
        }
        checks["resources"] = _configured_resource_inventory(config_for_inventory, unilm_source_dir)
    except Exception as exc:
        failures.append("FAIL_CONFIGURATION")
        checks["configuration"] = {"status": "FAIL_CONFIGURATION", "error": f"{type(exc).__name__}: {exc}"}
        checks["resources"] = []

    before_path = list(sys.path)
    before_modules = set(sys.modules)
    known_vendor_roots = {"models", "tools", "BEATs", "backbone", "modules", "laion_clap", "clap", "m2d", "transformers"}
    classes = []
    for expert_name in order:
        try:
            adapter_class = resolve_expert_adapter_class(expert_name)
            module = sys.modules[adapter_class.__module__]
            module_path = Path(module.__file__).resolve(strict=False)
            expected_root = (REPO_ROOT / "model" / "expert_adapters").resolve()
            if not module_path.is_relative_to(expected_root):
                raise ImportError(
                    f"{expert_name} adapter origin {module_path} is outside {expected_root}"
                )
            classes.append(adapter_class)
            checks["adapter_imports"].append(
                {
                    "expert": expert_name,
                    "status": "PASS",
                    "adapter_class": adapter_class.__name__,
                    "module_origin": str(module_path),
                    "factory_available": callable(getattr(adapter_class, "from_config", None)),
                    "factory_constructed": False,
                }
            )
        except Exception as exc:
            failures.append("FAIL_IMPORT_ORIGIN" if isinstance(exc, ImportError) else "FAIL_FACTORY_REGISTRATION")
            checks["adapter_imports"].append(
                {"expert": expert_name, "status": "FAIL_FACTORY_REGISTRATION", "error": f"{type(exc).__name__}: {exc}"}
            )

    added_modules = set(sys.modules) - before_modules
    unexpected_vendor_imports = sorted(
        name for name in added_modules if name.split(".", 1)[0] in known_vendor_roots
    )
    path_unchanged = sys.path == before_path
    aliases_before = before_modules & {"BEATs", "backbone", "modules"}
    aliases_after = set(sys.modules) & {"BEATs", "backbone", "modules"}
    alias_state_preserved = aliases_after == aliases_before
    checks["import_isolation"] = {
        "status": "PASS" if path_unchanged and alias_state_preserved and not unexpected_vendor_imports else "FAIL_IMPORT_COLLISION",
        "sys_path_unchanged": path_unchanged,
        "temporary_beats_alias_state_preserved": alias_state_preserved,
        "unexpected_vendor_modules_loaded": unexpected_vendor_imports,
        "laion_clap_loaded_by_adapter_import": "laion_clap" in unexpected_vendor_imports,
        "factory_imports_exercised": False,
    }
    if checks["import_isolation"]["status"] != "PASS":
        failures.append("FAIL_IMPORT_COLLISION")

    if len(classes) == len(order):
        try:
            batch_size = 2
            stub_outputs = {
                name: (
                    torch.full((batch_size, 1), float(index + 1)),
                    torch.full((batch_size, EXPERT_FEATURE_DIMS[name]), float(index + 1)),
                )
                for index, name in enumerate(EXPERT_ORDER)
            }
            scores, features = validate_and_assemble_expert_outputs(stub_outputs, batch_size=batch_size)
            checks["output_contract"] = {
                "status": "PASS",
                "score_shape": list(scores.shape),
                "feature_shape": list(features.shape),
                "feature_width": int(features.shape[1]),
                "expected_feature_width": EXPECTED_FEATURE_WIDTH,
                "score_order": list(EXPERT_ORDER),
                "feature_order": list(EXPERT_ORDER),
                "model_outputs_used": False,
            }
        except Exception as exc:
            failures.append("FAIL_OUTPUT_CONTRACT")
            checks["output_contract"] = {"status": "FAIL_OUTPUT_CONTRACT", "error": f"{type(exc).__name__}: {exc}"}
    else:
        failures.append("FAIL_FACTORY_REGISTRATION")
        checks["output_contract"] = {"status": "NOT_RUN_FACTORY_REGISTRATION_FAILURE"}

    checks["unilm_identity"] = _source_identity_check(unilm_source_dir)
    if checks["unilm_identity"]["status"] == "FAIL_CONFIGURATION":
        failures.append("FAIL_CONFIGURATION")
    if checks["unilm_identity"]["status"] == "BLOCKED_MISSING_RESOURCE":
        failures.append("BLOCKED_MISSING_RESOURCE")

    resource_statuses = {entry["status"] for entry in checks["resources"]}
    if "BLOCKED_MISSING_RESOURCE" in resource_statuses:
        resource_summary_status = "BLOCKED_MISSING_RESOURCE"
    elif "NOT_CONFIGURED" in resource_statuses:
        resource_summary_status = "NOT_CONFIGURED"
    else:
        resource_summary_status = "PASS"
    checks["configured_resources_summary"] = {
        "status": resource_summary_status,
        "status_counts": {status: sum(entry["status"] == status for entry in checks["resources"]) for status in sorted(resource_statuses)},
        "informational_only_in_lightweight_mode": True,
    }
    if "BLOCKED_MISSING_RESOURCE" in resource_statuses:
        failures.append("BLOCKED_MISSING_RESOURCE")
    checks["real_factory_coexistence"] = {
        "status": "NOT_VERIFIED_REAL_FACTORY",
        "reason": "LOCAL-08 imports adapter wrappers only; no pretrained expert factory was called.",
    }
    checks["real_inference"] = {
        "status": "NOT_VERIFIED_REAL_INFERENCE",
        "reason": "No heavyweight checkpoints or expert inference were run in LOCAL-08.",
    }
    checks["memory"] = {
        "status": "NOT_VERIFIED_MEMORY",
        "reason": "No pretrained models were initialized; individual expert peaks do not establish joint residency.",
    }
    return {
        "status": "PASS — LIGHTWEIGHT INTEGRATION FOUNDATION VERIFIED" if not failures else failures[0],
        "mode": "lightweight",
        "order": list(order),
        "offline": True,
        "real_models_constructed": False,
        "heavy_checkpoints_deserialized": False,
        "failures": failures,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    from model.expert_adapters.registry import EXPERT_ORDER

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lightweight",), default="lightweight")
    parser.add_argument("--order", type=lambda value: _parse_order(value, EXPERT_ORDER), default=EXPERT_ORDER)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "moe_local.example.yaml")
    parser.add_argument("--unilm-source-dir", type=str)
    parser.add_argument("--offline", action="store_true", help="Require offline mode (mandatory).")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    if not args.offline:
        parser.error("--offline is required; this diagnostic never accesses the network")

    summary = run_preflight(
        order=args.order,
        config_path=args.config,
        unilm_source_dir=args.unilm_source_dir,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.report_json is not None:
        report_path = args.report_json.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
