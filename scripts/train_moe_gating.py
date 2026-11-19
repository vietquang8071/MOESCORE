#!/usr/bin/env python3
"""Train only the LOCAL-10 gate from validated frozen-expert feature stores."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from data.moe_feature_store import preprocessing_contract
from data.audio_overlap import (
    build_train_validation_audio_overlap_diagnostic,
    emit_audio_overlap_warning,
    raise_for_audio_overlap,
    validate_source_audio_records,
)
from model.moe_gating_core import GatingNetwork
from model.expert_adapters.registry import profile_contract
from training.checkpoint_manager import (
    MoeCheckpointManager, full7_source_code_sha256, full8_source_code_sha256,
)
from training.config import load_moe_training_config
from training.moe_evaluator import MoeGateEvaluator
from training.moe_trainer import MoeGateTrainer
from training.sampler import StatefulRandomSampler
from scripts.audit_full7_scores import audit_store

EXPERIMENTAL_SKIPPED = "EXPERIMENTAL_SKIPPED"
HUMAN_REVIEWED = "HUMAN_REVIEWED"
SYNTHETIC_NOT_APPLICABLE = "SYNTHETIC_NOT_APPLICABLE"


def _experimental_skip_score_review_enabled() -> bool:
    return os.environ.get("MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW") == "1"


def _real_score_review(path: str | Path | None, manifest: dict, audit: dict) -> str:
    """Require an explicit train-bound human score review before real optimization."""
    if path is None:
        raise ValueError("real Full7 training requires a score review file after train-only audit")
    target = Path(path).expanduser().resolve()
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"real Full7 score review cannot be read: {exc}") from exc
    def fingerprint(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if (not isinstance(body, dict) or body.get("schema_version") != 1
            or body.get("train_store_identity_sha256") != manifest["store_identity_sha256"]
            or body.get("score_contract_sha256") != fingerprint(manifest["score_contracts"])
            or body.get("score_audit_sha256") != fingerprint(audit["diagnostics"])
            or not isinstance(body.get("reviewer"), str) or not body["reviewer"].strip()
            or not isinstance(body.get("evidence"), str) or not body["evidence"].strip()
            or not isinstance(body.get("experts"), dict)
            or set(body["experts"]) != set(manifest["expert_order"])
            or any(body["experts"][name] != "REVIEWED_FOR_GATE_PROXY" for name in manifest["expert_order"])):
        raise ValueError("real Full7 score review is incomplete or bound to different evidence")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _resolve_full7_score_review(
    store_root: str | Path,
    manifest: dict,
    store_identity: dict[str, Any],
    review_path: str | Path | None,
    *,
    skip_review: bool,
) -> tuple[dict[str, Any], str | None, str, bool]:
    """Run the train-only audit and bind the selected review policy to the store."""
    audit = audit_store(store_root)
    if manifest["assembly_identity"].get("synthetic_provenance"):
        policy, review_sha256, human_review_performed = SYNTHETIC_NOT_APPLICABLE, None, False
    elif skip_review:
        policy, review_sha256, human_review_performed = EXPERIMENTAL_SKIPPED, None, False
        print(
            "\n*** EXPERIMENTAL WARNING: HUMAN SCORE REVIEW SKIPPED ***\n"
            "The train-only score audit still ran, but expert score semantics were not "
            "human-reviewed. This is a research-only run and is not independently "
            "validated or a claim of competition compliance.\n",
            file=sys.stderr,
            flush=True,
        )
    else:
        review_sha256 = _real_score_review(review_path, manifest, audit)
        policy, human_review_performed = HUMAN_REVIEWED, True
    store_identity["score_review_policy"] = policy
    if review_sha256 is not None:
        store_identity["score_review_sha256"] = review_sha256
    else:
        store_identity.pop("score_review_sha256", None)
    return audit, review_sha256, policy, human_review_performed


def _reject_accidental_checkpoint_overwrite(output_dir: str | Path, *, resume: str | None) -> None:
    if resume is not None:
        return
    target = Path(output_dir)
    existing = [name for name in ("latest_gate.pt", "best_gate.pt") if (target / name).exists()]
    if existing:
        raise FileExistsError(
            f"Gate output directory already contains {', '.join(existing)}; "
            "use a fresh output directory or an explicit compatible resume"
        )


def _interval_checkpoint_due(*, global_step: int, interval: int,
                             sampler_cursor: int, sample_count: int, batch_size: int) -> bool:
    """Keep the final full batch's checkpoint until epoch validation completes."""
    return global_step % interval == 0 and sample_count - sampler_cursor >= batch_size


def _feature_loader(dataset, *, batch_size: int, sampler=None, collate_fn=collate_moe_features) -> DataLoader:
    """Build a feature loader that retains a final partial batch for every official row."""
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=0, drop_last=False, collate_fn=collate_fn)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device("cuda:0" if value == "cuda" else value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.pop("config_path", None)
    value.pop("output_dir", None)
    value.get("data", {}).pop("feature_store_root", None)
    value.get("runtime", {}).pop("device", None)
    return value


def _store_provenance(train: MoeTrainingDataset, validation: MoeTrainingDataset, *,
                      allow_train_validation_audio_overlap: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(allow_train_validation_audio_overlap) is not bool:
        raise ValueError("allow_train_validation_audio_overlap must be boolean")
    train_manifest, val_manifest = train.manifest, validation.manifest
    if (train_manifest.get("expert_profile", "top1_4") != val_manifest.get("expert_profile", "top1_4")
            or train_manifest["expert_order"] != val_manifest["expert_order"]
            or train_manifest["feature_dimensions"] != val_manifest["feature_dimensions"]):
        raise ValueError("train and validation feature-store expert profiles differ")
    train_provenance = train_manifest["checkpoint_provenance"]
    if train_provenance != val_manifest["checkpoint_provenance"]:
        raise ValueError("train and validation feature stores use different expert checkpoint provenance")
    train_contract, val_contract = preprocessing_contract(train_manifest), preprocessing_contract(val_manifest)
    if train_contract != val_contract:
        raise ValueError("train and validation preprocessing contracts differ")
    if train_manifest.get("expert_profile") in {"full_7", "full_8"}:
        train_rows = validate_source_audio_records(
            train_manifest["assembly_identity"].get("ordered_source_fingerprints", []), split="train",
        )
        validation_rows = validate_source_audio_records(
            val_manifest["assembly_identity"].get("ordered_source_fingerprints", []), split="validation",
        )
        overlap_diagnostic = build_train_validation_audio_overlap_diagnostic(
            train_rows, validation_rows,
            allow_train_validation_audio_overlap=allow_train_validation_audio_overlap,
        )
        if overlap_diagnostic["overlapping_sha256_count"] and not allow_train_validation_audio_overlap:
            raise_for_audio_overlap(overlap_diagnostic, stage="Global Gate")
        emit_audio_overlap_warning(overlap_diagnostic, stage="Global Gate training")
    elif allow_train_validation_audio_overlap:
        raise ValueError("allow_train_validation_audio_overlap is supported only for real Full7/Full8 feature stores")
    identity = {
        "train": train.identity,
        "validation": validation.identity,
        "preprocessing_contract": train_contract,
    }
    if train_manifest.get("expert_profile") in {"full_7", "full_8"}:
        identity["source_wav_sha256_by_split"] = {
            "train": [row["wav_sha256"] for row in train_rows],
            "validation": [row["wav_sha256"] for row in validation_rows],
        }
        identity["source_sample_ids_by_split"] = {
            "train": [row["sample_id"] for row in train_rows],
            "validation": [row["sample_id"] for row in validation_rows],
        }
        identity["audio_overlap_diagnostic"] = overlap_diagnostic
    return train_provenance, identity


def run_training(config: dict[str, Any], *, resume: str | None = None, device_override: str | None = None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    allow_audio_overlap = config["data"].get("allow_train_validation_audio_overlap", False)
    if type(allow_audio_overlap) is not bool:
        raise ValueError("data.allow_train_validation_audio_overlap must be boolean")
    config["data"]["allow_train_validation_audio_overlap"] = allow_audio_overlap
    device = _device(device_override or config["runtime"]["device"])
    torch.manual_seed(config["runtime"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["runtime"]["seed"])
    store_root = config["data"]["feature_store_root"]
    train_ds = MoeTrainingDataset(store_root, split="train", require_labels=True)
    validation_ds = MoeTrainingDataset(store_root, split="validation", require_labels=True)
    checkpoint_provenance, store_identity = _store_provenance(
        train_ds, validation_ds,
        allow_train_validation_audio_overlap=allow_audio_overlap,
    )
    profile = train_ds.manifest.get("expert_profile", "top1_4")
    overlap_diagnostic = store_identity.get("audio_overlap_diagnostic")
    validation_warning = (
        overlap_diagnostic.get("warning") if isinstance(overlap_diagnostic, dict)
        and overlap_diagnostic.get("overlapping_sha256_count") else None
    )
    skip_review = _experimental_skip_score_review_enabled()

    score_audit = None
    score_review_sha256 = None
    score_review_policy = None
    human_review_performed = False

    if profile in {"full_7", "full_8"}:
        (score_audit, score_review_sha256, score_review_policy,
         human_review_performed) = _resolve_full7_score_review(
            store_root,
            train_ds.manifest,
            store_identity,
            config["data"].get("score_review_file"),
            skip_review=skip_review,
        )
        config["data"]["score_review_policy"] = score_review_policy
    sampler = StatefulRandomSampler(train_ds, seed=config["runtime"]["seed"])
    train_loader = _feature_loader(train_ds, batch_size=config["training"]["batch_size"],
                                   sampler=sampler)
    validation_loader = _feature_loader(validation_ds,
                                        batch_size=config["validation"]["batch_size"])
    if config["data"].get("profile", profile) != profile:
        raise ValueError(f"configured expert profile differs from {profile} feature store")
    order, _, width = profile_contract(profile)
    gate = GatingNetwork(input_feature_dim=width, num_experts=len(order), profile=profile).to(device)
    trainer = MoeGateTrainer(
        gate,
        config["training"]["loss"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        diversity_weight=config["training"]["diversity_weight"],
    )
    evaluator = MoeGateEvaluator(gate, config["training"]["loss"])
    manager = MoeCheckpointManager()
    run_config = _checkpoint_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_gate.pt"
    best_path = output_dir / "best_gate.pt"
    if profile in {"full_7", "full_8"}:
        _reject_accidental_checkpoint_overwrite(output_dir, resume=resume)
    if profile in {"full_7", "full_8"} and resume is not None:
        for existing_path in (latest_path, best_path):
            if not existing_path.is_file():
                continue
            try:
                manager.load(
                    existing_path,
                    gating_network=gate,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    expected_config=run_config,
                    expected_feature_store_identity=store_identity,
                    expected_expert_checkpoint_provenance=checkpoint_provenance,
                    restore_rng=False,
                )
            except Exception as exc:
                raise ValueError(
                    f"existing Gate checkpoint is incompatible ({existing_path}); "
                    "refusing to overwrite it; use a fresh output directory"
                ) from exc
    start_epoch = 0
    best_srcc: float | None = None
    patience = 0
    if resume:
        state = manager.load(
            resume,
            gating_network=gate,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            expected_config=run_config,
            expected_feature_store_identity=store_identity,
            expected_expert_checkpoint_provenance=checkpoint_provenance,
        )
        trainer.global_step = int(state["global_step"])
        best_srcc = state.get("best_srcc")
        patience = int(state.get("patience", 0))
        sampler_state = state.get("sampler_state")
        if sampler_state is not None:
            sampler.load_state_dict(sampler_state)
            if sampler.cursor == len(sampler.order):
                start_epoch = sampler.epoch + 1
                sampler.set_epoch(start_epoch)
            else:
                start_epoch = sampler.epoch
        else:
            start_epoch = int(state["epoch"]) + 1
            sampler.set_epoch(start_epoch)

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        if sampler.epoch != epoch:
            sampler.set_epoch(epoch)
        gate.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_rows_seen = 0
        for batch in train_loader:
            result = trainer.train_step(batch)
            train_loss_sum += float(result["total_loss"].item())
            train_batches += 1
            train_rows_seen += int(batch["labels_raw"].shape[0])
            interval = config["training"]["checkpoint_interval_steps"]
            if (trainer.global_step % interval == 0 and
                    (profile not in {"full_7", "full_8"} or _interval_checkpoint_due(
                        global_step=trainer.global_step, interval=interval,
                        sampler_cursor=sampler.cursor, sample_count=len(train_ds),
                        batch_size=config["training"]["batch_size"]))):
                manager.save(
                    latest_path,
                    gating_network=gate,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    epoch=epoch,
                    global_step=trainer.global_step,
                    config=run_config,
                    feature_store_identity=store_identity,
                    expert_checkpoint_provenance=checkpoint_provenance,
                    sampler_state=sampler.state_dict(),
                    best_srcc=best_srcc,
                    patience=patience,
                    seed=config["runtime"]["seed"],
                )
        if train_batches == 0:
            raise ValueError("training loader produced no batches; reduce batch_size or check train store")
        validation = evaluator.evaluate(validation_loader)
        validation_loss = float(validation.get("regression_loss_normalized", validation["metrics"]["mse"]))
        trainer.scheduler_step(validation_loss)
        srcc = validation["metrics"]["srcc"]
        improved = srcc is not None and (best_srcc is None or srcc > best_srcc)
        if improved:
            best_srcc, patience = srcc, 0
            manager.save(
                best_path,
                gating_network=gate,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                global_step=trainer.global_step,
                config=run_config,
                feature_store_identity=store_identity,
                expert_checkpoint_provenance=checkpoint_provenance,
                sampler_state=sampler.state_dict(),
                best_srcc=best_srcc,
                patience=patience,
                seed=config["runtime"]["seed"],
            )
        elif srcc is not None:
            patience += 1
        history_row = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "train_rows_used": train_rows_seen,
            "train_rows_available": len(train_ds),
            "train_loss": train_loss_sum / train_batches,
            "validation_loss_normalized": validation_loss,
            "validation_metrics_raw": validation["metrics"],
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        }
        if validation_warning:
            history_row["validation_metrics_warning"] = validation_warning
        history.append(history_row)
        print(f"Epoch {epoch + 1}: step={trainer.global_step}, train_loss={history[-1]['train_loss']:.6f}, "
              f"validation={validation['metrics']}" +
              (f" [WARNING: {validation_warning}]" if validation_warning else "") +
              f", best_srcc={best_srcc}", file=sys.stderr, flush=True)
        sampler.set_epoch(epoch + 1)
        manager.save(
            latest_path,
            gating_network=gate,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=epoch + 1,
            global_step=trainer.global_step,
            config=run_config,
            feature_store_identity=store_identity,
            expert_checkpoint_provenance=checkpoint_provenance,
            sampler_state=sampler.state_dict(),
            best_srcc=best_srcc,
            patience=patience,
            seed=config["runtime"]["seed"],
        )
        if patience >= config["training"]["early_stop_patience"] > 0:
            break
    report = {
        "status": "PASS",
        "device": str(device),
        "epochs_completed": len(history),
        "global_step": trainer.global_step,
        "best_srcc_raw_scale": best_srcc,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path),
        "history": history,
        "expert_model_construction": False,
        "expert_checkpoint_deserialization": False,
        "score_audit": score_audit,
        "score_review_sha256": score_review_sha256,
    }
    if profile in {"full_7", "full_8"}:
        report.update({
            "score_review_policy": score_review_policy,
            "human_review_performed": human_review_performed,
            "run_purpose": "research_only",
        })
    if overlap_diagnostic is not None:
        report["allow_train_validation_audio_overlap"] = allow_audio_overlap
        report["audio_overlap_diagnostic"] = overlap_diagnostic
        if validation_warning:
            report["validation_warning"] = validation_warning
    if profile in {"full_7", "full_8"}:
        def file_hash(path: Path) -> str | None:
            if not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False)
        git_status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, check=False)
        source_hashes = full8_source_code_sha256() if profile == "full_8" else full7_source_code_sha256()
        manifest = {
            "schema_version": 1, "expert_profile": profile,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "working_tree_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
            "source_code_sha256": source_hashes,
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "config_sha256": hashlib.sha256(json.dumps(run_config, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "train_store_identity_sha256": train_ds.identity,
            "validation_store_identity_sha256": validation_ds.identity,
            "source_contract": store_identity["preprocessing_contract"],
            "expert_checkpoint_provenance": checkpoint_provenance,
            "gate_checkpoint_sha256": file_hash(best_path),
            "latest_gate_checkpoint_sha256": file_hash(latest_path),
            "seed": config["runtime"]["seed"], "device": str(device), "dtype": "float32",
            "split_policy": "train_gradient_validation_evaluation_test_inference",
            "selection_policy": "official_validation_srcc_strict_v1_provisional",
            "invocation": sys.argv if Path(sys.argv[0]).name == "train_moe_gating.py"
                          else ["run_training(programmatic)"],
            "synthetic_provenance": train_ds.manifest["assembly_identity"]["synthetic_provenance"],
            "score_audit": score_audit,
            "score_review_sha256": score_review_sha256,
            "score_review_policy": score_review_policy,
            "human_review_performed": human_review_performed,
            "run_purpose": "research_only",
            "allow_train_validation_audio_overlap": allow_audio_overlap,
            "audio_overlap_diagnostic": overlap_diagnostic,
            "validation_warning": validation_warning,
            "history": history,
        }
        temporary = output_dir / f"run_manifest.json.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output_dir / "run_manifest.json")
        report["run_manifest"] = str(output_dir / "run_manifest.json")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    try:
        report = run_training(load_moe_training_config(args.config), resume=args.resume, device_override=args.device)
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    body = json.dumps(report, indent=2, ensure_ascii=False)
    print(body)
    if args.report_json:
        target = Path(args.report_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
