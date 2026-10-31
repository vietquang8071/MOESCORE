"""Train MOESCORE Expert 3 with the required physical batch size of eight."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader

from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.losses.loss_function import get_loss_function
from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3 import MoescoreExpert3
from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.utils.expert3_checkpoint import (
    load_resume_checkpoint,
    save_best_checkpoint,
    save_resume_checkpoint,
)


ROOT = Path(__file__).resolve().parent
TRAIN_BATCH_SIZE = 8


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return config


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def prepare_config(config, dataset_root=None, checkpoint=None, output_dir=None, cache_dir=None, device=None):
    cfg = copy.deepcopy(config)
    if dataset_root is not None:
        cfg["dataset_root"] = str(dataset_root)
    if checkpoint is not None:
        cfg["m2d_checkpoint"] = str(checkpoint)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if cache_dir is not None:
        cfg["cache_dir"] = str(cache_dir)
    if device is not None:
        cfg["device"] = device
    for key in ("dataset_root", "m2d_checkpoint", "cache_dir", "output_dir"):
        cfg[key] = str(resolve_path(cfg[key]))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required = {
        "dataset_root", "train_csv", "validation_csv", "test_csv", "wav_dir",
        "filename_column", "caption_column", "score_column", "m2d_checkpoint",
        "cache_dir", "output_dir", "batch_size", "gradient_accumulation_steps",
        "drop_last", "val_batch_size", "lr", "optimizer", "scheduler", "num_workers",
        "epochs", "max_len", "early_stop_patience", "seed", "sample_rate", "amp",
        "activation_checkpointing", "concat_assignment", "loss", "device",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"Missing Expert 3 config keys: {missing}")
    if cfg["batch_size"] != TRAIN_BATCH_SIZE:
        raise ValueError("Expert 3 reproduction requires physical batch_size=8")
    if cfg["gradient_accumulation_steps"] != 1:
        raise ValueError("gradient_accumulation_steps must be 1; pairwise loss sees each real batch of eight")
    if not cfg["drop_last"]:
        raise ValueError("drop_last must be true so every training loss uses exactly eight examples")
    if cfg["val_batch_size"] != 1:
        raise ValueError("val_batch_size must be 1, following the MOESCORE configuration")
    if cfg["sample_rate"] != 16000 or cfg["max_len"] != 10:
        raise ValueError("M2D-CLAP training requires 16 kHz waveforms padded/truncated to 10 seconds")
    if cfg["num_workers"] != 0 or cfg["epochs"] <= 0 or cfg["early_stop_patience"] <= 0:
        raise ValueError("num_workers must be 0 and epochs/patience must be positive")
    if cfg["optimizer"].get("name") != "Adam" or tuple(cfg["optimizer"].get("betas", ())) != (0.9, 0.999):
        raise ValueError("This reproduction uses Adam with betas=(0.9, 0.999)")
    if cfg["scheduler"].get("name") != "linear_warmup_linear_decay":
        raise ValueError("scheduler.name must be linear_warmup_linear_decay")
    if not 0.0 <= float(cfg["scheduler"].get("warmup_ratio", -1)) < 1.0:
        raise ValueError("scheduler.warmup_ratio must be in [0,1)")
    if cfg["lr"] <= 0 or cfg["concat_assignment"] not in {"released", "semantic"}:
        raise ValueError("Invalid learning rate or concat assignment")
    if cfg["loss"].get("loss_weights") != [1.0, 0.5]:
        raise ValueError("MOESCORE Expert 3 uses clipped MSE and contrastive weights [1.0, 0.5]")
    if cfg["loss"]["clipped_mse"].get("tau") != 0.25 or cfg["loss"]["clipped_mse"].get("cbl") is not False:
        raise ValueError("Expected clipped-MSE tau=0.25 and CBL disabled")
    if cfg["loss"]["contrastive"].get("margin") != 0.1 or cfg["loss"]["contrastive"].get("cbl") is not False:
        raise ValueError("Expected contrastive margin=0.1 and CBL disabled")


def build_datasets(cfg, dataset_root=None, split_names=("train", "validation")):
    from MOESCORE.expert3_finetune.XACLE_2027_baselinemodel.datasets.moescore_expert3_dataset import MoescoreExpert3Dataset

    root = Path(dataset_root or cfg["dataset_root"])
    datasets = {}
    for split in split_names:
        csv_path = root / cfg[f"{split}_csv"]
        wav_dir = root / cfg["wav_dir"] / split
        dataset = MoescoreExpert3Dataset(
            csv_path,
            wav_dir,
            sample_rate=cfg["sample_rate"],
            max_len=cfg["max_len"],
            filename_column=cfg["filename_column"],
            caption_column=cfg["caption_column"],
            score_column=cfg["score_column"],
        )
        expected = cfg.get("expected_counts", {}).get(split)
        if expected is not None and len(dataset) != int(expected):
            raise ValueError(f"{split}: expected {expected} scored pairs, found {len(dataset)} in {csv_path}")
        datasets[split] = dataset
    if "train" in datasets and "validation" in datasets:
        pairs = [("train", "validation")]
        if "test" in datasets:
            pairs.extend((("train", "test"), ("validation", "test")))
        for left, right in pairs:
            overlap = datasets[left].wav_names & datasets[right].wav_names
            if overlap:
                raise ValueError(f"{left}/{right} split overlap: {len(overlap)} WAV names; first={sorted(overlap)[0]}")
    return datasets


def _split_manifest(dataset):
    """Hash split metadata bytes and the path-independent WAV name/size inventory."""
    metadata_digest = hashlib.sha256()
    with dataset.csv_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            metadata_digest.update(chunk)
    audio_digest = hashlib.sha256()
    for filename in sorted(dataset.wav_names):
        size = (dataset.wav_dir / filename).stat().st_size
        audio_digest.update(f"{filename}\0{size}\n".encode("utf-8"))
    return {
        "metadata_sha256": metadata_digest.hexdigest(),
        "wav_inventory_sha256": audio_digest.hexdigest(),
        "count": len(dataset),
    }


def build_dataloaders(cfg, dataset_root=None):
    # Validate all published splits on every run, but expose only train and
    # validation loaders to the optimizer and checkpoint-selection loop.
    datasets = build_datasets(cfg, dataset_root, ("train", "validation", "test"))
    cfg["split_manifest"] = {name: _split_manifest(dataset) for name, dataset in datasets.items()}
    if len(datasets["train"]) < TRAIN_BATCH_SIZE:
        raise ValueError("Training split must contain at least eight examples")
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        datasets["train"], batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=0, collate_fn=datasets["train"].collate_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=1, shuffle=False,
        drop_last=False, num_workers=0, collate_fn=datasets["validation"].collate_fn,
    )
    if len(train_loader) == 0:
        raise ValueError("DataLoader produced no full eight-example training batches")
    return train_loader, validation_loader, generator


def compute_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape or predictions.size < 2:
        raise ValueError("Metric inputs must be equal-length vectors containing at least two samples")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Metric inputs contain NaN or Inf")
    metrics = {
        "SRCC": float(spearmanr(targets, predictions).statistic),
        "LCC": float(pearsonr(targets, predictions).statistic),
        "KTAU": float(kendalltau(targets, predictions).statistic),
        "MSE": float(np.mean((predictions - targets) ** 2)),
        "N": int(predictions.size),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite evaluation metrics: {metrics}")
    return metrics


def write_predictions(path, filenames, predictions):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wav_file_name", "pred_score"))
        writer.writerows(zip(filenames, predictions))


def build_optimizer(model, cfg):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Expert 3 has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["lr"]), betas=tuple(cfg["optimizer"]["betas"]))
    expected = {id(parameter) for parameter in trainable}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("Optimizer parameter set does not match all and only trainable parameters")
    return optimizer


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * float(warmup_ratio))

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()


def _move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _loss_for_batch(model, batch, loss_fn, device, amp):
    batch = _move_batch(batch, device)
    with _autocast(device, amp):
        prediction, _ = model(batch)
    prediction = prediction.reshape(-1).float()
    target = batch["scores"].reshape(-1).float()
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    loss = loss_fn(prediction, target, batch["num_class"].reshape(-1))
    return batch, prediction, target, loss


def _finite_gradients(model):
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                missing.append(name)
            elif not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
    if missing or nonfinite:
        raise RuntimeError(f"Trainable gradient check failed; missing={missing[:5]}, nonfinite={nonfinite[:5]}")


def smoke_test(model, train_loader, optimizer, loss_fn, scaler, device, amp):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    batch = next(iter(train_loader))
    if batch["scores"].numel() != TRAIN_BATCH_SIZE:
        raise RuntimeError("Smoke test requires one physical batch of exactly eight pairs")
    before = next(parameter.detach().clone() for parameter in model.alignment_head.parameters() if parameter.requires_grad)
    optimizer.zero_grad(set_to_none=True)
    try:
        batch, prediction, target, loss = _loss_for_batch(model, batch, loss_fn, device, amp)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke-test loss is not finite: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _finite_gradients(model)
        scaler.step(optimizer)
        scaler.update()
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "A physical batch of eight exceeded available GPU memory, even with AMP and activation checkpointing. "
            "Training stopped without reducing the batch or using gradient accumulation."
        ) from exc
    after = next(parameter.detach() for parameter in model.alignment_head.parameters() if parameter.requires_grad)
    if torch.equal(before, after):
        raise RuntimeError("Smoke-test optimizer step did not update the regression head")
    report = {
        "batch_size": batch["scores"].numel(),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "amp": bool(amp and device.type == "cuda"),
        "activation_checkpointing": model.activation_checkpointing,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    return report


@torch.inference_mode()
def validate(model, loader, device, amp):
    model.eval()
    predictions, targets, names = [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        with _autocast(device, amp):
            prediction, _ = model(batch)
        prediction = prediction.reshape(-1).float().cpu() * 5.0 + 5.0
        target = batch["scores_raw"].reshape(-1).float().cpu()
        predictions.extend(prediction.tolist())
        targets.extend(target.tolist())
        names.extend(batch["wav_file_names"])
    return compute_metrics(predictions, targets), names, predictions


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("a", encoding="utf-8")

    def log(self, message):
        print(message, flush=True)
        self.stream.write(str(message) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def _write_trainable_parameter_report(model, path):
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write(f"trainable={sum(count for _, count in trainable):,} total={total:,}\n")
        for name, count in trainable:
            stream.write(f"{name}\t{count}\n")
    return sum(count for _, count in trainable), total, len(trainable)


def _new_run_dir(base):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = Path(base) / stamp
    suffix = 1
    while candidate.exists():
        candidate = Path(base) / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _check_resume_config(saved, current):
    keys = ("batch_size", "gradient_accumulation_steps", "lr", "optimizer", "scheduler",
            "epochs", "early_stop_patience", "max_len", "sample_rate", "loss", "amp",
            "activation_checkpointing", "concat_assignment", "seed", "train_csv",
            "validation_csv", "wav_dir", "filename_column", "caption_column", "score_column",
            "expected_counts", "split_manifest", "drop_last", "val_batch_size", "num_workers")
    differences = [key for key in keys if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError(f"Resume configuration changed for {differences}; use the saved training settings")


def train(config, dataset_root=None, checkpoint=None, output_dir=None, cache_dir=None,
          device=None, resume=None, smoke=False):
    cfg = prepare_config(config, dataset_root, checkpoint, output_dir, cache_dir, device)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select a CUDA Colab runtime or pass --device cpu")
    if not smoke and device.type != "cuda":
        raise RuntimeError("Full Expert 3 fine-tuning requires a CUDA GPU; CPU is supported for unit tests only")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    train_loader, validation_loader, loader_generator = build_dataloaders(cfg)
    try:
        model = MoescoreExpert3(
            checkpoint_path=cfg["m2d_checkpoint"],
            cache_dir=cfg["cache_dir"],
            activation_checkpointing=cfg["activation_checkpointing"],
            concat_assignment=cfg["concat_assignment"],
            expected_checkpoint_sha256=cfg.get("m2d_checkpoint_sha256"),
        ).to(device)
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU ran out of memory while initializing M2D-CLAP. No training batch size was changed; "
            "check available GPU VRAM and restart a clean runtime."
        ) from exc
    optimizer = build_optimizer(model, cfg)
    loss_fn = get_loss_function(cfg["loss"]).to(device)
    scaler = _make_scaler(bool(cfg["amp"] and device.type == "cuda"))
    total_steps = len(train_loader) * int(cfg["epochs"])
    scheduler = build_scheduler(optimizer, total_steps, cfg["scheduler"]["warmup_ratio"])

    if resume:
        resume_path = Path(resume).expanduser().resolve()
        run_dir = resume_path.parent
        state = load_resume_checkpoint(model, resume_path, optimizer, scheduler, scaler, loader_generator)
        _check_resume_config(state.get("config", {}), cfg)
        start_epoch = int(state["epoch"]) + 1
        best_srcc = float(state["best_srcc"])
        stale_epochs = int(state["stale_epochs"])
        del state
    else:
        run_dir = _new_run_dir(cfg["output_dir"])
        start_epoch = 0
        best_srcc = -math.inf
        stale_epochs = 0
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(cfg)
    resolved_config["run_dir"] = str(run_dir)
    resolved_config["physical_train_batch_size"] = TRAIN_BATCH_SIZE
    resolved_config["gradient_accumulation_steps"] = 1
    resolved_config["warmup_assumption"] = "10% of total optimizer steps unless configured otherwise"
    (run_dir / "config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
    logger = RunLogger(run_dir / "training_log.txt")
    trainable_count, total_count, trainable_tensors = _write_trainable_parameter_report(
        model, run_dir / "trainable_parameters.txt"
    )
    logger.log(f"Device: {device}; GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}")
    logger.log(f"Dataset root: {cfg['dataset_root']}; train pairs={len(train_loader.dataset)}; val pairs={len(validation_loader.dataset)}")
    logger.log(f"Physical batch size: {TRAIN_BATCH_SIZE}; gradient accumulation: 1; train batches/epoch={len(train_loader)}")
    logger.log(f"Trainable parameters: {trainable_count:,}/{total_count:,} across {trainable_tensors} tensors")
    logger.log(f"All trainable parameter names/counts: {run_dir / 'trainable_parameters.txt'}")
    logger.log(f"M2D checkpoint SHA-256: {model.base_checkpoint_identity['sha256']}")
    logger.log(f"AMP: {bool(cfg['amp'] and device.type == 'cuda')}; activation checkpointing: {model.activation_checkpointing}")

    if smoke:
        try:
            report = smoke_test(model, train_loader, optimizer, loss_fn, scaler, device, bool(cfg["amp"]))
            logger.log("Batch-8 smoke test passed: " + json.dumps(report))
            return str(run_dir)
        finally:
            logger.close()

    try:
        for epoch in range(start_epoch, int(cfg["epochs"])):
            start_time = time.time()
            model.train()
            losses = []
            for step, raw_batch in enumerate(train_loader, start=1):
                if raw_batch["scores"].numel() != TRAIN_BATCH_SIZE:
                    raise RuntimeError(f"Training step got {raw_batch['scores'].numel()} samples instead of eight")
                optimizer.zero_grad(set_to_none=True)
                try:
                    batch, prediction, target, loss = _loss_for_batch(
                        model, raw_batch, loss_fn, device, bool(cfg["amp"])
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    _finite_gradients(model)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "Physical batch size 8 ran out of GPU memory despite AMP and activation checkpointing. "
                        "No smaller batch or gradient accumulation was substituted."
                    ) from exc
                losses.append(float(loss.detach().cpu()))

            metrics, filenames, predictions = validate(
                model, validation_loader, device, bool(cfg["amp"])
            )
            improved = metrics["SRCC"] > best_srcc
            if improved:
                best_srcc = metrics["SRCC"]
                stale_epochs = 0
                save_best_checkpoint(
                    model, run_dir / "best_model.pt", epoch, metrics, resolved_config
                )
                (run_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                write_predictions(run_dir / "evaluation_result.csv", filenames, predictions)
            else:
                stale_epochs += 1
            save_resume_checkpoint(
                model, run_dir / "last_checkpoint.pt", optimizer, scheduler, scaler,
                epoch, best_srcc, stale_epochs, resolved_config, metrics, loader_generator,
            )
            elapsed = time.time() - start_time
            logger.log(
                f"epoch={epoch + 1}/{cfg['epochs']} train_loss={float(np.mean(losses)):.6f} "
                f"val={json.dumps(metrics, sort_keys=True)} best_SRCC={best_srcc:.6f} "
                f"stale={stale_epochs}/{cfg['early_stop_patience']} "
                f"lr={scheduler.get_last_lr()[0]:.3e} seconds={elapsed:.1f}"
            )
            if stale_epochs >= int(cfg["early_stop_patience"]):
                logger.log("Early stopping: validation SRCC has not improved within patience")
                break
    finally:
        logger.close()
    return str(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expert3_moescore.json")
    parser.add_argument("--data-root")
    parser.add_argument("--m2d-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device")
    parser.add_argument("--resume", help="Path to last_checkpoint.pt")
    parser.add_argument("--check-data", action="store_true", help="Validate all split CSVs and WAV files without loading M2D")
    parser.add_argument("--smoke-test", action="store_true", help="Run one real batch of eight through forward/backward/optimizer")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    if args.check_data:
        cfg = prepare_config(config, dataset_root=args.data_root, checkpoint=args.m2d_checkpoint,
                             output_dir=args.output_dir, cache_dir=args.cache_dir, device=args.device)
        if not Path(cfg["m2d_checkpoint"]).is_file():
            raise FileNotFoundError(f"M2D checkpoint not found: {cfg['m2d_checkpoint']}")
        splits = build_datasets(cfg, split_names=("train", "validation", "test"))
        samples = {name: dataset[0] for name, dataset in splits.items()}
        print(json.dumps({
            "checkpoint": cfg["m2d_checkpoint"],
            "split_counts": {name: len(dataset) for name, dataset in splits.items()},
            "sample_waveforms": {name: list(row["wav"].shape) for name, row in samples.items()},
            "columns": [cfg["filename_column"], cfg["caption_column"], cfg["score_column"]],
        }, indent=2))
        return
    run_dir = train(
        config, dataset_root=args.data_root, checkpoint=args.m2d_checkpoint,
        output_dir=args.output_dir, cache_dir=args.cache_dir, device=args.device,
        resume=args.resume, smoke=args.smoke_test,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
