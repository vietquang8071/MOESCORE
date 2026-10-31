"""Fine-tune the cosine MGA-CLAP expert on XACLE average relevance scores."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader

from MOESCORE.expert2_finetune.expert2_data import Expert2Dataset, collate_expert2
from MOESCORE.expert2_finetune.expert2_loss import hybrid_loss
from MOESCORE.expert2_finetune.expert2_model import create_expert2


ROOT = Path(__file__).resolve().parent


def compute_metrics(predictions, targets):
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape or len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Predictions and targets must be equal-length finite vectors with at least two items")
    result = {
        "SRCC": float(spearmanr(p, y).statistic),
        "LCC": float(pearsonr(p, y).statistic),
        "KTAU": float(kendalltau(p, y).statistic),
        "MSE": float(np.mean((p - y) ** 2)),
        "N": len(p),
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Validation metrics are not finite")
    return result


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_splits(config, data_root_override=None):
    data_root = resolve(data_root_override or config["data_root"])
    train = Expert2Dataset(data_root / "meta_data/train_average.csv", data_root / "wav/train")
    validation = Expert2Dataset(data_root / "meta_data/validation_average.csv", data_root / "wav/validation")
    for label, dataset, expected in (
        ("train", train, config.get("expected_train_pairs", 7500)),
        ("validation", validation, config.get("expected_validation_pairs", 3000)),
    ):
        if len(dataset) != expected:
            raise ValueError(f"{label}: expected {expected} average-score pairs, found {len(dataset)}")
        missing = [r[dataset.file_column] for r in dataset.rows if not (dataset.wav_dir / r[dataset.file_column]).is_file()]
        if missing:
            raise FileNotFoundError(f"{label}: {len(missing)} WAV files missing; first: {missing[0]}")
    overlap = {r[train.file_column] for r in train.rows} & {r[validation.file_column] for r in validation.rows}
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} WAV names")
    return train, validation


def get_eval_split(config, data_root_override=None, split="validation"):
    if split == "validation":
        return get_splits(config, data_root_override)[1]
    if split != "test":
        raise ValueError(f"Unsupported evaluation split: {split}")
    data_root = resolve(data_root_override or config["data_root"])
    dataset = Expert2Dataset(data_root / "meta_data/test_average.csv", data_root / "wav/test")
    expected = int(config.get("expected_test_pairs", 3000))
    if len(dataset) != expected:
        raise ValueError(f"test: expected {expected} average-score pairs, found {len(dataset)}")
    missing = [r[dataset.file_column] for r in dataset.rows if not (dataset.wav_dir / r[dataset.file_column]).is_file()]
    if missing:
        raise FileNotFoundError(f"test: {len(missing)} WAV files missing; first: {missing[0]}")
    return dataset


def validate(model, loader, device):
    model.eval()
    rows, predictions, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch["mga_wavs"] = batch["mga_wavs"].to(device)
            score, _ = model(batch)
            values = (score.reshape(-1) * 5 + 5).cpu().tolist()
            true = batch["scores_raw"].tolist()
            predictions.extend(values)
            targets.extend(true)
            rows.extend((name, pred) for name, pred in zip(batch["wav_file_name"], values))
    return compute_metrics(predictions, targets), rows


def save_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wav_file_name", "pred_score"))
        writer.writerows(rows)


def evaluate_checkpoint(config, data_root_override=None, checkpoint=None, split="validation"):
    device = torch.device(config.get("device", "cuda"))
    dataset = get_eval_split(config, data_root_override, split)
    loader = DataLoader(dataset, batch_size=int(config.get("val_batch_size", 8)), shuffle=False,
                        num_workers=int(config.get("num_workers", 0)), collate_fn=collate_expert2)
    with resolve(config["mga_config"]).open(encoding="utf-8") as stream:
        architecture = yaml.safe_load(stream)
    model = create_expert2(resolve(config["mga_source"]), architecture, resolve(config["mga_checkpoint"]),
                           device, freeze_policy=config.get("freeze_policy", "partial"))
    if checkpoint is not None:
        state = torch.load(resolve(checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
    metrics, rows = validate(model, loader, device)
    output = resolve(config["output"])
    run_dir = output.parent / f"{output.stem.removesuffix('_best')}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    label = "evaluation" if checkpoint is not None else "pretrained"
    if split == "test":
        label = f"test_{label}"
    save_predictions(run_dir / f"{label}_predictions.csv", rows)
    (run_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train(config, data_root_override=None, resume=None):
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set device: cpu or use a GPU")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds, val_ds = get_splits(config, data_root_override)
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True,
                              num_workers=int(config.get("num_workers", 0)), collate_fn=collate_expert2)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("val_batch_size", 8)), shuffle=False,
                            num_workers=int(config.get("num_workers", 0)), collate_fn=collate_expert2)
    with resolve(config["mga_config"]).open(encoding="utf-8") as stream:
        architecture = yaml.safe_load(stream)
    model = create_expert2(resolve(config["mga_source"]), architecture, resolve(config["mga_checkpoint"]),
                           device, freeze_policy=config.get("freeze_policy", "partial"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}", flush=True)
    if not parameters:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=float(config["lr"]), betas=(0.9, 0.999))
    epochs = int(config["epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    warmup = int(total_steps * float(config.get("warmup_ratio", 0.1)))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    output = resolve(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.parent / f"{output.stem.removesuffix('_best')}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_srcc, stale = 0, float("-inf"), 0
    if resume:
        state = torch.load(resolve(resume), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_srcc, stale = state["best_srcc"], state["stale"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch["mga_wavs"] = batch["mga_wavs"].to(device)
            target = batch["scores"].to(device)
            optimizer.zero_grad(set_to_none=True)
            score, _ = model(batch)
            loss, mse_loss, pair_loss = hybrid_loss(score.reshape(-1), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config.get("grad_clip", 2.0)))
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            if step % int(config.get("log_every", 100)) == 0 or step == len(train_loader):
                print(f"epoch={epoch+1} step={step}/{len(train_loader)} loss={running/step:.5f} "
                      f"mse={float(mse_loss.detach()):.5f} pair={float(pair_loss.detach()):.5f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        metrics, predictions = validate(model, val_loader, device)
        print(f"epoch={epoch+1} validation={metrics}", flush=True)
        improved = metrics["SRCC"] > best_srcc
        if improved:
            best_srcc, stale = metrics["SRCC"], 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
            save_predictions(run_dir / "validation_predictions.csv", predictions)
            (run_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            stale += 1
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_srcc": best_srcc, "stale": stale,
            "config": config, "metrics": metrics, "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        }
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, run_dir / "last.pt")
        if stale >= int(config.get("patience", 10)):
            print("Early stopping", flush=True)
            break
    return best_srcc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expert2_finetune.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate pretrained MGA or --checkpoint on a labeled split")
    parser.add_argument("--checkpoint", default=None, help="Raw Expert2 state dict for --eval-only")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Evaluation split for --eval-only")
    args = parser.parse_args()
    with resolve(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.split != "validation" and not args.eval_only:
        parser.error("--split test requires --eval-only")
    if args.check_data:
        train_ds, val_ds = get_splits(config, args.data_root)
        print(f"Data OK: train={len(train_ds)}, validation={len(val_ds)}")
        print("First train sample:", train_ds[0]["wav_file_name"])
        return
    if args.eval_only:
        print(json.dumps(evaluate_checkpoint(config, args.data_root, args.checkpoint, args.split), indent=2))
        return
    if args.checkpoint:
        parser.error("--checkpoint requires --eval-only")
    train(config, args.data_root, args.resume)


if __name__ == "__main__":
    main()
