"""Fine-tune the LAION-CLAP expert for XACLE text-audio relevance scores.

This script is independent of the MGA, M2D, and BEATs modules. It saves a raw
Expert1 state_dict compatible with MOEModel.load_pretrain_weight.
"""

import argparse
import random
from pathlib import Path

import laion_clap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from loss.loss_function import get_loss_function


SAMPLE_RATE = 48_000
MAX_SECONDS = 10


class Expert1Dataset(Dataset):
    def __init__(self, csv_path: Path, wav_dir: Path, training: bool):
        self.rows = pd.read_csv(csv_path)
        required = {"wav_file_name", "text", "average_score"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        if self.rows.empty:
            raise ValueError(f"{csv_path} has no rows")
        self.wav_dir = wav_dir
        self.training = training
        self.max_samples = SAMPLE_RATE * MAX_SECONDS

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        wav_path = self.wav_dir / str(row["wav_file_name"]).lstrip("/")
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)  # mono
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )
        if waveform.numel() == 0:
            raise ValueError(f"Empty audio file: {wav_path}")

        if waveform.numel() > self.max_samples:
            last_start = waveform.numel() - self.max_samples
            start = random.randint(0, last_start) if self.training else last_start // 2
            waveform = waveform[start : start + self.max_samples]
        elif waveform.numel() < self.max_samples:
            repeats = self.max_samples // waveform.numel()
            waveform = waveform.repeat(repeats)
            waveform = nn.functional.pad(
                waveform, (0, self.max_samples - waveform.numel())
            )

        score = float(row["average_score"])
        if not np.isfinite(score) or not 0 <= score <= 10:
            raise ValueError(f"Invalid average_score in row {index}: {score}")
        caption = str(row["text"])
        if pd.isna(row["text"]) or not caption.strip():
            raise ValueError(f"Missing text in row {index}")

        return {
            "clap_wav": waveform.unsqueeze(0),
            "caption": caption,
            "score": (score - 5.0) / 5.0,
            "score_raw": score,
            "num_class": int(row["num_class"]) if "num_class" in self.rows else 1,
        }


def collate_batch(items):
    return {
        "clap_wavs": torch.stack([item["clap_wav"] for item in items]),
        "caption_tokens": [item["caption"] for item in items],
        "scores": torch.tensor([item["score"] for item in items], dtype=torch.float32),
        "scores_raw": torch.tensor(
            [item["score_raw"] for item in items], dtype=torch.float32
        ),
        "num_class": torch.tensor(
            [item["num_class"] for item in items], dtype=torch.long
        ),
    }


class CLAPAudioTextEncoder(nn.Module):
    """Keep the same module nesting as Basic_comp.CLAP_audio_text_encoder."""

    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__()
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False, device=str(device)
        )
        self.model.load_ckpt(str(checkpoint))

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.model.model.named_parameters():
            if "projection" in name or "mlp" in name:
                parameter.requires_grad_(True)
        for name, parameter in self.model.model.audio_branch.named_parameters():
            if any(part in name for part in ("layers.2", "layers.3", "head", "norm")):
                parameter.requires_grad_(True)

    def forward(self, audio: torch.Tensor, captions: list[str]):
        audio_embedding = self.model.get_audio_embedding_from_data(
            audio.squeeze(1), use_tensor=True
        )
        text_embedding = self.model.get_text_embedding(captions, use_tensor=True)
        return audio_embedding, text_embedding


class Expert1(nn.Module):
    """Same trainable module names and score calculation as Basic_comp.Expert1."""

    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__()
        self.model = CLAPAudioTextEncoder(checkpoint, device)

    def forward(self, batch):
        audio_features, text_features = self.model(
            batch["clap_wavs"], batch["caption_tokens"]
        )
        score = (audio_features * text_features).sum(dim=-1)
        return score


def move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--train-wav-dir", type=Path)
    parser.add_argument("--val-wav-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/laion-clap.pt"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("weights/expert1_best.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print progress every N batches (0 disables batch logs)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1,
        help="Fraction of planned optimizer steps used for linear LR warm-up",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_csv = args.train_csv or args.data_root / "meta_data/train.csv"
    val_csv = args.val_csv or args.data_root / "meta_data/validation.csv"
    train_wav_dir = args.train_wav_dir or args.data_root / "wav/train"
    val_wav_dir = args.val_wav_dir or args.data_root / "wav/validation"
    for path in (args.checkpoint, args.config, train_csv, val_csv, train_wav_dir, val_wav_dir):
        if not path.exists():
            parser.error(f"Missing required path: {path}")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size and --patience must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu or use a GPU machine")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["loss"]["clipped_mse"]["cbl"] or config["loss"]["contrastive"]["cbl"]:
        parser.error("Class-balanced loss requires verified num_class values")
    device = torch.device(args.device)
    print("Reading training and validation CSV files...", flush=True)
    train_loader = DataLoader(
        Expert1Dataset(train_csv, train_wav_dir, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Expert1Dataset(val_csv, val_wav_dir, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    print(f"Loading LAION-CLAP checkpoint: {args.checkpoint}", flush=True)
    expert = Expert1(args.checkpoint, device).to(device)
    trainable = [parameter for parameter in expert.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Expert 1 has no trainable parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(
        f"LR schedule: {warmup_steps} warm-up steps, "
        f"{total_steps - warmup_steps} linear-decay steps "
        f"(peak lr={args.lr:.2e})",
        flush=True,
    )
    loss_fn = get_loss_function(config["loss"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_srcc = float("-inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        expert.train()
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}: training {len(train_loader)} batches", flush=True)
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = expert(batch)
            if not prediction.requires_grad:
                raise RuntimeError("CLAP embeddings are detached; check use_tensor=True")
            loss = loss_fn(prediction, batch["scores"], batch["num_class"])
            loss.backward()
            if epoch == 0 and step == 0 and not any(p.grad is not None for p in trainable):
                raise RuntimeError("No trainable parameter received a gradient")
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if args.log_every and (
                (step + 1) % args.log_every == 0 or step + 1 == len(train_loader)
            ):
                print(
                    f"Epoch {epoch + 1}: train {step + 1}/{len(train_loader)}, "
                    f"mean_loss={total_loss / (step + 1):.5f}, lr={current_lr:.2e}",
                    flush=True,
                )

        expert.eval()
        predictions, targets = [], []
        print(f"Epoch {epoch + 1}: validating {len(val_loader)} batches", flush=True)
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                batch = move_batch(batch, device)
                prediction = expert(batch)
                predictions.extend((prediction * 5.0 + 5.0).cpu().tolist())
                targets.extend(batch["scores_raw"].cpu().tolist())
                if args.log_every and (
                    (val_step + 1) % args.log_every == 0
                    or val_step + 1 == len(val_loader)
                ):
                    print(
                        f"Epoch {epoch + 1}: validation "
                        f"{val_step + 1}/{len(val_loader)}",
                        flush=True,
                    )
        srcc = float(spearmanr(predictions, targets).correlation)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / len(train_loader):.5f}, "
            f"val_srcc={srcc:.5f}, lr={current_lr:.2e}",
            flush=True,
        )
        if np.isfinite(srcc) and srcc > best_srcc:
            best_srcc = srcc
            stale_epochs = 0
            temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
            torch.save(
                {key: value.detach().cpu() for key, value in expert.state_dict().items()},
                temporary_output,
            )
            temporary_output.replace(args.output)
            print(f"Saved {args.output}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping")
                break

    if best_srcc == float("-inf"):
        raise RuntimeError("Validation SRCC was never finite; no checkpoint was saved")
    print(f"Best validation SRCC: {best_srcc:.5f}")


if __name__ == "__main__":
    main()
