"""Fixed 30-epoch E6 training on official train and validation splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.moe_expert_cache import validate_expert_cache_identity
from data.audio_overlap import (
    build_train_validation_audio_overlap_diagnostic,
    emit_audio_overlap_warning,
    raise_for_audio_overlap,
    validate_audio_overlap_diagnostic,
)
from data.set_b_cached_dataset import SetBCachedDataset
from model.composite_experts.set_b_residual import feature_order_contract
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter


SELECTION_POLICY = "official_validation_mse_min_v1"
CHECKPOINT_FORMAT_VERSION = 2
SCORE_NORMALIZATION = {"raw_to_normalized": "(raw - 5.0) / 5.0", "normalized_to_raw": "5.0 * score + 5.0"}


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"} or value == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"E6 runtime device unavailable: {value}")
    return torch.device(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {"component_outputs": {"top2_laion": {key: value.to(device) for key, value in batch["component_outputs"]["top2_laion"].items()}},
            "whisper_audio": batch["whisper_audio"].to(device), "whisper_text": batch["whisper_text"].to(device)}


def _training_loader(dataset, *, batch_size, num_workers, generator):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=num_workers, generator=generator)


def _evaluation_loader(dataset, *, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=num_workers)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _laion_cache_id(identity, *, context):
    try:
        cache_id = validate_expert_cache_identity(identity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"E6 {context} LAION cache identity is missing or invalid: {exc}") from exc
    if identity["expert_name"] != "top2_laion":
        raise ValueError(f"E6 {context} LAION cache identity names a different expert")
    return cache_id


def _source_audio_ids(records, *, split):
    by_hash = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"E6 {split} source audio record is invalid")
        digest = record.get("wav_sha256")
        sample_id = record.get("sample_id")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(sample_id, str):
            raise ValueError(f"E6 {split} source audio identity is missing or invalid")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"E6 {split} source audio SHA-256 is invalid")
        by_hash.setdefault(digest.lower(), []).append(sample_id)
    return by_hash


def load_set_b_checkpoint(path, *, model, expected_dataset=None, optimizer=None, map_location="cpu",
                          for_resume=False):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("unsupported E6 checkpoint")
    if for_resume and checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "legacy E6 checkpoint cannot resume because it does not record the explicit train/validation audio-overlap policy"
        )
    if checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION:
        settings = checkpoint.get("training_settings")
        diagnostic = validate_audio_overlap_diagnostic(checkpoint.get("audio_overlap_diagnostic"))
        saved_provenance = checkpoint.get("dataset_provenance")
        membership = checkpoint.get("training_membership")
        if (not isinstance(settings, dict)
                or type(settings.get("allow_train_validation_audio_overlap")) is not bool
                or checkpoint.get("audio_overlap_diagnostic_sha256") != diagnostic["diagnostic_sha256"]
                or diagnostic["allow_train_validation_audio_overlap"]
                != settings["allow_train_validation_audio_overlap"]
                or (diagnostic["overlapping_sha256_count"]
                    and not settings["allow_train_validation_audio_overlap"])):
            raise ValueError("E6 checkpoint overlap policy or diagnostic identity is inconsistent")
        if (not isinstance(saved_provenance, dict) or set(saved_provenance) != {"train", "validation"}
                or not isinstance(membership, dict) or set(membership) != {"train", "validation"}):
            raise ValueError("E6 checkpoint train/validation provenance is incomplete")
        for split in ("train", "validation"):
            provenance = saved_provenance[split]
            row_membership = membership[split]
            whisper_identity = provenance.get("whisper_dataset_identity") if isinstance(provenance, dict) else None
            if (not isinstance(provenance, dict) or provenance.get("split") != split
                    or not isinstance(row_membership, dict)
                    or type(row_membership.get("sample_count")) is not int
                    or row_membership["sample_count"] <= 0
                    or not isinstance(row_membership.get("ordered_sample_ids_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", row_membership["ordered_sample_ids_sha256"]) is None
                    or row_membership.get("source_manifest_sha256") != provenance.get("full9_source_manifest_sha256")
                    or not isinstance(whisper_identity, dict)
                    or whisper_identity.get("sample_count") != row_membership["sample_count"]):
                raise ValueError(f"E6 checkpoint {split} membership or source provenance is invalid")
        if (diagnostic["overlapping_sha256_count"]
                and checkpoint.get("validation_metrics_warning") != diagnostic.get("warning")):
            raise ValueError("E6 checkpoint validation metrics lack the audio-overlap warning")
    if checkpoint.get("architecture_config") != model.head.architecture_config or checkpoint.get("feature_contract") != feature_order_contract():
        raise ValueError("E6 architecture or feature contract mismatch")
    if checkpoint.get("score_normalization") != SCORE_NORMALIZATION or checkpoint.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("E6 score normalization or selection policy mismatch")
    if expected_dataset is not None:
        saved = checkpoint.get("dataset_provenance", {})
        split = expected_dataset.split
        if split in saved:
            if saved[split] != expected_dataset.provenance:
                raise ValueError(f"E6 {split} cache provenance mismatch")
        else:
            saved_train = saved.get("train") if isinstance(saved, dict) else None
            if not isinstance(saved_train, dict):
                raise ValueError("E6 inference requires saved training provenance")
            saved_laion_id = _laion_cache_id(saved_train.get("laion_cache_identity"), context="saved training")
            expected_laion_id = _laion_cache_id(expected_dataset.laion_cache_identity, context=f"{split} inference")
            if saved_laion_id != expected_laion_id:
                raise ValueError("E6 inference LAION cache identity mismatch")
            if (saved_train.get("laion_model_identity") != expected_dataset.laion_model_identity
                    or saved_train.get("whisper_checkpoint_identity") != expected_dataset.whisper_checkpoint_identity
                    or saved_train.get("whisper_preprocessing_identity") != expected_dataset.whisper_preprocessing_identity):
                raise ValueError("E6 inference component provenance mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class SetBResidualTrainer:
    def __init__(self, model: SetBResidualAdapter, *, learning_rate=1e-3, weight_decay=1e-4,
                 gradient_clip=1.0, seed=42, device="auto"):
        if not isinstance(model, SetBResidualAdapter):
            raise TypeError("model must be SetBResidualAdapter")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (learning_rate, weight_decay, gradient_clip)):
            raise ValueError("E6 optimizer settings must be finite numbers")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0 or type(seed) is not int or seed < 0:
            raise ValueError("invalid E6 training settings")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.seed = seed
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.global_step = 0
        self.completed_epochs = 0
        self.best_validation_mse = math.inf
        self.best_epoch = 0
        self.audio_overlap_diagnostic = None

    def evaluate_mse(self, dataset, *, batch_size=8, num_workers=0):
        self.model.eval()
        total = 0.0
        count = 0
        loader = _evaluation_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        with torch.inference_mode():
            for raw in loader:
                target = raw["target_normalized"].to(self.device)
                if target.dtype != torch.float32 or target.ndim != 1 or not torch.isfinite(target).all():
                    raise ValueError("E6 evaluation target must be finite float32 [B]")
                prediction, _ = self.model(move_batch(raw, self.device))
                loss = torch.nn.functional.mse_loss(prediction, target.unsqueeze(1), reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("E6 evaluation MSE is non-finite")
                total += float(loss)
                count += target.shape[0]
        if count == 0:
            raise ValueError("E6 evaluation dataset is empty")
        return total / count

    def fit(self, train_dataset, validation_dataset, *, output_dir, epochs=30, batch_size=8,
            num_workers=0, resume_from=None, allow_train_validation_audio_overlap=False):
        if not isinstance(train_dataset, SetBCachedDataset) or train_dataset.split != "train":
            raise ValueError("E6 requires official train dataset")
        if not isinstance(validation_dataset, SetBCachedDataset) or validation_dataset.split != "validation":
            raise ValueError("E6 requires official validation dataset")
        if epochs != 30 or type(batch_size) is not int or batch_size <= 0 or type(num_workers) is not int or num_workers < 0:
            raise ValueError("E6 requires exactly 30 epochs, positive batch_size and nonnegative num_workers")
        if set(train_dataset.sample_ids) & set(validation_dataset.sample_ids):
            raise ValueError("E6 train/validation sample identities overlap")
        if type(allow_train_validation_audio_overlap) is not bool:
            raise ValueError("E6 allow_train_validation_audio_overlap must be boolean")
        overlap_diagnostic = build_train_validation_audio_overlap_diagnostic(
            train_dataset.records, validation_dataset.records,
            allow_train_validation_audio_overlap=allow_train_validation_audio_overlap,
        )
        if overlap_diagnostic["overlapping_sha256_count"] and not allow_train_validation_audio_overlap:
            raise_for_audio_overlap(overlap_diagnostic, stage="E6")
        emit_audio_overlap_warning(overlap_diagnostic, stage="E6 training")
        self.audio_overlap_diagnostic = overlap_diagnostic
        if (train_dataset.laion_model_identity != validation_dataset.laion_model_identity
                or train_dataset.whisper_checkpoint_identity != validation_dataset.whisper_checkpoint_identity
                or train_dataset.whisper_preprocessing_identity != validation_dataset.whisper_preprocessing_identity):
            raise ValueError("E6 train/validation component or preprocessing identities differ")
        train_laion_id = _laion_cache_id(train_dataset.laion_cache_identity, context="train")
        validation_laion_id = _laion_cache_id(validation_dataset.laion_cache_identity, context="validation")
        if train_laion_id != validation_laion_id:
            raise ValueError("E6 train/validation LAION cache identity mismatch")
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        best_path, latest_path, final_path = (output / f"expert6_{role}.pt" for role in ("best", "latest", "final"))
        settings = {"epochs": 30, "batch_size": batch_size, "num_workers": num_workers,
                    "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                    "gradient_clip": self.gradient_clip, "seed": self.seed,
                    "selection_metric": "official_validation_mse", "selection_policy": SELECTION_POLICY,
                    "allow_train_validation_audio_overlap": allow_train_validation_audio_overlap}
        provenance = {"train": train_dataset.provenance, "validation": validation_dataset.provenance}
        membership = {
            split: {
                "sample_count": len(dataset.sample_ids),
                "ordered_sample_ids_sha256": hashlib.sha256(json.dumps(
                    dataset.sample_ids, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "source_manifest_sha256": dataset.provenance["full9_source_manifest_sha256"],
            }
            for split, dataset in (("train", train_dataset), ("validation", validation_dataset))
        }
        history = []
        diagnostic_path = output / "expert6_audio_overlap_diagnostic.json"
        diagnostic_path.write_text(json.dumps(overlap_diagnostic, indent=2, sort_keys=True,
                                              ensure_ascii=False) + "\n", encoding="utf-8")
        if resume_from is not None:
            resume_path = Path(resume_from).expanduser().resolve()
            if resume_path != latest_path:
                raise ValueError("resume requires output_dir/expert6_latest.pt")
            checkpoint = load_set_b_checkpoint(resume_path, model=self.model, optimizer=self.optimizer,
                                               expected_dataset=train_dataset, map_location=self.device,
                                               for_resume=True)
            if (checkpoint.get("checkpoint_role") != "latest" or checkpoint.get("training_settings") != settings
                    or checkpoint.get("dataset_provenance") != provenance
                    or checkpoint.get("training_membership") != membership
                    or checkpoint.get("audio_overlap_diagnostic") != overlap_diagnostic
                    or checkpoint.get("audio_overlap_diagnostic_sha256") != overlap_diagnostic["diagnostic_sha256"]):
                raise ValueError("E6 resume settings or validation cache provenance mismatch")
            if not best_path.is_file() or checkpoint.get("best_checkpoint_sha256") != _sha256(best_path):
                raise ValueError("E6 best checkpoint missing or modified")
            self.completed_epochs = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.best_validation_mse = checkpoint["best_validation_mse"]
            self.best_epoch = checkpoint["best_epoch"]
            history = list(checkpoint["validation_history"])
            if len(history) != self.completed_epochs or self.completed_epochs >= 30:
                raise ValueError("E6 resume epoch/history invalid or training already complete")
        for epoch in range(self.completed_epochs + 1, 31):
            seed_everything(self.seed + epoch - 1)
            generator = torch.Generator().manual_seed(self.seed + epoch - 1)
            loader = _training_loader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      generator=generator)
            self.model.train()
            for raw in loader:
                target = raw["target_normalized"].to(self.device)
                prediction, _ = self.model(move_batch(raw, self.device))
                loss = torch.nn.functional.mse_loss(prediction, target.unsqueeze(1))
                if not torch.isfinite(loss):
                    raise ValueError("E6 training MSE is non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.global_step += 1
            train_mse = self.evaluate_mse(train_dataset, batch_size=batch_size, num_workers=num_workers)
            validation_mse = self.evaluate_mse(validation_dataset, batch_size=batch_size, num_workers=num_workers)
            self.completed_epochs = epoch
            if validation_mse < self.best_validation_mse:
                self.best_validation_mse = validation_mse
                self.best_epoch = epoch
                improved = True
            else:
                improved = False
            metric_row = {"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse}
            if overlap_diagnostic["overlapping_sha256_count"]:
                metric_row["validation_metrics_warning"] = overlap_diagnostic["warning"]
            history.append(metric_row)
            payload = {"format_version": CHECKPOINT_FORMAT_VERSION, "model_state_dict": self.model.state_dict(),
                       "optimizer_state_dict": self.optimizer.state_dict(),
                       "architecture_config": self.model.head.architecture_config,
                       "feature_contract": feature_order_contract(), "score_normalization": SCORE_NORMALIZATION,
                       "epoch": epoch, "global_step": self.global_step, "seed": self.seed,
                       "training_settings": settings, "dataset_provenance": provenance,
                       "training_membership": membership,
                       "audio_overlap_diagnostic": overlap_diagnostic,
                       "audio_overlap_diagnostic_sha256": overlap_diagnostic["diagnostic_sha256"],
                       "train_mse": train_mse, "validation_mse": validation_mse,
                       "best_validation_mse": self.best_validation_mse, "best_epoch": self.best_epoch,
                       "validation_history": list(history), "selection_metric": "official_validation_mse",
                       "selection_policy": SELECTION_POLICY}
            if overlap_diagnostic["overlapping_sha256_count"]:
                payload["validation_metrics_warning"] = overlap_diagnostic["warning"]
            if improved:
                _atomic_save({**payload, "checkpoint_role": "best"}, best_path)
            best_hash = _sha256(best_path)
            _atomic_save({**payload, "checkpoint_role": "latest", "best_checkpoint_sha256": best_hash}, latest_path)
            if epoch == 30:
                _atomic_save({**payload, "checkpoint_role": "final", "best_checkpoint_sha256": best_hash}, final_path)
        return history
