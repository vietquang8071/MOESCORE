import pytest
import torch
import subprocess
import sys
import csv
from pathlib import Path

from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter
from model.composite_experts.set_b_residual import SetBResidualHead
from data.set_b_cached_dataset import SetBCachedDataset
from test_set_b_cached_dataset import synthetic_caches
from training.set_b_trainer import SetBResidualTrainer, load_set_b_checkpoint


def test_training_rejects_wrong_split_before_optimizer(tmp_path):
    trainer = SetBResidualTrainer(SetBResidualAdapter(), device="cpu")
    with pytest.raises(ValueError, match="official train"):
        trainer.fit(object(), object(), output_dir=tmp_path)


def test_trainer_uses_adamw_for_head_only():
    model = SetBResidualAdapter()
    trainer = SetBResidualTrainer(model, device="cpu")
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert {id(p) for group in trainer.optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}


def test_synthetic_30_epoch_training_resume_and_inference_cli(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train", labels=[5.1, 6.2])
    synthetic_caches(tmp_path, split="validation", labels=[5.5, 7.5])
    synthetic_caches(tmp_path, split="test")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    model = SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0.0))
    trainer = SetBResidualTrainer(model, device="cpu")
    output = tmp_path / "out"
    history = trainer.fit(train, validation, output_dir=output, batch_size=2)
    assert len(history) == trainer.completed_epochs == 30
    assert trainer.global_step == 30
    checkpoints = {role: torch.load(output / f"expert6_{role}.pt", weights_only=False)
                   for role in ("best", "latest", "final")}
    assert checkpoints["latest"]["epoch"] == checkpoints["final"]["epoch"] == 30
    assert checkpoints["best"]["epoch"] == min(history, key=lambda row: row["validation_mse"])["epoch"]
    assert checkpoints["latest"]["best_checkpoint_sha256"]
    with pytest.raises(ValueError, match="already complete"):
        trainer.fit(train, validation, output_dir=output, batch_size=2, resume_from=output / "expert6_latest.pt")
    model2 = SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0.0))
    load_set_b_checkpoint(output / "expert6_best.pt", model=model2, expected_dataset=validation)
    config = tmp_path / "e6.yaml"
    config.write_text(f"feature_store_root: {full9}\nwhisper_cache_root: {whisper}\noutput_dir: {output}\n"
                      "runtime:\n  device: cpu\nmodel:\n  hidden_dims: [8, 4]\n  dropout: 0.0\n"
                      "training:\n  batch_size: 2\n", encoding="utf-8")
    csv_path = tmp_path / "scores.csv"
    script = Path(__file__).resolve().parents[1] / "scripts/infer_set_b_residual.py"
    subprocess.run([sys.executable, str(script), "--config", str(config), "--checkpoint",
                    str(output / "expert6_best.pt"), "--split", "test", "--output", str(csv_path)], check=True)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["sample_id"] for r in rows] == ["test:0.wav", "test:1.wav"]
    assert all(float(row["score_raw"]) == pytest.approx(5 * float(row["score_normalized"]) + 5) for row in rows)


def test_partial_resume_preserves_best_and_rejects_changed_validation(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    output = tmp_path / "out"
    trainer = SetBResidualTrainer(SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), device="cpu")
    original = trainer.evaluate_mse
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    trainer.evaluate_mse = interrupted
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.fit(train, validation, output_dir=output, batch_size=2)
    latest = torch.load(output / "expert6_latest.pt", weights_only=False)
    assert latest["epoch"] == 2
    resumed = SetBResidualTrainer(SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), device="cpu")
    history = resumed.fit(train, validation, output_dir=output, batch_size=2,
                          resume_from=output / "expert6_latest.pt")
    assert resumed.completed_epochs == 30
    assert [row["epoch"] for row in history] == list(range(1, 31))
    assert resumed.global_step == 30
    changed = SetBResidualTrainer(SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), device="cpu")
    with pytest.raises(ValueError, match="already complete"):
        changed.fit(train, validation, output_dir=output, batch_size=2,
                    resume_from=output / "expert6_latest.pt")


def test_validation_ties_keep_earliest_best_and_final_is_epoch_30(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    trainer = SetBResidualTrainer(SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), device="cpu")
    trainer.evaluate_mse = lambda dataset, **kwargs: 1.0 if dataset.split == "validation" else 0.5
    trainer.fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    best = torch.load(tmp_path / "out/expert6_best.pt", weights_only=False)
    final = torch.load(tmp_path / "out/expert6_final.pt", weights_only=False)
    assert best["epoch"] == best["best_epoch"] == 1
    assert final["epoch"] == 30
    assert final["best_epoch"] == 1
    assert best["checkpoint_role"] == "best" and final["checkpoint_role"] == "final"
