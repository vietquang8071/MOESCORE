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
from training.set_b_trainer import (
    SetBResidualTrainer, _evaluation_loader,
    _training_loader, load_set_b_checkpoint,
)
from training.set_b_config import load_set_b_config


def _small_trainer():
    return SetBResidualTrainer(SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), device="cpu")


@pytest.mark.parametrize("field,changed", [("laion_implementation", "different-extractor"),
                                           ("laion_preprocessing", "different-preprocessing")])
def test_train_validation_reject_different_laion_cache_with_same_checkpoint(tmp_path, field, changed):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation", **{field: changed})
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    assert train.laion_model_identity == validation.laion_model_identity
    assert train.laion_cache_identity["cache_id"] != validation.laion_cache_identity["cache_id"]
    trainer = _small_trainer()
    with pytest.raises(ValueError, match="LAION cache identity mismatch"):
        trainer.fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    assert trainer.global_step == 0


def test_training_rejects_incomplete_laion_identity_before_optimizer(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    validation.laion_cache_identity = {"cache_id": train.laion_cache_identity["cache_id"]}
    trainer = _small_trainer()
    with pytest.raises(ValueError, match="LAION cache identity"):
        trainer.fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    assert trainer.global_step == 0


def test_e6_config_rejects_removed_overlap_policy(tmp_path):
    import yaml
    source = Path(__file__).resolve().parents[1] / "configs/set_b_residual.example.yaml"
    raw = yaml.safe_load(source.read_text())
    for key in ("feature_store_root", "whisper_cache_root", "output_dir"):
        raw[key] = str(tmp_path / key)
    config = tmp_path / "e6.yaml"
    config.write_text(yaml.safe_dump(raw))
    assert "allow_train_validation_audio_overlap" not in load_set_b_config(config)["training"]
    raw["training"]["allow_train_validation_audio_overlap"] = True
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unsupported E6 training setting"):
        load_set_b_config(config)


def test_e6_loaders_retain_all_7500_train_and_3000_validation_rows():
    train_ids = range(7500)
    validation_ids = range(3000)
    train_loader = _training_loader(
        train_ids, batch_size=8, num_workers=0, generator=torch.Generator().manual_seed(4),
    )
    validation_loader = _evaluation_loader(validation_ids, batch_size=8, num_workers=0)
    train_seen = [int(value) for batch in train_loader for value in batch]
    validation_seen = [int(value) for batch in validation_loader for value in batch]
    assert len(train_seen) == len(set(train_seen)) == 7500
    assert len(validation_seen) == len(set(validation_seen)) == 3000


def test_compatible_laion_identity_accepts_different_split_fingerprints(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    assert train.laion_cache_identity["cache_id"] == validation.laion_cache_identity["cache_id"]
    assert train.store.identity != validation.store.identity
    assert train.provenance["full9_source_manifest_sha256"] != validation.provenance["full9_source_manifest_sha256"]
    history = _small_trainer().fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    assert len(history) == 30


def test_unseen_split_rejects_different_laion_cache_before_loading_weights(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    synthetic_caches(tmp_path, split="test", laion_implementation="different-extractor")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    test = SetBCachedDataset(full9, whisper, split="test")
    assert train.laion_model_identity == test.laion_model_identity
    assert train.laion_cache_identity["cache_id"] != test.laion_cache_identity["cache_id"]
    _small_trainer().fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    model = SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="LAION cache identity mismatch"):
        load_set_b_checkpoint(tmp_path / "out/expert6_best.pt", model=model, expected_dataset=test)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    test.laion_cache_identity = {"cache_id": "0" * 64}
    with pytest.raises(ValueError, match="LAION cache identity"):
        load_set_b_checkpoint(tmp_path / "out/expert6_best.pt", model=model, expected_dataset=test)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("bad_identity", [None, {"cache_id": "0" * 64}])
def test_unseen_split_rejects_missing_or_incomplete_saved_laion_identity(tmp_path, bad_identity):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    synthetic_caches(tmp_path, split="test")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    test = SetBCachedDataset(full9, whisper, split="test")
    _small_trainer().fit(train, validation, output_dir=tmp_path / "out", batch_size=2)
    payload = torch.load(tmp_path / "out/expert6_best.pt", weights_only=False)
    if bad_identity is None:
        del payload["dataset_provenance"]["train"]["laion_cache_identity"]
    else:
        payload["dataset_provenance"]["train"]["laion_cache_identity"] = bad_identity
    damaged = tmp_path / "damaged.pt"
    torch.save(payload, damaged)
    with pytest.raises((ValueError, TypeError), match="LAION cache identity|expert cache identity"):
        load_set_b_checkpoint(damaged, model=SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)), expected_dataset=test)


def test_duplicate_wav_content_across_e6_splits_trains_all_rows_without_diagnostics(tmp_path):
    shared = b"accepted experimental duplicate"
    full9, whisper, _ = synthetic_caches(
        tmp_path, split="train", filenames=["03768.wav", "train-only.wav"],
        audio_payloads=[shared, b"train-only"],
    )
    synthetic_caches(
        tmp_path, split="validation", filenames=["07603.wav", "validation-only.wav"],
        audio_payloads=[shared, b"validation-only"],
    )
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    original_train_ids = list(train.sample_ids)
    original_validation_ids = list(validation.sample_ids)
    output = tmp_path / "out"
    trainer = _small_trainer()
    history = trainer.fit(train, validation, output_dir=output, batch_size=2)
    assert len(train) == 2 and len(validation) == 2
    assert train.sample_ids == original_train_ids
    assert validation.sample_ids == original_validation_ids
    assert trainer.global_step == 30
    assert len(history) == 30
    best = torch.load(output / "expert6_best.pt", weights_only=False)
    assert best["format_version"] == 3
    assert "allow_train_validation_audio_overlap" not in best["training_settings"]
    assert "audio_overlap_diagnostic" not in best
    assert best["dataset_provenance"]["train"] == train.provenance
    assert best["dataset_provenance"]["validation"] == validation.provenance


def test_e6_legacy_checkpoint_loads_for_inference_but_cannot_resume(tmp_path):
    full9, whisper, _ = synthetic_caches(tmp_path, split="train")
    synthetic_caches(tmp_path, split="validation")
    train = SetBCachedDataset(full9, whisper, split="train")
    validation = SetBCachedDataset(full9, whisper, split="validation")
    output = tmp_path / "out"
    _small_trainer().fit(train, validation, output_dir=output, batch_size=2)
    latest_path = output / "expert6_latest.pt"
    payload = torch.load(latest_path, weights_only=False)
    payload["format_version"] = 1
    payload["training_membership"] = None
    torch.save(payload, latest_path)
    loaded = load_set_b_checkpoint(
        latest_path, model=SetBResidualAdapter(SetBResidualHead(hidden_dims=(8, 4), dropout=0)),
        expected_dataset=validation,
    )
    assert loaded["format_version"] == 1
    with pytest.raises(ValueError, match="legacy E6 checkpoint.*cannot resume"):
        _small_trainer().fit(
            train, validation, output_dir=output, batch_size=2,
            resume_from=latest_path,
        )


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
    # Version 2 checkpoints from the previous implementation can resume after
    # dropping only the retired overlap setting; optimizer and row provenance
    # remain bound to the same train/validation caches.
    latest["format_version"] = 2
    latest["training_settings"]["allow_train_validation_audio_overlap"] = True
    latest["audio_overlap_diagnostic"] = {"legacy": True}
    latest["audio_overlap_diagnostic_sha256"] = "a" * 64
    torch.save(latest, output / "expert6_latest.pt")
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
