import csv
import hashlib

import pytest
import torch
import numpy as np

from data.dss_dataset import load_aligned_metadata
from training.dss_runner import best_epoch_update, train_step
from model.moe_gating_core import GatingNetwork


def _metadata_fixture(path, *, duplicate=False, wrong_caption=False):
    rows = [
        {"wav_file_name": "b.wav", "text": "second" if not wrong_caption else "wrong",
         "average_score": "7", "model_ID": "1", "model_name": "audiogen"},
        {"wav_file_name": "a.wav", "text": "first", "average_score": "5",
         "model_ID": "0", "model_name": "audiocaps"},
    ]
    if duplicate:
        rows.append(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _manifest():
    return {"shards": [{"sample_ids": ["train:a.wav", "train:b.wav"],
                        "wave_file_names": ["a.wav", "b.wav"]}],
            "assembly_identity": {"ordered_source_fingerprints": [
                {"sample_id": "train:a.wav", "caption_sha256": hashlib.sha256(b"first").hexdigest()},
                {"sample_id": "train:b.wav", "caption_sha256": hashlib.sha256(b"second").hexdigest()},
            ]}}


def test_metadata_join_is_identity_based_and_checks_caption(tmp_path):
    path = tmp_path / "metadata.csv"
    _metadata_fixture(path)
    rows = load_aligned_metadata(path, _manifest(), "train", expected_counts={0: 1, 1: 1})
    assert [r["sample_id"] for r in rows] == ["train:a.wav", "train:b.wav"]
    assert [r["model_id"] for r in rows] == [0, 1]
    _metadata_fixture(path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_aligned_metadata(path, _manifest(), "train", expected_counts={0: 1, 1: 1})
    _metadata_fixture(path, wrong_caption=True)
    with pytest.raises(ValueError, match="caption"):
        load_aligned_metadata(path, _manifest(), "train", expected_counts={0: 1, 1: 1})


def test_preflight_hashes_official_wav_against_cache_fingerprint(tmp_path):
    from types import SimpleNamespace
    from training.dss_runner import verify_official_audio_hashes

    (tmp_path / "meta_data").mkdir()
    wave_root = tmp_path / "wav" / "train"
    wave_root.mkdir(parents=True)
    wave = wave_root / "a.wav"
    wave.write_bytes(b"official audio")
    sample = {"wave_file_name": "a.wav", "sample_id": "train:a.wav"}
    manifest = {"assembly_identity": {"ordered_source_fingerprints": [
        {"sample_id": "train:a.wav", "wav_sha256": hashlib.sha256(wave.read_bytes()).hexdigest()}]}}
    dataset = SimpleNamespace(metadata=[sample], manifest=manifest)
    verify_official_audio_hashes({"train": tmp_path / "meta_data" / "train_average.csv"},
                                 {"train": dataset})
    wave.write_bytes(b"different audio")
    with pytest.raises(ValueError, match="WAV SHA"):
        verify_official_audio_hashes({"train": tmp_path / "meta_data" / "train_average.csv"},
                                     {"train": dataset})


def test_selection_is_strict_and_train_step_updates_fresh_gate():
    assert best_epoch_update(0.5, 0.5) is False
    assert best_epoch_update(float("nan"), 0.5) is False
    assert best_epoch_update(0.51, 0.5) is True
    torch.manual_seed(2027)
    gate = GatingNetwork(33032, 8, profile="full_8")
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-5, weight_decay=1e-4)
    batch = {
        "features": torch.randn(32, 33032),
        "scores": torch.randn(32, 8) * 0.1,
        "labels_raw": torch.linspace(1, 9, 32),
        "model_ids": torch.tensor([0] * 16 + [1] * 16),
    }
    before = gate.net[0].weight.detach().clone()
    record = train_step(gate, optimizer, batch, torch.device("cpu"))
    assert torch.isfinite(gate.net[0].weight).all()
    assert not torch.equal(before, gate.net[0].weight)
    assert record["valid_group_count"] == 2
    assert record["total_loss"] == pytest.approx(0.5 * (record["global_loss"] + record["domain_loss"]))


def test_synthetic_full8_cache_joins_official_rows(tmp_path, monkeypatch):
    from data import dss_dataset
    from data.moe_feature_store import MoeFeatureStoreWriter
    from model.expert_adapters.registry import FULL7_EXPERT_ORDER
    from test_full8_pipeline import _synthetic_full8_assembly

    sample_ids = ["train:a.wav", "train:b.wav"]
    assembly, cache_identity = _synthetic_full8_assembly("train", sample_ids, "2" * 64)
    for row, caption in zip(assembly["ordered_source_fingerprints"], ("first", "second")):
        row["caption_sha256"] = hashlib.sha256(caption.encode()).hexdigest()
    provenance = {name: {"checkpoint_sha256": f"{i + 1:064x}"}
                  for i, name in enumerate(FULL7_EXPERT_ORDER)}
    provenance["expert8"] = cache_identity["checkpoint_provenance"]
    writer = MoeFeatureStoreWriter(
        tmp_path / "store", split="train", expected_sample_ids=sample_ids,
        source_manifest_sha256="2" * 64, checkpoint_provenance=provenance,
        preprocessing_identity={"full8": "fixture"}, profile="full_8", assembly_identity=assembly,
    )
    writer.write_shard(0, sample_ids=sample_ids, wave_file_names=["a.wav", "b.wav"],
                       scores=np.zeros((2, 8), dtype=np.float32),
                       features=np.zeros((2, 33032), dtype=np.float32),
                       labels_raw=np.array([5., 7.], dtype=np.float32))
    writer.finalize()
    path = tmp_path / "metadata.csv"
    _metadata_fixture(path)
    monkeypatch.setitem(dss_dataset.EXPECTED_COUNTS, "train", {0: 1, 1: 1})
    dataset = dss_dataset.DSSDataset(tmp_path / "store", path, split="train")
    batch = dss_dataset.collate_dss([dataset[0], dataset[1]])
    assert batch["sample_ids"] == sample_ids
    assert batch["features"].shape == (2, 33032)
    assert batch["scores"].shape == (2, 8)
    assert batch["model_ids"].tolist() == [0, 1]
    assert batch["labels_raw"].tolist() == [5, 7]


def test_runner_records_failed_status(tmp_path, monkeypatch):
    import json
    from training import dss_runner

    def fail(*args, **kwargs):
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(dss_runner, "_run_dss", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        dss_runner.run_dss("data", "artifacts", tmp_path)
    status = json.loads((tmp_path / "experiment_metadata.json").read_text())
    assert status["status"] == "FAILED"
    assert "preflight failed" in status["failure_reason"]


def test_rng_checkpoint_loads_safely_and_restores_streams(tmp_path):
    import random
    import numpy as np
    from training.dss_runner import _rng_state, _restore_rng, _atomic_save

    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027)
    state = _rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    path = tmp_path / "last.pt"
    _atomic_save({"rng": state}, path)
    loaded = torch.load(path, weights_only=True)
    _restore_rng(loaded["rng"])
    observed = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert observed == expected


def test_two_epoch_runner_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from training import dss_runner
    from torch import nn

    class TinyGate(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.Softmax(dim=-1))

        def forward(self, features):
            return self.net(features)

    class TinyDataset:
        def __init__(self, split, count):
            self.split = split
            self.model_ids = [i // 32 if split == "train" else i % 9 for i in range(count)]
            self.samples = []
            generator = torch.Generator().manual_seed(100 if split == "train" else 200)
            for i in range(count):
                self.samples.append({"features": torch.randn(2, generator=generator),
                                     "scores": torch.randn(8, generator=generator) * .3,
                                     "label_raw": torch.tensor(float(i % 31) / 3),
                                     "model_id": self.model_ids[i], "sample_id": f"{split}:{i}.wav"})

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    datasets = {"train": TinyDataset("train", 288),
                "validation": TinyDataset("validation", 3000),
                "test": TinyDataset("test", 1)}
    report = {"source_commit": "synthetic", "feature_store_identities": {s: s for s in datasets},
              "metadata_sha256": {s: s for s in datasets}, "dataset_root": "synthetic",
              "artifacts_root": "synthetic"}
    monkeypatch.setattr(dss_runner, "preflight", lambda *_: (datasets, report))
    monkeypatch.setattr(dss_runner, "GatingNetwork", TinyGate)

    def collate(samples):
        return {"features": torch.stack([r["features"] for r in samples]),
                "scores": torch.stack([r["scores"] for r in samples]),
                "labels_raw": torch.stack([r["label_raw"] for r in samples]),
                "model_ids": torch.tensor([r["model_id"] for r in samples])}

    monkeypatch.setattr(dss_runner, "collate_dss", collate)

    def predict(gate, dataset, device, **kwargs):
        with torch.inference_mode():
            gate.eval()
            features = torch.stack([r["features"] for r in dataset.samples])
            scores = torch.stack([r["scores"] for r in dataset.samples])
            raw = ((gate(features) * scores).sum(1) * 5 + 5).tolist()
        return [{"sample_id": r["sample_id"], "model_id": r["model_id"],
                 "model_name": str(r["model_id"]), "ground_truth_raw": float(r["label_raw"]),
                 "prediction_raw": p, **{f"gate_weight_{j+1}": .125 for j in range(8)}}
                for r, p in zip(dataset.samples, raw)]

    monkeypatch.setattr(dss_runner, "predict_rows", predict)
    original_train_step = dss_runner.train_step
    calls = 0

    def interrupt_after_first_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 60:
            raise RuntimeError("synthetic interruption")
        return original_train_step(*args, **kwargs)

    interrupted = tmp_path / "interrupted"
    monkeypatch.setattr(dss_runner, "train_step", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        dss_runner.run_dss("unused", "unused", interrupted, run_mode="smoke", device_name="cpu")
    assert (interrupted / "checkpoints/last.pt").is_file()
    monkeypatch.setattr(dss_runner, "train_step", original_train_step)
    resumed = dss_runner.run_dss("unused", "unused", interrupted, run_mode="smoke", device_name="cpu",
                                 resume=interrupted / "checkpoints/last.pt")
    uninterrupted = dss_runner.run_dss("unused", "unused", tmp_path / "uninterrupted",
                                       run_mode="smoke", device_name="cpu")
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"] == 118
    assert resumed["best_epoch"] == uninterrupted["best_epoch"]
    assert resumed["best_validation_srcc"] == uninterrupted["best_validation_srcc"]
    left = torch.load(interrupted / "checkpoints/last.pt", weights_only=True)["gate"]
    right = torch.load(tmp_path / "uninterrupted/checkpoints/last.pt", weights_only=True)["gate"]
    assert all(torch.equal(left[key], right[key]) for key in left)
