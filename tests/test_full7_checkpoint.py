import torch
import pytest

from model.moe_gating_core import GatingNetwork
from model.expert_adapters.registry import profile_contract
from training.checkpoint_manager import MoeCheckpointManager
from training.moe_trainer import MoeGateTrainer
from training.sampler import StatefulRandomSampler


def test_full7_checkpoint_v4_resume_and_provenance(tmp_path):
    gate = GatingNetwork(29959, 7, profile="full_7")
    optimizer = torch.optim.AdamW(gate.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    order, _, _ = profile_contract("full_7")
    provenance = {name: {"sha256": f"{index + 1:064x}"} for index, name in enumerate(order)}
    from data.audio_overlap import build_train_validation_audio_overlap_diagnostic
    diagnostic = build_train_validation_audio_overlap_diagnostic([], [], allow_train_validation_audio_overlap=False)
    identity = {"train": "a" * 64, "validation": "b" * 64,
                "preprocessing_contract": {"score_contracts": {"policy": "synthetic"}},
                "source_wav_sha256_by_split": {"train": ["a" * 64], "validation": ["b" * 64]},
                "source_sample_ids_by_split": {"train": ["train:a.wav"], "validation": ["validation:b.wav"]},
                "audio_overlap_diagnostic": diagnostic}
    manager = MoeCheckpointManager()
    path = tmp_path / "latest_gate.pt"
    manager.save(path, gating_network=gate, optimizer=optimizer, scheduler=scheduler,
                 epoch=1, global_step=3,
                 config={"seed": 2027, "data": {"allow_train_validation_audio_overlap": False}},
                 feature_store_identity=identity, expert_checkpoint_provenance=provenance,
                 sampler_state={"epoch": 1, "cursor": 2})
    saved = torch.load(path, weights_only=True)
    assert saved["format_version"] == 4
    assert saved["expert_profile"] == "full_7"
    assert saved["expert_order"] == list(order)
    assert saved["source_code_sha256"]["model/moe_gating_core.py"]
    restored = GatingNetwork(29959, 7, profile="full_7")
    opt = torch.optim.AdamW(restored.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    loaded = manager.load(path, gating_network=restored, optimizer=opt, scheduler=sched,
                          expected_config={"seed": 2027, "data": {"allow_train_validation_audio_overlap": False}},
                          expected_feature_store_identity=identity,
                          expected_expert_checkpoint_provenance=provenance)
    assert loaded["sampler_state"]["cursor"] == 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        manager.load(path, gating_network=restored, optimizer=opt, scheduler=sched,
                     expected_config={"seed": 2027, "data": {"allow_train_validation_audio_overlap": True}},
                     expected_feature_store_identity=identity,
                     expected_expert_checkpoint_provenance=provenance)
    with pytest.raises(ValueError, match="provenance"):
        manager.load_for_inference(path, gating_network=restored,
            expected_expert_checkpoint_provenance={**provenance, "expert7": {"sha256": "f" * 64}})
    saved["gate_architecture"]["hidden_width"] = 256
    altered = tmp_path / "altered_gate.pt"
    torch.save(saved, altered)
    with pytest.raises(ValueError, match="architecture"):
        manager.load_for_inference(altered, gating_network=restored,
            expected_expert_checkpoint_provenance=provenance)
    saved["gate_architecture"]["hidden_width"] = 512
    saved["source_code_sha256"]["model/moe_gating_core.py"] = "0" * 64
    torch.save(saved, altered)
    with pytest.raises(ValueError, match="source code"):
        manager.load_for_inference(altered, gating_network=restored,
            expected_expert_checkpoint_provenance=provenance)

    from training.checkpoint_manager import _LEGACY_FULL7_SOURCE_CODE_SHA256, _fingerprint
    legacy = torch.load(path, weights_only=True)
    legacy["format_version"] = 3
    legacy["source_code_sha256"] = dict(_LEGACY_FULL7_SOURCE_CODE_SHA256)
    legacy["config"]["data"].pop("allow_train_validation_audio_overlap")
    legacy["training_config"] = legacy["config"]
    legacy["config_sha256"] = _fingerprint(legacy["config"], "config")
    legacy["feature_store_identity"].pop("audio_overlap_diagnostic")
    legacy["feature_store_identity"].pop("source_sample_ids_by_split")
    legacy["feature_store_identity_sha256"] = _fingerprint(
        legacy["feature_store_identity"], "feature_store_identity",
    )
    legacy_path = tmp_path / "legacy_gate.pt"
    torch.save(legacy, legacy_path)
    loaded_legacy = manager.load_for_inference(
        legacy_path, gating_network=restored,
        expected_expert_checkpoint_provenance=provenance,
    )
    assert loaded_legacy["format_version"] == 3
    with pytest.raises(ValueError, match="legacy Full7 gate checkpoint.*cannot resume"):
        manager.load(
            legacy_path, gating_network=restored, optimizer=opt, scheduler=sched,
            expected_config={"seed": 2027},
            expected_feature_store_identity=legacy["feature_store_identity"],
            expected_expert_checkpoint_provenance=provenance,
        )


def test_full7_interrupted_step_matches_continuous_cpu(tmp_path):
    torch.manual_seed(2027)
    gate = GatingNetwork(29959, 7, profile="full_7")
    loss = {"loss_weights": [1.0, 0.5],
            "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
            "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False}}
    trainer = MoeGateTrainer(gate, loss)
    batch = {"split": "train", "expert_profile": "full_7", "labels_scale": "raw_0_to_10",
             "scores": torch.tensor([[0.1] * 7, [0.7] * 7], dtype=torch.float32),
             "features": torch.randn(2, 29959), "labels_raw": torch.tensor([5.0, 8.0])}
    trainer.train_step(batch)
    trainer.scheduler_step(0.2)
    order, _, _ = profile_contract("full_7")
    provenance = {name: {"sha256": f"{index + 1:064x}"} for index, name in enumerate(order)}
    identity = {"train": "a" * 64, "validation": "b" * 64,
                "preprocessing_contract": {"synthetic": True},
                "source_wav_sha256_by_split": {"train": ["a" * 64], "validation": ["b" * 64]},
                "source_sample_ids_by_split": {"train": ["train:a.wav"], "validation": ["validation:b.wav"]}}
    from data.audio_overlap import build_train_validation_audio_overlap_diagnostic
    identity["audio_overlap_diagnostic"] = build_train_validation_audio_overlap_diagnostic(
        [], [], allow_train_validation_audio_overlap=False,
    )
    sampler = StatefulRandomSampler(range(2), seed=2027)
    path = tmp_path / "latest_gate.pt"
    manager = MoeCheckpointManager()
    manager.save(path, gating_network=gate, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
                 epoch=0, global_step=1,
                 config={"seed": 2027, "data": {"allow_train_validation_audio_overlap": False}},
                 feature_store_identity=identity, expert_checkpoint_provenance=provenance,
                 sampler_state=sampler.state_dict())
    trainer.train_step(batch)
    continuous = {key: value.clone() for key, value in gate.state_dict().items()}

    resumed_gate = GatingNetwork(29959, 7, profile="full_7")
    resumed = MoeGateTrainer(resumed_gate, loss)
    state = manager.load(path, gating_network=resumed_gate, optimizer=resumed.optimizer,
                         scheduler=resumed.scheduler,
                         expected_config={"seed": 2027, "data": {"allow_train_validation_audio_overlap": False}},
                         expected_feature_store_identity=identity,
                         expected_expert_checkpoint_provenance=provenance)
    restored_sampler = StatefulRandomSampler(range(2), seed=2027)
    restored_sampler.load_state_dict(state["sampler_state"])
    assert restored_sampler.state_dict() == sampler.state_dict()
    resumed.train_step(batch)
    for key, expected in continuous.items():
        torch.testing.assert_close(resumed_gate.state_dict()[key], expected, rtol=0, atol=0)


def test_drop_last_interval_checkpoint_defers_remainder_only_cursor():
    from scripts.train_moe_gating import _interval_checkpoint_due
    assert not _interval_checkpoint_due(global_step=1, interval=1,
                                        sampler_cursor=8, sample_count=9, batch_size=8)
    assert _interval_checkpoint_due(global_step=1, interval=1,
                                    sampler_cursor=8, sample_count=16, batch_size=8)
