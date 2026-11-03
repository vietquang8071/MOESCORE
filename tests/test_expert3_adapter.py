import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from model.expert_adapters.expert3_adapter import (
    Expert3Adapter,
    Expert3CheckpointError,
    validate_expert3_module_origin,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
E3_SOURCE = REPO_ROOT / "expert3_finetune" / "XACLE_2027_baselinemodel"
M2D_DIRECTORY = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"


class StubExpert3(nn.Module):
    """Tiny model with deterministic released-assignment features."""

    def __init__(self, dim=3, invalid=None):
        super().__init__()
        self.embedding_dim = dim
        self.cross_feature_dim = 5 * dim
        self.concat_assignment = "released"
        self.model_config = {
            "embedding_dim": dim,
            "cross_feature_dim": 5 * dim,
            "concat_assignment": "released",
            "activation_checkpointing": True,
            "trainable_audio_blocks": [-2, -1],
            "trainable_text_layers": [-2, -1],
            "has_text_projector": True,
        }
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p=0.9)
        self.calls = 0
        self.last_batch = None
        self.invalid = invalid

    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        wavs = batch["wavs"]
        captions = batch["captions"]
        batch_size = wavs.shape[0]
        dim = self.embedding_dim
        # Deliberately distinguish text and audio blocks for ordering checks.
        audio = torch.arange(batch_size * dim, dtype=wavs.dtype, device=wavs.device)
        audio = audio.reshape(batch_size, dim) + 10.0
        text_values = [float(ord(caption[0])) for caption in captions]
        text = torch.tensor(text_values, dtype=wavs.dtype, device=wavs.device)
        text = text[:, None] + torch.arange(dim, dtype=wavs.dtype, device=wavs.device)[None, :]
        # The original source returns (text_emb, audio_emb), then the released
        # assignment binds them as (audio_feats, text_feats).
        audio_feats, text_feats = text, audio
        cosine = (audio_feats * text_feats).sum(dim=-1, keepdim=True)
        features = torch.cat(
            [audio_feats, text_feats, (audio_feats-text_feats).abs(),
             audio_feats*text_feats, cosine.expand(-1, dim)], dim=-1
        )
        score = torch.arange(batch_size, dtype=wavs.dtype, device=wavs.device) * 4.0 - 2.5
        score = score[:, None]
        if self.invalid == "score_rank":
            score = score[:, 0]
        elif self.invalid == "score_nan":
            score[0, 0] = float("nan")
        elif self.invalid == "feature_rank":
            features = features.unsqueeze(1)
        elif self.invalid == "feature_width":
            features = features[:, :-1]
        elif self.invalid == "feature_inf":
            features[0, 0] = float("inf")
        return score, features


def valid_batch(batch_size=1, dim=3):
    wavs = torch.arange(batch_size * 160000, dtype=torch.float32).reshape(
        batch_size, 1, 160000
    )
    captions = [f"  caption-{index}  " for index in range(batch_size)]
    return {"expert3_wavs": wavs, "captions_raw": captions}


def make_factory_files(root: Path):
    base = root / "m2d-base.bin"
    base.write_bytes(b"tiny trusted base fixture")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    config_path = root / "expert3.json"
    config_path.write_text(
        json.dumps({
            "sample_rate": 16000,
            "max_len": 10,
            "activation_checkpointing": True,
            "concat_assignment": "released",
            "m2d_checkpoint_sha256": digest,
        }),
        encoding="utf-8",
    )
    fine = root / "best_model.pt"
    model = FactoryStubExpert(digest, base.stat().st_size)
    payload = {
        "format_version": 1,
        "checkpoint_kind": "best",
        "model_trainable": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "model_config": dict(model.model_config),
        "base_checkpoint_identity": {
            "path": "/training/machine/checkpoint-30.pth",
            "size_bytes": base.stat().st_size,
            "sha256": digest,
            "model_directory": M2D_DIRECTORY,
        },
        "epoch": 3,
        "best_srcc": 0.5,
        "metrics": {"SRCC": 0.5},
        "config": {
            "sample_rate": 16000,
            "max_len": 10,
            "concat_assignment": "released",
            "m2d_checkpoint_sha256": digest,
            "activation_checkpointing": True,
        },
    }
    torch.save(payload, fine)
    return base, fine, config_path, digest, payload


class FactoryStubExpert(nn.Module):
    def __init__(self, digest, base_size):
        super().__init__()
        self.embedding_dim = 1
        self.cross_feature_dim = 5
        self.concat_assignment = "released"
        self.model_config = {
            "embedding_dim": 1,
            "cross_feature_dim": 5,
            "concat_assignment": "released",
            "activation_checkpointing": True,
            "trainable_audio_blocks": [-2, -1],
            "trainable_text_layers": [-2, -1],
            "has_text_projector": True,
        }
        self.alignment_head = nn.Linear(1, 1, bias=False)
        self.m2d_base_parameter = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.base_checkpoint_identity = {
            "path": "/runtime/path/m2d-base.bin",
            "size_bytes": base_size,
            "sha256": digest,
            "model_directory": M2D_DIRECTORY,
        }

    def forward(self, batch):
        raise NotImplementedError


class Expert3AdapterForwardTests(unittest.TestCase):
    def test_forward_contract_batches_released_order_and_single_model_call(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                model = StubExpert3(dim=3)
                adapter = Expert3Adapter(model=model)
                batch = valid_batch(batch_size)
                original_audio = batch["expert3_wavs"]
                score, features = adapter(batch)

                self.assertEqual(model.calls, 1)
                self.assertIs(model.last_batch["wavs"], original_audio)
                self.assertEqual(model.last_batch["captions"], [f"caption-{i}" for i in range(batch_size)])
                self.assertEqual(tuple(score.shape), (batch_size, 1))
                self.assertEqual(tuple(features.shape), (batch_size, 15))
                self.assertEqual(adapter.feature_dim, 15)
                self.assertTrue(torch.equal(score[:, 0], torch.arange(batch_size).float()*4.0-2.5))
                # Released assignment's first block is the model's text embedding.
                expected_text = torch.tensor(
                    [[float(ord(f"caption-{i}"[0])) + j for j in range(3)] for i in range(batch_size)]
                )
                expected_audio = torch.arange(batch_size*3, dtype=torch.float32).reshape(batch_size, 3)+10
                self.assertTrue(torch.equal(features[:, :3], expected_text))
                self.assertTrue(torch.equal(features[:, 3:6], expected_audio))
                expected_cosine = (expected_text * expected_audio).sum(dim=-1, keepdim=True)
                expected_features = torch.cat(
                    [expected_text, expected_audio, (expected_text-expected_audio).abs(),
                     expected_text*expected_audio, expected_cosine.expand(-1, 3)], dim=-1
                )
                self.assertTrue(torch.equal(features, expected_features))
                self.assertTrue(torch.isfinite(features).all())

    def test_score_is_not_bounded_or_transformed(self):
        adapter = Expert3Adapter(StubExpert3())
        score, _ = adapter(valid_batch(2))
        self.assertTrue(torch.equal(score[:, 0], torch.tensor([-2.5, 1.5])))
        self.assertGreater(float(score[1, 0]), 1.0)

    def test_dimension_is_derived_and_explicit_width_is_validated(self):
        model = StubExpert3(dim=7)
        self.assertEqual(Expert3Adapter(model).feature_dim, 35)
        self.assertEqual(Expert3Adapter(model, feature_dim=35).feature_dim, 35)
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            Expert3Adapter(model, feature_dim=2048)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Expert3Adapter(model, feature_dim=0)

    def test_invalid_model_metadata_is_rejected(self):
        model = StubExpert3(dim=3)
        model.cross_feature_dim = 14
        with self.assertRaisesRegex(ValueError, "5 \* embedding_dim"):
            Expert3Adapter(model)
        model = StubExpert3(dim=3)
        model.concat_assignment = "semantic"
        with self.assertRaisesRegex(ValueError, "released"):
            Expert3Adapter(model)
        model = StubExpert3(dim=3)
        del model.embedding_dim
        model.model_config.pop("embedding_dim")
        with self.assertRaisesRegex(ValueError, "metadata"):
            Expert3Adapter(model)

    def test_output_contract_failures_are_reported(self):
        for invalid, message in (
            ("score_rank", "score must have shape"),
            ("score_nan", "finite"),
            ("feature_rank", "features must have shape"),
            ("feature_width", "width mismatch"),
            ("feature_inf", "finite"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, message):
                    Expert3Adapter(StubExpert3(invalid=invalid))(valid_batch())

    def test_input_validation(self):
        adapter = Expert3Adapter(StubExpert3())
        cases = [
            ({"captions_raw": ["x"]}, KeyError),
            ({"expert3_wavs": torch.zeros(1, 1, 160000)}, KeyError),
            ({"expert3_wavs": torch.zeros(1, 160000), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.zeros(1, 2, 160000), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.zeros(1, 1, 159999), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.zeros(1, 1, 160000, dtype=torch.float64), "captions_raw": ["x"]}, TypeError),
            ({"expert3_wavs": torch.zeros(0, 1, 160000), "captions_raw": []}, ValueError),
            ({"expert3_wavs": torch.full((1, 1, 160000), float("nan")), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.full((1, 1, 160000), float("inf")), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.zeros(2, 1, 160000), "captions_raw": ["x"]}, ValueError),
            ({"expert3_wavs": torch.zeros(1, 1, 160000), "captions_raw": torch.tensor([1])}, TypeError),
            ({"expert3_wavs": torch.zeros(1, 1, 160000), "captions_raw": [1]}, TypeError),
            ({"expert3_wavs": torch.zeros(1, 1, 160000), "captions_raw": ["  "]}, ValueError),
        ]
        for batch, error_type in cases:
            with self.subTest(batch_keys=tuple(batch)):
                with self.assertRaises(error_type):
                    adapter(batch)

    def test_model_device_mismatch_is_rejected(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable for the cross-device input validation case")
        adapter = Expert3Adapter(StubExpert3().cuda())
        with self.assertRaisesRegex(ValueError, "same device"):
            adapter(valid_batch())

    def test_freeze_and_eval_survive_parent_train_calls(self):
        torch.manual_seed(2027)
        model = StubExpert3()
        adapter = Expert3Adapter(model)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        adapter.train()
        self.assertTrue(adapter.training)
        self.assertFalse(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        adapter.train(False)
        self.assertFalse(adapter.training)
        self.assertFalse(model.training)
        batch = valid_batch()
        with torch.no_grad():
            score_a, feat_a = adapter(batch)
            score_b, feat_b = adapter(batch)
        self.assertTrue(torch.equal(score_a, score_b))
        self.assertTrue(torch.equal(feat_a, feat_b))

    def test_non_module_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            Expert3Adapter(object())


class Expert3FactoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base, self.fine, self.config, self.digest, self.payload = make_factory_files(self.root)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.temp.cleanup()

    def factory_kwargs(self):
        return {
            "source_dir": E3_SOURCE,
            "base_checkpoint": self.base,
            "finetuned_checkpoint": self.fine,
            "model_config": self.config,
            "cache_dir": self.cache,
            "device": "cpu",
            "offline": True,
        }

    def build_with_payload(self, mutate, message):
        payload = dict(self.payload)
        payload["model_trainable"] = dict(payload["model_trainable"])
        payload["model_config"] = dict(payload["model_config"])
        payload["base_checkpoint_identity"] = dict(payload["base_checkpoint_identity"])
        payload["metrics"] = dict(payload["metrics"])
        payload["config"] = dict(payload["config"])
        mutate(payload)
        torch.save(payload, self.fine)
        with mock.patch(
            "model.expert_adapters.expert3_adapter._construct_original_expert3",
            return_value=FactoryStubExpert(self.digest, self.base.stat().st_size),
        ):
            with self.assertRaisesRegex((ValueError, Expert3CheckpointError), message):
                Expert3Adapter.from_finetuned(**self.factory_kwargs())

    def test_factory_uses_source_loader_and_restores_partial_state(self):
        trained_model = FactoryStubExpert(self.digest, self.base.stat().st_size)
        original_value = trained_model.alignment_head.weight.detach().clone()
        payload = dict(self.payload)
        payload["model_trainable"] = {"alignment_head.weight": torch.tensor([[9.0]])}
        torch.save(payload, self.fine)
        with mock.patch(
            "model.expert_adapters.expert3_adapter._construct_original_expert3",
            return_value=trained_model,
        ) as construct:
            adapter = Expert3Adapter.from_finetuned(**self.factory_kwargs())
        construct.assert_called_once()
        self.assertEqual(float(trained_model.alignment_head.weight), 9.0)
        self.assertNotEqual(float(original_value), 9.0)
        self.assertEqual(adapter.feature_dim, 5)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.model.parameters()))
        self.assertFalse(adapter.model.training)

    def test_source_identity_helper_accepts_only_modules_from_source_tree(self):
        valid = SimpleNamespace(__file__=str(E3_SOURCE / "models" / "moescore_expert3.py"))
        validate_expert3_module_origin(valid, E3_SOURCE, "models.moescore_expert3")
        wrong = SimpleNamespace(__file__="/tmp/other/models/moescore_expert3.py")
        with self.assertRaisesRegex(ImportError, "origin"):
            validate_expert3_module_origin(wrong, E3_SOURCE, "models.moescore_expert3")

    def test_factory_rejects_invalid_partial_checkpoint_contracts(self):
        cases = [
            (lambda p: p.__setitem__("checkpoint_kind", "resume"), "checkpoint_kind"),
            (lambda p: p.__setitem__("format_version", 2), "format_version"),
            (lambda p: p["model_trainable"].clear(), "missing"),
            (lambda p: p["model_trainable"].__setitem__("unexpected", torch.ones(1)), "unexpected"),
            (lambda p: p["model_trainable"].__setitem__("alignment_head.weight", torch.ones(2, 1)), "shape"),
            (lambda p: p["model_config"].__setitem__("concat_assignment", "semantic"), "model_config"),
            (lambda p: p["base_checkpoint_identity"].__setitem__("sha256", "0" * 64), "base SHA-256"),
            (lambda p: p["base_checkpoint_identity"].__setitem__("size_bytes", 1), "size does not match"),
            (lambda p: p["config"].__setitem__("concat_assignment", "semantic"), "concat_assignment"),
            (lambda p: p.__delitem__("metrics"), "top-level keys mismatch"),
            (lambda p: p.__setitem__("unexpected_top_level", True), "top-level"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                make_factory_files(self.root)
                self.base, self.fine, self.config, self.digest, self.payload = make_factory_files(self.root)
                self.build_with_payload(mutate, message)

    def test_factory_rejects_malformed_config_hash_and_assignment(self):
        data = json.loads(self.config.read_text(encoding="utf-8"))
        data["m2d_checkpoint_sha256"] = "not-a-digest"
        self.config.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sha256"):
            with mock.patch("model.expert_adapters.expert3_adapter._construct_original_expert3") as construct:
                Expert3Adapter.from_finetuned(**self.factory_kwargs())
        construct.assert_not_called()

        make_factory_files(self.root)
        self.base, self.fine, self.config, self.digest, self.payload = make_factory_files(self.root)
        kwargs = self.factory_kwargs()
        kwargs["concat_assignment"] = "semantic"
        with self.assertRaisesRegex(ValueError, "released"):
            Expert3Adapter.from_finetuned(**kwargs)

    def test_factory_rejects_conflicting_expected_hash(self):
        kwargs = self.factory_kwargs()
        kwargs["expected_base_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "conflicts"):
            with mock.patch("model.expert_adapters.expert3_adapter._construct_original_expert3") as construct:
                Expert3Adapter.from_finetuned(**kwargs)
        construct.assert_not_called()

    def test_factory_missing_resources_fail_before_model_construction(self):
        for field in ("base_checkpoint", "finetuned_checkpoint", "model_config"):
            kwargs = self.factory_kwargs()
            kwargs[field] = self.root / "missing-resource"
            with self.subTest(field=field):
                with mock.patch("model.expert_adapters.expert3_adapter._construct_original_expert3") as construct:
                    with self.assertRaisesRegex((FileNotFoundError, ValueError), "Expert 3"):
                        Expert3Adapter.from_finetuned(**kwargs)
                construct.assert_not_called()

    def test_factory_requires_absolute_paths_and_offline(self):
        kwargs = self.factory_kwargs()
        kwargs["base_checkpoint"] = "weights/checkpoint-30.pth"
        with self.assertRaisesRegex(ValueError, "absolute"):
            Expert3Adapter.from_finetuned(**kwargs)
        kwargs = self.factory_kwargs()
        kwargs["offline"] = False
        with self.assertRaisesRegex(ValueError, "offline"):
            Expert3Adapter.from_finetuned(**kwargs)

    def test_factory_accepts_a_new_nested_cache_directory(self):
        kwargs = self.factory_kwargs()
        kwargs["cache_dir"] = self.root / "new-cache" / "hf" / "nested"
        model = FactoryStubExpert(self.digest, self.base.stat().st_size)
        with mock.patch(
            "model.expert_adapters.expert3_adapter._construct_original_expert3",
            return_value=model,
        ):
            adapter = Expert3Adapter.from_finetuned(**kwargs)
        self.assertEqual(adapter.feature_dim, 5)
        self.assertFalse(kwargs["cache_dir"].exists())

    def test_factory_rejects_untrusted_source_directory(self):
        kwargs = self.factory_kwargs()
        kwargs["source_dir"] = self.root
        with self.assertRaisesRegex(ValueError, "authoritative checked-in source"):
            Expert3Adapter.from_finetuned(**kwargs)

    def test_from_config_uses_only_expert3_resources(self):
        resolved = {
            "runtime": {"device": "cpu", "offline": True, "cache_dir": self.cache},
            "experts": {"expert3": {
                "source_dir": E3_SOURCE,
                "base_checkpoint": self.base,
                "finetuned_checkpoint": self.fine,
                "model_config": self.config,
                "expected_base_sha256": None,
                "concat_assignment": "released",
            }},
        }
        expected = dict(resolved["experts"]["expert3"])
        with mock.patch.object(Expert3Adapter, "from_finetuned", return_value="adapter") as factory:
            self.assertEqual(Expert3Adapter.from_config(resolved), "adapter")
        args = factory.call_args.kwargs
        self.assertEqual(args["base_checkpoint"], expected["base_checkpoint"])
        self.assertEqual(args["finetuned_checkpoint"], expected["finetuned_checkpoint"])
        self.assertEqual(args["expected_base_sha256"], None)
        self.assertEqual(args["cache_dir"], self.cache)
        self.assertTrue(args["offline"])


class Expert3ImportTests(unittest.TestCase):
    def test_import_is_lightweight_and_does_not_mutate_import_state(self):
        code = r'''
import os, sys
import socket
def blocked_connect(*args, **kwargs):
    raise AssertionError("network access attempted during adapter import")
socket.socket.connect = blocked_connect
before_path = list(sys.path)
before_cwd = os.getcwd()
import model.expert_adapters
from model.expert_adapters.expert3_adapter import Expert3Adapter
assert list(sys.path) == before_path
assert os.getcwd() == before_cwd
assert "expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3" not in sys.modules
assert "expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d" not in sys.modules
assert Expert3Adapter.__name__ == "Expert3Adapter"
'''
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, text=True,
            capture_output=True, timeout=30,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
