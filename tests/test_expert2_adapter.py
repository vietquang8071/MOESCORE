"""Checkpoint-free contract tests for the fine-tuned MGA-CLAP adapter."""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

from model.expert_adapters.expert2_adapter import (
    Expert2Adapter,
    MGAImportNamespaceError,
    validate_mga_import_namespace,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SAMPLES = 320_000


class DummyASE(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))
        self.word_proj = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))
        self.codebook = nn.Parameter(torch.ones(8, embedding_dim))


class DummyMGAEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = DummyASE(embedding_dim)


class DummyExpert2(nn.Module):
    def __init__(self, embedding_dim=4, *, invalid_output=None, dropout=False):
        super().__init__()
        self.model = DummyMGAEncoder(embedding_dim)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.audio_values = nn.Parameter(
            torch.arange(1, embedding_dim + 1, dtype=torch.float32),
            requires_grad=True,
        )
        self.text_values = nn.Parameter(
            torch.arange(11, 11 + embedding_dim, dtype=torch.float32),
            requires_grad=True,
        )
        self.invalid_output = invalid_output
        self.calls = 0
        self.last_batch = None

    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        batch_size = batch["mga_wavs"].shape[0]
        audio = self.dropout(self.audio_values.unsqueeze(0).expand(batch_size, -1))
        text = self.text_values.unsqueeze(0).expand(batch_size, -1)
        score = (audio * text).sum(dim=-1, keepdim=True)
        features = torch.cat((audio, text), dim=-1)
        if self.invalid_output == "score_rank":
            score = score.squeeze(-1)
        elif self.invalid_output == "score_matrix":
            score = score.expand(batch_size, batch_size)
        elif self.invalid_output == "feature_width":
            features = features[:, :-1]
        elif self.invalid_output == "feature_rank":
            features = features.unsqueeze(-1)
        elif self.invalid_output == "nonfinite":
            features = features.clone()
            features[0, 0] = float("nan")
        elif self.invalid_output == "batch":
            features = features[:0]
        elif self.invalid_output == "device":
            features = features.to("meta")
        return score, features


def make_batch(batch_size=1, *, audio=None, captions=None):
    if audio is None:
        audio = torch.zeros(batch_size, 1, AUDIO_SAMPLES, dtype=torch.float32)
    if captions is None:
        captions = [f"  Caption {index}  " for index in range(batch_size)]
    return {"expert2_wavs": audio, "captions_raw": captions}


class Expert2AdapterImportTests(unittest.TestCase):
    def test_import_is_lightweight_and_does_not_mutate_sys_path(self):
        script = (
            "import os, sys; cwd=os.getcwd(); paths=list(sys.path); "
            "import model.expert_adapters; "
            "import model.expert_adapters.expert2_adapter; "
            "assert 'models' not in sys.modules; "
            "assert 'expert2_finetune.expert2_model' not in sys.modules; "
            "assert os.getcwd()==cwd and sys.path==paths"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Expert 2 import must remain lightweight. stdout={result.stdout}\n"
            f"stderr={result.stderr}",
        )

    def test_namespace_checker_accepts_vendor_modules_from_mga_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "MGA-CLAP"
            models = source / "models"
            tools = source / "tools"
            models.mkdir(parents=True)
            tools.mkdir()
            registry = {
                "models": _module(models / "__init__.py", package_path=models),
                "models.ase_model": _module(models / "ase_model.py"),
                "tools": _module(tools / "__init__.py", package_path=tools),
                "tools.losses": _module(tools / "losses.py"),
            }
            validate_mga_import_namespace(source, modules=registry)

    def test_namespace_checker_rejects_wrong_models_and_tools_origins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MGA-CLAP"
            other = root / "expert4_vendor"
            source.mkdir()
            (other / "models").mkdir(parents=True)
            (other / "tools").mkdir()
            for name, file_path in (
                ("models", other / "models" / "__init__.py"),
                ("tools", other / "tools" / "__init__.py"),
            ):
                with self.subTest(namespace=name):
                    registry = {name: _module(file_path, package_path=file_path.parent)}
                    with self.assertRaisesRegex(MGAImportNamespaceError, name):
                        validate_mga_import_namespace(source, modules=registry)

    def test_namespace_checker_does_not_mutate_module_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "MGA-CLAP"
            models = source / "models"
            models.mkdir(parents=True)
            registry = {"models": _module(models / "__init__.py", package_path=models)}
            before = dict(registry)
            validate_mga_import_namespace(source, modules=registry)
            self.assertEqual(registry, before)

class Expert2AdapterForwardTests(unittest.TestCase):
    def test_forwards_once_and_preserves_score_features_and_caption_alignment(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                model = DummyExpert2(embedding_dim=5)
                adapter = Expert2Adapter(model)
                batch = make_batch(batch_size)
                score, features = adapter(batch)

                audio = model.audio_values.detach().expand(batch_size, -1)
                text = model.text_values.detach().expand(batch_size, -1)
                expected_score = (audio * text).sum(dim=-1, keepdim=True)
                expected_features = torch.cat((audio, text), dim=-1)
                self.assertTrue(torch.equal(score, expected_score))
                self.assertTrue(torch.equal(features, expected_features))
                self.assertEqual(tuple(score.shape), (batch_size, 1))
                self.assertEqual(tuple(features.shape), (batch_size, 10))
                self.assertEqual(adapter.feature_dim, 10)
                self.assertEqual(model.calls, 1)
                self.assertIs(model.last_batch["mga_wavs"], batch["expert2_wavs"])
                self.assertEqual(
                    model.last_batch["caption_tokens"],
                    [f"Caption {index}" for index in range(batch_size)],
                )
                self.assertTrue(torch.isfinite(score).all())
                self.assertTrue(torch.isfinite(features).all())

    def test_derives_width_from_both_msc_projection_outputs_and_codebook(self):
        for dimension in (3, 7):
            with self.subTest(dimension=dimension):
                adapter = Expert2Adapter(DummyExpert2(embedding_dim=dimension))
                self.assertEqual(adapter.feature_dim, 2 * dimension)

    def test_rejects_explicit_feature_width_that_disagrees_with_model(self):
        with self.assertRaisesRegex(ValueError, "feature_dim.*metadata"):
            Expert2Adapter(DummyExpert2(embedding_dim=5), feature_dim=9)

    def test_rejects_non_module_or_unknown_feature_width(self):
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            Expert2Adapter(object(), feature_dim=6)
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            Expert2Adapter(nn.Identity())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Expert2Adapter(DummyExpert2(), feature_dim=0)

    def test_adapter_preserves_original_model_score_without_recalculation(self):
        model = DummyExpert2(embedding_dim=4)
        adapter = Expert2Adapter(model)
        expected_score = (model.audio_values * model.text_values).sum().reshape(1, 1)
        score, _ = adapter(make_batch())
        self.assertTrue(torch.equal(score, expected_score))
        self.assertEqual(model.calls, 1)

    def test_shared_validator_rejects_invalid_original_outputs(self):
        for invalid_output, message in (
            ("score_rank", "score must have shape"),
            ("score_matrix", "score must have shape"),
            ("feature_width", "features width mismatch"),
            ("feature_rank", "features must have shape"),
            ("nonfinite", "finite values"),
            ("batch", "batch dimension mismatch"),
            ("device", "same device"),
        ):
            with self.subTest(invalid_output=invalid_output):
                adapter = Expert2Adapter(
                    DummyExpert2(embedding_dim=4, invalid_output=invalid_output)
                )
                with self.assertRaisesRegex(ValueError, message):
                    adapter(make_batch(2 if invalid_output == "score_matrix" else 1))


class Expert2AdapterInputTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Expert2Adapter(DummyExpert2())

    def test_rejects_missing_fields_and_legacy_caption_alias(self):
        with self.assertRaisesRegex(KeyError, "expert2_wavs"):
            self.adapter({"captions_raw": ["caption"]})
        with self.assertRaisesRegex(KeyError, "captions_raw"):
            self.adapter({"expert2_wavs": make_batch()["expert2_wavs"]})
        with self.assertRaisesRegex(KeyError, "captions_raw"):
            self.adapter(
                {"expert2_wavs": make_batch()["expert2_wavs"], "caption_tokens": ["x"]}
            )

    def test_rejects_wrong_audio_rank_channel_or_length(self):
        captions = ["caption"]
        invalid_audio = (
            (torch.zeros(1, AUDIO_SAMPLES), "rank 3"),
            (torch.zeros(1, 2, AUDIO_SAMPLES), "one channel"),
            (torch.zeros(1, 1, AUDIO_SAMPLES - 1), "320000 samples"),
        )
        for audio, message in invalid_audio:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter({"expert2_wavs": audio, "captions_raw": captions})

    def test_rejects_empty_batch_wrong_dtype_and_nonfinite_waveforms(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.adapter(
                {"expert2_wavs": torch.zeros(0, 1, AUDIO_SAMPLES), "captions_raw": []}
            )
        with self.assertRaisesRegex(TypeError, "float32"):
            self.adapter(
                {
                    "expert2_wavs": torch.zeros(1, 1, AUDIO_SAMPLES, dtype=torch.int16),
                    "captions_raw": ["caption"],
                }
            )
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.adapter(
                        {
                            "expert2_wavs": torch.full(
                                (1, 1, AUDIO_SAMPLES), value, dtype=torch.float32
                            ),
                            "captions_raw": ["caption"],
                        }
                    )

    def test_rejects_caption_count_types_tokens_and_empty_text(self):
        audio = make_batch()["expert2_wavs"]
        invalid_captions = (
            (["one", "two"], "exactly 1 caption"),
            (torch.ones(1, 4, dtype=torch.long), "sequence of raw strings"),
            ([3], "every caption must be a string"),
            (["   "], "empty after Expert 2 whitespace stripping"),
            ("caption", "sequence of raw strings"),
        )
        for captions, message in invalid_captions:
            with self.subTest(captions=repr(captions)):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.adapter({"expert2_wavs": audio, "captions_raw": captions})

    def test_rejects_audio_on_a_different_device_from_wrapped_model(self):
        audio = torch.empty((1, 1, AUDIO_SAMPLES), device="meta", dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "same device"):
            self.adapter({"expert2_wavs": audio, "captions_raw": ["caption"]})


class Expert2AdapterFreezeTests(unittest.TestCase):
    def test_freezes_every_parameter_and_keeps_all_modules_eval_across_train_calls(self):
        model = DummyExpert2(dropout=True)
        adapter = Expert2Adapter(model)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
            0,
        )
        for mode in (True, False, True):
            self.assertIs(adapter.train(mode), adapter)
            self.assertTrue(all(not module.training for module in model.modules()))
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_no_grad_forward_returns_valid_tensors_and_repeat_is_deterministic(self):
        adapter = Expert2Adapter(DummyExpert2(dropout=True))
        with torch.no_grad():
            first = adapter(make_batch(2))
            second = adapter(make_batch(2))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(first[0].requires_grad)
        self.assertFalse(first[1].requires_grad)


class Expert2AdapterFactoryTests(unittest.TestCase):
    def _resources(self, root, *, finetuned_state=None, freeze_policy="partial"):
        source = root / "MGA-CLAP"
        (source / "models").mkdir(parents=True)
        (source / "models" / "ase_model.py").write_text("# test fixture\n", encoding="utf-8")
        architecture = root / "inference.yaml"
        architecture.write_text(
            yaml.safe_dump(
                {
                    "embed_size": 4,
                    "audio_encoder_args": {
                        "type": "transformer",
                        "model": "htsat",
                        "pretrained": False,
                        "freeze": False,
                    },
                    "text_encoder_args": {"type": "bert-base-uncased", "freeze": False},
                    "training": {"spec_augmentation": True},
                }
            ),
            encoding="utf-8",
        )
        training = root / "training.yaml"
        training.write_text(yaml.safe_dump({"freeze_policy": freeze_policy}), encoding="utf-8")
        pretrained = root / "mga-pretrained.pt"
        pretrained.write_bytes(b"trusted test fixture; not deserialized by stub")
        finetuned = root / "expert2-best.pt"
        if finetuned_state is None:
            finetuned_state = DummyExpert2().state_dict()
        torch.save(finetuned_state, finetuned)
        return source, architecture, training, pretrained, finetuned

    def _factory_patch(self, model=None, *, calls=None):
        if model is None:
            model = DummyExpert2()

        def create(source, architecture, pretrained, device, freeze_policy):
            if calls is not None:
                calls.append((source, architecture, pretrained, device, freeze_policy))
            return model

        return patch(
            "model.expert_adapters.expert2_adapter._create_original_expert2",
            side_effect=create,
        )

    def test_factory_failure_reports_wrong_vendor_origin_loaded_during_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._resources(root)
            conflicting_root = root / "other-vendor" / "models"
            conflicting_root.mkdir(parents=True)
            wrong_module = _module(
                conflicting_root / "__init__.py", package_path=conflicting_root
            )
            factory_calls = []

            def failing_factory(*_args, **_kwargs):
                factory_calls.append(True)
                sys.modules["models"] = wrong_module
                raise ImportError("construction failed after conflicting module import")

            with patch.dict(sys.modules, {}, clear=False):
                with patch(
                    "model.expert_adapters.expert2_adapter._create_original_expert2",
                    side_effect=failing_factory,
                ):
                    with self.assertRaisesRegex(MGAImportNamespaceError, "outside MGA source"):
                        Expert2Adapter.from_finetuned(
                            source_dir=paths[0], architecture_yaml=paths[1],
                            training_config=paths[2], pretrained_checkpoint=paths[3],
                            finetuned_checkpoint=paths[4], device="cpu",
                        )
            self.assertEqual(factory_calls, [True])

    def test_factory_restores_raw_full_state_strictly_and_freezes_after_partial_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = DummyExpert2()
            expected_state = {key: torch.full_like(value, 0.25) for key, value in template.state_dict().items()}
            paths = self._resources(root, finetuned_state=expected_state)
            model = DummyExpert2()
            calls = []
            with self._factory_patch(model, calls=calls):
                adapter = Expert2Adapter.from_finetuned(
                    source_dir=paths[0],
                    architecture_yaml=paths[1],
                    training_config=paths[2],
                    pretrained_checkpoint=paths[3],
                    finetuned_checkpoint=paths[4],
                    device="cpu",
                    offline=True,
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][-1], "partial")
            self.assertEqual(calls[0][1]["embed_size"], 4)
            self.assertEqual(calls[0][1]["training"]["spec_augmentation"], True)
            self.assertTrue(all(torch.equal(adapter.model.state_dict()[k], v) for k, v in expected_state.items()))
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
            self.assertTrue(all(not module.training for module in adapter.model.modules()))

    def test_factory_rejects_training_freeze_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._resources(Path(temporary), freeze_policy="partial")
            with self._factory_patch() as factory:
                with self.assertRaisesRegex(ValueError, "freeze_policy.*does not match"):
                    Expert2Adapter.from_finetuned(
                        source_dir=paths[0], architecture_yaml=paths[1],
                        training_config=paths[2], pretrained_checkpoint=paths[3],
                        finetuned_checkpoint=paths[4], freeze_policy="full", device="cpu",
                    )
                factory.assert_not_called()

    def test_factory_rejects_missing_resources_before_original_model_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._resources(root)
            missing_cases = (
                {"source_dir": root / "absent-source"},
                {"architecture_yaml": root / "absent-architecture.yaml"},
                {"training_config": root / "absent-training.yaml"},
                {"pretrained_checkpoint": root / "absent-pretrained.pt"},
                {"finetuned_checkpoint": root / "absent-finetuned.pt"},
            )
            base = dict(
                source_dir=paths[0], architecture_yaml=paths[1], training_config=paths[2],
                pretrained_checkpoint=paths[3], finetuned_checkpoint=paths[4], device="cpu",
            )
            for override in missing_cases:
                with self.subTest(field=next(iter(override))):
                    with self._factory_patch() as factory:
                        with self.assertRaises((FileNotFoundError, NotADirectoryError)):
                            Expert2Adapter.from_finetuned(**(base | override))
                        factory.assert_not_called()

    def test_factory_rejects_relative_paths_and_unsupported_structured_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            Expert2Adapter.from_finetuned(
                source_dir="relative/MGA-CLAP", architecture_yaml="x.yaml",
                training_config="t.yaml", pretrained_checkpoint="p.pt",
                finetuned_checkpoint="f.pt",
            )
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._resources(Path(temporary), finetuned_state={"model": {"x": torch.ones(1)}})
            with self._factory_patch():
                with self.assertRaisesRegex(ValueError, "raw tensor state_dict"):
                    Expert2Adapter.from_finetuned(
                        source_dir=paths[0], architecture_yaml=paths[1],
                        training_config=paths[2], pretrained_checkpoint=paths[3],
                        finetuned_checkpoint=paths[4], device="cpu",
                    )

    def test_factory_rejects_missing_unexpected_and_incompatible_state(self):
        good = DummyExpert2().state_dict()
        cases = (
            ({key: value for index, (key, value) in enumerate(good.items()) if index > 0}, "Missing key"),
            ({**good, "unexpected.weight": torch.ones(1)}, "Unexpected key"),
            ({**good, "audio_values": torch.ones(3)}, "size mismatch"),
        )
        for state, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                paths = self._resources(Path(temporary), finetuned_state=state)
                with self._factory_patch():
                    with self.assertRaisesRegex(RuntimeError, message):
                        Expert2Adapter.from_finetuned(
                            source_dir=paths[0], architecture_yaml=paths[1],
                            training_config=paths[2], pretrained_checkpoint=paths[3],
                            finetuned_checkpoint=paths[4], device="cpu",
                        )

    def test_from_config_accepts_full_resolved_config_and_scopes_resources_to_expert2(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._resources(Path(temporary))
            config = {
                "runtime": {"device": "cpu", "offline": True},
                "experts": {
                    "expert2": {
                        "source_dir": paths[0],
                        "architecture_yaml": paths[1],
                        "training_config": paths[2],
                        "pretrained_checkpoint": paths[3],
                        "finetuned_checkpoint": paths[4],
                        "freeze_policy": "partial",
                    }
                },
            }
            with self._factory_patch():
                adapter = Expert2Adapter.from_config(config)
            self.assertEqual(adapter.feature_dim, 8)
            config["experts"]["expert2"]["finetuned_checkpoint"] = paths[4].parent / "missing.pt"
            with self._factory_patch() as factory:
                with self.assertRaisesRegex(FileNotFoundError, "fine-tuned"):
                    Expert2Adapter.from_config(config)
                factory.assert_not_called()


def _module(file_path, *, package_path=None):
    module = types.ModuleType(file_path.stem)
    module.__file__ = str(file_path)
    if package_path is not None:
        module.__path__ = [str(package_path)]
    return module


if __name__ == "__main__":
    unittest.main()
