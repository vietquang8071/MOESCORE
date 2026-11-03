"""Checkpoint-free tests for the LAION-CLAP Expert 1 adapter contract."""

import os
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from model.expert_adapters.expert1_adapter import Expert1Adapter


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SAMPLES = 480_000


class DummyEncoder(nn.Module):
    def __init__(
        self,
        audio_embeddings,
        text_embeddings,
        *,
        dropout=False,
        metadata_dim=None,
        fixed_batch=False,
    ):
        super().__init__()
        self.register_buffer("audio_embeddings", audio_embeddings)
        self.register_buffer("text_embeddings", text_embeddings)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.calls = 0
        self.last_audio = None
        self.last_captions = None
        self.fixed_batch = fixed_batch
        self.model = SimpleNamespace(
            model=SimpleNamespace(
                joint_embed_shape=(
                    audio_embeddings.shape[-1] if metadata_dim is None else metadata_dim
                )
            )
        )

    def forward(self, audio, captions):
        self.calls += 1
        self.last_audio = audio
        self.last_captions = captions
        batch_size = self.audio_embeddings.shape[0] if self.fixed_batch else audio.shape[0]
        audio_embedding = self.dropout(self.audio_embeddings[:batch_size])
        text_embedding = self.text_embeddings[:batch_size]
        return audio_embedding, text_embedding


class DummyExpert1(nn.Module):
    def __init__(
        self,
        audio_embeddings,
        text_embeddings,
        *,
        dropout=False,
        metadata_dim=None,
        fixed_batch=False,
    ):
        super().__init__()
        self.model = DummyEncoder(
            audio_embeddings,
            text_embeddings,
            dropout=dropout,
            metadata_dim=metadata_dim,
            fixed_batch=fixed_batch,
        )
        self.unused_parameter = nn.Parameter(torch.ones(()))

    def forward(self, batch):
        raise AssertionError("Adapter must extract both embeddings in one encoder call")


def make_adapter(batch_capacity=4, embedding_dim=3, *, dropout=False):
    audio = torch.arange(1, batch_capacity * embedding_dim + 1, dtype=torch.float32)
    audio = audio.reshape(batch_capacity, embedding_dim)
    text = torch.arange(101, 101 + batch_capacity * embedding_dim, dtype=torch.float32)
    text = text.reshape(batch_capacity, embedding_dim)
    expert = DummyExpert1(audio, text, dropout=dropout)
    return Expert1Adapter(expert, embedding_dim=embedding_dim), expert


def make_batch(batch_size=1):
    audio = torch.zeros(batch_size, 1, AUDIO_SAMPLES, dtype=torch.float32)
    captions = [f"  Caption {index}  " for index in range(batch_size)]
    return {"expert1_wavs": audio, "captions_raw": captions}


class Expert1AdapterImportTests(unittest.TestCase):
    def test_module_import_does_not_import_laion_or_construct_models(self):
        script = (
            "import os, sys; before=os.getcwd(); paths=list(sys.path); "
            "import model.expert_adapters; "
            "import model.expert_adapters.expert1_adapter; "
            "assert 'laion_clap' not in sys.modules; "
            "assert 'expert1_finetune.Train_Expert1' not in sys.modules; "
            "assert os.getcwd()==before and sys.path==paths"
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
            f"Lightweight Expert 1 import failed. stdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_offline_hub_context_sets_and_restores_offline_environment(self):
        from huggingface_hub import constants as hub_constants
        from model.expert_adapters.expert1_adapter import _huggingface_offline

        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        previous = {key: os.environ.get(key) for key in keys}
        previous_hub_offline = hub_constants.HF_HUB_OFFLINE
        with _huggingface_offline(True):
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertTrue(hub_constants.HF_HUB_OFFLINE)
        for key in keys:
            self.assertEqual(os.environ.get(key), previous[key])
        self.assertEqual(hub_constants.HF_HUB_OFFLINE, previous_hub_offline)


class Expert1AdapterForwardTests(unittest.TestCase):
    def test_returns_original_dot_score_and_audio_then_text_features(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                adapter, expert = make_adapter()
                batch = make_batch(batch_size)
                score, features = adapter(batch)

                audio_embedding = expert.model.audio_embeddings[:batch_size]
                text_embedding = expert.model.text_embeddings[:batch_size]
                expected_score = (audio_embedding * text_embedding).sum(dim=-1)
                expected_features = torch.cat(
                    [audio_embedding, text_embedding], dim=-1
                )

                self.assertTrue(torch.equal(score.squeeze(-1), expected_score))
                self.assertTrue(torch.equal(features, expected_features))
                self.assertEqual(tuple(score.shape), (batch_size, 1))
                self.assertEqual(tuple(features.shape), (batch_size, 6))
                self.assertEqual(adapter.feature_dim, 6)
                self.assertEqual(expert.model.calls, 1)
                self.assertIs(expert.model.last_audio, batch["expert1_wavs"])
                self.assertEqual(expert.model.last_captions, batch["captions_raw"])
                self.assertEqual(expert.model.last_captions[0], "  Caption 0  ")
                self.assertTrue(torch.isfinite(score).all())
                self.assertTrue(torch.isfinite(features).all())

    def test_embedding_dimension_is_read_from_constructed_clap_metadata(self):
        adapter, _ = make_adapter(embedding_dim=5)
        inferred = Expert1Adapter(make_adapter(embedding_dim=5)[1])
        self.assertEqual(adapter.feature_dim, 10)
        self.assertEqual(inferred.feature_dim, 10)

    def test_explicit_dimension_must_match_model_metadata(self):
        _, expert = make_adapter(embedding_dim=3)
        with self.assertRaisesRegex(ValueError, "metadata.*3"):
            Expert1Adapter(expert, embedding_dim=4)

    def test_inference_mode_output_contract(self):
        adapter, _ = make_adapter()
        with torch.no_grad():
            score, features = adapter(make_batch(2))
        self.assertEqual(tuple(score.shape), (2, 1))
        self.assertEqual(tuple(features.shape), (2, 6))

    def test_rejects_non_module_model_or_unknown_dimension(self):
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            Expert1Adapter(object(), embedding_dim=3)
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            Expert1Adapter(nn.Identity())

    def test_rejects_wrong_embedding_ranks_widths_and_batch_sizes(self):
        cases = [
            (torch.zeros(1, 3, 1), torch.zeros(1, 3), "rank-2"),
            (torch.zeros(1, 4), torch.zeros(1, 3), "width"),
            (torch.zeros(2, 3), torch.zeros(1, 3), "batch dimension"),
        ]
        for audio_embedding, text_embedding, message in cases:
            with self.subTest(message=message):
                expert = DummyExpert1(
                    audio_embedding,
                    text_embedding,
                    metadata_dim=3,
                    fixed_batch=message == "batch dimension",
                )
                adapter = Expert1Adapter(expert, embedding_dim=3)
                with self.assertRaisesRegex(ValueError, message):
                    adapter(make_batch(1))

    def test_rejects_non_finite_encoder_outputs(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                audio = torch.full((1, 3), value)
                text = torch.ones(1, 3)
                adapter = Expert1Adapter(DummyExpert1(audio, text), embedding_dim=3)
                with self.assertRaisesRegex(ValueError, "finite"):
                    adapter(make_batch())

    def test_rejects_embedding_dtype_or_device_mismatch(self):
        audio = torch.ones(1, 3, dtype=torch.float32)
        for text, exception, message in (
            (torch.ones(1, 3, dtype=torch.float64), TypeError, "same dtype"),
            (torch.ones(1, 3, device="meta"), ValueError, "same device"),
        ):
            with self.subTest(message=message):
                adapter = Expert1Adapter(
                    DummyExpert1(audio, text, metadata_dim=3), embedding_dim=3
                )
                with self.assertRaisesRegex(exception, message):
                    adapter(make_batch())


class Expert1AdapterInputTests(unittest.TestCase):
    def setUp(self):
        self.adapter, _ = make_adapter()

    def test_rejects_missing_batch_fields(self):
        with self.assertRaisesRegex(KeyError, "expert1_wavs"):
            self.adapter({"captions_raw": ["text"]})
        with self.assertRaisesRegex(KeyError, "captions_raw"):
            self.adapter({"expert1_wavs": make_batch()["expert1_wavs"]})

    def test_rejects_audio_with_wrong_rank_channel_or_length(self):
        captions = ["caption"]
        invalid = [
            (torch.zeros(1, AUDIO_SAMPLES), "rank-3"),
            (torch.zeros(1, 2, AUDIO_SAMPLES), "one channel"),
            (torch.zeros(1, 1, AUDIO_SAMPLES - 1), "480000 samples"),
        ]
        for audio, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter({"expert1_wavs": audio, "captions_raw": captions})

    def test_rejects_empty_batch_non_float32_and_non_finite_audio(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.adapter({"expert1_wavs": torch.zeros(0, 1, AUDIO_SAMPLES), "captions_raw": []})
        with self.assertRaisesRegex(TypeError, "float32"):
            self.adapter(
                {"expert1_wavs": torch.zeros(1, 1, AUDIO_SAMPLES, dtype=torch.int16), "captions_raw": ["x"]}
            )
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.adapter(
                        {"expert1_wavs": torch.full((1, 1, AUDIO_SAMPLES), value), "captions_raw": ["x"]}
                    )

    def test_rejects_invalid_caption_container_count_and_content(self):
        audio = torch.zeros(1, 1, AUDIO_SAMPLES)
        invalid_captions = [
            (torch.tensor([[1, 2]]), "sequence of strings"),
            (["one", "two"], "same batch size"),
            ([1], "must be a string"),
            (["   "], "empty caption"),
            ("not a sequence of captions", "sequence of strings"),
        ]
        for captions, message in invalid_captions:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.adapter({"expert1_wavs": audio, "captions_raw": captions})


class Expert1AdapterFreezeTests(unittest.TestCase):
    def test_model_is_frozen_and_stays_eval_when_parent_enters_train_mode(self):
        adapter, expert = make_adapter(dropout=True)
        self.assertTrue(all(not parameter.requires_grad for parameter in expert.parameters()))
        self.assertFalse(expert.training)
        self.assertFalse(expert.model.dropout.training)
        adapter.train()
        self.assertTrue(adapter.training)
        self.assertFalse(expert.training)
        self.assertFalse(expert.model.dropout.training)
        adapter.eval()
        self.assertFalse(adapter.training)
        self.assertFalse(expert.training)

    def test_forward_is_deterministic_after_adapter_train_call(self):
        adapter, _ = make_adapter(dropout=True)
        adapter.train()
        first = adapter(make_batch(2))
        second = adapter(make_batch(2))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(
            [name for name, param in adapter.named_parameters() if param.requires_grad], []
        )


class Expert1AdapterCheckpointTests(unittest.TestCase):
    def test_rejects_invalid_resource_path_type(self):
        with self.assertRaisesRegex(TypeError, "pretrained_checkpoint"):
            Expert1Adapter.from_finetuned(None, Path("/unused/fine.pt"))

    def test_missing_pretrained_or_finetuned_resources_fail_before_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "pretrained.pt"
            fine = root / "finetuned.pt"
            with self.assertRaisesRegex(FileNotFoundError, "Expert 1.*pretrained_checkpoint"):
                Expert1Adapter.from_finetuned(base, fine)
            base.touch()
            with self.assertRaisesRegex(FileNotFoundError, "Expert 1.*finetuned_checkpoint"):
                Expert1Adapter.from_finetuned(base, fine)

    def test_config_factory_checks_only_expert1_and_requires_resolved_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "pretrained.pt"
            fine = root / "finetuned.pt"
            base.touch()
            fine.touch()
            resolved = {
                "runtime": {"device": "cpu"},
                "experts": {
                    "expert1": {
                        "pretrained_checkpoint": base,
                        "finetuned_checkpoint": fine,
                        "enable_fusion": False,
                    },
                    "expert2": {"pretrained_checkpoint": root / "missing-e2.pt"},
                    "expert3": {"base_checkpoint": root / "missing-e3.pt"},
                    "expert4": {"beats_checkpoint": root / "missing-e4.pt"},
                },
            }
            with patch.object(Expert1Adapter, "from_finetuned", return_value="adapter") as factory:
                self.assertEqual(Expert1Adapter.from_config(resolved), "adapter")
            factory.assert_called_once_with(base, fine, device="cpu", offline=True)

            resolved["runtime"]["offline"] = False
            with patch.object(Expert1Adapter, "from_finetuned", return_value="adapter") as factory:
                self.assertEqual(Expert1Adapter.from_config(resolved), "adapter")
            factory.assert_called_once_with(base, fine, device="cpu", offline=False)

            resolved["experts"]["expert1"]["pretrained_checkpoint"] = "relative.pt"
            with self.assertRaisesRegex(ValueError, "absolute path"):
                Expert1Adapter.from_config(resolved)

    def test_config_factory_requires_fusion_disabled(self):
        with self.assertRaisesRegex(ValueError, "enable_fusion must be false"):
            Expert1Adapter.from_config({"expert1": {"enable_fusion": True}})

    def test_strict_restore_accepts_raw_state_dict_and_rejects_incompatible_keys(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "state.pt"
            torch.save(source.state_dict(), checkpoint)
            Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)
            for source_tensor, target_tensor in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(source_tensor, target_tensor))

            broken = OrderedDict(source.state_dict())
            broken.pop("bias")
            torch.save(broken, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "strict.*fine-tuned"):
                Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)

            unexpected = OrderedDict(source.state_dict())
            unexpected["unexpected"] = torch.ones(1)
            torch.save(unexpected, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "strict.*fine-tuned"):
                Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)

            wrong_shape = OrderedDict(source.state_dict())
            wrong_shape["weight"] = torch.ones(3, 3)
            torch.save(wrong_shape, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "strict.*fine-tuned"):
                Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)

            wrong_shape = OrderedDict(source.state_dict())
            wrong_shape["weight"] = torch.ones(5, 3)
            torch.save(wrong_shape, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "strict.*fine-tuned"):
                Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)

    def test_strict_restore_rejects_structured_or_non_tensor_checkpoint(self):
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "bundle.pt"
            torch.save({"state_dict": target.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "raw tensor state_dict"):
                Expert1Adapter._restore_finetuned_state_dict(target, checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
