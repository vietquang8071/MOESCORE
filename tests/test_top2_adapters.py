from __future__ import annotations

import builtins
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


MODULES = (
    "model.expert_adapters.top2_m2d_adapter",
    "model.expert_adapters.top2_msclap_adapter",
    "model.expert_adapters.top2_laion_adapter",
    "model.expert_adapters.top2_mga_adapter",
)


class TinyM2D(nn.Module):
    def encode_clap_audio(self, audio):
        return audio.mean(dim=-1, keepdim=True).expand(-1, 4)

    def encode_clap_text(self, captions):
        return torch.arange(len(captions) * 4, dtype=torch.float32).reshape(-1, 4) + 1


class TinyFactoryM2D(TinyM2D):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(sample_rate=16_000)

    def get_clap_text_encoder(self):
        return None


class TinyMSCLAP:
    def __init__(self):
        self.clap = nn.Linear(1, 1)
        self.args = SimpleNamespace(d_proj=4)
        self.audio_paths = None

    def get_audio_embeddings(self, wav_paths):
        self.audio_paths = list(wav_paths)
        return torch.ones((len(wav_paths), 4), dtype=torch.float32)

    def get_text_embeddings(self, captions):
        return torch.arange(len(captions) * 4, dtype=torch.float32).reshape(-1, 4) + 1


class TinyLAION(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(projection_dim=4)
        self.seen = None

    def forward(self, *, input_features, input_ids, attention_mask, is_longer=None):
        batch_size = input_features.shape[0]
        self.seen = (input_features.shape[1], input_ids.shape[0])
        return SimpleNamespace(
            audio_embeds=torch.ones((batch_size, 4), device=input_features.device),
            text_embeds=torch.arange(batch_size * 4, device=input_ids.device).reshape(-1, 4) + 1,
        )


class TinyNativeLAION(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.model = SimpleNamespace(joint_embed_shape=4)
        self.loaded_checkpoint = None

    def load_ckpt(self, ckpt=None, model_id=-1, verbose=True):
        self.loaded_checkpoint = (ckpt, verbose)

    def get_audio_embedding_from_data(self, audio, use_tensor=False):
        assert use_tensor is True
        return torch.ones((audio.shape[0], 4), device=audio.device)

    def get_text_embedding(self, captions, use_tensor=False):
        assert use_tensor is True
        return torch.arange(len(captions) * 4, dtype=torch.float32).reshape(-1, 4) + 1


class TinyProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=48_000)

    def __call__(self, *, audio=None, text=None, sampling_rate=None, return_tensors=None, padding=None):
        if audio is not None:
            assert sampling_rate == self.feature_extractor.sampling_rate
            return {
                "input_features": torch.zeros((len(audio), 8, 4)),
                "is_longer": torch.zeros((len(audio), 1), dtype=torch.bool),
            }
        return {
            "input_ids": torch.ones((len(text), 5), dtype=torch.int64),
            "attention_mask": torch.ones((len(text), 5), dtype=torch.int64),
        }


class TinyMGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.codebook = nn.Parameter(torch.ones((2, 4)))

    def encode_audio(self, audio):
        batch_size = audio.shape[0]
        return None, torch.ones((batch_size, 3, 4), device=audio.device)

    def encode_text(self, captions):
        batch_size = len(captions)
        words = torch.arange(batch_size * 12, device=self.anchor.device).reshape(batch_size, 3, 4) + 1
        return None, words, torch.ones((batch_size, 3), device=self.anchor.device)

    def msc(self, embeddings, codebook, attention_mask=None):
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        return embeddings.mean(dim=1)


def _batch(batch_size=2):
    return {
        "captions_raw": [f"caption {index}" for index in range(batch_size)],
        "expert2_wavs": torch.zeros(batch_size, 1, 320_000),
        "expert3_wavs": torch.zeros(batch_size, 1, 160_000),
        "expert1_wavs": torch.zeros(batch_size, 1, 480_000),
        "wave_file_names": [f"audio-{index}.wav" for index in range(batch_size)],
        "split": "test",
        "wav_paths": [f"/tmp/audio-{index}.wav" for index in range(batch_size)],
    }


class Top2AdapterSmokeTests(unittest.TestCase):
    def test_adapter_modules_import_without_model_dependencies(self):
        blocked = {"msclap", "transformers", "timm", "nnAudio", "sentence_transformers"}
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in blocked:
                raise AssertionError(f"unexpected eager model dependency import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", guarded_import):
            for module_name in MODULES:
                module = importlib.import_module(module_name)
                importlib.reload(module)

    def test_all_adapter_outputs_are_cosine_finite_and_frozen(self):
        cases = (
            (MODULES[0], "Top2M2DAdapter", TinyM2D, "expert3_wavs", False),
            (MODULES[1], "Top2MSCLAPAdapter", TinyMSCLAP, "expert2_wavs", False),
            (MODULES[2], "Top2LAIONAdapter", TinyLAION, "expert1_wavs", True),
            (MODULES[3], "Top2MGAAdapter", TinyMGA, "expert2_wavs", False),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp = Path(temporary_dir)
            paths = []
            for index in range(2):
                path = temp / f"audio-{index}.wav"
                path.write_bytes(b"placeholder")
                paths.append(str(path.resolve()))

            for module_name, class_name, model_type, batch_key, uses_processor in cases:
                with self.subTest(adapter=class_name):
                    module = importlib.import_module(module_name)
                    adapter_type = getattr(module, class_name)
                    model = model_type()
                    batch = _batch()
                    self.assertIn(batch_key, batch)
                    if class_name == "Top2MSCLAPAdapter":
                        batch["wav_paths"] = paths
                        adapter = adapter_type(model, embedding_dim=4)
                    elif uses_processor:
                        adapter = adapter_type(model, TinyProcessor())
                    else:
                        adapter = adapter_type(model, embedding_dim=4)

                    adapter.train()
                    model_module = model if isinstance(model, nn.Module) else model.clap
                    self.assertFalse(model_module.training)
                    self.assertTrue(
                        all(not parameter.requires_grad for parameter in model_module.parameters())
                    )

                    score, features = adapter(batch)
                    self.assertEqual(tuple(score.shape), (2, 1))
                    self.assertEqual(tuple(features.shape), (2, 8))
                    self.assertEqual(score.dtype, torch.float32)
                    self.assertEqual(features.dtype, torch.float32)
                    self.assertEqual(score.device, features.device)
                    self.assertTrue(bool(torch.isfinite(score).all()))
                    self.assertTrue(bool(torch.isfinite(features).all()))
                    self.assertTrue(bool((score.abs() <= 1.000001).all()))
                    expected = torch.nn.functional.cosine_similarity(features[:, :4], features[:, 4:])
                    torch.testing.assert_close(score[:, 0], expected)

                    if class_name == "Top2MSCLAPAdapter":
                        self.assertEqual(model.audio_paths, paths)
                    if class_name == "Top2LAIONAdapter":
                        self.assertEqual(model.seen, (8, 2))

    def test_huggingface_cache_dir_updates_already_imported_clients_temporarily(self):
        from huggingface_hub import constants as hub_constants
        from transformers.utils import hub as transformers_hub

        from model.expert_adapters.top2_common import huggingface_environment

        hub_names = ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE")
        transformer_names = ("HF_MODULES_CACHE", "TRANSFORMERS_CACHE", "default_cache_path")
        previous_hub = {name: getattr(hub_constants, name, None) for name in hub_names}
        previous_transformers = {
            name: getattr(transformers_hub, name, None) for name in transformer_names
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_root = Path(temporary_dir).resolve()
            expected = {
                "HF_HOME": cache_root,
                "HF_HUB_CACHE": cache_root / "hub",
                "HF_ASSETS_CACHE": cache_root / "assets",
                "HF_XET_CACHE": cache_root / "xet",
            }
            with huggingface_environment(offline=True, cache_dir=cache_root):
                for name, value in expected.items():
                    self.assertEqual(Path(getattr(hub_constants, name)), value)
                if hasattr(transformers_hub, "HF_MODULES_CACHE"):
                    self.assertEqual(
                        Path(transformers_hub.HF_MODULES_CACHE), cache_root / "modules"
                    )
                if hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
                    self.assertEqual(
                        Path(transformers_hub.TRANSFORMERS_CACHE), cache_root / "hub"
                    )
                if hasattr(transformers_hub, "default_cache_path"):
                    self.assertEqual(
                        Path(transformers_hub.default_cache_path), cache_root / "hub"
                    )

        for name, value in previous_hub.items():
            if value is not None:
                self.assertEqual(getattr(hub_constants, name), value)
        for name, value in previous_transformers.items():
            if value is not None:
                self.assertEqual(getattr(transformers_hub, name), value)

    def test_msclap_resolves_batch_names_under_selected_split(self):
        module = importlib.import_module(MODULES[1])
        adapter_type = module.Top2MSCLAPAdapter
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_root = Path(temporary_dir) / "audio" / "test"
            audio_root.mkdir(parents=True)
            paths = []
            for name in ("one.wav", "two.wav"):
                path = audio_root / name
                path.write_bytes(b"placeholder")
                paths.append(str(path.resolve()))
            adapter = adapter_type(
                TinyMSCLAP(),
                embedding_dim=4,
                audio_roots={"test": audio_root},
            )
            batch = _batch()
            batch.pop("wav_paths")
            batch["wave_file_names"] = ["one.wav", "two.wav"]

            adapter(batch)

            self.assertEqual(adapter.clap_api.audio_paths, paths)
            batch["wave_file_names"] = ["../outside.wav", "two.wav"]
            with self.assertRaisesRegex(ValueError, "basename|escape"):
                adapter(batch)

    def test_msclap_factory_restores_the_configured_local_checkpoint(self):
        module = importlib.import_module(MODULES[1])
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "CLAP_weights_2023.pth"
            checkpoint.write_bytes(b"checkpoint")
            calls = []

            def clap_factory(**kwargs):
                calls.append(kwargs)
                return TinyMSCLAP()

            with patch.dict("sys.modules", {"msclap": SimpleNamespace(CLAP=clap_factory)}):
                adapter = module.Top2MSCLAPAdapter.from_config(
                    {"checkpoint": str(checkpoint), "runtime": {"device": "cpu"}}
                )

            self.assertEqual(
                calls,
                [{"model_fp": str(checkpoint.resolve()), "version": "2023", "use_cuda": False}],
            )
            self.assertFalse(adapter.clap.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.clap.parameters()))

    def test_m2d_factory_preserves_model_named_checkpoint_symlink_parent(self):
        module = importlib.import_module(MODULES[0])
        source_module = (
            "expert3_finetune.XACLE_2027_baselinemodel.models.m2d.vendor.portable_m2d"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp = Path(temporary_dir)
            real_checkpoint = temp / "checkpoint-30.pth"
            real_checkpoint.write_bytes(b"checkpoint")
            model_dir = temp / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
            model_dir.mkdir()
            logical_checkpoint = model_dir / "checkpoint-30.pth"
            logical_checkpoint.symlink_to(real_checkpoint)
            calls = []

            def portable_m2d(**kwargs):
                calls.append(kwargs)
                return TinyFactoryM2D()

            fake_source = SimpleNamespace(PortableM2D=portable_m2d)
            with patch.dict("sys.modules", {source_module: fake_source}):
                adapter = module.Top2M2DAdapter.from_config(
                    {
                        "checkpoint": str(logical_checkpoint),
                        "embedding_dim": 4,
                        "runtime": {"device": "cpu"},
                    }
                )

            self.assertEqual(
                calls,
                [{"weight_file": str(logical_checkpoint), "flat_features": True}],
            )
            self.assertFalse(adapter.model.training)

    def test_laion_factory_uses_local_native_checkpoint_and_paired_embeddings(self):
        module = importlib.import_module(MODULES[2])
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "laion-clap.pt"
            checkpoint.write_bytes(b"checkpoint")
            model = TinyNativeLAION()
            calls = []

            def clap_factory(**kwargs):
                calls.append(kwargs)
                return model

            with patch.dict("sys.modules", {"laion_clap": SimpleNamespace(CLAP_Module=clap_factory)}):
                adapter = module.Top2LAIONAdapter.from_config(
                    {"checkpoint": str(checkpoint), "runtime": {"device": "cpu"}}
                )

            self.assertEqual(
                calls,
                [{"enable_fusion": False, "device": "cpu", "amodel": "HTSAT-tiny"}],
            )
            self.assertEqual(model.loaded_checkpoint, (str(checkpoint.resolve()), False))
            batch = _batch(batch_size=1)
            score, features = adapter(batch)
            self.assertEqual(tuple(score.shape), (1, 1))
            self.assertEqual(tuple(features.shape), (1, 8))
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
