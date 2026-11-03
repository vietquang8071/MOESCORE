"""Lightweight registry, import-order, and four-adapter coexistence tests."""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.expert_adapters.expert1_adapter import Expert1Adapter
from model.expert_adapters.expert2_adapter import Expert2Adapter
from model.expert_adapters.expert3_adapter import Expert3Adapter
from model.expert_adapters.expert4_adapter import Expert4Adapter
from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.registry import (
    EXPERT_FEATURE_DIMS,
    EXPERT_INPUT_CONTRACTS,
    EXPERT_ORDER,
    EXPECTED_FEATURE_WIDTH,
    create_expert_adapter,
    resolve_expert_adapter_class,
    validate_module_origins,
    validate_and_assemble_expert_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ORDERS = (
    ("expert1", "expert2", "expert3", "expert4"),
    ("expert4", "expert3", "expert2", "expert1"),
    ("expert2", "expert4"),
    ("expert4", "expert2"),
    ("expert1", "expert4"),
    ("expert4", "expert1"),
    ("expert2", "expert3"),
    ("expert3", "expert2"),
)


class _ParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.last_batch = None


class _ClapCore(nn.Module):
    joint_embed_shape = 512


class _ClapModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ClapCore()


class _Expert1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ClapModule()
        self.calls = 0

    def forward(self, audio, captions):
        self.calls += 1
        self.last_captions = list(captions)
        batch = audio.shape[0]
        audio_features = torch.ones(batch, 512, dtype=audio.dtype, device=audio.device)
        text_features = torch.full_like(audio_features, 1.0 / 512.0)
        return audio_features, text_features


class _OriginalExpert1(_ParameterModel):
    def __init__(self):
        super().__init__()
        self.model = _Expert1Encoder()


class _OriginalExpert2(_ParameterModel):
    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        size = batch["mga_wavs"].shape[0]
        return torch.full((size, 1), 2.0), torch.full((size, 2048), 2.0)


class _OriginalExpert3(_ParameterModel):
    embedding_dim = 768
    cross_feature_dim = 3840
    concat_assignment = "released"

    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        size = batch["wavs"].shape[0]
        return torch.full((size, 1), 3.0), torch.full((size, 3840), 3.0)


class _SeqCoAttn(nn.Module):
    dim = 512


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 4), nn.ReLU(), nn.Linear(4, 1))


class _OriginalExpert4(_ParameterModel):
    def __init__(self):
        super().__init__()
        self.seq_coattn = _SeqCoAttn()
        self.head = _Head()

    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        size = batch["wavs"].shape[0]
        return torch.full((size,), 4.0), torch.full((size, 1024), 4.0)


def _adapter_set():
    models = {
        "expert1": _OriginalExpert1(),
        "expert2": _OriginalExpert2(),
        "expert3": _OriginalExpert3(),
        "expert4": _OriginalExpert4(),
    }
    adapters = {
        "expert1": Expert1Adapter(models["expert1"]),
        "expert2": Expert2Adapter(models["expert2"], feature_dim=2048),
        "expert3": Expert3Adapter(models["expert3"]),
        "expert4": Expert4Adapter(models["expert4"]),
    }
    return models, adapters


def _batches(batch_size):
    captions = [f"caption {index}" for index in range(batch_size)]
    return {
        "expert1": {"expert1_wavs": torch.zeros(batch_size, 1, 480_000), "captions_raw": captions},
        "expert2": {"expert2_wavs": torch.zeros(batch_size, 1, 320_000), "captions_raw": captions},
        "expert3": {"expert3_wavs": torch.zeros(batch_size, 1, 160_000), "captions_raw": captions},
        "expert4": {
            "expert4_wavs": torch.zeros(batch_size, 1, 160_000),
            "expert4_wav_lens": torch.tensor(
                [80_000 if index % 2 == 0 else 160_000 for index in range(batch_size)],
                dtype=torch.int64,
            ),
            "captions_raw": [f"  {caption}  " for caption in captions],
        },
    }


class ExpertRegistryTests(unittest.TestCase):
    def test_registry_has_stable_order_dimensions_and_input_contracts(self):
        self.assertEqual(EXPERT_ORDER, ("expert1", "expert2", "expert3", "expert4"))
        self.assertEqual(
            dict(EXPERT_FEATURE_DIMS),
            {"expert1": 1024, "expert2": 2048, "expert3": 3840, "expert4": 1024},
        )
        self.assertEqual(EXPECTED_FEATURE_WIDTH, 7936)
        self.assertEqual(EXPERT_INPUT_CONTRACTS["expert1"].audio_key, "expert1_wavs")
        self.assertEqual(EXPERT_INPUT_CONTRACTS["expert2"].audio_shape, (None, 1, 320_000))
        self.assertEqual(EXPERT_INPUT_CONTRACTS["expert3"].sample_rate_hz, 16_000)
        self.assertEqual(EXPERT_INPUT_CONTRACTS["expert4"].lengths_key, "expert4_wav_lens")

    def test_registry_resolves_all_adapter_classes_lazily(self):
        expected = (Expert1Adapter, Expert2Adapter, Expert3Adapter, Expert4Adapter)
        self.assertEqual(
            tuple(resolve_expert_adapter_class(name) for name in EXPERT_ORDER), expected
        )

    def test_factory_passes_only_requested_expert_section_and_runtime(self):
        created = []

        class _FactoryAdapter(ExpertAdapter):
            def __init__(self):
                super().__init__(1024)

            def _forward(self, batch):
                raise NotImplementedError

            @classmethod
            def from_config(cls, config, *, device=None):
                created.append((config, device))
                return cls()

        config = {
            "runtime": {"device": "cpu", "offline": True, "cache_dir": "/cache"},
            "experts": {
                "expert1": {"finetuned_checkpoint": "/e1.pt"},
                "expert2": {"secret": "must not leak"},
            },
        }
        with patch("model.expert_adapters.registry.resolve_expert_adapter_class", return_value=_FactoryAdapter):
            result = create_expert_adapter("expert1", config, device="cpu")
        self.assertIsInstance(result, _FactoryAdapter)
        passed, device = created[0]
        self.assertEqual(passed["experts"], {"expert1": {"finetuned_checkpoint": "/e1.pt"}})
        self.assertEqual(passed["runtime"], config["runtime"])
        self.assertEqual(device, "cpu")

    def test_unknown_expert_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "expert5"):
            resolve_expert_adapter_class("expert5")


class OutputContractTests(unittest.TestCase):
    def _outputs(self, batch_size=2):
        return {
            name: (
                torch.full((batch_size, 1), float(index + 1)),
                torch.full((batch_size, EXPERT_FEATURE_DIMS[name]), float(index + 1)),
            )
            for index, name in enumerate(EXPERT_ORDER)
        }

    def test_assembly_uses_stable_order_and_expected_widths(self):
        ordered = OrderedDict(reversed(list(self._outputs().items())))
        scores, features = validate_and_assemble_expert_outputs(ordered, batch_size=2)
        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertEqual(tuple(features.shape), (2, 7936))
        self.assertTrue(torch.equal(scores[0], torch.tensor([1.0, 2.0, 3.0, 4.0])))
        for index, name in enumerate(EXPERT_ORDER):
            start = sum(EXPERT_FEATURE_DIMS[item] for item in EXPERT_ORDER[:index])
            self.assertTrue(torch.all(features[:, start : start + EXPERT_FEATURE_DIMS[name]] == index + 1))

    def test_missing_and_unexpected_expert_outputs_are_rejected(self):
        outputs = self._outputs()
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_and_assemble_expert_outputs({k: v for k, v in outputs.items() if k != "expert3"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_and_assemble_expert_outputs({**outputs, "expert5": outputs["expert1"]})

    def test_wrong_score_and_feature_shapes_are_rejected(self):
        for name, output, message in (
            ("expert1", (torch.zeros(2), torch.zeros(2, 1024)), "score"),
            ("expert2", (torch.zeros(2, 2), torch.zeros(2, 2048)), "score"),
            ("expert3", (torch.zeros(2, 1), torch.zeros(2, 3, 1280)), r"shape \[B, D\]"),
            ("expert4", (torch.zeros(2, 1), torch.zeros(2, 1023)), "width mismatch"),
        ):
            with self.subTest(name=name):
                outputs = self._outputs()
                outputs[name] = output
                with self.assertRaisesRegex(ValueError, message):
                    validate_and_assemble_expert_outputs(outputs)

    def test_batch_device_dtype_and_finiteness_must_match(self):
        cases = []
        output = self._outputs()
        output["expert2"] = (torch.zeros(1, 1), torch.zeros(1, 2048))
        cases.append((output, "batch"))
        output = self._outputs()
        output["expert2"] = (torch.zeros(2, 1, dtype=torch.float64), torch.zeros(2, 2048, dtype=torch.float64))
        cases.append((output, "dtype"))
        output = self._outputs()
        output["expert3"] = (torch.zeros(2, 1, device="meta"), torch.zeros(2, 3840, device="meta"))
        cases.append((output, "device"))
        output = self._outputs()
        output["expert4"] = (torch.full((2, 1), float("nan")), output["expert4"][1])
        cases.append((output, "finite"))
        for output, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate_and_assemble_expert_outputs(output)


class ModuleOriginTests(unittest.TestCase):
    def test_expected_origins_are_accepted_and_conflicts_are_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_root = root / "mga" / "models"
            expected_root.mkdir(parents=True)
            valid = types.ModuleType("models.ase_model")
            valid.__file__ = str(expected_root / "ase_model.py")
            registry = {"models.ase_model": valid}
            origins = validate_module_origins({"models": expected_root}, modules=registry)
            self.assertEqual(origins["models.ase_model"][0], str(Path(valid.__file__).resolve()))

            wrong = types.ModuleType("models.ase_model")
            wrong.__file__ = str(root / "other" / "ase_model.py")
            registry["models.ase_model"] = wrong
            before = dict(registry)
            with self.assertRaisesRegex(ImportError, "models.*expected.*observed"):
                validate_module_origins({"models": expected_root}, modules=registry)
            self.assertEqual(registry, before)


class StubCoexistenceTests(unittest.TestCase):
    def test_all_four_adapters_coexist_for_multiple_batch_sizes(self):
        models, adapters = _adapter_set()
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                batches = _batches(batch_size)
                models["expert1"].model.calls = 0
                for model in models.values():
                    model.calls = 0
                outputs = {
                    name: adapters[name](batches[name])
                    for name in EXPERT_ORDER
                }
                scores, features = validate_and_assemble_expert_outputs(
                    outputs, batch_size=batch_size
                )
                self.assertEqual(tuple(scores.shape), (batch_size, 4))
                self.assertEqual(tuple(features.shape), (batch_size, 7936))
                self.assertTrue(torch.equal(scores[0], torch.tensor([1.0, 2.0, 3.0, 4.0])))
                self.assertEqual(models["expert1"].model.calls, 1)
                for name in ("expert2", "expert3", "expert4"):
                    self.assertEqual(models[name].calls, 1)
                self.assertIs(models["expert2"].last_batch["mga_wavs"], batches["expert2"]["expert2_wavs"])
                self.assertEqual(models["expert3"].last_batch["captions"], batches["expert3"]["captions_raw"])
                self.assertIs(models["expert4"].last_batch["wav_lens"], batches["expert4"]["expert4_wav_lens"])

        for adapter in adapters.values():
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))
            adapter.train()
            self.assertTrue(all(not module.training for module in adapter.model.modules()))
            adapter.train(False)
            self.assertTrue(all(not module.training for module in adapter.model.modules()))


class ImportOrderTests(unittest.TestCase):
    def test_eight_adapter_import_orders_are_isolated_in_clean_processes(self):
        script = r'''
import importlib, json, sys
from pathlib import Path
from model.expert_adapters.registry import resolve_expert_adapter_class
order = json.loads(sys.argv[1])
before_path = list(sys.path)
before_modules = set(sys.modules)
classes = []
for name in order:
    classes.append(resolve_expert_adapter_class(name))
assert sys.path == before_path, "adapter imports mutated sys.path"
forbidden = {"models", "tools", "BEATs", "backbone", "modules", "laion_clap", "transformers", "m2d"}
new_forbidden = sorted(name for name in set(sys.modules) - before_modules if name.split(".", 1)[0] in forbidden)
assert not new_forbidden, f"heavy vendor import during adapter import: {new_forbidden}"
assert len(classes) == len(order)
print(json.dumps({"status": "PASS", "order": order, "loaded": [c.__name__ for c in classes]}))
'''
        for order in IMPORT_ORDERS:
            with self.subTest(order=order):
                env = os.environ.copy()
                env.update(
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_DATASETS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1",
                )
                result = subprocess.run(
                    [sys.executable, "-c", script, json.dumps(order)],
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"order={order}\nstdout={result.stdout}\nstderr={result.stderr}",
                )
                summary = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(summary["status"], "PASS")
                self.assertEqual(tuple(summary["order"]), order)

    def test_lightweight_diagnostic_emits_json_and_marks_real_checks_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "preflight.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "check_expert_factory_coexistence.py"),
                    "--mode",
                    "lightweight",
                    "--order",
                    "expert2,expert4",
                    "--config",
                    str(REPO_ROOT / "configs" / "moe_local.example.yaml"),
                    "--offline",
                    "--report-json",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary, saved)
            self.assertEqual(summary["status"], "PASS — LIGHTWEIGHT INTEGRATION FOUNDATION VERIFIED")
            self.assertFalse(summary["real_models_constructed"])
            checks = summary["checks"]
            self.assertEqual(checks["real_factory_coexistence"]["status"], "NOT_VERIFIED_REAL_FACTORY")
            self.assertEqual(checks["real_inference"]["status"], "NOT_VERIFIED_REAL_INFERENCE")
            self.assertEqual(checks["memory"]["status"], "NOT_VERIFIED_MEMORY")
            self.assertEqual(checks["output_contract"]["feature_shape"], [2, 7936])


if __name__ == "__main__":
    unittest.main(verbosity=2)
