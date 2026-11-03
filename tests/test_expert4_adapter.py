import importlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from model.expert_adapters.expert4_adapter import (
    Expert4Adapter,
    _scoped_beats_import,
    _validate_restored_trainable_parameters,
    validate_expert4_module_origins,
    _validate_best_checkpoint_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 1))


class _StubExpert4(nn.Module):
    def __init__(self, dim=4, *, output_mode="valid"):
        super().__init__()
        self.seq_coattn = types.SimpleNamespace(dim=dim)
        self.head = _Head(2 * dim)
        self.dropout = nn.Dropout(0.75)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.output_mode = output_mode
        self.calls = 0
        self.last_batch = None
        self.last_scores = None
        self.last_features = None

    def forward(self, batch):
        self.calls += 1
        self.last_batch = batch
        batch_size = batch["wavs"].shape[0]
        device = batch["wavs"].device
        self.last_scores = torch.tensor(
            [1.25 + index for index in range(batch_size)],
            dtype=torch.float32,
            device=device,
        ) * self.scale
        width = self.seq_coattn.dim * 2
        self.last_features = torch.arange(
            batch_size * width, dtype=torch.float32, device=device
        ).reshape(batch_size, width) * self.scale
        if self.output_mode == "score_rank2":
            return self.last_scores[:, None], self.last_features
        if self.output_mode == "bad_score":
            return self.last_scores[:, None].expand(-1, 2), self.last_features
        if self.output_mode == "bad_feature_width":
            return self.last_scores, self.last_features[:, :-1]
        if self.output_mode == "bad_feature_batch":
            return self.last_scores, self.last_features[:0]
        if self.output_mode == "bad_feature_rank":
            return self.last_scores, self.last_features[:, :, None]
        if self.output_mode == "nonfinite_score":
            return self.last_scores * torch.tensor(float("nan"), device=device), self.last_features
        if self.output_mode == "nonfinite_features":
            return self.last_scores, self.last_features * torch.tensor(float("inf"), device=device)
        return self.last_scores, self.last_features


def _batch(batch_size=2, *, samples=160_000, length_values=None):
    if length_values is None:
        length_values = [max(1, samples - 1000 * index) for index in range(batch_size)]
    return {
        "expert4_wavs": torch.zeros(batch_size, 1, samples, dtype=torch.float32),
        "expert4_wav_lens": torch.tensor(length_values, dtype=torch.int64),
        "captions_raw": [f"  caption {index}  " for index in range(batch_size)],
    }


def _checkpoint_bundle(model_state=None):
    zeros = "0" * 64
    return {
        "model_state_dict": model_state or {"scale": torch.tensor(1.0)},
        "config": {"resources": {}, "training": {}, "loss": {}},
        "fingerprints": {
            "beats_checkpoint": zeros,
            "clap_checkpoint": zeros,
            "train_csv": zeros,
            "validation_csv": zeros,
            "unilm_source": {"git_commit": None, "sha256": zeros},
        },
        "epoch": 0,
        "best_srcc": 0.25,
    }


class Expert4AdapterForwardTests(unittest.TestCase):
    def test_batch_sizes_scores_features_and_single_model_call(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                original = _StubExpert4(dim=3)
                adapter = Expert4Adapter(original)
                batch = _batch(batch_size)
                score, features = adapter(batch)
                self.assertEqual(tuple(score.shape), (batch_size, 1))
                self.assertEqual(tuple(features.shape), (batch_size, 6))
                self.assertEqual(original.calls, 1)
                self.assertTrue(torch.equal(score[:, 0], original.last_scores))
                self.assertIs(features, original.last_features)
                self.assertIs(original.last_batch["wavs"], batch["expert4_wavs"])
                self.assertIs(original.last_batch["wav_lens"], batch["expert4_wav_lens"])
                self.assertEqual(
                    original.last_batch["captions"],
                    [f"caption {index}" for index in range(batch_size)],
                )
                self.assertTrue(torch.isfinite(score).all())
                self.assertTrue(torch.isfinite(features).all())

    def test_accepts_original_score_shapes_without_changing_values(self):
        for output_mode in ("valid", "score_rank2"):
            with self.subTest(output_mode=output_mode):
                original = _StubExpert4(dim=2, output_mode=output_mode)
                adapter = Expert4Adapter(original)
                score, features = adapter(_batch(2))
                expected = torch.tensor([[1.25], [2.25]])
                self.assertTrue(torch.equal(score, expected))
                self.assertTrue(torch.equal(features, original.last_features))

    def test_dynamic_feature_dimension_and_explicit_validation(self):
        for dim in (2, 7, 16):
            with self.subTest(dim=dim):
                model = _StubExpert4(dim=dim)
                adapter = Expert4Adapter(model)
                self.assertEqual(adapter.feature_dim, 2 * dim)
                self.assertEqual(adapter(_batch(1))[1].shape[-1], 2 * dim)
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            Expert4Adapter(_StubExpert4(dim=4), feature_dim=7)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Expert4Adapter(_StubExpert4(dim=4), feature_dim=0)

    def test_rejects_inconsistent_feature_metadata(self):
        model = _StubExpert4(dim=4)
        model.head.net[0] = nn.Linear(9, 1)
        with self.assertRaisesRegex(ValueError, "head|metadata|width"):
            Expert4Adapter(model)

    def test_rejects_missing_or_malformed_batch_fields(self):
        adapter = Expert4Adapter(_StubExpert4())
        good = _batch(2)
        cases = [
            ({"expert4_wav_lens": good["expert4_wav_lens"], "captions_raw": good["captions_raw"]}, "expert4_wavs"),
            ({"expert4_wavs": good["expert4_wavs"], "captions_raw": good["captions_raw"]}, "expert4_wav_lens"),
            ({"expert4_wavs": good["expert4_wavs"], "expert4_wav_lens": good["expert4_wav_lens"]}, "captions_raw"),
            ({**good, "expert4_wavs": good["expert4_wavs"].squeeze(1)}, "shape"),
            ({**good, "expert4_wavs": torch.zeros(2, 2, 160_000)}, "channel"),
            ({**good, "expert4_wavs": torch.zeros(2, 1, 0)}, "samples"),
            ({**good, "expert4_wavs": torch.zeros(2, 1, 160_001)}, "maximum|160000|10 seconds"),
            ({**good, "expert4_wavs": good["expert4_wavs"].double()}, "float32"),
            ({**good, "expert4_wavs": torch.full((2, 1, 160_000), float("nan"))}, "finite"),
            ({**good, "expert4_wavs": torch.full((2, 1, 160_000), float("inf"))}, "finite"),
            ({**good, "expert4_wav_lens": torch.tensor([10])}, "shape|batch"),
            ({**good, "expert4_wav_lens": torch.tensor([10.0, 20.0])}, "int64"),
            ({**good, "expert4_wav_lens": torch.tensor([0, 10])}, "positive"),
            ({**good, "expert4_wav_lens": torch.tensor([-1, 10])}, "positive"),
            ({**good, "expert4_wav_lens": torch.tensor([160_001, 10])}, "length|maximum|padded"),
            ({**good, "captions_raw": ["only one"]}, "caption"),
            ({**good, "captions_raw": "a string, not a list"}, "caption"),
            ({**good, "captions_raw": ["valid", 42]}, "string"),
            ({**good, "captions_raw": ["  ", "valid"]}, "empty|non-empty"),
            ([], "mapping"),
        ]
        for malformed, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex((TypeError, ValueError, KeyError), expected):
                    adapter(malformed)

    def test_mixed_lengths_are_forwarded_unchanged(self):
        original = _StubExpert4()
        adapter = Expert4Adapter(original)
        batch = _batch(4, length_values=[80_000, 111_000, 159_999, 160_000])
        adapter(batch)
        self.assertIs(original.last_batch["wav_lens"], batch["expert4_wav_lens"])
        self.assertEqual(original.last_batch["wav_lens"].tolist(), [80_000, 111_000, 159_999, 160_000])

    def test_single_sample_short_and_full_valid_lengths(self):
        for valid_length in (80_000, 160_000):
            with self.subTest(valid_length=valid_length):
                original = _StubExpert4()
                adapter = Expert4Adapter(original)
                batch = _batch(1, length_values=[valid_length])
                adapter(batch)
                self.assertEqual(original.last_batch["wav_lens"].tolist(), [valid_length])

    def test_rejects_empty_batch_and_length_device_mismatch(self):
        adapter = Expert4Adapter(_StubExpert4())
        empty = {
            "expert4_wavs": torch.empty((0, 1, 160_000), dtype=torch.float32),
            "expert4_wav_lens": torch.empty((0,), dtype=torch.int64),
            "captions_raw": [],
        }
        with self.assertRaisesRegex(ValueError, "non-empty"):
            adapter(empty)
        if hasattr(torch, "device"):
            mismatched = _batch(1)
            mismatched["expert4_wav_lens"] = torch.empty((1,), dtype=torch.int64, device="meta")
            with self.assertRaisesRegex(ValueError, "share a device"):
                adapter(mismatched)

    def test_model_device_mismatch_is_rejected(self):
        model = _StubExpert4()
        model.scale = nn.Parameter(torch.empty((), device="meta"))
        adapter = Expert4Adapter(model)
        with self.assertRaisesRegex(ValueError, "device"):
            adapter(_batch(1))

    def test_invalid_original_outputs_are_rejected_by_shared_contract(self):
        for mode, message in (
            ("bad_score", "score must have shape"),
            ("bad_feature_width", "width mismatch"),
            ("bad_feature_batch", "batch dimension mismatch"),
            ("bad_feature_rank", "features must have shape"),
            ("nonfinite_score", "finite"),
            ("nonfinite_features", "finite"),
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, message):
                    Expert4Adapter(_StubExpert4(output_mode=mode))(_batch(2))

    def test_frozen_and_eval_even_after_parent_mode_changes(self):
        model = _StubExpert4()
        adapter = Expert4Adapter(model)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad), 0)
        adapter.train()
        self.assertTrue(adapter.training)
        self.assertFalse(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        adapter.train(False)
        self.assertFalse(model.training)
        adapter.eval()
        self.assertFalse(model.training)
        batch = _batch(2)
        with torch.no_grad():
            score_a, features_a = adapter(batch)
            score_b, features_b = adapter(batch)
        self.assertTrue(torch.equal(score_a, score_b))
        self.assertTrue(torch.equal(features_a, features_b))


class Expert4CheckpointValidationTests(unittest.TestCase):
    def test_source_best_bundle_schema_is_accepted(self):
        state = _checkpoint_bundle()
        _validate_best_checkpoint_bundle(state)

    def test_resume_bundle_and_malformed_best_bundles_are_rejected(self):
        state = _checkpoint_bundle()
        invalid = []
        missing = dict(state)
        missing.pop("fingerprints")
        invalid.append((missing, "missing"))
        resume = {**state, "optimizer_state_dict": {}}
        invalid.append((resume, "unexpected"))
        wrong_state = {**state, "model_state_dict": {"weight": 1}}
        invalid.append((wrong_state, "tensor"))
        bad_hash = _checkpoint_bundle()
        bad_hash["fingerprints"] = {**bad_hash["fingerprints"], "clap_checkpoint": "not-a-sha"}
        invalid.append((bad_hash, "fingerprint|SHA-256"))
        bad_epoch = {**state, "epoch": -1}
        invalid.append((bad_epoch, "epoch"))
        bad_srcc = {**state, "best_srcc": float("nan")}
        invalid.append((bad_srcc, "best_srcc|finite"))
        bad_config = {**state, "config": []}
        invalid.append((bad_config, "config"))
        for bundle, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _validate_best_checkpoint_bundle(bundle)

    def test_import_origin_guard_accepts_expected_and_rejects_conflicting_beats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beats = root / "beats" / "BEATs.py"
            beats.parent.mkdir()
            beats.touch()
            module = types.ModuleType("BEATs")
            module.__file__ = str(beats)
            module.__spec__ = importlib.machinery.ModuleSpec("BEATs", loader=None)
            module.__spec__.origin = str(beats)
            validate_expert4_module_origins(root, modules={"BEATs": module})
            module.__file__ = str(root / "other" / "BEATs.py")
            with self.assertRaisesRegex(ImportError, "BEATs.*outside|origin"):
                validate_expert4_module_origins(root, modules={"BEATs": module})

    def test_scoped_beats_import_supports_package_relative_vendor_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unilm"
            beats_dir = source / "beats"
            beats_dir.mkdir(parents=True)
            (beats_dir / "backbone.py").write_text("MARKER = 'relative-import-ok'\n")
            (beats_dir / "BEATs.py").write_text(
                "from .backbone import MARKER\nVALUE = MARKER\n"
            )
            original_path = list(sys.path)
            with _scoped_beats_import(source) as module:
                self.assertEqual(module.VALUE, "relative-import-ok")
                self.assertEqual(Path(module.__file__).resolve(), (beats_dir / "BEATs.py").resolve())
                self.assertIs(sys.modules["BEATs"], module)
            self.assertNotIn("BEATs", sys.modules)
            self.assertEqual(sys.path, original_path)

    def test_scoped_beats_import_preserves_valid_existing_module_and_cleans_added_path(self):
        with tempfile.TemporaryDirectory() as directory:
            beats_dir = Path(directory) / "beats"
            beats_dir.mkdir()
            beats_file = beats_dir / "BEATs.py"
            beats_file.write_text("# existing module fixture\n")
            module = types.ModuleType("BEATs")
            module.__file__ = str(beats_file)
            module.__package__ = "fixture_beats"
            module.__spec__ = importlib.machinery.ModuleSpec("BEATs", loader=None)
            module.__spec__.origin = str(beats_file)
            original_path = list(sys.path)
            sys.modules["BEATs"] = module
            self.addCleanup(lambda: sys.modules.pop("BEATs", None))
            with _scoped_beats_import(directory) as loaded:
                self.assertIs(loaded, module)
                sys.path.insert(0, str(beats_dir))
            self.assertIs(sys.modules["BEATs"], module)
            self.assertEqual(sys.path, original_path)

    def test_import_origin_guard_checks_absolute_backbone_and_modules_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beats_dir = root / "beats"
            beats_dir.mkdir()
            modules = {}
            for name in ("BEATs", "backbone", "modules"):
                path = beats_dir / f"{name}.py"
                path.touch()
                module = types.ModuleType(name)
                module.__file__ = str(path)
                module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
                module.__spec__.origin = str(path)
                modules[name] = module
            validate_expert4_module_origins(root, modules=modules)
            modules["backbone"].__file__ = str(root / "other" / "backbone.py")
            with self.assertRaisesRegex(ImportError, "backbone.*outside|backbone.*origin"):
                validate_expert4_module_origins(root, modules=modules)

    def test_scoped_beats_import_supports_original_absolute_imports_and_both_import_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unilm"
            beats_dir = source / "beats"
            beats_dir.mkdir(parents=True)
            (beats_dir / "modules.py").write_text("MARKER = 'absolute-import-ok'\n")
            (beats_dir / "backbone.py").write_text("from modules import MARKER\nVALUE = MARKER\n")
            (beats_dir / "BEATs.py").write_text("from backbone import VALUE\nRESULT = VALUE\n")
            script = r'''
import importlib, pathlib, sys
from model.expert_adapters.expert4_adapter import _scoped_beats_import
source = pathlib.Path(sys.argv[1])
beats = source / 'beats'
before_path = list(sys.path)
models_before = sys.modules.get('models')
if sys.argv[2] == 'expert2_first':
    importlib.import_module('model.expert_adapters.expert2_adapter')
with _scoped_beats_import(source) as module:
    assert module.RESULT == 'absolute-import-ok'
    for name, filename in [('BEATs', 'BEATs.py'), ('backbone', 'backbone.py'), ('modules', 'modules.py')]:
        current = sys.modules[name]
        assert pathlib.Path(current.__file__).resolve() == (beats / filename).resolve()
    sys.path.insert(0, str(beats))  # emulate the original AudioEncoder constructor
assert all(name not in sys.modules for name in ('BEATs', 'backbone', 'modules'))
assert sys.path == before_path
assert sys.modules.get('models') is models_before
try:
    with _scoped_beats_import(source):
        sys.path.insert(0, str(beats))
        raise RuntimeError('controlled bridge cleanup failure')
except RuntimeError as exc:
    assert 'controlled bridge cleanup failure' in str(exc)
else:
    raise AssertionError('controlled exception was not propagated')
assert all(name not in sys.modules for name in ('BEATs', 'backbone', 'modules'))
assert sys.path == before_path
if sys.argv[2] == 'expert4_first':
    importlib.import_module('model.expert_adapters.expert2_adapter')
assert sys.path == before_path
print('PASS')
'''
            for import_order in ("expert2_first", "expert4_first"):
                with self.subTest(import_order=import_order):
                    result = subprocess.run(
                        [sys.executable, "-c", script, str(source), import_order],
                        cwd=REPO_ROOT,
                        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("PASS", result.stdout)

    def test_scoped_beats_import_rejects_conflicting_absolute_namespace_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unilm"
            beats_dir = source / "beats"
            beats_dir.mkdir(parents=True)
            (beats_dir / "modules.py").write_text("MARKER = 'expected'\n")
            (beats_dir / "backbone.py").write_text("from modules import MARKER\n")
            (beats_dir / "BEATs.py").write_text("from backbone import MARKER\n")
            script = r'''
import pathlib, sys, types
from model.expert_adapters.expert4_adapter import _scoped_beats_import
source = pathlib.Path(sys.argv[1])
foreign = types.ModuleType('modules')
foreign.__file__ = str(source / 'foreign' / 'modules.py')
sys.modules['modules'] = foreign
try:
    with _scoped_beats_import(source):
        raise AssertionError('conflicting modules namespace was accepted')
except ImportError as exc:
    assert 'modules' in str(exc) and 'origin' in str(exc).lower()
else:
    raise AssertionError('origin collision was not rejected')
assert sys.modules['modules'] is foreign
assert 'BEATs' not in sys.modules and 'backbone' not in sys.modules
print('PASS')
'''
            result = subprocess.run(
                [sys.executable, "-c", script, str(source)],
                cwd=REPO_ROOT,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)

    def test_restored_trainable_tensor_set_shape_and_values_are_checked(self):
        model = _StubExpert4(dim=2)
        expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        _validate_restored_trainable_parameters(model, expected)
        with self.assertRaisesRegex(ValueError, "missing=.*unexpected"):
            _validate_restored_trainable_parameters(model, {"scale": expected["scale"]})
        wrong_shape = dict(expected)
        wrong_shape["scale"] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _validate_restored_trainable_parameters(model, wrong_shape)
        wrong_value = dict(expected)
        wrong_value["scale"] = torch.tensor(8.0)
        with self.assertRaisesRegex(ValueError, "restoring fine-tuned tensors"):
            _validate_restored_trainable_parameters(model, wrong_value)

    def test_factory_scopes_resources_and_uses_original_loader_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beats = root / "beats.pt"
            clap = root / "clap.pt"
            beats.write_bytes(b"beats")
            clap.write_bytes(b"clap")
            unilm = root / "unilm"
            beats_dir = unilm / "beats"
            beats_dir.mkdir(parents=True)
            beats_source = beats_dir / "BEATs.py"
            beats_source.write_text("# synthetic origin fixture\n")
            checkpoint = root / "best.pt"
            model = _StubExpert4(dim=5)
            trained_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            state = _checkpoint_bundle(trained_state)
            torch.save(state, checkpoint)
            original_loader = importlib.import_module("expert4_finetune.inference").load_finetuned
            calls = []

            def fake_loader(path, resources, device):
                calls.append((path, resources, device))
                vendor = types.ModuleType("BEATs")
                vendor.__file__ = str(beats_source)
                vendor.__spec__ = importlib.machinery.ModuleSpec("BEATs", loader=None)
                vendor.__spec__.origin = str(beats_source)
                self.addCleanup(lambda: sys.modules.pop("BEATs", None))
                sys.modules["BEATs"] = vendor
                return model, state

            with mock.patch("expert4_finetune.inference.load_finetuned", side_effect=fake_loader) as loader:
                adapter = Expert4Adapter.from_finetuned(
                    beats_checkpoint=beats,
                    clap_checkpoint=clap,
                    unilm_source_dir=unilm,
                    finetuned_checkpoint=checkpoint,
                    device="cpu",
                )
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(calls), 1)
            self.assertIs(adapter.model, model)
            self.assertEqual(adapter.feature_dim, 10)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(set(calls[0][1]), {"beats_checkpoint", "clap_checkpoint", "unilm_source_dir"})
            self.assertEqual(calls[0][2], torch.device("cpu"))
            self.assertIsNotNone(original_loader)

    def test_factory_rejects_missing_resources_and_missing_finetuned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = {
                "beats_checkpoint": root / "beats.pt",
                "clap_checkpoint": root / "clap.pt",
                "unilm_source_dir": root / "unilm",
                "finetuned_checkpoint": root / "best.pt",
            }
            with mock.patch("expert4_finetune.inference.load_finetuned") as loader:
                with self.assertRaises(FileNotFoundError):
                    Expert4Adapter.from_finetuned(**args)
                loader.assert_not_called()

    def test_from_config_accepts_resolved_expert4_section_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beats, clap, checkpoint = root / "beats.pt", root / "clap.pt", root / "best.pt"
            beats.write_bytes(b"beats")
            clap.write_bytes(b"clap")
            unilm = root / "unilm"
            (unilm / "beats").mkdir(parents=True)
            beats_source = unilm / "beats" / "BEATs.py"
            beats_source.write_text("# fixture\n")
            model = _StubExpert4(dim=8)
            state = _checkpoint_bundle({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
            torch.save(state, checkpoint)

            def fake_loader(path, resources, device):
                vendor = types.ModuleType("BEATs")
                vendor.__file__ = str(beats_source)
                vendor.__spec__ = importlib.machinery.ModuleSpec("BEATs", loader=None)
                vendor.__spec__.origin = str(beats_source)
                sys.modules["BEATs"] = vendor
                self.addCleanup(lambda: sys.modules.pop("BEATs", None))
                return model, state

            section = {
                "beats_checkpoint": beats,
                "clap_checkpoint": clap,
                "unilm_source_dir": unilm,
                "finetuned_checkpoint": checkpoint,
            }
            with mock.patch("expert4_finetune.inference.load_finetuned", side_effect=fake_loader):
                adapter = Expert4Adapter.from_config(
                    {"experts": {"expert4": section}, "runtime": {"device": "cpu", "offline": True}}
                )
            self.assertEqual(adapter.feature_dim, 16)


class Expert4ImportIsolationTests(unittest.TestCase):
    def test_adapter_import_is_lightweight_and_does_not_mutate_sys_path(self):
        script = r'''
import sys
before = list(sys.path)
from model.expert_adapters.expert4_adapter import Expert4Adapter
assert sys.path == before
assert "expert4_finetune.model" not in sys.modules
assert "BEATs" not in sys.modules
assert "laion_clap" not in sys.modules
print(Expert4Adapter.__name__)
'''
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Expert4Adapter", result.stdout)

    def test_adapter_module_import_orders_with_prior_experts(self):
        for modules in (
            ("model.expert_adapters.expert2_adapter", "model.expert_adapters.expert4_adapter"),
            ("model.expert_adapters.expert4_adapter", "model.expert_adapters.expert2_adapter"),
            ("model.expert_adapters.expert3_adapter", "model.expert_adapters.expert4_adapter"),
            ("model.expert_adapters.expert4_adapter", "model.expert_adapters.expert3_adapter"),
        ):
            with self.subTest(modules=modules):
                script = "import importlib,sys; before=list(sys.path); " + "; ".join(
                    f"importlib.import_module({module!r})" for module in modules
                ) + "; assert sys.path == before; print('PASS')"
                result = subprocess.run(
                    [sys.executable, "-c", script], cwd=REPO_ROOT,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                    capture_output=True, text=True, timeout=90, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
