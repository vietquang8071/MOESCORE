"""P1 regression tests for M2D-LoRA cache provenance and Top2 LAION inputs."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import torch

from model.expert_adapters.registry import (
    TOP2_EXPERT_INPUT_CONTRACTS,
    ExpertInputContract,
)
from scripts import extract_moe_feature_store as extraction
from scripts.extract_moe_feature_store import (
    _TEXT_RESOURCE_IDS,
    compute_expert_cache_identity,
    prepare_expert_inputs,
    projected_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
M2D_TEXT_MODEL_ID = "google-bert/bert-base-uncased"


def _copy_file(source_root: Path, target_root: Path, relative: str) -> None:
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / relative, target)


def _cache_identity_project_fixture(root: Path) -> Path:
    """Copy only source trees read by real cache identity computation."""
    project = root / "project"
    files = (
        "model/expert_adapters/base.py",
        "model/expert_adapters/paired_cross_features.py",
        "model/expert_adapters/top2_common.py",
        "model/expert_adapters/m2d_lora_adapter.py",
        "model/expert_adapters/m2d_lora_runtime.py",
        "model/expert_adapters/expert1_adapter.py",
        "model/expert_adapters/top2_laion_adapter.py",
        "model/expert_adapters/expert8_m2d_clap_adapter.py",
        "model/expert_adapters/expert8_m2d_clap_runtime.py",
        "scripts/expert8_text_resource.py",
    )
    for relative in files:
        _copy_file(ROOT, project, relative)
    for relative in (
        "expert1_finetune",
        "expert3_finetune/XACLE_2027_baselinemodel",
    ):
        shutil.copytree(ROOT / relative, project / relative)
    return project


def _write_hf_snapshot(cache_root: Path, model_id: str, revision: str, marker: str) -> Path:
    model_root = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (model_root / "refs").mkdir(exist_ok=True)
    for name, value in {
        "config.json": '{"model_type":"bert"}',
        "tokenizer_config.json": '{"do_lower_case":true}',
        "tokenizer.json": f'{{"marker":"{marker}"}}',
        "vocab.txt": f"[PAD]\n{marker}\n",
        "model.safetensors": f"weights-{marker}",
    }.items():
        (snapshot / name).write_text(value, encoding="utf-8")
    return model_root


def _identity_config(root: Path, model_cache: Path) -> dict:
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    backbone = checkpoint_root / "checkpoint-30.pth"
    finetuned = checkpoint_root / "best.pt"
    backbone.write_bytes(b"temporary-backbone-checkpoint")
    finetuned.write_bytes(b"temporary-e34-checkpoint")
    return {
        "runtime": {"device": "cpu", "cache_dir": str(model_cache)},
        "experts": {
            "m2d_lora": {
                "base_checkpoint": str(backbone),
                "finetuned_checkpoint": str(finetuned),
            }
        },
    }


def _m2d_contract() -> ExpertInputContract:
    return ExpertInputContract(
        audio_key="expert3_wavs",
        audio_shape=(None, 1, 160_000),
        sample_rate_hz=16_000,
        max_samples=160_000,
    )


def _write_dataset(root: Path, *, rate: int, samples: np.ndarray) -> tuple[Path, Path]:
    audio_root = root / "wav" / "train"
    audio_root.mkdir(parents=True, exist_ok=True)
    audio_path = audio_root / "clip.wav"
    with wave.open(str(audio_path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    metadata = root / "meta_data" / "train_average.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wav_file_name", "text", "average_score"])
        writer.writerow(["clip.wav", "  Exact caption  ", "5"])
    return audio_path, metadata


def _record_for(dataset, audio_path: Path) -> dict:
    row = dataset.rows[0]
    return {
        "sample_id": row["sample_id"],
        "wave_file_name": row["wave_file_name"],
        "raw_caption": row["captions_raw"],
        "wav_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
    }


class M2DLoRACacheIdentityRegressionTests(unittest.TestCase):
    def _compute(self, project: Path, config: dict, *, expert: str = "m2d_lora"):
        if expert == "m2d_lora":
            return compute_expert_cache_identity(
                expert,
                config,
                contract=_m2d_contract(),
                feature_dim=3073,
                device="cpu",
            )
        return compute_expert_cache_identity(expert, config, device="cpu")

    def test_runtime_source_changes_m2d_identity_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _cache_identity_project_fixture(root)
            hub = root / "hf" / "hub"
            bert_root = _write_hf_snapshot(hub, M2D_TEXT_MODEL_ID, "revision-a", "a")
            _write_hf_snapshot(hub, "roberta-base", "revision-r", "r")
            (bert_root / "refs" / "main").write_text("revision-a", encoding="utf-8")
            (hub / "models--roberta-base" / "refs" / "main").write_text(
                "revision-r", encoding="utf-8"
            )
            config = _identity_config(root, hub)
            config["experts"]["expert1"] = {
                "pretrained_checkpoint": str(root / "checkpoints" / "expert1-base.pt"),
                "finetuned_checkpoint": str(root / "checkpoints" / "expert1-best.pt"),
            }
            Path(config["experts"]["expert1"]["pretrained_checkpoint"]).write_bytes(b"e1-base")
            Path(config["experts"]["expert1"]["finetuned_checkpoint"]).write_bytes(b"e1-best")

            with (
                patch.object(extraction, "PROJECT_ROOT", project),
                patch.object(extraction, "_TEXT_SNAPSHOT_CACHE", {}),
                patch.dict(os.environ, {"HF_HOME": "", "HF_HUB_CACHE": ""}),
                patch("pathlib.Path.home", return_value=root / "empty-home"),
            ):
                m2d_before = self._compute(project, config)
                expert1_before = self._compute(project, config, expert="expert1")
                runtime_file = project / "model/expert_adapters/m2d_lora_runtime.py"
                original = runtime_file.read_bytes()
                runtime_file.write_bytes(original + b"\n# fixture-only runtime change\n")
                m2d_after = self._compute(project, config)
                expert1_after = self._compute(project, config, expert="expert1")

            self.assertNotEqual(m2d_before["cache_id"], m2d_after["cache_id"])
            self.assertEqual(expert1_before["cache_id"], expert1_after["cache_id"])
            self.assertIn(
                "model/expert_adapters/m2d_lora_runtime.py",
                m2d_after["implementation_provenance"],
            )
            self.assertEqual(
                list(m2d_after["relevant_dependencies"]["text_encoder_tokenizer_resources"]),
                [M2D_TEXT_MODEL_ID],
            )
            self.assertEqual(
                set(m2d_after["checkpoint_provenance"]),
                {"base_checkpoint", "finetuned_checkpoint"},
            )
            self.assertIn(
                "expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py",
                m2d_after["implementation_provenance"],
            )

    def test_expert8_identity_hashes_local_bert_without_hub_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _cache_identity_project_fixture(root)
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            base = checkpoint_root / "checkpoint-30.pth"
            finetuned = checkpoint_root / "expert8-best.pt"
            base.write_bytes(b"frozen M2D checkpoint")
            finetuned.write_bytes(b"frozen E8 LoRA checkpoint")
            model_dir = root / "models" / "bert-base-uncased"
            model_dir.mkdir(parents=True)
            for name, value in {
                "config.json": '{"model_type":"bert"}',
                "tokenizer_config.json": '{"do_lower_case":true}',
                "tokenizer.json": '{"fixture":true}',
                "vocab.txt": "[PAD]\n[UNK]\n",
                "model.safetensors": "local frozen weights",
            }.items():
                (model_dir / name).write_text(value, encoding="utf-8")
            config = {
                "runtime": {"device": "cpu", "cache_dir": str(root / "empty-hub-cache")},
                "experts": {"expert8": {
                    "base_checkpoint": str(base), "finetuned_checkpoint": str(finetuned),
                    "text_model_dir": str(model_dir),
                    "text_model_repo_id": M2D_TEXT_MODEL_ID,
                    "text_model_revision": "a" * 40,
                }},
            }
            with (
                patch.object(extraction, "PROJECT_ROOT", project),
                patch.object(extraction, "_TEXT_SNAPSHOT_CACHE", {}),
                patch.dict(os.environ, {"HF_HOME": "", "HF_HUB_CACHE": ""}),
                patch("pathlib.Path.home", return_value=root / "empty-home"),
            ):
                before = compute_expert_cache_identity("expert8", config, feature_dim=3073, device="cpu")
                (model_dir / "model.safetensors").write_text("changed local weights", encoding="utf-8")
                after = compute_expert_cache_identity("expert8", config, feature_dim=3073, device="cpu")

            resource = before["relevant_dependencies"]["text_encoder_tokenizer_resources"]["expert8_text_model"]
            self.assertEqual(resource["original_huggingface_repo_id"], M2D_TEXT_MODEL_ID)
            self.assertEqual(resource["resolved_revision"], "a" * 40)
            self.assertEqual(resource["model_directory"], str(model_dir.resolve()))
            self.assertEqual(set(resource["files_sha256"]), {
                "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors",
            })
            self.assertIn("scripts/expert8_text_resource.py", before["implementation_provenance"])
            self.assertNotEqual(before["cache_id"], after["cache_id"])
            self.assertNotIn("expert8", _TEXT_RESOURCE_IDS)
            self.assertEqual(_TEXT_RESOURCE_IDS["m2d_lora"], (M2D_TEXT_MODEL_ID,))

    def test_selected_bert_snapshot_content_and_revision_change_m2d_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _cache_identity_project_fixture(root)
            hub = root / "hf" / "hub"
            model_root = _write_hf_snapshot(hub, M2D_TEXT_MODEL_ID, "revision-a", "a")
            _write_hf_snapshot(hub, M2D_TEXT_MODEL_ID, "revision-b", "b")
            refs = model_root / "refs"
            (refs / "main").write_text("revision-a", encoding="utf-8")
            config = _identity_config(root, hub)

            with (
                patch.object(extraction, "PROJECT_ROOT", project),
                patch.object(extraction, "_TEXT_SNAPSHOT_CACHE", {}),
                patch.dict(os.environ, {"HF_HOME": "", "HF_HUB_CACHE": ""}),
                patch("pathlib.Path.home", return_value=root / "empty-home"),
            ):
                first = self._compute(project, config)
                (model_root / "snapshots" / "revision-a" / "tokenizer.json").write_text(
                    '{"tokenizer":"changed"}', encoding="utf-8"
                )
                extraction._TEXT_SNAPSHOT_CACHE.clear()
                content_changed = self._compute(project, config)
                (refs / "main").write_text("revision-b", encoding="utf-8")
                extraction._TEXT_SNAPSHOT_CACHE.clear()
                revision_changed = self._compute(project, config)

            self.assertNotEqual(first["cache_id"], content_changed["cache_id"])
            self.assertNotEqual(content_changed["cache_id"], revision_changed["cache_id"])
            self.assertEqual(_TEXT_RESOURCE_IDS["m2d_lora"], (M2D_TEXT_MODEL_ID,))


class Top2LAIONPreprocessingRegressionTests(unittest.TestCase):
    def test_laion_cache_identity_records_center_evaluation_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _cache_identity_project_fixture(root)
            hub = root / "hf" / "hub"
            model_root = _write_hf_snapshot(hub, "roberta-base", "revision-r", "r")
            (model_root / "refs" / "main").write_text("revision-r", encoding="utf-8")
            checkpoint = root / "laion.pt"
            checkpoint.write_bytes(b"temporary-laion-checkpoint")
            config = {
                "runtime": {"device": "cpu", "cache_dir": str(hub)},
                "experts": {"top2_laion": {"checkpoint": str(checkpoint)}},
            }
            with (
                patch.object(extraction, "PROJECT_ROOT", project),
                patch.object(extraction, "_TEXT_SNAPSHOT_CACHE", {}),
                patch.dict(os.environ, {"HF_HOME": "", "HF_HUB_CACHE": ""}),
                patch("pathlib.Path.home", return_value=root / "empty-home"),
            ):
                identity = compute_expert_cache_identity(
                    "top2_laion",
                    config,
                    contract=TOP2_EXPERT_INPUT_CONTRACTS["top2_laion"],
                    feature_dim=4,  # Test-only fixture dimension; no model is loaded.
                    device="cpu",
                )

            self.assertEqual(
                identity["preprocessing_provenance"]["dataset_method"],
                "_expert1_audio",
            )
            self.assertEqual(
                identity["preprocessing_provenance"]["train_crop_policy"],
                "center_validation_mode",
            )
            self.assertEqual(
                identity["preprocessing_provenance"]["effective_sample_rate_hz"],
                48_000,
            )
            self.assertEqual(
                identity["preprocessing_provenance"]["effective_sample_count"],
                480_000,
            )

    def test_laion_uses_real_expert1_center_crop_and_validated_48khz_input(self):
        self.assertIn("top2_laion", TOP2_EXPERT_INPUT_CONTRACTS)
        contract = TOP2_EXPERT_INPUT_CONTRACTS["top2_laion"]
        self.assertEqual(contract.audio_key, "expert1_wavs")
        self.assertEqual(contract.audio_shape, (None, 1, 480_000))
        self.assertEqual(contract.sample_rate_hz, 48_000)
        self.assertEqual(contract.max_samples, 480_000)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (np.arange(484_801, dtype=np.int64) % 30_000 - 15_000).astype(np.int16)
            audio_path, _ = _write_dataset(root, rate=48_000, samples=source)
            dataset, _ = projected_dataset(root, "train", root / "projection", limit=1)
            # Extraction deliberately applies deterministic evaluation crop policy for every split.
            dataset.split = "validation"
            record = _record_for(dataset, audio_path)

            with patch.object(dataset, "_expert1_audio", wraps=dataset._expert1_audio) as original_method:
                baseline = prepare_expert_inputs(
                    dataset, "expert1", [record], start=0, device="cpu"
                )
                laion = prepare_expert_inputs(
                    dataset, "top2_laion", [record], start=0, device="cpu"
                )
                self.assertEqual(original_method.call_count, 2)
                self.assertEqual(original_method.call_args_list[0].args[1:], (0,))
                self.assertEqual(original_method.call_args_list[1].args[1:], (0,))

            self.assertEqual(tuple(laion["expert1_wavs"].shape), (1, 1, 480_000))
            self.assertEqual(laion["expert1_wavs"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(laion["expert1_wavs"]).all())
            self.assertEqual(laion["captions_raw"], ["  Exact caption  "])
            torch.testing.assert_close(laion["expert1_wavs"], baseline["expert1_wavs"])
            center_start = (len(source) - 480_000) // 2
            expected = torch.from_numpy(
                source[center_start : center_start + 480_000].astype(np.float32) / 32768.0
            ).reshape(1, 1, -1)
            torch.testing.assert_close(laion["expert1_wavs"], expected)

            with self.assertRaisesRegex(ValueError, "WAV changed"):
                prepare_expert_inputs(
                    dataset,
                    "top2_laion",
                    [{**record, "wav_sha256": "0" * 64}],
                    start=0,
                    device="cpu",
                )
            with self.assertRaisesRegex(ValueError, "caption"):
                prepare_expert_inputs(
                    dataset,
                    "top2_laion",
                    [{**record, "raw_caption": "different caption"}],
                    start=0,
                    device="cpu",
                )

    def test_existing_preprocessing_dispatch_and_expert4_lengths_remain_compatible(self):
        self.assertEqual(extraction._PREPROCESS_METHODS["expert1"], "_expert1_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["top2_laion"], "_expert1_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["expert2"], "_expert2_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["top2_mga"], "_expert2_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["top2_msclap"], "_expert2_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["expert3"], "_expert3_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["top2_m2d"], "_expert3_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["m2d_lora"], "_expert3_audio")
        self.assertEqual(extraction._PREPROCESS_METHODS["expert4"], "_expert4_audio")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.arange(8_000, dtype=np.int16)
            audio_path, _ = _write_dataset(root, rate=16_000, samples=source)
            dataset, _ = projected_dataset(root, "train", root / "projection", limit=1)
            record = _record_for(dataset, audio_path)
            inputs = prepare_expert_inputs(
                dataset, "expert4", [record], start=0, device="cpu"
            )
            self.assertEqual(tuple(inputs["expert4_wavs"].shape), (1, 1, 160_000))
            self.assertEqual(inputs["expert4_wavs"].dtype, torch.float32)
            self.assertEqual(inputs["expert4_wav_lens"].dtype, torch.int64)
            self.assertEqual(inputs["expert4_wav_lens"].tolist(), [8_000])


if __name__ == "__main__":
    unittest.main()
