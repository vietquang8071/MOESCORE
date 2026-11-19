"""Cache identity regression tests for the paired cross-feature representation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model.expert_adapters.registry import ExpertInputContract
from scripts import extract_moe_feature_store as extraction


class PairedCrossCacheIdentityTests(unittest.TestCase):
    DIMENSIONS = {
        "expert1": 2049,
        "expert2": 4097,
        "top2_m2d": 3073,
        "top2_msclap": 4097,
        "top2_laion": 2049,
        "top2_mga": 4097,
        "expert3": 3840,
        "expert4": 1024,
        "m2d_lora": 3073,
        "m2d_lora_coral": 3073,
    }
    LEGACY_PAIRED_DIMENSIONS = {
        "expert1": 2560,
        "expert2": 5120,
        "top2_m2d": 3840,
        "top2_msclap": 5120,
        "top2_laion": 2560,
        "top2_mga": 5120,
    }
    UNCHANGED = ("expert3", "expert4", "m2d_lora", "m2d_lora_coral")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "project"
        self.cache_root = self.root / "hf-cache" / "hub"
        self._create_sources()
        self._create_checkpoints()
        self._create_text_snapshots()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_source(self, relative_path: str, content: str = "fixture source v1\n") -> Path:
        path = self.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _create_sources(self):
        self._write_source("model/expert_adapters/base.py")
        for expert, relative_path in extraction._ADAPTER_SOURCE_FILES.items():
            self._write_source(relative_path)
            if expert.startswith("top2_") or expert in {"m2d_lora", "m2d_lora_coral"}:
                self._write_source("model/expert_adapters/top2_common.py")
            if expert == "m2d_lora":
                self.runtime_source = self._write_source(
                    "model/expert_adapters/m2d_lora_runtime.py"
                )
            if expert == "m2d_lora_coral":
                self.coral_runtime_source = self._write_source(
                    "model/expert_adapters/m2d_lora_coral_runtime.py"
                )
                self._write_source("model/expert_adapters/m2d_lora_runtime.py")
            if expert == "expert2":
                self._write_source("expert2_finetune/expert2_model.py")
            if expert == "expert4":
                for filename in (
                    "inference.py", "trainer.py", "model.py", "audio_encoder.py",
                    "text_encoder.py", "seq_coattn.py", "source_compatibility.py",
                ):
                    self._write_source(f"expert4_finetune/{filename}")
        self.cross_source = self._write_source(
            "model/expert_adapters/paired_cross_features.py"
        )

    def _create_checkpoints(self):
        self.config = {"runtime": {"device": "cpu", "cache_dir": str(self.cache_root)}, "experts": {}}
        for expert in self.DIMENSIONS:
            fields = extraction.FULL10_CHECKPOINT_FIELDS.get(
                expert, extraction.CHECKPOINT_FIELDS.get(expert, ("checkpoint",))
            )
            section = {}
            for field in fields:
                path = self.root / "weights" / f"{expert}-{field}.pth"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{expert}:{field}:v1".encode())
                section[field] = str(path)
            self.config["experts"][expert] = section

    def _create_text_snapshots(self):
        for model_id in set(
            model_id
            for names in extraction._TEXT_RESOURCE_IDS.values()
            for model_id in names
        ):
            model_root = self.cache_root / ("models--" + model_id.replace("/", "--"))
            snapshot = model_root / "snapshots" / "fixture-revision"
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text('{"fixture": 1}', encoding="utf-8")
            (snapshot / "tokenizer.json").write_text('{"vocab": ["fixture"]}', encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"fixture-weights-v1")
            refs = model_root / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("fixture-revision", encoding="utf-8")

    def _identity(self, expert: str, *, feature_dim: int | None = None, config=None):
        contract = extraction.EXPERT_INPUT_CONTRACTS.get(expert)
        if contract is None:
            contract = extraction.TOP2_EXPERT_INPUT_CONTRACTS.get(expert)
        if contract is None:
            audio_contract = (
                extraction.EXPERT_INPUT_CONTRACTS["expert2"]
                if expert in {"top2_mga"}
                else extraction.EXPERT_INPUT_CONTRACTS["expert3"]
            )
            contract = ExpertInputContract(
                audio_key=audio_contract.audio_key,
                audio_shape=audio_contract.audio_shape,
                sample_rate_hz=audio_contract.sample_rate_hz,
                max_samples=audio_contract.max_samples,
                caption_policy=audio_contract.caption_policy,
            )
        with (
            patch.object(extraction, "PROJECT_ROOT", self.source_root),
            patch.object(extraction, "_source_files_under", return_value=[]),
            patch.object(extraction, "_TEXT_SNAPSHOT_CACHE", {}),
        ):
            return extraction.compute_expert_cache_identity(
                expert,
                self.config if config is None else config,
                contract=contract,
                feature_dim=feature_dim or self.DIMENSIONS[expert],
                device="cpu",
            )

    def test_all_six_changed_experts_receive_distinct_real_cache_identities(self):
        for expert in extraction._PAIRED_CROSS_EXPERTS:
            with self.subTest(expert=expert):
                new_identity = self._identity(expert)
                old_width = self.LEGACY_PAIRED_DIMENSIONS[expert]
                with patch.object(extraction, "_PAIRED_CROSS_EXPERTS", frozenset()):
                    old_identity = self._identity(expert, feature_dim=old_width)
                self.assertNotEqual(old_identity["cache_id"], new_identity["cache_id"])
                self.assertEqual(
                    new_identity["score_contract"]["feature_representation"],
                    "paired_cross_4d_plus_1_v2",
                )
                self.assertIn(
                    "model/expert_adapters/paired_cross_features.py",
                    new_identity["implementation_provenance"],
                )

    def test_helper_source_change_changes_selected_expert_cache_identity(self):
        before = self._identity("top2_m2d")
        self.cross_source.write_text("fixture source v2\n", encoding="utf-8")
        after = self._identity("top2_m2d")
        self.assertNotEqual(before["cache_id"], after["cache_id"])

    def test_text_snapshot_content_change_changes_actual_m2d_lora_identity(self):
        before = self._identity("m2d_lora")
        bert_root = self.cache_root / "models--google-bert--bert-base-uncased"
        weight_file = bert_root / "snapshots" / "fixture-revision" / "model.safetensors"
        weight_file.write_bytes(b"fixture-weights-v2")
        after = self._identity("m2d_lora")
        self.assertNotEqual(before["cache_id"], after["cache_id"])
        self.assertEqual(
            after["relevant_dependencies"]["text_encoder_tokenizer_resources"][
                "google-bert/bert-base-uncased"
            ]["model_id"],
            "google-bert/bert-base-uncased",
        )

    def test_e10_identity_uses_its_own_runtime_and_profile_addition_preserves_old_ids(self):
        full10 = self.config
        full9 = {
            "runtime": full10["runtime"],
            "experts": {name: value for name, value in full10["experts"].items()
                        if name != "m2d_lora_coral"},
        }
        for expert in (name for name in self.DIMENSIONS if name != "m2d_lora_coral"):
            self.assertEqual(
                self._identity(expert, config=full9)["cache_id"],
                self._identity(expert, config=full10)["cache_id"],
                expert,
            )
        before = self._identity("m2d_lora_coral")
        self.coral_runtime_source.write_text("fixture E10 runtime v2\n", encoding="utf-8")
        after = self._identity("m2d_lora_coral")
        self.assertNotEqual(before["cache_id"], after["cache_id"])
        self.assertEqual(after["preprocessing_provenance"]["dataset_method"], "_expert3_audio")
        self.assertEqual(after["input_contract"]["sample_rate_hz"], 16000)

    def test_unchanged_expert_identities_do_not_depend_on_new_helper_policy(self):
        for expert in self.UNCHANGED:
            with self.subTest(expert=expert):
                current = self._identity(expert)
                with patch.object(extraction, "_PAIRED_CROSS_EXPERTS", frozenset()):
                    baseline = self._identity(expert)
                self.assertEqual(current["cache_id"], baseline["cache_id"])
                self.assertNotIn("feature_representation", current["score_contract"])
                self.assertNotIn(
                    "model/expert_adapters/paired_cross_features.py",
                    current["implementation_provenance"],
                )


if __name__ == "__main__":
    unittest.main()
