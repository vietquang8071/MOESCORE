import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts.kaggle_moe_pipeline import check_text_cache
from scripts.prepare_moe_text_cache import prepare_text_cache


class KaggleTextCacheTests(unittest.TestCase):
    @staticmethod
    def fake_modules():
        config = types.SimpleNamespace(model_type="unit-test")
        transformers = types.SimpleNamespace(**{name: types.SimpleNamespace(from_pretrained=Mock(return_value=config))
            for name in ("AutoConfig", "AutoTokenizer", "BertConfig", "BertTokenizer")})
        hub = types.SimpleNamespace(snapshot_download=Mock(), try_to_load_from_cache=Mock(return_value="cached.bin"))
        return transformers, hub

    def test_cache_setup_uses_a_fresh_process_and_explicit_cache_environment(self):
        fake_hub = types.SimpleNamespace(snapshot_download=Mock(), try_to_load_from_cache=Mock(return_value="cached.bin"))
        fake_transformers = types.SimpleNamespace(AutoConfig=Mock(), AutoTokenizer=Mock())

        def child(command, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "0")
            self.assertEqual(kwargs["env"]["HF_HUB_CACHE"], kwargs["env"]["TRANSFORMERS_CACHE"])
            self.assertIn("--download", command)
            report = {"status": "PASS", "model_ids": ["roberta-base", "bert-base-uncased", "google-bert/bert-base-uncased"]}
            Path(command[command.index("--report-json") + 1]).write_text(json.dumps(report))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.dict("sys.modules", {"huggingface_hub": fake_hub, "transformers": fake_transformers}), \
             patch("scripts.kaggle_moe_pipeline.subprocess.run", side_effect=child) as process:
            result = check_text_cache(Path(tmp), allow_download=True)
            process.assert_called_once()
            self.assertIn("bert-base-uncased", result["model_ids"])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_exact_e2_alias_is_prepared_without_downloading_its_model_weights(self):
        transformers, hub = self.fake_modules()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.dict("sys.modules", {"huggingface_hub": hub, "transformers": transformers}):
            report = prepare_text_cache(tmp, allow_download=True)
        calls = {call.args[0]: call.kwargs for call in hub.snapshot_download.call_args_list}
        self.assertEqual(set(calls), {"roberta-base", "bert-base-uncased", "google-bert/bert-base-uncased"})
        self.assertEqual(calls["bert-base-uncased"]["allow_patterns"], ["*.json", "*.txt", "*.model"])
        self.assertIn("*.safetensors", calls["google-bert/bert-base-uncased"]["allow_patterns"])
        transformers.BertConfig.from_pretrained.assert_called_once_with("bert-base-uncased", local_files_only=True)
        transformers.BertTokenizer.from_pretrained.assert_called_once_with("bert-base-uncased", local_files_only=True)
        self.assertTrue(report["offline_extraction"])
        self.assertFalse(report["model_construction"])

    def test_missing_short_alias_config_is_rejected_even_when_namespaced_bert_exists(self):
        transformers, hub = self.fake_modules()
        transformers.BertConfig.from_pretrained.side_effect = OSError("config.json not cached")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.dict("sys.modules", {"huggingface_hub": hub, "transformers": transformers}):
            with self.assertRaisesRegex(RuntimeError, "bert-base-uncased"):
                prepare_text_cache(tmp)
        hub.snapshot_download.assert_not_called()

    def test_offline_cache_validation_does_not_enable_network(self):
        def child(command, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
            self.assertNotIn("--download", command)
            Path(command[command.index("--report-json") + 1]).write_text('{"status":"PASS"}')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch("scripts.kaggle_moe_pipeline.subprocess.run", side_effect=child):
            self.assertEqual(check_text_cache(tmp)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
