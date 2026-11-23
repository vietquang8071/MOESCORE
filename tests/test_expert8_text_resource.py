from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.expert8_text_resource import (
    EXPERT8_TEXT_MODEL_ID,
    expert8_cache_text_resource_matches,
    local_expert8_text_resource_identity,
    prepare_expert8_text_resource,
)


REVISION = "a" * 40


def _write_bert_files(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "config.json": b'{"model_type":"bert"}',
        "tokenizer.json": b'{"version":"fixture"}',
        "tokenizer_config.json": b'{"do_lower_case":true}',
        "vocab.txt": b"[PAD]\n[UNK]\n",
        "model.safetensors": b"local-frozen-weights",
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}


class Expert8TextResourceTests(unittest.TestCase):
    def test_download_preflight_hashes_local_files_and_loads_both_resources_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "models" / "bert-base-uncased"
            calls = []

            class FakeHubAPI:
                def model_info(self, repo_id, *, revision):
                    calls.append(("model_info", repo_id, revision))
                    return SimpleNamespace(sha=REVISION)

            def snapshot_download(**kwargs):
                calls.append(("snapshot_download", kwargs))
                return str(model_dir)

            class FakeAutoTokenizer:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls.append(("tokenizer", path, kwargs))
                    return object()

            class FakeAutoModel:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls.append(("model", path, kwargs))
                    return SimpleNamespace(config=SimpleNamespace(model_type="bert"))

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = FakeHubAPI

            def fake_download(**kwargs):
                result = snapshot_download(**kwargs)
                _write_bert_files(model_dir)
                return result

            fake_hub.snapshot_download = fake_download
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            fake_transformers.AutoModel = FakeAutoModel
            with patch.dict(sys.modules, {
                "huggingface_hub": fake_hub,
                "transformers": fake_transformers,
            }):
                identity = prepare_expert8_text_resource(model_dir)

            self.assertEqual(identity["original_huggingface_repo_id"], EXPERT8_TEXT_MODEL_ID)
            self.assertEqual(identity["resolved_revision"], REVISION)
            self.assertEqual(identity["model_directory"], str(model_dir.resolve()))
            self.assertEqual(identity["files_sha256"], _write_expected_hashes())
            download = next(call[1] for call in calls if call[0] == "snapshot_download")
            self.assertEqual(download["repo_id"], EXPERT8_TEXT_MODEL_ID)
            self.assertEqual(download["revision"], REVISION)
            self.assertEqual(download["local_dir"], str(model_dir.resolve()))
            self.assertIn("model.safetensors", download["allow_patterns"])
            self.assertEqual(
                [call[0] for call in calls if call[0] in {"tokenizer", "model"}],
                ["tokenizer", "model"],
            )
            for call in calls:
                if call[0] in {"tokenizer", "model"}:
                    self.assertEqual(call[1], str(model_dir.resolve()))
                    self.assertEqual(call[2], {"local_files_only": True})

    def test_local_identity_rejects_missing_model_assets_and_rehashes_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "bert-base-uncased"
            _write_bert_files(model_dir)
            identity = local_expert8_text_resource_identity(
                model_dir, revision=REVISION,
            )
            (model_dir / "model.safetensors").write_bytes(b"changed local weights")
            changed = local_expert8_text_resource_identity(model_dir, revision=REVISION)
            self.assertNotEqual(identity["files_sha256"], changed["files_sha256"])
            (model_dir / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model weights"):
                local_expert8_text_resource_identity(model_dir, revision=REVISION)

    def test_local_resource_identity_matches_only_content_and_revision_compatible_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "bert-base-uncased"
            _write_bert_files(model_dir)
            expected = local_expert8_text_resource_identity(model_dir, revision=REVISION)
            direct_identity = {
                "relevant_dependencies": {
                    "text_encoder_tokenizer_resources": {
                        "expert8_text_model": expected,
                    },
                },
            }
            legacy_identity = {
                "relevant_dependencies": {
                    "text_encoder_tokenizer_resources": {
                        EXPERT8_TEXT_MODEL_ID: {
                            "model_id": EXPERT8_TEXT_MODEL_ID,
                            "cache_views": [{"snapshots": [{
                                "revision": REVISION,
                                "files_sha256": expected["files_sha256"],
                            }]}],
                        },
                    },
                },
            }
            self.assertTrue(expert8_cache_text_resource_matches(direct_identity, expected))
            self.assertTrue(expert8_cache_text_resource_matches(legacy_identity, expected))
            legacy_identity["relevant_dependencies"]["text_encoder_tokenizer_resources"][EXPERT8_TEXT_MODEL_ID]["cache_views"][0]["snapshots"][0]["revision"] = "b" * 40
            self.assertFalse(expert8_cache_text_resource_matches(legacy_identity, expected))

    def test_download_failure_explains_that_kaggle_internet_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_hub = types.ModuleType("huggingface_hub")

            class FailedAPI:
                def model_info(self, *_args, **_kwargs):
                    raise OSError("network disabled")

            fake_hub.HfApi = FailedAPI
            fake_hub.snapshot_download = lambda **_kwargs: self.fail("download must not proceed")
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with self.assertRaisesRegex(RuntimeError, "enable Kaggle Internet"):
                    prepare_expert8_text_resource(Path(temporary) / "bert-base-uncased")


def _write_expected_hashes() -> dict[str, str]:
    values = {
        "config.json": b'{"model_type":"bert"}',
        "tokenizer.json": b'{"version":"fixture"}',
        "tokenizer_config.json": b'{"do_lower_case":true}',
        "vocab.txt": b"[PAD]\n[UNK]\n",
        "model.safetensors": b"local-frozen-weights",
    }
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in values.items()}


if __name__ == "__main__":
    unittest.main()
