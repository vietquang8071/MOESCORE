"""Privacy and integrity checks for Kaggle feature-cache uploads."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import kaggle_full9_artifacts as artifacts


class FakeKaggleError(Exception):
    def __init__(self, status, message="Kaggle request failed"):
        super().__init__(message)
        self.status = status
        self.response = SimpleNamespace(status_code=status)


class MetadataApi:
    def __init__(self, metadata=None, error=None, auth_error=None):
        self.metadata = metadata
        self.error = error
        self.auth_error = auth_error
        self.auth_calls = 0
        self.metadata_calls = 0

    def authenticate(self):
        self.auth_calls += 1
        if self.auth_error:
            raise self.auth_error

    def dataset_metadata(self, handle, path):
        self.metadata_calls += 1
        if self.error:
            raise self.error
        metadata_path = Path(path) / "dataset-metadata.json"
        metadata_path.write_text(json.dumps(self.metadata))
        return str(metadata_path)


class KagglePrivacyTests(unittest.TestCase):
    def test_new_dataset_only_allows_verified_http_404(self):
        api = MetadataApi(error=FakeKaggleError(404, "not found"))
        self.assertFalse(artifacts.check_private_upload_target("owner/private-cache", api=api))
        self.assertEqual(api.auth_calls, 1)

    def test_existing_private_dataset_is_allowed(self):
        api = MetadataApi(metadata={"isPrivate": True})
        self.assertTrue(artifacts.check_private_upload_target("owner/private-cache", api=api))

    def test_existing_public_dataset_is_rejected(self):
        api = MetadataApi(metadata={"isPrivate": False})
        with self.assertRaisesRegex(ValueError, "public"):
            artifacts.check_private_upload_target("owner/public-cache", api=api)

    def test_missing_visibility_metadata_fails_closed(self):
        api = MetadataApi(metadata={"title": "cache"})
        with self.assertRaisesRegex(RuntimeError, "visibility"):
            artifacts.check_private_upload_target("owner/unknown-cache", api=api)

    def test_invalid_handle_is_rejected_before_api_call(self):
        api = MetadataApi(metadata={"isPrivate": True})
        for handle in ("owner", "owner/slug/extra", "/slug", "owner/", "owner name/slug"):
            with self.subTest(handle=handle), self.assertRaisesRegex(ValueError, "owner/dataset-slug"):
                artifacts.check_private_upload_target(handle, api=api)
        self.assertEqual(api.auth_calls, 0)

    def test_placeholder_username_is_rejected_with_configuration_guidance(self):
        api = MetadataApi(metadata={"isPrivate": True})
        with self.assertRaisesRegex(ValueError, "real Kaggle username"):
            artifacts.check_private_upload_target(
                "YOUR_KAGGLE_USERNAME/moescore-full9-feature-cache", api=api
            )
        self.assertEqual(api.auth_calls, 0)

    def test_authentication_failure_is_clear_and_stops_metadata_lookup(self):
        api = MetadataApi(metadata={"isPrivate": True}, auth_error=SystemExit(1))
        with self.assertRaisesRegex(RuntimeError, "authentication"):
            artifacts.check_private_upload_target("owner/cache", api=api)
        self.assertEqual(api.metadata_calls, 0)

    def test_server_error_is_not_mistaken_for_missing_dataset(self):
        api = MetadataApi(error=FakeKaggleError(503, "unavailable"))
        with self.assertRaisesRegex(RuntimeError, "503"):
            artifacts.check_private_upload_target("owner/cache", api=api)

    def test_timeout_is_not_mistaken_for_missing_dataset(self):
        api = MetadataApi(error=TimeoutError("request timed out"))
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            artifacts.check_private_upload_target("owner/cache", api=api)

    def test_text_404_without_http_status_is_not_treated_as_missing(self):
        api = MetadataApi(error=RuntimeError("404 not found"))
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            artifacts.check_private_upload_target("owner/cache", api=api)

    def test_incompatible_kaggle_api_is_reported(self):
        api = SimpleNamespace(authenticate=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "dataset_metadata"):
            artifacts.check_private_upload_target("owner/cache", api=api)


class UploadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "source/feature_cache/experts/expert1").mkdir(parents=True)
        (self.root / "source/feature_cache/experts/expert1/shard.npz").write_bytes(b"verified shard")
        self.upload = self.root / "upload"
        artifacts.package_feature_cache(self.root / "source", self.upload)

    def tearDown(self):
        self.temporary.cleanup()

    def kagglehub_stub(self, calls):
        return SimpleNamespace(
            dataset_upload=lambda **kwargs: calls.append(kwargs),
            # Verification must bypass Kagglehub's notebook datasource resolver.
            dataset_download=lambda *args, **kwargs: self.fail("resolver-based download must not be used"),
        )

    def upload_api(self, remote_manifest=None, remote_archive=None, metadata=None):
        api = MetadataApi(metadata=metadata or {"isPrivate": True})

        def dataset_download_file(handle, file_name, path, force, quiet, unzip):
            if file_name == "upload_manifest.json" and remote_manifest is not None:
                content = json.dumps(remote_manifest).encode()
            elif file_name == "feature_cache.tar" and remote_archive is not None:
                content = remote_archive
            else:
                content = (self.upload / file_name).read_bytes()
            return Path(path, file_name).write_bytes(content)

        api.dataset_download_file = dataset_download_file
        return api

    def test_public_dataset_prevents_kagglehub_upload(self):
        api = MetadataApi(metadata={"isPrivate": False})
        calls = []
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            with self.assertRaisesRegex(ValueError, "public"):
                artifacts.upload_feature_cache("owner/cache", self.upload, api=api)
        self.assertEqual(calls, [])

    def test_placeholder_handle_fails_before_upload_directory_or_sdk_access(self):
        with self.assertRaisesRegex(ValueError, "real Kaggle username"):
            artifacts.upload_feature_cache(
                "YOUR_KAGGLE_USERNAME/moescore-full9-feature-cache", self.root / "missing-upload"
            )

    def test_missing_direct_download_method_prevents_upload(self):
        calls = []
        api = MetadataApi(metadata={"isPrivate": True})
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            with self.assertRaisesRegex(RuntimeError, "dataset_download_file"):
                artifacts.upload_feature_cache("owner/cache", self.upload, api=api)
        self.assertEqual(calls, [])

    def test_successful_upload_checks_local_and_remote_sha256(self):
        calls = []
        api = self.upload_api()
        manifest = json.loads((self.upload / "upload_manifest.json").read_text())
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            result = artifacts.upload_feature_cache("owner/cache", self.upload, version_notes="smoke", api=api)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["archive_sha256"], manifest["archive_sha256"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(api.auth_calls, 1)

    def test_remote_manifest_mismatch_fails(self):
        calls = []
        api = self.upload_api(remote_manifest={**json.loads((self.upload / "upload_manifest.json").read_text()), "stores": {"wrong": True}})
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            with patch.object(artifacts.time, "sleep"):
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    artifacts.upload_feature_cache("owner/cache", self.upload, api=api)
        self.assertEqual(len(calls), 1)

    def test_remote_archive_sha256_mismatch_fails(self):
        calls = []
        api = self.upload_api(remote_archive=b"different archive")
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                artifacts.upload_feature_cache("owner/cache", self.upload, api=api)
        self.assertEqual(len(calls), 1)

    def test_changed_local_archive_fails_before_upload(self):
        calls = []
        api = self.upload_api()
        with (self.upload / "feature_cache.tar").open("ab") as stream:
            stream.write(b"changed")
        with patch.dict(sys.modules, {"kagglehub": self.kagglehub_stub(calls)}):
            with self.assertRaisesRegex(ValueError, "Local archive.*SHA-256"):
                artifacts.upload_feature_cache("owner/cache", self.upload, api=api)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
