"""Transport only verified feature-cache artifacts between Kaggle sessions."""
from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from data.moe_expert_cache import sha256_file
from data.moe_feature_store import _atomic_json, validate_feature_store
from model.expert_adapters.registry import profile_contract


def validate_profile_store(store_root, *, profile, expected_counts=None):
    if profile not in {"full_9", "full_10"}:
        raise ValueError("Kaggle feature-cache packaging supports full_9 and full_10 stores")
    result = {}
    expected_order, _, expected_width = profile_contract(profile)
    for split in ("train", "validation", "test"):
        manifest = validate_feature_store(store_root, split=split, require_labels=split != "test")
        count = sum(shard["count"] for shard in manifest["shards"])
        if (manifest.get("expert_profile") != profile or tuple(manifest["expert_order"]) != expected_order
                or manifest["feature_width"] != expected_width or (split == "test" and manifest["labels_present"])):
            raise ValueError(f"{split}: incompatible {profile} feature store")
        if expected_counts and count != expected_counts[split]:
            raise ValueError(f"{split}: expected {expected_counts[split]} samples, found {count}")
        result[split] = {"samples": count, "store_identity": manifest["store_identity_sha256"]}
    return result


def validate_full9_store(store_root, expected_counts=None):
    """Backward-compatible Full9 validator used by the existing notebook."""
    return validate_profile_store(store_root, profile="full_9", expected_counts=expected_counts)


def validate_full10_store(store_root, expected_counts=None):
    return validate_profile_store(store_root, profile="full_10", expected_counts=expected_counts)


def package_feature_cache(output_root, upload_directory, *, stores=None):
    cache = Path(output_root).resolve() / "feature_cache"
    if not cache.is_dir():
        raise FileNotFoundError(f"Feature cache is missing: {cache}")
    target = Path(upload_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    allowed = {"feature_cache.tar", "upload_manifest.json"}
    unrelated = [p.name for p in target.iterdir() if p.name not in allowed]
    if unrelated:
        raise ValueError(f"Upload directory contains unrelated files: {unrelated}")
    archive = target / "feature_cache.tar"
    temporary = target / "feature_cache.tar.tmp"
    try:
        with tarfile.open(temporary, "w") as tar:
            for path in sorted(cache.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Cache contains a symlink, refusing upload: {path}")
                if path.is_file():
                    tar.add(path, arcname=str(Path("feature_cache") / path.relative_to(cache)), recursive=False)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {"schema_version": 1, "archive_name": archive.name,
                "archive_sha256": sha256_file(archive), "archive_size_bytes": archive.stat().st_size,
                "stores": stores or {}}
    _atomic_json(target / "upload_manifest.json", manifest)
    return manifest


def restore_feature_cache(upload_directory, output_root):
    source = Path(upload_directory).resolve()
    manifest = json.loads((source / "upload_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("archive_name") != "feature_cache.tar":
        raise ValueError("Unsupported feature-cache upload manifest")
    archive = source / "feature_cache.tar"
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise ValueError("Feature-cache archive SHA-256 mismatch")
    target = Path(output_root).resolve()
    if (target / "feature_cache").exists():
        raise FileExistsError("Destination feature_cache already exists; validate it instead of overwriting")
    target.mkdir(parents=True, exist_ok=True)
    partial = target / "feature_cache.restore-partial"
    if partial.exists():
        raise FileExistsError(f"Previous incomplete cache restore exists: {partial}")
    partial.mkdir()
    try:
        with tarfile.open(archive, "r") as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or not name.parts
                        or name.parts[0] != "feature_cache" or not member.isfile()):
                    raise ValueError(f"Unsafe feature-cache archive member: {member.name}")
                relative = Path(*name.parts[1:])
                destination = partial / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as reader, destination.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
        partial.replace(target / "feature_cache")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


_HANDLE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def _validate_dataset_handle(handle):
    if not isinstance(handle, str) or handle.count("/") != 1:
        raise ValueError("Kaggle dataset handle must use the format owner/dataset-slug")
    owner, slug = handle.split("/")
    if not _HANDLE_PART_RE.fullmatch(owner) or not _HANDLE_PART_RE.fullmatch(slug):
        raise ValueError("Kaggle dataset handle must use the format owner/dataset-slug")
    if owner.upper() in {"YOUR_KAGGLE_USERNAME", "INSERT_OWNER_SLUG_HERE"} or owner.upper().startswith("YOUR_"):
        raise ValueError(
            "Replace the placeholder owner in the Kaggle dataset handle with your real Kaggle username "
            "(for example, set KAGGLE_USERNAME in the notebook configuration)."
        )
    return owner, slug


def _resolve_kaggle_api(api=None):
    if api is not None:
        return api
    try:
        from kaggle import api as kaggle_api
    except ImportError as exc:
        raise RuntimeError("The Kaggle API package is unavailable; install the declared kaggle dependency.") from exc
    return kaggle_api


def _http_status(exc):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        candidates = (
            getattr(response, "status_code", None),
            getattr(current, "status", None),
            getattr(current, "status_code", None),
        )
        for value in candidates:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                match = re.match(r"^\s*(\d{3})\b", value)
                if match:
                    return int(match.group(1))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return None


def _authenticate_kaggle_api(api):
    authenticate = getattr(api, "authenticate", None)
    if not callable(authenticate):
        raise RuntimeError("Installed Kaggle API is incompatible: expected callable authenticate().")
    try:
        authenticate()
    except SystemExit as exc:
        raise RuntimeError(
            "Kaggle authentication failed; configure a valid Kaggle token or kaggle.json credentials."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Kaggle authentication failed; verify the configured Kaggle credentials: {exc}") from exc


def _read_dataset_privacy(api, handle):
    metadata_method = getattr(api, "dataset_metadata", None)
    if not callable(metadata_method):
        raise RuntimeError(
            "Installed Kaggle API is incompatible: expected dataset_metadata(dataset, path) "
            "for supported dataset visibility lookup."
        )
    with tempfile.TemporaryDirectory(prefix="moescore-kaggle-metadata-") as temporary:
        try:
            metadata_file = metadata_method(handle, temporary)
        except Exception as exc:
            status = _http_status(exc)
            if status == 404:
                return None  # Verified HTTP 404: handle is new.
            if status in (401, 403):
                raise RuntimeError(
                    f"Kaggle authentication/authorization failed while reading dataset metadata for {handle} "
                    f"(HTTP {status}): {exc}"
                ) from exc
            status_text = f"HTTP {status}: " if status is not None else ""
            raise RuntimeError(
                f"Cannot verify Kaggle dataset visibility for {handle}; {status_text}{exc}"
            ) from exc

        if not isinstance(metadata_file, (str, Path)):
            raise RuntimeError("Kaggle dataset_metadata() returned an incompatible response; expected a metadata path.")
        metadata_path = Path(metadata_file)
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Kaggle dataset_metadata() did not produce a readable metadata file for {handle}."
            )
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot parse Kaggle dataset metadata for {handle}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Kaggle dataset metadata for {handle} has an incompatible response type.")
    info = metadata.get("info", metadata)
    if not isinstance(info, dict):
        raise RuntimeError(f"Kaggle dataset metadata for {handle} has no valid info object.")
    private = info.get("isPrivate", info.get("is_private"))
    if type(private) is not bool:
        raise RuntimeError(f"Cannot verify Kaggle dataset visibility for {handle}: isPrivate metadata is missing/unknown.")
    return private


def check_private_upload_target(handle, *, api=None):
    """Allow only verified-private existing datasets or a verified HTTP 404 new handle."""
    _validate_dataset_handle(handle)
    api = _resolve_kaggle_api(api)
    _authenticate_kaggle_api(api)
    private = _read_dataset_privacy(api, handle)
    if private is None:
        return False  # kagglehub.dataset_upload creates new datasets as private.
    if private is False:
        raise ValueError(f"Refusing upload to public Kaggle dataset {handle}; make it private before uploading.")
    return True


def _load_and_validate_upload_manifest(directory):
    manifest_path = directory / "upload_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read a valid upload_manifest.json: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or malformed feature-cache upload manifest")
    if manifest.get("archive_name") != "feature_cache.tar":
        raise ValueError("Upload manifest must identify feature_cache.tar")
    expected_sha256 = manifest.get("archive_sha256")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("Upload manifest contains an invalid archive SHA-256")
    expected_size = manifest.get("archive_size_bytes")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("Upload manifest contains an invalid archive size")
    if not isinstance(manifest.get("stores"), dict):
        raise ValueError("Upload manifest contains invalid store metadata")
    archive_path = directory / "feature_cache.tar"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Feature-cache archive is missing: {archive_path}")
    if archive_path.stat().st_size != expected_size or sha256_file(archive_path) != expected_sha256:
        raise ValueError("Local archive changed after packaging (size/SHA-256 mismatch)")
    return manifest


@contextmanager
def _download_remote_dataset_file(api, handle, file_name):
    download_method = getattr(api, "dataset_download_file", None)
    if not callable(download_method):
        raise RuntimeError(
            "Installed Kaggle API is incompatible: expected dataset_download_file() for direct remote verification."
        )
    last_error = None
    for attempt in range(12):
        temporary = tempfile.TemporaryDirectory(prefix="moescore-kaggle-direct-download-")
        target = Path(temporary.name) / file_name
        try:
            download_method(
                handle,
                file_name,
                path=temporary.name,
                force=True,
                quiet=True,
                unzip=True,
            )
        except TypeError as exc:
            temporary.cleanup()
            raise RuntimeError(
                "Installed Kaggle API dataset_download_file() has an incompatible signature."
            ) from exc
        except Exception as exc:
            status = _http_status(exc)
            temporary.cleanup()
            if status in (401, 403):
                raise RuntimeError(f"Kaggle authentication/authorization failed during remote verification: {exc}") from exc
            last_error = exc
        else:
            if target.is_file():
                try:
                    yield target
                finally:
                    temporary.cleanup()
                return
            temporary.cleanup()
            last_error = RuntimeError(f"dataset_download_file() did not write the expected file {file_name}")
        if attempt < 11:
            time.sleep(10)
    raise RuntimeError(f"Kaggle upload could not be verified with direct dataset download: {last_error}") from last_error


def upload_feature_cache(handle, upload_directory, *, version_notes="", api=None):
    """Upload a private-by-default Kaggle Dataset and verify uploaded manifest/archive bytes."""
    _validate_dataset_handle(handle)
    directory = Path(upload_directory).resolve()
    local = _load_and_validate_upload_manifest(directory)
    kaggle_api = _resolve_kaggle_api(api)
    check_private_upload_target(handle, api=kaggle_api)
    if not callable(getattr(kaggle_api, "dataset_download_file", None)):
        raise RuntimeError(
            "Installed Kaggle API is incompatible: expected dataset_download_file() for direct remote verification."
        )

    import kagglehub

    kagglehub.dataset_upload(handle=handle, local_dataset_dir=str(directory), version_notes=version_notes)
    with _download_remote_dataset_file(kaggle_api, handle, "upload_manifest.json") as downloaded_manifest:
        try:
            remote = json.loads(downloaded_manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Uploaded dataset manifest is unreadable: {exc}") from exc
    if remote != local:
        raise ValueError("Uploaded dataset manifest differs from local cache identity")
    with _download_remote_dataset_file(kaggle_api, handle, "feature_cache.tar") as downloaded_archive:
        remote_sha256 = sha256_file(downloaded_archive)
        if remote_sha256 != local["archive_sha256"]:
            raise ValueError("Uploaded dataset archive SHA-256 differs from local cache")
    return {
        "status": "PASS",
        "url": f"https://www.kaggle.com/datasets/{handle}",
        "archive_sha256": local["archive_sha256"],
    }
