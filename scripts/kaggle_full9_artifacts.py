"""Transport only verified feature-cache artifacts between Kaggle sessions."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time

from data.moe_expert_cache import sha256_file
from data.moe_feature_store import _atomic_json, validate_feature_store
from model.expert_adapters.registry import profile_contract


def validate_full9_store(store_root, expected_counts=None):
    result = {}
    expected_width = profile_contract("full_9")[2]
    for split in ("train", "validation", "test"):
        manifest = validate_feature_store(store_root, split=split, require_labels=split != "test")
        count = sum(shard["count"] for shard in manifest["shards"])
        if (manifest.get("expert_profile") != "full_9" or len(manifest["expert_order"]) != 9
                or manifest["feature_width"] != expected_width or (split == "test" and manifest["labels_present"])):
            raise ValueError(f"{split}: incompatible full_9 feature store")
        if expected_counts and count != expected_counts[split]:
            raise ValueError(f"{split}: expected {expected_counts[split]} samples, found {count}")
        result[split] = {"samples": count, "store_identity": manifest["store_identity_sha256"]}
    return result


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


def check_private_upload_target(handle, *, api=None):
    """Reject updates to a public dataset; new kagglehub datasets are private."""
    if api is None:
        from kaggle import api as api
        api.authenticate()
    try:
        view = api.dataset_view(handle)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", getattr(exc, "status", None))
        if status == 404:
            return False  # New handle; kagglehub creates it private.
        raise RuntimeError(f"Cannot verify Kaggle dataset visibility for {handle}: {exc}") from exc
    private = (view.get("isPrivate", view.get("is_private")) if isinstance(view, dict)
               else getattr(view, "isPrivate", getattr(view, "is_private", None)))
    if private is not True:
        raise ValueError(f"Refusing upload to public or unverifiable Kaggle dataset {handle}")
    return True


def upload_feature_cache(handle, upload_directory, *, version_notes=""):
    """Upload a private-by-default Kaggle Dataset and verify its manifest online."""
    import kagglehub
    directory = Path(upload_directory).resolve()
    local = json.loads((directory / "upload_manifest.json").read_text())
    if sha256_file(directory / local["archive_name"]) != local["archive_sha256"]:
        raise ValueError("Local archive changed after packaging")
    check_private_upload_target(handle)
    kagglehub.dataset_upload(handle=handle, local_dataset_dir=str(directory), version_notes=version_notes)
    remote = None
    last_error = None
    for _ in range(12):
        try:
            with tempfile.TemporaryDirectory(prefix="moescore-kaggle-upload-check-") as temporary:
                downloaded = Path(kagglehub.dataset_download(handle, path="upload_manifest.json",
                    output_dir=temporary, force_download=True))
                remote = json.loads(downloaded.read_text())
            break
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(10)
    if remote is None:
        raise RuntimeError(f"Kaggle upload could not be verified: {last_error}")
    if remote != local:
        raise ValueError("Uploaded dataset manifest differs from local cache identity")
    with tempfile.TemporaryDirectory(prefix="moescore-kaggle-archive-check-") as temporary:
        downloaded_archive = Path(kagglehub.dataset_download(handle, path="feature_cache.tar",
            output_dir=temporary, force_download=True))
        if sha256_file(downloaded_archive) != local["archive_sha256"]:
            raise ValueError("Uploaded dataset archive SHA-256 differs from local cache")
    return {"status": "PASS", "url": f"https://www.kaggle.com/datasets/{handle}",
            "archive_sha256": local["archive_sha256"]}
