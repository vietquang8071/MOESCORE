"""Download and fingerprint the fixed Expert8 BERT tokenizer/model files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


EXPERT8_TEXT_MODEL_ID = "google-bert/bert-base-uncased"
EXPERT8_TEXT_MODEL_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def local_expert8_text_resource_identity(
    model_dir: str | Path,
    *,
    revision: str,
    repo_id: str = EXPERT8_TEXT_MODEL_ID,
) -> dict[str, Any]:
    """Return the content identity of the files the Expert8 BERT loader can use."""
    if repo_id != EXPERT8_TEXT_MODEL_ID:
        raise ValueError(f"Expert8 text resource must be {EXPERT8_TEXT_MODEL_ID!r}, got {repo_id!r}")
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("Expert8 BERT revision must be a resolved 40-character Hugging Face commit SHA")
    if not isinstance(model_dir, (str, Path)) or not str(model_dir):
        raise TypeError("Expert8 text_model_dir must be an explicit local filesystem path")
    root = Path(model_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Expert8 BERT model directory is not a directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"Expert8 BERT resource file escapes its model directory: {path}")
        files[relative.as_posix()] = _sha256_file(resolved)
    missing = []
    if "config.json" not in files:
        missing.append("config.json")
    if "tokenizer_config.json" not in files:
        missing.append("tokenizer_config.json")
    if not {"tokenizer.json", "vocab.txt"}.intersection(files):
        missing.append("tokenizer.json or vocab.txt")
    if not any(
        name in {"model.safetensors", "pytorch_model.bin"}
        or (Path(name).suffix in {".safetensors", ".bin"}
            and Path(name).name.startswith(("model-", "pytorch_model-")))
        for name in files
    ):
        missing.append("model weights (.safetensors or .bin)")
    if missing:
        raise FileNotFoundError(f"Expert8 BERT model directory is missing required files: {missing}")
    return {
        "original_huggingface_repo_id": repo_id,
        "resolved_revision": revision,
        "model_directory": str(root),
        "files_sha256": files,
    }


def expert8_cache_text_resource_matches(
    cache_identity: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Accept a cached E8 output only when it used this BERT revision and content."""
    repo_id = expected.get("original_huggingface_repo_id")
    revision = expected.get("resolved_revision")
    files = expected.get("files_sha256")
    if (repo_id != EXPERT8_TEXT_MODEL_ID or not isinstance(revision, str)
            or not isinstance(files, Mapping) or not files):
        return False
    dependencies = cache_identity.get("relevant_dependencies")
    resources = dependencies.get("text_encoder_tokenizer_resources") if isinstance(dependencies, Mapping) else None
    if not isinstance(resources, Mapping):
        return False
    local = resources.get("expert8_text_model")
    if isinstance(local, Mapping):
        local_files = local.get("files_sha256")
        return (
            local.get("original_huggingface_repo_id") == repo_id
            and local.get("resolved_revision") == revision
            and isinstance(local_files, Mapping)
            and all(local_files.get(name) == digest for name, digest in files.items())
        )
    cached = resources.get(repo_id)
    if not isinstance(cached, Mapping) or cached.get("model_id") != repo_id:
        return False
    views = cached.get("cache_views")
    if not isinstance(views, list):
        return False
    return any(
        snapshot.get("revision") == revision
        and isinstance(snapshot.get("files_sha256"), Mapping)
        and all(snapshot["files_sha256"].get(name) == digest for name, digest in files.items())
        for view in views if isinstance(view, Mapping)
        for snapshot in view.get("snapshots", []) if isinstance(snapshot, Mapping)
    )


def prepare_expert8_text_resource(model_dir: str | Path) -> dict[str, Any]:
    """Download a pinned BERT snapshot and verify its tokenizer/model load offline."""
    target = Path(model_dir).expanduser().resolve(strict=False)
    try:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(EXPERT8_TEXT_MODEL_ID, revision="main").sha
        if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
            raise ValueError(f"Hugging Face returned an invalid resolved revision: {revision!r}")
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=EXPERT8_TEXT_MODEL_ID,
            revision=revision,
            local_dir=str(target),
            allow_patterns=list(EXPERT8_TEXT_MODEL_ALLOW_PATTERNS),
        )
    except Exception as exc:
        raise RuntimeError(
            "Expert8 BERT preflight could not resolve/download google-bert/bert-base-uncased; "
            "enable Kaggle Internet in kernel metadata before starting this notebook"
        ) from exc

    identity = local_expert8_text_resource_identity(target, revision=revision)
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
        model = AutoModel.from_pretrained(str(target), local_files_only=True)
        if getattr(getattr(model, "config", None), "model_type", None) != "bert":
            raise ValueError("Downloaded Expert8 text model config is not BERT")
        del tokenizer, model
    except Exception as exc:
        raise RuntimeError(
            f"Expert8 BERT resources were downloaded to {target} but failed local-only loading: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "EXPERT8_TEXT_MODEL_ALLOW_PATTERNS",
    "EXPERT8_TEXT_MODEL_ID",
    "expert8_cache_text_resource_matches",
    "local_expert8_text_resource_identity",
    "prepare_expert8_text_resource",
]
