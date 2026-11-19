from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_E10_SHA256 = "d6edbec6b724be48a18f618c98a2e25d044f06108e8715d86b58c59e97e6cbdb"
TEST_HF_REQUIRED_MODELS = (
    "roberta-base", "bert-base-uncased", "google-bert/bert-base-uncased", "gpt2", "facebook/bart-base"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _support_namespace():
    notebook = json.loads((ROOT / "notebooks/MOESCORE_Full10_Kaggle.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "# MOE10_KAGGLE_SUPPORT" in source:
            namespace = {"__name__": "moe10_notebook_support_test"}
            exec(compile(source, "MOESCORE_Full10_Kaggle.ipynb:support", "exec"), namespace)
            return namespace
    raise AssertionError("Full10 notebook has no self-contained MOE10_KAGGLE_SUPPORT cell")


def _make_hf_archive(root: Path, *, member_name="hub/models--roberta-base/blobs/blob1", link_target=""):
    archive = root / "hf_hub_cache.tar"
    with tarfile.open(archive, "w") as tar:
        for model_id in TEST_HF_REQUIRED_MODELS:
            model_dir = "models--" + model_id.replace("/", "--")
            payload = f"resource:{model_id}".encode()
            info = tarfile.TarInfo(f"hub/{model_dir}/blobs/resource.bin")
            info.size = len(payload)
            import io

            tar.addfile(info, io.BytesIO(payload))
            snapshot_dir = f"hub/{model_dir}/snapshots/test-revision"
            directory = tarfile.TarInfo(snapshot_dir)
            directory.type = tarfile.DIRTYPE
            tar.addfile(directory)
            link = tarfile.TarInfo(f"{snapshot_dir}/resource.bin")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../blobs/resource.bin"
            tar.addfile(link)
        if member_name:
            payload = b"extra"
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            tar.addfile(info, __import__("io").BytesIO(payload))
        if link_target:
            link = tarfile.TarInfo("hub/unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = link_target
            tar.addfile(link)
    (root / "manifest.json").write_text(
        json.dumps({
            "archive": archive.name,
            "sha256": _sha(archive),
            "models": list(TEST_HF_REQUIRED_MODELS),
            "offline_preflight": "PASS",
        }),
        encoding="utf-8",
    )
    return archive


def test_expert10_checkpoint_discovery_requires_unique_exact_file_and_sha(tmp_path):
    resolve_expert10_checkpoint = _support_namespace()["resolve_expert10_checkpoint"]
    checkpoint = tmp_path / "datasets" / "private-e10" / "m2d-lora-Nam.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    expected = _sha(checkpoint)

    resolved = resolve_expert10_checkpoint(tmp_path, expected_sha256=expected)
    assert resolved == checkpoint.resolve()

    with pytest.raises(FileNotFoundError, match="attach.*m2d-lora-Nam.pt"):
        resolve_expert10_checkpoint(tmp_path / "empty", expected_sha256=expected)

    duplicate = tmp_path / "other-dataset" / "m2d-lora-Nam.pt"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="multiple.*m2d-lora-Nam.pt"):
        resolve_expert10_checkpoint(tmp_path, expected_sha256=expected)


def test_expert10_checkpoint_sha_mismatch_is_rejected(tmp_path):
    resolve_expert10_checkpoint = _support_namespace()["resolve_expert10_checkpoint"]
    checkpoint = tmp_path / "m2d-lora-Nam.pt"
    checkpoint.write_bytes(b"not the verified checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_expert10_checkpoint(tmp_path, expected_sha256=EXPECTED_E10_SHA256)


def test_hf_cache_restore_checks_manifest_and_preserves_safe_symlinks(tmp_path):
    support = _support_namespace()
    HF_REQUIRED_MODELS = support["HF_REQUIRED_MODELS"]
    restore_hf_cache = support["restore_hf_cache"]
    source = tmp_path / "hf-input"
    source.mkdir()
    archive = _make_hf_archive(source, member_name="")

    result = restore_hf_cache(archive, source / "manifest.json", tmp_path / "hf-home")
    assert result["status"] == "PASS"
    assert result["models"] == list(HF_REQUIRED_MODELS)
    assert result["archive_sha256"] == _sha(archive)
    hub = tmp_path / "hf-home" / "hub"
    for model_id in HF_REQUIRED_MODELS:
        model_dir = hub / ("models--" + model_id.replace("/", "--"))
        link = model_dir / "snapshots" / "test-revision" / "resource.bin"
        assert link.is_symlink()
        assert link.read_bytes() == f"resource:{model_id}".encode()


def test_hf_cache_restore_rejects_traversal_even_with_valid_manifest(tmp_path):
    restore_hf_cache = _support_namespace()["restore_hf_cache"]
    source = tmp_path / "hf-input"
    source.mkdir()
    archive = _make_hf_archive(source, member_name="../outside.txt")
    with pytest.raises(ValueError, match="Unsafe HF cache archive"):
        restore_hf_cache(archive, source / "manifest.json", tmp_path / "hf-home")
    assert not (tmp_path / "outside.txt").exists()


def test_hf_cache_restore_rejects_symlink_escape(tmp_path):
    restore_hf_cache = _support_namespace()["restore_hf_cache"]
    source = tmp_path / "hf-input"
    source.mkdir()
    archive = _make_hf_archive(source, member_name="", link_target="../../../../escape")
    with pytest.raises(ValueError, match="symlink escapes hub"):
        restore_hf_cache(archive, source / "manifest.json", tmp_path / "hf-home")


def test_source_checkout_is_pinned_and_stale_checkout_fails_without_replacement(tmp_path):
    ensure_source_checkout = _support_namespace()["ensure_source_checkout"]
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-b", "feature/moe-10-m2d-lora-coral"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "codex@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Codex Test"], cwd=source, check=True)
    (source / "required.py").write_text("source = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "required.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "test source"], cwd=source, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()

    checkout = tmp_path / "checkout"
    actual = ensure_source_checkout(str(source), "feature/moe-10-m2d-lora-coral", commit, checkout)
    assert actual == commit
    assert (checkout / "required.py").is_file()

    other_commit = "0" * 40
    with pytest.raises(ValueError, match="stale checkout|expected commit"):
        ensure_source_checkout(str(source), "feature/moe-10-m2d-lora-coral", other_commit, checkout)


def test_run_summary_records_complete_skipped_and_failed_stages(tmp_path):
    RunSummary = _support_namespace()["RunSummary"]
    path = tmp_path / "run_summary.json"
    summary = RunSummary(path, stage_names=("preflight", "cache", "extract"), metadata={"profile": "full_10"})
    with summary.stage("preflight"):
        summary.update_metadata(source_sha="abc")
    summary.skip("cache", "EXISTING_FEATURE_CACHE=None")
    with pytest.raises(RuntimeError, match="expected stage failure"):
        with summary.stage("extract"):
            raise RuntimeError("expected stage failure")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stages"]["preflight"]["status"] == "PASS"
    assert payload["stages"]["cache"]["status"] == "SKIPPED"
    assert payload["stages"]["extract"]["status"] == "FAIL"
    assert payload["source_sha"] == "abc"
    assert payload["errors"]


def test_cache_reuse_statistics_report_actual_validated_and_missing_shards():
    cache_shard_statistics = _support_namespace()["cache_shard_statistics"]
    assert cache_shard_statistics({"reused_validated_shards": 12}, 20) == {
        "expected_shards": 20,
        "reused_shards": 12,
        "extracted_shards": 8,
    }
    with pytest.raises(ValueError, match="exceeds expected shards"):
        cache_shard_statistics({"reused_validated_shards": 21}, 20)


def test_full9_cache_identity_comparison_reports_the_changed_fields():
    from scripts.extract_moe_feature_store import compare_expert_cache_identity

    previous = {
        "cache_id": "a" * 64,
        "checkpoint_provenance": {"checkpoint": {"sha256": "old"}},
        "preprocessing_provenance": {"audio": "center-crop-v1"},
        "inference_policy": {"device": {"requested_device": "cuda:0"}},
    }
    expected = {
        "cache_id": "b" * 64,
        "checkpoint_provenance": {"checkpoint": {"sha256": "new"}},
        "preprocessing_provenance": {"audio": "center-crop-v1"},
        "inference_policy": {"device": {"requested_device": "cuda:1"}},
    }
    result = compare_expert_cache_identity(previous, expected)
    assert result["compatible"] is False
    assert result["differences"] == [
        "cache_id",
        "checkpoint_provenance.checkpoint.sha256",
        "inference_policy.device.requested_device",
    ]


def test_full10_cache_audit_reports_full9_identity_mismatches(tmp_path):
    from model.expert_adapters.registry import EXPERT_PROFILES
    from scripts.extract_moe_feature_store import audit_full10_cache_identities, job_identity

    def identity(letter):
        return {
            "cache_id": letter * 64,
            "checkpoint_provenance": {"checkpoint": {"sha256": letter * 64}},
            "preprocessing_provenance": {"audio": "center-crop-v1"},
            "inference_policy": {"device": {"requested_device": "cuda:0"}},
        }

    order9 = list(EXPERT_PROFILES["full_9"])
    previous = {
        "expert_profile": "full_9",
        "expert_order": order9,
        "assembly_ids": {"train": "old-full9"},
        "expert_cache_identities": {name: identity("a") for name in order9},
    }
    current_order = list(EXPERT_PROFILES["full_10"])
    expected = {
        "expert_profile": "full_10",
        "expert_order": current_order,
        "assembly_ids": {"train": "new-full10"},
        "expert_cache_identities": {name: identity("a") for name in order9},
    }
    expected["expert_cache_identities"]["expert1"] = identity("b")
    expected["expert_cache_identities"]["m2d_lora_coral"] = identity("c")
    jobs = tmp_path / "feature_cache" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / f"{job_identity(previous)}.json").write_text(json.dumps(previous))

    report = audit_full10_cache_identities(expected, tmp_path)
    assert report["status"] == "PASS"
    assert len(report["candidates"]) == 1
    compared = report["candidates"][0]["experts"]
    assert compared["expert1"]["compatible"] is False
    assert compared["expert1"]["differences"] == [
        "cache_id", "checkpoint_provenance.checkpoint.sha256"
    ]
    assert compared["expert2"]["compatible"] is True
    assert "m2d_lora_coral" not in compared


def test_corrupt_matching_cache_shard_is_quarantined_without_discarding_bytes(tmp_path):
    from scripts.extract_moe_feature_store import quarantine_invalid_cache_shard

    cache_root = tmp_path / "feature_cache"
    shard = cache_root / "experts" / "expert1" / "shard.npz"
    sidecar = shard.with_suffix(".json")
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"corrupt-npz")
    sidecar.write_bytes(b"corrupt-metadata")

    quarantine = Path(quarantine_invalid_cache_shard(shard, cache_root=cache_root, reason="bad hash"))
    assert not shard.exists()
    assert not sidecar.exists()
    assert (quarantine / shard.name).read_bytes() == b"corrupt-npz"
    assert (quarantine / sidecar.name).read_bytes() == b"corrupt-metadata"
    assert json.loads((quarantine / "reason.json").read_text())["reason"] == "bad hash"


def test_dual_gpu_report_counts_actual_cached_and_new_shards():
    from scripts.kaggle_full9_dual_gpu import worker_cache_statistics

    assert worker_cache_statistics(
        {"reused_validated_shards": 12, "quarantined_invalid_shards": 1}, 20
    ) == {
        "expected_shards": 20,
        "reused_shards": 12,
        "extracted_shards": 8,
        "quarantined_invalid_shards": 1,
    }
