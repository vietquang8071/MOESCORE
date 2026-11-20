from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "MOESCORE_Full8_T4x2_Kaggle.ipynb"


def _cells():
    body = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return body, ["".join(cell.get("source", [])) for cell in body["cells"]]


def _function(name, namespace=None):
    _, cells = _cells()
    for cell_index, source in enumerate(cells):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                scope = dict(namespace or {})
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"notebook-cell-{cell_index}", "exec"), scope)
                return scope[name], source
    raise AssertionError(f"Notebook function {name!r} is missing")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_full8_kaggle_notebook_json_code_cells_and_source_pin():
    notebook, cells = _cells()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"Full8 notebook cell {index}", "exec")
    config = next(cell for cell in cells if "SOURCE_COMMIT =" in cell)
    match = re.search(r'SOURCE_COMMIT\s*=\s*["\']([0-9a-f]{40})["\']', config)
    assert match, "Full8 notebook must pin a complete implementation commit SHA"
    subprocess.run(["git", "cat-file", "-e", f"{match.group(1)}^{{commit}}"], check=True)


def test_full8_notebook_uses_exact_e8_checkpoint_and_preflights_before_expensive_stages():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert '"m2d-lora-Nam.pt"' in combined
    assert "E8_LORA_CHECKPOINT_OVERRIDE = None" in combined
    assert 'input_root.rglob(filename)' in combined
    assert "WEIGHTS_DATASET_ROOT" not in combined
    assert "/kaggle/input/datasets/qnv8071/xacle-weights" not in combined
    assert "Ambiguous" in combined
    assert "load_expert8_checkpoint(E8_LORA_CHECKPOINT)" in combined
    assert "validate_base_m2d_checkpoint(E8_BASE_CHECKPOINT)" in combined
    assert "EXPECTED_LORA_SHAPES" in combined and 'len(e8_preflight["lora"]) != 24' in combined
    all_source = "\n".join(cells)
    preflight_index = all_source.index("validate_base_m2d_checkpoint(E8_BASE_CHECKPOINT)")
    e5_index = all_source.index("E5_CONFIG = CONFIG_ROOT")
    whisper_index = all_source.index("def whisper_command(split")
    assert preflight_index < e5_index and preflight_index < whisper_index


def test_full8_notebook_has_cache_reuse_and_full_pipeline_in_order():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "verify_full7_candidate" in combined
    assert "NEED_FULL7_BUILD" in combined
    assert "Full7 Gate is never trained" in combined
    ordered = (
        "from scripts.kaggle_full9_artifacts import restore_feature_cache, validate_full9_store",
        "E5_CONFIG = CONFIG_ROOT", "def whisper_command(split", "E6_CONFIG = CONFIG_ROOT",
        "FULL7_BUILD_REPORT =", "E8_CONFIG = CONFIG_ROOT", "FULL8_BUILD_REPORT=",
        "full8_train_only_score_audit", 'run_logged("full8_gate_train"',
        'if RUN_VALIDATION:', 'if RUN_TEST_INFERENCE:',
        "FROZEN_AFTER_LABEL_FREE_INFERENCE", "evaluate_post_inference_metrics",
    )
    positions = [combined.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert "E8_BATCH_SIZE = 1" in combined
    assert "E8_SHARD_SIZE = 8" in combined
    assert "--allow-train-validation-audio-overlap" in combined


def test_full8_notebook_propagates_experimental_review_and_never_fabricates_review():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "SKIP_SCORE_REVIEW_EXPERIMENTAL = True" in combined
    assert 'gate_env["MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW"]="1"' in combined
    assert "EXPERIMENTAL_SKIPPED" in combined
    assert "RUN_SCORE_AUDIT = True" in combined
    assert "_real_score_review" in combined
    assert "score_review_file" in combined
    assert "REVIEWED_FOR_GATE_PROXY" not in combined
    assert "reviewer\"] =" not in combined
    assert "evidence\"] =" not in combined


def test_test_ground_truth_is_accessed_only_after_inference_freeze():
    _, cells = _cells()
    notebook, _ = _cells()
    accesses = [i for i, cell in enumerate(notebook["cells"])
                if cell.get("cell_type") == "code" and "test_average.csv" in "".join(cell.get("source", []))]
    assert len(accesses) == 1
    inference_cell = next(i for i, text in enumerate(cells) if 'run_logged("full8_test_inference"' in text)
    freeze_cell = next(i for i, text in enumerate(cells) if "FROZEN_AFTER_LABEL_FREE_INFERENCE" in text)
    metric_cell = next(i for i, text in enumerate(cells) if "evaluate_post_inference_metrics" in text)
    assert inference_cell == freeze_cell < accesses[0] == metric_cell
    assert cells[inference_cell].index('run_logged("full8_test_inference"') < cells[inference_cell].index("FROZEN_AFTER_LABEL_FREE_INFERENCE")
    assert '["wave_file_name","pred_score"]' in cells[inference_cell]
    assert "ALLOW_POST_INFERENCE_TEST_LABEL_ACCESS" in cells[metric_cell]


def test_full7_kaggle_notebook_remains_unmodified():
    root = Path(__file__).parents[1]
    actual = (root / "notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb").read_bytes()
    expected = subprocess.run(
        ["git", "show", "HEAD:notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb"],
        check=True, capture_output=True,
    ).stdout
    assert actual == expected


def test_central_config_and_full_profile_contract_are_explicit():
    _, cells = _cells()
    combined = "\n".join(cells)
    config = next(source for source in cells if "E8_LORA_CHECKPOINT_OVERRIDE" in source)
    assert "SCORE_REVIEW_FILE_OVERRIDE = None" in config
    assert 'EXPECTED_COUNTS = {"train": 7500, "validation": 3000, "test": 3000}' in combined
    compact = re.sub(r"\s+", "", combined)
    assert '("full_8",33032,8,expected)' in compact
    assert '(manifest.get("expert_profile"),manifest.get("feature_width"),manifest.get("score_width"),count)!=("full_8",33032,8,expected)' in compact
    assert '"selection_metric": "official_validation_mse"' in combined
    assert 'official_validation_srcc_strict_v1_provisional' in combined


def test_full7_cache_accepts_distinct_split_source_manifest_hashes():
    def row(split, name, digest):
        return {"sample_id": f"{split}:{name}", "wave_file_name": name, "wav_sha256": digest}

    manifests = {
        "train": {"expert_profile": "full_7", "feature_width": 29959, "score_width": 7,
                  "labels_present": True, "source_manifest_sha256": "a" * 64,
                  "shards": [{"count": 1, "sample_ids": ["train:a.wav"], "wave_file_names": ["a.wav"]}],
                  "assembly_identity": {"ordered_source_fingerprints": [row("train", "a.wav", "a" * 64)]}},
        "validation": {"expert_profile": "full_7", "feature_width": 29959, "score_width": 7,
                       "labels_present": True, "source_manifest_sha256": "b" * 64,
                       "shards": [{"count": 1, "sample_ids": ["validation:b.wav"], "wave_file_names": ["b.wav"]}],
                       "assembly_identity": {"ordered_source_fingerprints": [row("validation", "b.wav", "b" * 64)]}},
        "test": {"expert_profile": "full_7", "feature_width": 29959, "score_width": 7,
                 "labels_present": False, "source_manifest_sha256": "c" * 64,
                 "shards": [{"count": 1, "sample_ids": ["test:c.wav"], "wave_file_names": ["c.wav"]}],
                 "assembly_identity": {"ordered_source_fingerprints": [row("test", "c.wav", "c" * 64)]}},
    }
    calls = []

    def validate(_root, *, split, require_labels):
        calls.append((split, require_labels))
        return manifests[split]

    diagnostic = {"overlapping_sha256_count": 0, "allow_train_validation_audio_overlap": False}
    fn, _ = _function("verify_full7_candidate", {
        "Path": Path,
        "EXPECTED_COUNTS": {"train": 1, "validation": 1, "test": 1},
        "ALLOW_TRAIN_VALIDATION_AUDIO_OVERLAP": False,
        "validate_feature_store": validate,
        "build_train_validation_audio_overlap_diagnostic": lambda *a, **k: diagnostic,
    })
    with tempfile.TemporaryDirectory() as tmp:
        result, observed = fn(tmp)
    assert set(result) == {"train", "validation", "test"}
    assert [observed for observed, _ in calls] == ["train", "validation", "test"]
    assert [labels for _, labels in calls] == [True, True, False]

    manifests["validation"]["feature_width"] = 29958
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="incompatible Full7"):
        fn(tmp)


def test_e8_bert_snapshot_download_and_identity_share_the_hub_cache(tmp_path):
    fn, source = _function("prepare_e8_bert_snapshot", {
        "Path": Path, "os": os, "json": json,
    })
    hub_root = tmp_path / "e8" / "hub"
    snapshot = hub_root / "models--google-bert--bert-base-uncased" / "snapshots" / "fixture"
    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if kwargs["local_files_only"]:
            raise FileNotFoundError("fixture cache miss")
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"model_type":"bert","hidden_size":768}')
        (snapshot / "tokenizer_config.json").write_text("{}")
        (snapshot / "vocab.txt").write_text("[PAD]\n")
        (snapshot / "model.safetensors").write_bytes(b"pytorch weights")
        return snapshot

    identities = []

    def identity(model_id, runtime, _snapshot_cache):
        identities.append((model_id, runtime))
        return {"model_id": model_id, "cache_views": [{"snapshots": [{"revision": "fixture"}]}]}

    snapshot_path, observed = fn(
        tmp_path / "e8", allow_download=True,
        snapshot_download_fn=download, snapshot_identity_fn=identity,
    )
    assert snapshot_path == snapshot
    assert observed["model_id"] == "google-bert/bert-base-uncased"
    assert calls[1][1]["cache_dir"] == str(hub_root)
    assert calls[1][1]["allow_patterns"] == [
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "added_tokens.json", "model.safetensors", "model.safetensors.index.json",
        "model-*.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    ]
    assert identities[0][1]["cache_dir"] == str(hub_root)


def test_json_writer_emits_real_newline_and_valid_json(tmp_path):
    fn, _ = _function("atomic_write_json", {
        "Path": Path, "os": os, "tempfile": tempfile, "json": json,
    })
    destination = tmp_path / "report.json"
    fn(destination, {"status": "PASS", "count": 3})
    content = destination.read_bytes()
    assert content.endswith(b"\n")
    assert not content.endswith(b"\\n")
    assert json.loads(content) == {"status": "PASS", "count": 3}


def test_whisper_snapshot_verifier_requires_complete_checkpoint_and_tokenizer(tmp_path):
    from model.feature_extractors.whisper_top2 import _local_model_files

    verify, _ = _function("verify_whisper_snapshot", {
        "Path": Path,
        "WHISPER_REQUIRED_ASSETS": ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    })
    snapshot = tmp_path / "whisper"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "whisper", "d_model": 1280,
        "_name_or_path": "openai/whisper-large-v2",
    }))
    (snapshot / "preprocessor_config.json").write_text(json.dumps({"feature_size": 80, "sampling_rate": 16000}))
    (snapshot / "tokenizer_config.json").write_text("{}")
    (snapshot / "vocab.json").write_text("{}")
    (snapshot / "merges.txt").write_text("#version: 0.2\n")
    (snapshot / "pytorch_model.bin").write_bytes(b"weights")
    files = verify(snapshot, local_model_files_fn=_local_model_files)
    assert any(path.name == "pytorch_model.bin" for path in files)
    (snapshot / "pytorch_model.bin").unlink()
    with pytest.raises(FileNotFoundError, match="no model weights"):
        verify(snapshot, local_model_files_fn=_local_model_files)


def test_whisper_workers_use_gpu_order_and_stop_other_workers_on_failure(tmp_path):
    import subprocess as subprocess_module
    fn, source = _function("run_whisper_workers", {
        "time": __import__("time"), "subprocess": subprocess_module, "Path": Path,
    })

    class Process:
        def __init__(self, results, *, wait_hangs=False):
            self.results = list(results)
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.wait_hangs = wait_hangs

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if self.results:
                value = self.results.pop(0)
                if value is not None:
                    self.returncode = value
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            if self.wait_hangs and not self.killed:
                raise subprocess_module.TimeoutExpired("whisper", timeout)
            if self.returncode is None:
                raise subprocess_module.TimeoutExpired("whisper", timeout)
            return self.returncode

    class Stream:
        closed = False
        def close(self):
            self.closed = True

    started = []
    processes = {
        "train": Process([None, None, 0]),
        "validation": Process([0]),
        "test": Process([0]),
    }

    def start(name, gpu):
        started.append((name, gpu))
        log = tmp_path / f"{name}.log"
        log.write_text(f"{name} log")
        return processes[name], Stream(), log

    statuses = fn(start, poll_interval=0.001, heartbeat_seconds=0.001, terminate_timeout=0.001)
    assert started == [("train", 0), ("validation", 1), ("test", 1)]
    assert statuses == {"train": 0, "validation": 0, "test": 0}
    assert "heartbeat" in source.lower()

    started.clear()
    processes["train"] = Process([None, None, None], wait_hangs=True)
    processes["validation"] = Process([17])
    processes.pop("test", None)
    with pytest.raises(RuntimeError, match="validation"):
        fn(start, poll_interval=0.001, heartbeat_seconds=0.001, terminate_timeout=0.001)
    assert ("test", 1) not in started
    assert processes["train"].terminated
    assert processes["train"].killed


def test_whisper_cache_archive_is_atomic_and_verifies_contents(tmp_path):
    from data.whisper_feature_cache import WhisperFeatureCacheWriter

    cache_root = tmp_path / "whisper_cache"
    split_summaries = {}
    for split in ("train", "validation", "test"):
        writer = WhisperFeatureCacheWriter(
            cache_root / split, split=split,
            source_checkpoint_identity={"model_id": "openai/whisper-large-v2"},
            preprocessing_identity={"batch_size": 1},
            dataset_identity={"split": split, "sample_count": 1,
                              "ordered_samples_sha256": hashlib.sha256(split.encode()).hexdigest()},
        )
        writer.append([f"{split}:a.wav"], np.zeros((1, 1280), np.float32), np.ones((1, 1280), np.float32))
        manifest_path = cache_root / split / "manifest.json"
        cache_manifest = json.loads(manifest_path.read_text())
        split_summaries[split] = {
            "sample_count": 1,
            "manifest_sha256": _sha256(manifest_path),
            "dataset_identity": cache_manifest["dataset_identity"],
            "source_checkpoint_identity": cache_manifest["source_checkpoint_identity"],
            "preprocessing_identity": cache_manifest["preprocessing_identity"],
            "sample_ids": [f"{split}:a.wav"],
        }
    namespace = {"atomic_write_json": lambda path, value: Path(path).write_text(json.dumps(value) + "\n")}
    fn, _ = _function("archive_verified_whisper_cache", {
        "Path": Path, "os": os, "json": json, "tarfile": tarfile,
        "tempfile": tempfile, "hashlib": hashlib, "np": np, **namespace,
    })
    archive = tmp_path / "whisper.tar"
    manifest = tmp_path / "whisper_archive.json"
    report = fn(cache_root, archive, manifest, split_summaries)
    assert report["status"] == "PASS"
    assert report["archive_sha256"] == _sha256(archive)
    assert json.loads(manifest.read_text())["archive_sha256"] == _sha256(archive)
    with tarfile.open(archive) as stream:
        assert {member.name for member in stream.getmembers()} == {
            f"whisper_cache/{split}/{name}"
            for split in ("train", "validation", "test")
            for name in ("manifest.json", "shard-00000000.npz")
        }
    old_digest = _sha256(archive)
    shard = cache_root / "train/shard-00000000.npz"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        fn(cache_root, archive, manifest, split_summaries)
    assert _sha256(archive) == old_digest


def test_e8_cache_skip_path_revalidates_source_rows_and_every_shard(tmp_path):
    from data.moe_expert_cache import (
        build_expert_cache_identity,
        save_expert_shard,
        sha256_file,
    )
    from scripts.extract_expert8_features import (
        _shard_sample_identity,
        make_expert8_cache_manifest,
    )
    from types import SimpleNamespace

    base_sha, lora_sha = "a" * 64, "b" * 64
    identity = build_expert_cache_identity(
        expert_name="expert8",
        checkpoint_provenance={
            "base_checkpoint": {"sha256": base_sha},
            "finetuned_checkpoint": {"sha256": lora_sha},
        },
        implementation_provenance={"adapter": "c" * 64},
        preprocessing_provenance={"sample_rate_hz": 16000},
        input_contract={"waveform_samples": 160000},
        inference_policy={"device": "cuda:0", "frozen": True},
        score_contract={"score_shape": [None, 1]},
        relevant_dependencies={"torch": "fixture"},
    )
    rows_by_split = {
        split: [{
            "sample_id": f"{split}:sample.wav",
            "wave_file_name": "sample.wav",
            "raw_caption": f"caption {split}",
            "wav_sha256": hashlib.sha256(split.encode()).hexdigest(),
            "audio_path": str(tmp_path / f"{split}.wav"),
        }]
        for split in ("train", "validation", "test")
    }
    output_root = tmp_path / "cache"
    cache_base = output_root / "expert8" / identity["cache_id"]
    for split, rows in rows_by_split.items():
        split_root = cache_base / split
        records_for_identity = _shard_sample_identity(split, rows, identity)
        shard_path = split_root / "shards" / "fixture.npz"
        save_expert_shard(
            shard_path,
            expert_identity=identity,
            sample_identity=records_for_identity,
            scores=np.zeros((1, 1), dtype=np.float32),
            features=np.ones((1, 3073), dtype=np.float32),
            inference_metadata={"factory_restoration": "PASS_FROM_EXISTING_SOURCE_FACTORY", "frozen_eval": True},
        )
        shard = {
            "index": 0, "start": 0, "count": 1,
            "sample_identity_sha256": records_for_identity["sample_identity_sha256"],
            "relative_path": str(shard_path.relative_to(split_root)),
            "file_sha256": sha256_file(shard_path),
        }
        manifest = make_expert8_cache_manifest(
            split=split, records=rows, cache_identity=identity, shards=[shard],
        )
        (split_root / "manifest.json").write_text(json.dumps(manifest))

    fake_config = {"experts": {"expert8": {}}}

    def fake_source_rows(_dataset_root, split, _work_root, *, limit):
        assert limit is None
        return SimpleNamespace(split=split), rows_by_split[split]

    fn, _ = _function("verify_existing_e8_cache", {
        "_e8_config": lambda *_args, **_kwargs: fake_config,
        "compute_expert_cache_identity": lambda *_args, **_kwargs: identity,
        "torch": SimpleNamespace(device=lambda value: value),
        "Path": Path,
        "E8_BASE_SHA256": base_sha,
        "E8_LORA_SHA256": lora_sha,
        "EXPECTED_COUNTS": {"train": 1, "validation": 1, "test": 1},
        "_e8_source_rows": fake_source_rows,
        "json": json,
        "validate_expert8_cache_manifest": __import__(
            "scripts.extract_expert8_features", fromlist=["validate_expert8_cache_manifest"]
        ).validate_expert8_cache_manifest,
        "_e8_shard_sample_identity": _shard_sample_identity,
        "load_expert_shard": __import__(
            "data.moe_expert_cache", fromlist=["load_expert_shard"]
        ).load_expert_shard,
        "sha256_file": sha256_file,
        "CORAL_FEATURE_DIM": 3073,
        "gc": __import__("gc"),
        "atomic_write_json": lambda path, value: Path(path).write_text(json.dumps(value) + "\n"),
    })
    report = fn("fixture.yaml", tmp_path, output_root, tmp_path / "report.json")
    assert report["status"] == "PASS"
    assert report["expert_cache_identity"] == identity
    assert {split: item["sample_count"] for split, item in report["splits"].items()} == {
        "train": 1, "validation": 1, "test": 1,
    }

    corrupt = cache_base / "validation" / "shards" / "fixture.npz"
    corrupt.write_bytes(corrupt.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        fn("fixture.yaml", tmp_path, output_root, tmp_path / "report.json")


def test_notebook_orders_dependency_install_before_torch_and_whisper_archive_before_e6():
    _, cells = _cells()
    combined = "\n".join(cells)
    source_cell = next(i for i, source in enumerate(cells) if 'checkout", "--detach", SOURCE_COMMIT' in source)
    dependency_cell = next(i for i, source in enumerate(cells) if "INSTALL_MISSING_DEPENDENCIES" in source and "install_kaggle_dependencies.py" in source)
    preflight_cell = next(i for i, source in enumerate(cells) if "device_count()" in source)
    assert source_cell < dependency_cell <= preflight_cell
    dependency_source = cells[dependency_cell]
    assert dependency_source.index("subprocess.run(install_command") < dependency_source.index("\nimport torch\n")
    assert "torch already imported" in combined.lower()
    assert "gpu_stack_preserved" in combined
    assert any("SCORE_REVIEW_FILE_OVERRIDE = None" in source for source in cells)
    extraction_cell = next(i for i, source in enumerate(cells) if "archive_verified_whisper_cache" in source)
    e6_cell = next(i for i, source in enumerate(cells) if "E6_CONFIG = CONFIG_ROOT" in source)
    assert extraction_cell < e6_cell
    assert "source_manifest_sha256\" for m in manifests.values()" not in combined


def test_notebook_verifies_e5_e6_and_e8_cache_skip_without_changing_policies():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "load_set_a_checkpoint" in combined
    assert 'for checkpoint_path, checkpoint_role in ((E5_BEST, "best"), (E5_FINAL, "final_best"))' in combined
    assert "validation_provenance" in combined and "validation_sample_ids" in combined
    assert "load_set_b_checkpoint" in combined
    assert "expert6_final.pt" in combined and "audio_overlap_diagnostic_sha256" in combined
    assert "verify_existing_e8_cache" in combined
    skip_branch = next(source for source in cells if "verify_existing_e8_cache" in source and "RUN_E8_EXTRACTION" in source)
    assert "else" in skip_branch and "verify_existing_e8_cache" in skip_branch
    assert "source_audio_dir=DATA_ROOT / \"wav\" / split" in combined
    assert "WHISPER_BATCH_SIZE = 1" in cells[2]
    assert "official_validation_mse" in combined


def test_test_freeze_and_metrics_reports_use_atomic_json_writer_and_all_five_metrics():
    _, cells = _cells()
    combined = "\n".join(cells)
    assert "TEST_FREEZE_MANIFEST.write_text(json.dumps" not in combined
    assert "TEST_METRICS_PATH.write_text(json.dumps" not in combined
    assert "atomic_write_json(TEST_FREEZE_MANIFEST" in combined
    assert "atomic_write_json(TEST_METRICS_PATH" in combined
    metric_source = (Path(__file__).parents[1] / "scripts/full7_test_metrics.py").read_text(encoding="utf-8")
    for metric in ("srcc", "lcc", "kendall_tau", "mse", "mae"):
        assert metric in metric_source.lower()


def test_expected_full8_counts_and_feature_dimensions_are_verified():
    _, cells = _cells()
    assert 'EXPECTED_COUNTS = {"train": 7500, "validation": 3000, "test": 3000}' in cells[2]
    compact = re.sub(r"\s+", "", "\n".join(cells))
    assert '("full_8",33032,8,expected)' in compact
    assert '(manifest.get("expert_profile"),manifest.get("feature_width"),manifest.get("score_width"),count)!=("full_8",33032,8,expected)' in compact
