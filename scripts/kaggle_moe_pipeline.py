"""Reusable setup and final evaluation for the Kaggle notebook.

Training, expert restoration, metrics and score fusion stay in their existing
modules. Test labels are accessed only by evaluate_best after predictions.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.moe_feature_store import validate_feature_store, _atomic_json, _sha256_file
from data.moe_training_dataset import MoeTrainingDataset, collate_moe_features
from model.moe_gating_core import GatingNetwork
from training.checkpoint_manager import MoeCheckpointManager
from training.moe_evaluator import MoeGateEvaluator, compute_metrics, write_submission_csv
from scripts.extract_moe_feature_store import (
    EXPECTED_COUNTS, CHECKPOINT_FIELDS, fingerprint, projected_dataset,
)


def discover_dataset(input_root, explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        root = Path(input_root)
        candidates = [root, *sorted(root.glob("*")), *sorted(root.glob("*/XACLE_dataset"))]
    found = sorted({p.resolve() for p in candidates if (p / "meta_data").is_dir() and (p / "wav").is_dir()})
    if len(found) != 1:
        raise ValueError(f"Set DATASET_ROOT explicitly; found {len(found)} XACLE roots: {found}")
    return found[0]


def generate_configs(project_root, dataset_root, weights_root, cache_dir, feature_store_root, output_dir, config_dir):
    project, dataset, weights = map(lambda p: Path(p).resolve(), (project_root, dataset_root, weights_root))
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    expert = yaml.safe_load((project / "configs/moe_local.example.yaml").read_text())
    expert["paths"] = {"project_root": str(project), "data_root": str(dataset), "resource_root": str(project)}
    expert["runtime"] = {"device": "cuda:0", "offline": True, "cache_dir": str(Path(cache_dir).resolve())}
    expert["data"] = {"train_csv": "meta_data/train_average.csv", "validation_csv": "meta_data/validation_average.csv",
                      "test_csv": "meta_data/test_average.csv", "audio_root": "wav"}
    paths = {
        "expert1": {"pretrained_checkpoint": "pretrained/laion-clap.pt", "finetuned_checkpoint": "finetuned_weights/expert1_best.pt"},
        "expert2": {"pretrained_checkpoint": "pretrained/mga-clap.pt", "finetuned_checkpoint": "finetuned_weights/expert2_best.pt"},
        "expert3": {"base_checkpoint": "pretrained/checkpoint-30.pth", "finetuned_checkpoint": "finetuned_weights/expert3_best.pt"},
        "expert4": {"beats_checkpoint": "pretrained/BEATs_iter3_plus_AS2M.pt", "clap_checkpoint": "pretrained/laion-clap.pt",
                    "finetuned_checkpoint": "finetuned_weights/expert4_best.pt"},
    }
    for name, fields in paths.items():
        expert["experts"][name].update({key: str(weights / value) for key, value in fields.items()})
    expert["experts"]["expert4"]["unilm_source_dir"] = str(project / "expert4_finetune/unilm")
    train = yaml.safe_load((project / "configs/moe_cloud.example.yaml").read_text())
    # Explicit finalized user configuration, checked against the current schema.
    train["runtime"] = {"device": "auto", "seed": 2027}
    train["data"]["feature_store_root"] = str(Path(feature_store_root).resolve())
    train["output_dir"] = str(Path(output_dir).resolve())
    train["training"].update(batch_size=8, max_epochs=50, learning_rate=1e-5, weight_decay=1e-4,
                             diversity_weight=0.2, early_stop_patience=50, checkpoint_interval_steps=1000)
    train["training"]["loss"] = {"loss_weights": [1.0, 0.5],
        "clipped_mse": {"tau": 0.25, "mode": "utt", "beta": 0.99, "cbl": False},
        "contrastive": {"margin": 0.1, "beta": 0.99, "cbl": False}}
    train["validation"]["batch_size"] = 8
    train["inference"]["batch_size"] = 64
    expert_path, train_path = config_dir / "moe_expert_config.yaml", config_dir / "moe_gating_config.yaml"
    expert_path.write_text(yaml.safe_dump(expert, sort_keys=False))
    train_path.write_text(yaml.safe_dump(train, sort_keys=False))
    from configs.moe_config import load_moe_config, resolve_moe_paths
    from training.config import load_moe_training_config
    resolved = resolve_moe_paths(load_moe_config(expert_path), expert_path)
    load_moe_training_config(train_path)
    return expert_path, train_path, resolved


def environment_report(require_cuda=True):
    report = {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if require_cuda and not report["cuda_available"]:
        raise RuntimeError("CUDA unavailable: enable a GPU accelerator in the Kaggle notebook settings")
    if report["cuda_available"]:
        free, total = torch.cuda.mem_get_info(0)
        report.update(gpu=torch.cuda.get_device_name(0), free_vram_bytes=free, total_vram_bytes=total)
    return report


def check_dependencies():
    required = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "yaml": "PyYAML",
                "soundfile": "soundfile", "torch": "torch", "torchaudio": "torchaudio",
                "transformers": "transformers", "laion_clap": "laion-clap", "timm": "timm",
                "librosa": "librosa", "einops": "einops", "tqdm": "tqdm",
                "torchlibrosa": "torchlibrosa", "nnAudio": "nnAudio", "matplotlib": "matplotlib"}
    versions = {}
    for module, distribution in required.items():
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            raise RuntimeError(f"Dependency {distribution} unavailable/incompatible: {exc}. "
                               "Supply compatible wheels or install explicitly; do not replace PyTorch/CUDA automatically.") from exc
    return versions


def check_text_cache(cache_dir, *, allow_download=False):
    """One explicit hub-cache location, including E3's direct cache_dir API."""
    cache = str(Path(cache_dir).resolve())
    os.environ.update(HF_HOME=cache, HF_HUB_CACHE=cache, TRANSFORMERS_CACHE=cache)
    if allow_download:
        os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
        from huggingface_hub import snapshot_download
        for name in ("roberta-base", "google-bert/bert-base-uncased"):
            snapshot_download(name, cache_dir=cache,
                              allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import try_to_load_from_cache
    for name in ("roberta-base", "google-bert/bert-base-uncased"):
        try:
            AutoConfig.from_pretrained(name, cache_dir=cache, local_files_only=True)
            AutoTokenizer.from_pretrained(name, cache_dir=cache, local_files_only=True)
            if not any(isinstance(try_to_load_from_cache(name, file, cache_dir=cache), str)
                       for file in ("model.safetensors", "pytorch_model.bin")):
                raise FileNotFoundError(f"Missing pretrained text model weights for {name}")
        except Exception as exc:
            raise RuntimeError(f"Missing approved tokenizer/model cache {name} in {cache}. "
                               "Attach a prepared hub cache or explicitly enable ALLOW_CACHE_DOWNLOAD.") from exc
    return {"status": "PASS", "cache_dir": cache, "offline_extraction": True}


def preflight(config, dataset_root, output_root, *, require_cuda=True, check_cache=True, allow_download=False):
    from expert4_finetune.source_compatibility import (
        ORIGINAL_COMMIT, ORIGINAL_SHA256, verify_bundled_import_equivalence,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment_report(require_cuda), "checkpoints": {}, "splits": {}}
    for name, fields in CHECKPOINT_FIELDS.items():
        report["checkpoints"][name] = {}
        for field in fields:
            path = Path(config["experts"][name][field])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty checkpoint: {path}")
            report["checkpoints"][name][field] = {"path": str(path), "size_bytes": path.stat().st_size}
    report["unilm"] = verify_bundled_import_equivalence(config["experts"]["expert4"]["unilm_source_dir"],
        {"git_commit": ORIGINAL_COMMIT, "sha256": ORIGINAL_SHA256})
    for split, expected in EXPECTED_COUNTS.items():
        dataset, headers = projected_dataset(dataset_root, split, root / "metadata")
        if len(dataset) != expected:
            raise ValueError(f"{split}: expected {expected} pairs, observed {len(dataset)}")
        report["splits"][split] = {"count": len(dataset), "headers": headers, "test_labels_read": False}
    if check_cache:
        report["text_cache"] = check_text_cache(config["runtime"]["cache_dir"], allow_download=allow_download)
    report["disk_free_bytes"] = shutil.disk_usage(root).free
    report["status"] = "PASS"
    _atomic_json(root / "preflight.json", report)
    return report


def verified_source_manifest(store_root, split):
    manifest = validate_feature_store(store_root, split=split, require_labels=split != "test")
    source = json.loads((Path(store_root) / split / "source_manifest.json").read_text())
    body = {key: value for key, value in source.items() if key != "sha256"}
    if fingerprint(body) != source.get("sha256") or source["sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("Feature store source manifest identity differs")
    ids = [item for shard in manifest["shards"] for item in shard["sample_ids"]]
    if [r["sample_id"] for r in source["records"]] != ids:
        raise ValueError("Feature store source sample order differs")
    return source


def match_test_labels(csv_path, prediction_rows, source_records):
    """Post-hoc only: join labels by identity after label-free prediction."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"wav_file_name", "text", "average_score"}.issubset(reader.fieldnames or []):
            raise ValueError("Final test evaluation needs wav_file_name,text,average_score")
        rows = list(reader)
    by_id = {f"test:{r['wav_file_name']}": r for r in rows}
    ids = [r["sample_id"] for r in prediction_rows]
    if len(ids) != len(set(ids)) or len(by_id) != len(rows) or set(ids) != set(by_id):
        raise ValueError("Missing, duplicate or mismatched test label/sample identities")
    sources = {r["sample_id"]: r for r in source_records}
    if len(sources) != len(source_records) or set(sources) != set(ids):
        raise ValueError("Test source manifest sample identities differ from predictions")
    labels = []
    for prediction in prediction_rows:
        sample_id = prediction["sample_id"]
        row, source = by_id[sample_id], sources[sample_id]
        if row["wav_file_name"] != prediction["wave_file_name"] or row["text"] != source["raw_caption"]:
            raise ValueError("Test caption/filename differs from extracted sample identity")
        label = float(row["average_score"])
        if not np.isfinite(label) or not 0 <= label <= 10:
            raise ValueError("Test average_score must be finite in [0,10]")
        labels.append(label)
    return np.asarray(labels, dtype=np.float64)


def display_metrics(metrics):
    return {"sample_count": metrics["sample_count"], **{key.upper(): metrics[key] for key in ("srcc", "lcc", "ktau", "mse")},
            "MAE": metrics["mae"], **{f"{key.upper()}_status": metrics[f"{key}_status"] for key in ("srcc", "lcc", "ktau")}}


def evaluate_best(config, dataset_root, *, checkpoint=None):
    """Evaluate both full splits; no optimizer, training or checkpoint selection."""
    from scripts.evaluate_moe_gating import _device
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint or output / "best_gate.pt")
    device = _device(config["runtime"]["device"])
    sources = {split: verified_source_manifest(config["data"]["feature_store_root"], split) for split in ("validation", "test")}
    stores = {split: validate_feature_store(config["data"]["feature_store_root"], split=split) for split in sources}
    if stores["validation"]["preprocessing_identity"] != stores["test"]["preprocessing_identity"]:
        raise ValueError("Test preprocessing identity differs from the verified validation store")
    reports = {}
    for split in ("validation", "test"):
        dataset = MoeTrainingDataset(config["data"]["feature_store_root"], split=split, require_labels=split == "validation")
        if split == "test" and dataset.manifest["labels_present"]:
            raise ValueError("Final test inference requires a label-free feature store")
        gate = GatingNetwork().to(device)
        payload = MoeCheckpointManager().load_for_inference(checkpoint, gating_network=gate,
                    expected_expert_checkpoint_provenance=dataset.manifest["checkpoint_provenance"])
        observed = dataset.manifest["preprocessing_identity"]
        if payload["feature_store_identity"].get("preprocessing_contract") != observed.get("contract"):
            raise ValueError("Evaluation preprocessing contract differs from gate checkpoint")
        if split == "validation" and payload["feature_store_identity"].get("validation") != dataset.identity:
            raise ValueError("Validation store identity differs from checkpoint selection data")
        loader = DataLoader(dataset, batch_size=config["validation" if split == "validation" else "inference"]["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_moe_features)
        evaluator = MoeGateEvaluator(gate)
        if split == "validation":
            result = evaluator.evaluate(loader)
            metrics = result["metrics"]
            rows = [{"sample_id": sid, "wave_file_name": name, "pred_score": float(value)}
                    for sid, name, value in zip(result["sample_ids"], result["wave_file_names"], result["prediction_raw"].tolist())]
        else:
            rows = evaluator.predict(loader, round_scores=False)
        # Save full-precision predictions before reading ANY test labels.
        write_submission_csv(rows, output / f"{split}_predictions.csv")
        if split == "test":
            labels = match_test_labels(Path(dataset_root) / "meta_data/test_average.csv", rows, sources[split]["records"])
            metrics = compute_metrics([row["pred_score"] for row in rows], labels)
            write_submission_csv([{**row, "pred_score": round(row["pred_score"], 2)} for row in rows], output / "submission.csv")
        report = {**display_metrics(metrics), "metrics_raw_scale": metrics, "checkpoint": checkpoint.name,
                  "checkpoint_sha256": _sha256_file(checkpoint), "best_epoch_zero_based": payload["epoch"],
                  "training_steps": payload["global_step"], "seed": config["runtime"]["seed"],
                  "feature_store_identity": dataset.identity, "training_configuration": config,
                  "evaluation_timestamp": datetime.now(timezone.utc).isoformat(), "optimizer_steps": 0}
        try:
            report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            report["git_commit"] = None
        _atomic_json(output / f"{split}_metrics.json", report)
        reports[split] = report
    combined = {"model": "MOESCORE", "dataset": "XACLE_2027", "checkpoint": checkpoint.name,
                "checkpoint_selection": "validation_srcc", "score_scale": "raw_0_to_10", **reports}
    _atomic_json(output / "metrics.json", combined)
    return combined


def record_training(report_path, output_dir):
    report = json.loads(Path(report_path).read_text())
    path = Path(output_dir) / "training_history.json"
    previous = json.loads(path.read_text()) if path.exists() else []
    by_epoch = {r["epoch"]: r for r in previous}
    by_epoch.update({r["epoch"]: r for r in report["history"]})
    path.write_text(json.dumps([by_epoch[e] for e in sorted(by_epoch)], indent=2, allow_nan=False) + "\n")
    return report


def smoke_store(store_root, config):
    """Exactly three optimizer steps on at most four checkpoint-backed rows."""
    from training.moe_trainer import MoeGateTrainer
    dataset = MoeTrainingDataset(store_root, split="train", require_labels=True)
    if len(dataset) > 4:
        raise ValueError("Smoke mode requires its separate store with at most four samples")
    torch.manual_seed(2027)
    trainer = MoeGateTrainer(GatingNetwork(), config["training"]["loss"])
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_moe_features))
    results = []
    for step in range(3):
        result = trainer.train_step(batches[step % len(batches)])
        results.append({key: float(result[key]) for key in ("total_loss", "regression_loss", "diversity_loss", "gradient_norm")})
    return {"status": "PASS", "scope": "SMOKE_ONLY", "sample_count": len(dataset), "optimizer_steps": 3,
            "source": "REAL_CHECKPOINT_EXTRACTION", "gate_device": "cpu", "steps": results,
            "full_training_completed": False}
