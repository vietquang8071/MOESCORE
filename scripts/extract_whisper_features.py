#!/usr/bin/env python3
"""Extract source-faithful Whisper component features for one MOESCORE split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/whisper_top2.example.yaml"))
    parser.add_argument("--metadata-csv", required=True, help="CSV for exactly one dataset split")
    parser.add_argument("--audio-dir", required=True, help="Split-specific audio directory")
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "result/whisper_feature_cache"),
        help="Cache root; output is written under <root>/<split>/",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run one item and print feature shapes without caching")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--cache-dir", default=None, help="Override runtime.cache_dir")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads; default behavior is offline-only",
    )
    return parser


def _resolve_optional_path(value: Any, *, base: Path) -> Any:
    if value is None or not isinstance(value, (str, Path)):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve(strict=False))
    candidate = (base / path).resolve(strict=False)
    return str(candidate)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_preprocessing_identity(extractor: Any, *, batch_size: int) -> dict[str, Any]:
    model = extractor.model
    first_parameter = next(model.parameters(), None)
    return {
        **extractor.preprocessing_identity,
        "unified_waveform_preprocessing": {
            "source": "data/unified_dataset.py:UnifiedXACLEDataset._expert3_audio",
            "source_sha256": _sha256_file(PROJECT_ROOT / "data/unified_dataset.py"),
            "mono": "mean over channels",
            "resampling": "scipy.signal.resample_poly to 16000 Hz when needed",
            "truncation": "first 160000 samples",
            "padding": "constant zero to 160000 samples",
        },
        "collation_source_sha256": _sha256_file(PROJECT_ROOT / "data/unified_collate.py"),
        "extraction_batch_size": batch_size,
        "inference_device": str(extractor.device),
        "model_parameter_dtype": str(first_parameter.dtype) if first_parameter is not None else "unknown",
        "batch_order": "stable dataset row order",
        "decoder_padding_scope": "per extraction batch; padded positions participate in mean pooling",
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the Whisper example configuration") from exc
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read Whisper configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Whisper configuration must be a YAML mapping")
    return config


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be a non-negative integer")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be a positive integer")
    config_path = Path(args.config).expanduser().resolve(strict=True)
    config = _load_config(config_path)
    section = config.setdefault("whisper_top2", {})
    runtime = config.setdefault("runtime", {})
    if not isinstance(section, dict) or not isinstance(runtime, dict):
        raise ValueError("configuration whisper_top2 and runtime values must be mappings")
    if args.model_name_or_path is not None:
        section["model_name_or_path"] = args.model_name_or_path
    if args.cache_dir is not None:
        runtime["cache_dir"] = args.cache_dir
    if args.allow_download:
        runtime["offline"] = False
    runtime.setdefault("offline", True)
    runtime["cache_dir"] = _resolve_optional_path(runtime.get("cache_dir"), base=PROJECT_ROOT)

    model_reference = section.get("model_name_or_path", "openai/whisper-large-v2")
    if model_reference != "openai/whisper-large-v2":
        section["model_name_or_path"] = _resolve_optional_path(model_reference, base=PROJECT_ROOT)

    try:
        from torch.utils.data import DataLoader, Subset
        from data.unified_collate import collate_unified
        from data.unified_dataset import UnifiedXACLEDataset
        from model.feature_extractors import WhisperTop2FeatureExtractor
        from data.whisper_feature_cache import (
            WhisperFeatureCacheWriter,
            build_whisper_dataset_identity,
            validate_whisper_cache_resume,
        )
    except ImportError as exc:
        raise RuntimeError(f"Whisper extraction dependencies are unavailable: {exc}") from exc

    dataset = UnifiedXACLEDataset(
        args.metadata_csv,
        args.audio_dir,
        split=args.split,
        require_labels=False,
    )
    sample_limit = 1 if args.smoke_test else args.limit_samples
    selected_indices = list(range(len(dataset)))
    if sample_limit is not None:
        selected_indices = selected_indices[:sample_limit]
        if (
            not args.smoke_test
            and sample_limit < len(dataset)
            and sample_limit % args.batch_size != 0
        ):
            raise ValueError(
                "--limit-samples must end on an extraction batch boundary; use --smoke-test for one item"
            )

    extractor = WhisperTop2FeatureExtractor.from_config(config)
    try:
        if args.smoke_test:
            loader = DataLoader(
                Subset(dataset, selected_indices[:1]),
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_unified,
            )
            batch = next(iter(loader))
            audio_features, text_features = extractor(batch)
            result = {
                "status": "PASS",
                "split": args.split,
                "sample_id": batch["sample_ids"][0],
                "audio_shape": list(audio_features.shape),
                "text_shape": list(text_features.shape),
                "audio_dtype": str(audio_features.dtype),
                "text_dtype": str(text_features.dtype),
                "checkpoint_identity": extractor.checkpoint_identity,
                "feature_extraction_version": extractor.feature_extraction_version,
            }
            return result

        cache_path = Path(args.output_dir).expanduser().resolve(strict=False) / args.split
        dataset_identity = build_whisper_dataset_identity(dataset.rows, split=args.split)
        preprocessing_identity = _cache_preprocessing_identity(
            extractor, batch_size=args.batch_size
        )
        cache = WhisperFeatureCacheWriter(
            cache_path,
            split=args.split,
            source_checkpoint_identity=extractor.checkpoint_identity,
            preprocessing_identity=preprocessing_identity,
            dataset_identity=dataset_identity,
            feature_extraction_version=extractor.feature_extraction_version,
        )
        dataset_ids = [row["sample_id"] for row in dataset.rows]
        start_index = validate_whisper_cache_resume(
            dataset_ids,
            cache.sample_ids,
            batch_size=args.batch_size,
        )
        missing_indices = selected_indices[start_index:] if start_index < len(selected_indices) else []
        loader = DataLoader(
            Subset(dataset, missing_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_unified,
        )
        newly_extracted = 0
        for batch in loader:
            sample_ids = batch["sample_ids"]
            audio_features, text_features = extractor(batch)
            cache.append(sample_ids, audio_features, text_features)
            newly_extracted += len(sample_ids)

        result = {
            "status": "PASS",
            "split": args.split,
            "cache_dir": str(cache_path),
            "cached_samples": len(cache.sample_ids),
            "newly_extracted": newly_extracted,
            "audio_feature_dim": 1280,
            "text_feature_dim": 1280,
            "total_feature_dim": 2560,
            "feature_extraction_version": extractor.feature_extraction_version,
        }
        return result
    finally:
        extractor.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:
        print(f"Whisper extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
