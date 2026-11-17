#!/usr/bin/env python3
"""Run sequential real-checkpoint inference for the four pretrained Top-2 adapters."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M2D_MODEL_DIR = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
ADAPTERS = {
    "top2_m2d": (
        "model.expert_adapters.top2_m2d_adapter",
        "Top2M2DAdapter",
        "checkpoint-30.pth",
    ),
    "top2_msclap": (
        "model.expert_adapters.top2_msclap_adapter",
        "Top2MSCLAPAdapter",
        "CLAP_weights_2023.pth",
    ),
    "top2_laion": (
        "model.expert_adapters.top2_laion_adapter",
        "Top2LAIONAdapter",
        "laion-clap.pt",
    ),
    "top2_mga": (
        "model.expert_adapters.top2_mga_adapter",
        "Top2MGAAdapter",
        "mga-clap.pt",
    ),
}
AUDIO_KEYS = {
    "top2_m2d": "expert3_wavs",
    "top2_msclap": "expert2_wavs",
    "top2_laion": "expert1_wavs",
    "top2_mga": "expert2_wavs",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--adapter", choices=("all", *ADAPTERS), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument("--verbose-errors", action="store_true")
    return parser.parse_args()


def _detect_dataset_root(value: Path, split: str) -> tuple[Path, Path]:
    """Find the directory that contains meta_data/ and wav/ for the selected split."""
    root = value.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    queue: list[tuple[Path, int]] = [(root, 0)]
    visited: set[Path] = set()
    while queue:
        candidate, depth = queue.pop(0)
        if candidate in visited:
            continue
        visited.add(candidate)
        metadata_dir = candidate / "meta_data"
        audio_dir = candidate / "wav" / split
        csv_candidates = (
            metadata_dir / f"{split}_average.csv",
            metadata_dir / f"{split}.csv",
        )
        metadata_csv = next((path for path in csv_candidates if path.is_file()), None)
        if metadata_csv is not None and audio_dir.is_dir():
            return candidate, metadata_csv

        # Dataset wrappers sometimes add one or two directory levels. Avoid
        # traversing wav trees, which can contain hundreds of thousands of files.
        if depth < 4:
            try:
                children = sorted(
                    (path for path in candidate.iterdir() if path.is_dir() and path.name != "wav"),
                    key=lambda path: path.name,
                )
            except PermissionError:
                children = []
            queue.extend((child, depth + 1) for child in children)
    raise FileNotFoundError(
        f"Could not detect XACLE dataset layout under {root}; expected meta_data/{split}[_average].csv "
        f"and wav/{split}/"
    )


def _load_batch(dataset_root: Path, split: str, sample_index: int) -> tuple[dict[str, Any], str, str]:
    if sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    detected_root, metadata_csv = _detect_dataset_root(dataset_root, split)

    # Reuse the existing four-rate preprocessing so each expert receives its
    # established MOESCORE waveform, all derived from this same metadata row.
    from data.unified_dataset import UnifiedXACLEDataset
    from data.unified_collate import collate_unified

    dataset = UnifiedXACLEDataset(
        metadata_csv,
        detected_root / "wav" / split,
        split=split,
        require_labels=False,
    )
    if sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {sample_index} is outside {split} metadata ({len(dataset)} rows)"
        )
    row = dataset.rows[sample_index]
    sample = dataset[sample_index]
    audio_path = str(row["audio_path"])
    caption = sample["captions_raw"]
    sample.pop("label", None)
    batch = collate_unified([sample])
    batch["wav_paths"] = [audio_path]
    return batch, audio_path, caption


def _model_config(
    name: str,
    checkpoint: Path,
    *,
    device: Any,
    project_root: Path,
    m2d_checkpoint: Path | None = None,
) -> dict[str, Any]:
    section: dict[str, Any]
    selected_checkpoint = m2d_checkpoint or checkpoint
    if name == "top2_m2d":
        section = {"checkpoint": str(selected_checkpoint)}
    elif name == "top2_msclap":
        section = {"checkpoint": str(selected_checkpoint)}
    elif name == "top2_laion":
        section = {"checkpoint": str(selected_checkpoint)}
    else:
        section = {
            "pretrained_checkpoint": str(selected_checkpoint),
            "source_dir": str(project_root / "expert2_finetune" / "MGA-CLAP"),
            "architecture_yaml": str(
                project_root / "expert2_finetune" / "MGA-CLAP" / "settings" / "inference_example.yaml"
            ),
        }
    return {name: section, "runtime": {"device": str(device), "offline": True}}


def _checkpoint_is_confirmed(name: str, adapter: Any, checkpoint: Path) -> None:
    if name == "top2_msclap":
        loaded = Path(adapter.clap_api.model_fp).resolve()
        if loaded != checkpoint.resolve():
            raise RuntimeError(f"MS-CLAP restored {loaded}, expected {checkpoint.resolve()}")
    elif name == "top2_laion":
        loaded = Path(adapter.checkpoint_path).resolve()
        if loaded != checkpoint.resolve():
            raise RuntimeError(f"LAION-CLAP restored {loaded}, expected {checkpoint.resolve()}")
    elif name == "top2_mga":
        report = adapter.checkpoint_load_report
        if not report or report["matched_tensors"] < 0.9 * report["model_tensors"]:
            raise RuntimeError("MGA-CLAP did not report a compatible pretrained checkpoint restore")


def _missing_resource(exc: BaseException) -> bool:
    messages = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current).lower())
        if isinstance(current, (FileNotFoundError, NotADirectoryError, ModuleNotFoundError)):
            return True
        current = current.__cause__ or current.__context__
    resource_phrases = (
        "no module named",
        "couldn't connect to",
        "could not connect to",
        "cannot find the requested files in the disk cache",
        "couldn't find them in the cached files",
        "not installed",
        "expected a readable",
    )
    return any(phrase in message for message in messages for phrase in resource_phrases)


def _run_one(
    name: str,
    checkpoint: Path,
    batch: dict[str, Any],
    *,
    project_root: Path,
    device: Any,
    verbose_errors: bool,
) -> dict[str, Any]:
    import torch

    module_name, class_name, _ = ADAPTERS[name]
    adapter = score = features = None
    previous_numba_cache = os.environ.get("NUMBA_CACHE_DIR")
    numba_cache = None
    if name in {"top2_msclap", "top2_mga"}:
        numba_cache = tempfile.TemporaryDirectory(
            prefix=".top2-msclap-numba-cache-", dir=project_root
        )
        os.environ["NUMBA_CACHE_DIR"] = numba_cache.name
    layout_context: Any = nullcontext(None)
    if name == "top2_m2d":
        layout_context = tempfile.TemporaryDirectory(
            prefix=".top2-m2d-checkpoint-layout-", dir=project_root
        )

    try:
        with layout_context as layout_root:
            model_checkpoint = checkpoint
            if name == "top2_m2d":
                model_dir = Path(layout_root) / M2D_MODEL_DIR
                model_dir.mkdir(parents=True)
                model_checkpoint = model_dir / checkpoint.name
                model_checkpoint.symlink_to(checkpoint.resolve())

            module = importlib.import_module(module_name)
            adapter_class = getattr(module, class_name)
            config = _model_config(
                name,
                checkpoint,
                device=device,
                project_root=project_root,
                m2d_checkpoint=model_checkpoint if name == "top2_m2d" else None,
            )
            adapter = adapter_class.from_config(config, device=device)
            _checkpoint_is_confirmed(name, adapter, checkpoint)
            if name == "top2_m2d":
                adapter.checkpoint_path = str(checkpoint.resolve())

            if adapter.training:
                raise AssertionError("adapter is not in evaluation mode")
            if any(parameter.requires_grad for parameter in adapter.parameters()):
                raise AssertionError("adapter contains trainable model parameters")

            adapter_batch = dict(batch)
            audio_key = AUDIO_KEYS[name]
            adapter_batch[audio_key] = batch[audio_key].to(device)
            score, features = adapter(adapter_batch)
            if tuple(score.shape) != (1, 1):
                raise AssertionError(f"unexpected score shape: {tuple(score.shape)}")
            if features.ndim != 2 or features.shape[0] != 1:
                raise AssertionError(f"unexpected feature shape: {tuple(features.shape)}")
            if features.shape[1] != adapter.feature_dim:
                raise AssertionError(
                    f"feature width {features.shape[1]} != declared {adapter.feature_dim}"
                )
            if score.dtype != torch.float32 or features.dtype != torch.float32:
                raise AssertionError(f"expected float32 outputs, got {score.dtype}, {features.dtype}")
            device_mismatch = score.device.type != device.type or (
                device.index is not None and score.device.index != device.index
            )
            if score.device != features.device or device_mismatch:
                raise AssertionError(
                    f"output devices differ or do not match model device: {score.device}, {features.device}, {device}"
                )
            if not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(features).all()):
                raise AssertionError("score or features contain NaN/Infinity")
            score_value = float(score.item())
            if not -1.00001 <= score_value <= 1.00001:
                raise AssertionError(f"cosine score outside [-1,1]: {score_value}")
            result = {
                "expert": name,
                "checkpoint": str(checkpoint.resolve()),
                "score": score_value,
                "score_shape": tuple(score.shape),
                "feature_shape": tuple(features.shape),
                "feature_dim": adapter.feature_dim,
                "status": "PASS",
            }
            if name == "top2_mga":
                result["checkpoint_match"] = adapter.checkpoint_load_report
            print(
                f"{name}: PASS checkpoint={result['checkpoint']} score={score_value:.8f} "
                f"score_shape={result['score_shape']} feature_shape={result['feature_shape']} "
                f"dtype={score.dtype} device={score.device} eval={not adapter.training} frozen=True"
            )
            return result
    except Exception as exc:
        status = "BLOCKED" if _missing_resource(exc) else "FAIL"
        message = f"{type(exc).__name__}: {exc}"
        print(f"{name}: {status} checkpoint={checkpoint} reason={message}", file=sys.stderr)
        if verbose_errors:
            traceback.print_exc()
        return {
            "expert": name,
            "checkpoint": str(checkpoint),
            "score": None,
            "score_shape": None,
            "feature_shape": None,
            "status": status,
            "reason": message,
        }
    finally:
        del adapter, score, features
        gc.collect()
        if name in {"top2_msclap", "top2_mga"}:
            if previous_numba_cache is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = previous_numba_cache
            if numba_cache is not None:
                numba_cache.cleanup()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.expanduser().resolve()
    checkpoint_dir = (args.checkpoint_dir or project_root / "weights" / "pretrained").expanduser().resolve()
    if not project_root.is_dir():
        print(f"Project root is not a directory: {project_root}", file=sys.stderr)
        return 2
    if not checkpoint_dir.is_dir():
        print(f"Checkpoint directory is not a directory: {checkpoint_dir}", file=sys.stderr)
        return 2

    project_path = str(project_root)
    if project_path in sys.path:
        sys.path.remove(project_path)
    sys.path.insert(0, project_path)

    try:
        import torch

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
            if device.type not in {"cpu", "cuda"}:
                raise ValueError(f"only CPU and CUDA devices are supported, got {device}")
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        batch, audio_path, caption = _load_batch(args.dataset_root, args.split, args.sample_index)
    except Exception as exc:
        print(f"Input preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose_errors:
            traceback.print_exc()
        return 2

    selected = tuple(ADAPTERS) if args.adapter == "all" else (args.adapter,)
    print(f"Python: {sys.executable}")
    print(f"Device: {device}")
    print(f"Dataset split: {args.split}")
    print(f"Audio file: {audio_path}")
    print(f"Caption: {caption}")

    results = []
    for name in selected:
        _, _, filename = ADAPTERS[name]
        checkpoint = checkpoint_dir / filename
        if not checkpoint.is_file() or not os.access(checkpoint, os.R_OK):
            result = {
                "expert": name,
                "checkpoint": str(checkpoint),
                "score": None,
                "score_shape": None,
                "feature_shape": None,
                "status": "BLOCKED",
                "reason": "required pretrained checkpoint is missing or unreadable",
            }
            print(f"{name}: BLOCKED checkpoint={checkpoint} reason={result['reason']}", file=sys.stderr)
            results.append(result)
            continue
        results.append(
            _run_one(
                name,
                checkpoint,
                batch,
                project_root=project_root,
                device=device,
                verbose_errors=args.verbose_errors,
            )
        )

    print("\n| Expert | Checkpoint loaded | Cosine score | Score shape | Feature shape | Status |")
    print("|---|---|---:|---|---|---|")
    for result in results:
        print(
            f"| {result['expert']} | `{result['checkpoint']}` | "
            f"{result['score'] if result['score'] is not None else '—'} | "
            f"{result['score_shape'] or '—'} | {result['feature_shape'] or '—'} | {result['status']} |"
        )
    print(f"Successful adapters: {sum(result['status'] == 'PASS' for result in results)}/{len(results)}")
    return 0 if results and all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
