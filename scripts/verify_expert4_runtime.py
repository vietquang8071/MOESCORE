#!/usr/bin/env python3
"""Offline runtime verification for the real Expert 4 checkpoints.

This script reuses the original Expert4 factory/loaders through
Expert4Adapter.from_finetuned. It never trains, changes precision, or writes to
model/checkpoint resources. The same restored model instance is used for the
standalone-versus-adapter comparisons.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any

# Enforce offline behavior before importing Transformers or any model code.
for _offline_key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_offline_key] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATOL_DEFAULT = 1e-6
RTOL_DEFAULT = 1e-5
MAX_SAMPLES = 160_000


class VerificationFailure(RuntimeError):
    def __init__(self, status: str, stage: str, message: str):
        super().__init__(message)
        self.status = status
        self.stage = stage


def emit(message: str) -> None:
    print(message, flush=True)


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=False)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "readable": os.access(path, os.R_OK),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_local": __import__("datetime").datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if include_sha256:
        emit(f"HASH_START: {path}")
        record["sha256"] = file_sha256(path)
        emit(f"HASH_DONE: {path} {record['sha256']}")
    return record


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise VerificationFailure("BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION", "device", "CUDA requested but unavailable")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise VerificationFailure("BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION", "device", f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
    return device


def cuda_memory(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    a, e = actual.detach().to("cpu"), expected.detach().to("cpu")
    same_shape = tuple(a.shape) == tuple(e.shape)
    if not same_shape:
        return {
            "shape_match": False,
            "actual_shape": list(a.shape),
            "expected_shape": list(e.shape),
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "max_relative_diff": None,
            "passed": False,
        }
    delta = (a - e).abs()
    relative = delta / e.abs().clamp_min(1e-12)
    try:
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol)
        passed = True
    except AssertionError:
        passed = False
    return {
        "shape_match": True,
        "actual_shape": list(a.shape),
        "expected_shape": list(e.shape),
        "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_relative_diff": float(relative.max().item()) if relative.numel() else 0.0,
        "passed": passed,
    }


def compare_state_after_finetuned_overlay(
    restored: dict[str, torch.Tensor],
    pretrained: dict[str, torch.Tensor],
    fine_tuned_overlay: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare base tensors while accounting for explicitly fine-tuned keys.

    Runtime verification happens after the source loader overlays trainable
    tensors. Those keys must be checked against the fine-tuned artifact, not
    against their original pretrained values.
    """
    restored_keys, pretrained_keys, overlay_keys = set(restored), set(pretrained), set(fine_tuned_overlay)
    missing = sorted(restored_keys - pretrained_keys)
    unexpected = sorted(pretrained_keys - restored_keys)
    shape_mismatches = sorted(
        key for key in restored_keys & pretrained_keys
        if tuple(restored[key].shape) != tuple(pretrained[key].shape)
    )
    overlay_missing = sorted(overlay_keys - pretrained_keys)
    overlay_shapes = sorted(
        key for key in overlay_keys & pretrained_keys
        if tuple(fine_tuned_overlay[key].shape) != tuple(pretrained[key].shape)
    )
    comparable = sorted(pretrained_keys - overlay_keys)
    mismatches = [
        key for key in comparable
        if key in restored and key in pretrained and tuple(restored[key].shape) == tuple(pretrained[key].shape)
        and not torch.equal(
            restored[key].detach().to("cpu"),
            pretrained[key].detach().to("cpu", dtype=restored[key].dtype),
        )
    ]
    passed = not (missing or unexpected or shape_mismatches or overlay_missing or overlay_shapes or mismatches)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "overlay_missing_from_base": overlay_missing,
        "overlay_shape_mismatches": overlay_shapes,
        "fine_tuned_overlay_keys": len(overlay_keys),
        "pretrained_values_compared": len(comparable),
        "mismatched_pretrained_keys": mismatches,
        "passed": passed,
    }


def cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone()


def mask_summary(mask: torch.Tensor) -> dict[str, Any]:
    value = mask.detach().to("cpu")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "true_padding_counts": value.to(torch.int64).sum(dim=1).tolist(),
        "valid_counts": (~value.bool()).to(torch.int64).sum(dim=1).tolist(),
        "all_rows_have_valid_positions": bool((~value.bool()).any(dim=1).all().item()),
        "padding_is_suffix": all(
            not bool((row[torch.where(row)[0][0]:] == 0).any().item())
            if bool(row.any().item()) else True
            for row in value.bool()
        ),
    }


class ForwardProbe:
    """Temporary hooks for actual call counts, masks and fused feature blocks."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.events: dict[str, Any] = {}
        self.handles = []
        modules = {
            "expert4": model,
            "audio_encoder": model.audio_encoder,
            "text_encoder": model.text_encoder,
            "seq_coattn": model.seq_coattn,
            "audio_to_text_attn": model.seq_coattn.audio_to_text_attn,
            "text_to_audio_attn": model.seq_coattn.text_to_audio_attn,
            "pool": model.seq_coattn.pool,
            "head": model.head,
        }
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))
        for name, module in (
            ("audio_to_text_attn", model.seq_coattn.audio_to_text_attn),
            ("text_to_audio_attn", model.seq_coattn.text_to_audio_attn),
        ):
            try:
                handle = module.register_forward_pre_hook(self._attention_pre_hook(name), with_kwargs=True)
            except TypeError as exc:
                raise RuntimeError("Installed PyTorch does not support kwargs-aware MHA hooks") from exc
            self.handles.append(handle)
        self.reset()

    def reset(self) -> None:
        self.events = {
            "counts": {},
            "pool_inputs": [],
            "pool_outputs": [],
            "attention_outputs": {},
            "attention_padding_masks": {},
        }

    def _hook(self, name: str):
        def hook(_module, inputs, output):
            self.events["counts"][name] = self.events["counts"].get(name, 0) + 1
            if name == "audio_encoder":
                wavs, lengths = inputs[:2]
                self.events["audio_input"] = {
                    "shape": list(wavs.shape),
                    "dtype": str(wavs.dtype),
                    "lengths": cpu_copy(lengths),
                }
                self.events["audio_features"] = cpu_copy(output[0])
                self.events["audio_mask"] = cpu_copy(output[1])
            elif name == "text_encoder":
                self.events["captions"] = list(inputs[0])
                self.events["text_features"] = cpu_copy(output[0])
                self.events["text_mask"] = cpu_copy(output[1])
            elif name == "seq_coattn":
                self.events["seq_audio_mask"] = cpu_copy(inputs[2])
                self.events["seq_text_mask"] = cpu_copy(inputs[3])
                self.events["seq_features"] = cpu_copy(output)
            elif name in {"audio_to_text_attn", "text_to_audio_attn"}:
                self.events["attention_outputs"][name] = cpu_copy(output[0])
            elif name == "pool":
                self.events["pool_outputs"].append(cpu_copy(output))
            elif name == "head":
                self.events["head_features"] = cpu_copy(inputs[0])
                self.events["head_score"] = cpu_copy(output)
        return hook

    def _attention_pre_hook(self, name: str):
        def hook(_module, _inputs, kwargs):
            mask = kwargs.get("key_padding_mask")
            if mask is None:
                self.events["attention_padding_masks"][name] = None
            else:
                self.events["attention_padding_masks"][name] = cpu_copy(mask)
        return hook

    @contextmanager
    def instrument_beats_mask_reduction(self):
        beats = self.model.audio_encoder.model
        original = beats.forward_padding_mask
        reductions: list[dict[str, Any]] = []

        def observed(features, padding_mask):
            result = original(features, padding_mask)
            reductions.append({
                "feature_shape": list(features.shape),
                "input_mask_shape": list(padding_mask.shape),
                "input_true_counts": padding_mask.detach().to("cpu", dtype=torch.int64).sum(dim=1).tolist(),
                "output_mask_shape": list(result.shape),
                "output_true_counts": result.detach().to("cpu", dtype=torch.int64).sum(dim=1).tolist(),
                "output_dtype": str(result.dtype),
            })
            return result

        beats.forward_padding_mask = observed
        try:
            yield reductions
        finally:
            del beats.forward_padding_mask

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def assert_probe_contract(probe: ForwardProbe, *, lengths: torch.Tensor, expect_two_audio_masks: bool = True) -> dict[str, Any]:
    events = probe.events
    expected_counts = {
        "expert4": 1,
        "audio_encoder": 1,
        "text_encoder": 1,
        "seq_coattn": 1,
        "audio_to_text_attn": 1,
        "text_to_audio_attn": 1,
        "pool": 2,
        "head": 1,
    }
    if events["counts"] != expected_counts:
        raise VerificationFailure(
            "FAIL — NUMERICAL PARITY OR MASKING", "single_pass",
            f"Unexpected Expert 4 module call counts: {events['counts']} (expected {expected_counts})",
        )
    if not torch.equal(events["audio_input"]["lengths"], lengths.detach().cpu()):
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "lengths", "AudioEncoder did not receive original wav_lens unchanged")

    audio_mask, text_mask = events["audio_mask"], events["text_mask"]
    if audio_mask.dtype != torch.bool or text_mask.dtype != torch.bool:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "masks", "Expected bool audio and text padding masks")
    if not torch.equal(audio_mask, events["seq_audio_mask"]) or not torch.equal(text_mask, events["seq_text_mask"]):
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "masks", "SeqCoAttn did not receive the exact encoder masks")
    attention_masks = events["attention_padding_masks"]
    if not torch.equal(attention_masks["audio_to_text_attn"], text_mask):
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "masks", "Audio-query attention did not use text key-padding mask")
    if not torch.equal(attention_masks["text_to_audio_attn"], audio_mask):
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "masks", "Text-query attention did not use audio key-padding mask")

    features = events["seq_features"]
    dim = int(getattr(probe.model.seq_coattn, "dim"))
    blocks = []
    for index, pooled in enumerate(events["pool_outputs"]):
        if pooled.ndim != 3 or pooled.shape[1:] != (dim, 1):
            raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "feature_blocks", f"Unexpected pooled block shape {tuple(pooled.shape)}")
        block = pooled.squeeze(-1)
        actual = features[:, index * dim:(index + 1) * dim].detach().cpu()
        metrics = compare_tensors(actual, block, atol=0.0, rtol=0.0)
        if not metrics["passed"]:
            raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "feature_blocks", f"Fused feature block {index} differs from its source pooled branch")
        blocks.append({"index": index + 1, "shape": list(block.shape), "max_abs_diff": metrics["max_abs_diff"]})
    if len(blocks) != 2 or features.shape[1] != 2 * dim:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "feature_blocks", "Runtime fused feature width does not equal two SeqCoAttn pooled branches")

    reduction_records = events.get("mask_reductions", [])
    if expect_two_audio_masks and len(reduction_records) != 2:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "masks", f"Expected two source BEATs mask reductions, observed {len(reduction_records)}")
    return {
        "module_invocation_counts": events["counts"],
        "audio_encoder_received_lengths_unchanged": True,
        "audio_padding_mask": mask_summary(audio_mask),
        "text_padding_mask": mask_summary(text_mask),
        "audio_mask_reductions": reduction_records,
        "seqcoattn_received_encoder_masks_unchanged": True,
        "mha_key_padding_masks_match_source_masks": True,
        "released_fused_blocks": [
            {"index": 1, "source": "audio-query attending to text, then original adaptive max pool", **blocks[0]},
            {"index": 2, "source": "text-query attending to audio, then original adaptive max pool", **blocks[1]},
        ],
    }


def run_one_path(model, adapter, probe, wavs, lengths, captions, *, use_adapter, label, atol, rtol):
    probe.reset()
    with probe.instrument_beats_mask_reduction() as reductions, torch.no_grad():
        if use_adapter:
            score, features = adapter({
                "expert4_wavs": wavs,
                "expert4_wav_lens": lengths,
                "captions_raw": captions,
            })
        else:
            score, features = model({"wavs": wavs, "wav_lens": lengths, "captions": captions})
    probe.events["mask_reductions"] = reductions
    mask_check = assert_probe_contract(probe, lengths=lengths)
    if not torch.isfinite(score).all().item() or not torch.isfinite(features).all().item():
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", label, "Non-finite score or fused features")
    return {
        "score": score.detach(),
        "features": features.detach(),
        "score_shape": list(score.shape),
        "feature_shape": list(features.shape),
        "score_values": score.detach().cpu().reshape(-1).tolist(),
        "score_dtype": str(score.dtype),
        "feature_dtype": str(features.dtype),
        "device": str(features.device),
        "finite": True,
        "masking_and_invocation": mask_check,
        "caption_values_seen_by_original": probe.events["captions"],
    }


def run_parity_case(model, adapter, probe, *, label, wavs, lengths, captions, atol, rtol):
    standalone = run_one_path(model, adapter, probe, wavs, lengths, captions, use_adapter=False, label=f"{label}_standalone", atol=atol, rtol=rtol)
    adapter_result = run_one_path(model, adapter, probe, wavs, lengths, captions, use_adapter=True, label=f"{label}_adapter", atol=atol, rtol=rtol)
    if standalone["score_shape"] != [len(captions)] or adapter_result["score_shape"] != [len(captions), 1]:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", label, f"Unexpected score shapes: {standalone['score_shape']} vs {adapter_result['score_shape']}")
    if standalone["feature_shape"] != adapter_result["feature_shape"]:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", label, "Standalone and adapter feature shapes differ")
    score_metrics = compare_tensors(adapter_result["score"].squeeze(-1), standalone["score"], atol=atol, rtol=rtol)
    feature_metrics = compare_tensors(adapter_result["features"], standalone["features"], atol=atol, rtol=rtol)
    if not score_metrics["passed"] or not feature_metrics["passed"]:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", label, f"Parity failed: score={score_metrics}, features={feature_metrics}")
    return {
        "label": label,
        "batch_size": len(captions),
        "waveform_shape": list(wavs.shape),
        "waveform_dtype": str(wavs.dtype),
        "valid_lengths": lengths.detach().cpu().tolist(),
        "captions": list(captions),
        "standalone_score_shape": standalone["score_shape"],
        "adapter_score_shape": adapter_result["score_shape"],
        "feature_shape": standalone["feature_shape"],
        "score_values": standalone["score_values"],
        "score_parity": score_metrics,
        "feature_parity": feature_metrics,
        "output_dtype": standalone["score_dtype"],
        "output_device": standalone["device"],
        "standalone_masking_and_counts": standalone["masking_and_invocation"],
        "adapter_masking_and_counts": adapter_result["masking_and_invocation"],
        "passed": True,
    }


def preflight_text_cache() -> dict[str, Any]:
    from transformers import RobertaConfig, RobertaTokenizer

    config = RobertaConfig.from_pretrained("roberta-base", local_files_only=True)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True)
    return {
        "status": "PASS",
        "identifier": "roberta-base",
        "local_files_only": True,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "config_model_type": config.model_type,
        "config_hidden_size": config.hidden_size,
        "model_weights_source": "LAION-CLAP checkpoint supplies text branch weights; source constructs architecture-only RoBERTa config",
    }


def validate_checkpoint_resources(args, summary):
    from model.expert_adapters.expert4_adapter import _load_best_bundle_preflight
    from expert4_finetune.trainer import _source_identity

    beats = repo_path(args.beats_checkpoint)
    clap = repo_path(args.clap_checkpoint)
    finetuned = repo_path(args.finetuned_checkpoint)
    unilm = repo_path(args.unilm_source_dir)
    config = repo_path(args.training_config)
    source_files = [unilm / "beats" / name for name in ("BEATs.py", "backbone.py", "modules.py")]
    for path in (beats, clap, finetuned, config, *source_files):
        if not path.is_file() or not os.access(path, os.R_OK):
            raise VerificationFailure("BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION", "resources", f"Required readable resource is missing: {path}")
    if not unilm.is_dir():
        raise VerificationFailure("BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION", "resources", f"UniLM source directory is missing: {unilm}")

    inventory = {name: file_record(path, include_sha256=name in {"BEATs.py", "backbone.py", "modules.py"}) for name, path in {
        "beats_checkpoint": beats,
        "clap_checkpoint": clap,
        "finetuned_checkpoint": finetuned,
        "training_config": config,
        "BEATs.py": source_files[0],
        "backbone.py": source_files[1],
        "modules.py": source_files[2],
    }.items()}
    source_dir_identity = _source_identity(unilm)
    bundle = _load_best_bundle_preflight(finetuned)
    state = bundle["model_state_dict"]
    expected = bundle["fingerprints"]
    summary["resources"] = inventory
    summary["fine_tuned_checkpoint_schema"] = {
        "status": "PASS",
        "top_level_keys": sorted(bundle),
        "expected_source_best_schema": ["best_srcc", "config", "epoch", "fingerprints", "model_state_dict"],
        "format_version": "not defined by source best-checkpoint writer",
        "checkpoint_kind": "not defined by source best-checkpoint writer; exact five-key schema distinguishes best from resume",
        "epoch": bundle["epoch"],
        "best_srcc": float(bundle["best_srcc"]),
        "model_state_key_count": len(state),
        "non_tensor_values": sum(not isinstance(value, torch.Tensor) for value in state.values()),
        "non_finite_tensors": sum(not bool(torch.isfinite(value).all().item()) for value in state.values()),
        "checkpoint_config_resource_keys": sorted(bundle["config"].get("resources", {})),
        "checkpoint_recorded_resource_paths": {
            key: bundle["config"].get("resources", {}).get(key)
            for key in ("unilm_source_dir", "beats_checkpoint", "clap_checkpoint")
        },
        "fingerprints": expected,
    }
    summary["unilm_source_identity"] = {
        "implementation": "expert4_finetune.trainer._source_identity",
        "checkpoint_expected": expected["unilm_source"],
        "actual": source_dir_identity,
        "matches": source_dir_identity == expected["unilm_source"],
        "individual_source_file_sha256": {
            name: inventory[name]["sha256"] for name in ("BEATs.py", "backbone.py", "modules.py")
        },
    }

    emit("HASH_START: BEATs pretrained checkpoint")
    beats_sha = file_sha256(beats)
    emit(f"HASH_DONE: BEATs {beats_sha}")
    emit("HASH_START: LAION-CLAP pretrained checkpoint")
    clap_sha = file_sha256(clap)
    emit(f"HASH_DONE: CLAP {clap_sha}")
    identities = {
        "beats_checkpoint": {"expected": expected["beats_checkpoint"], "actual": beats_sha, "matches": beats_sha == expected["beats_checkpoint"]},
        "clap_checkpoint": {"expected": expected["clap_checkpoint"], "actual": clap_sha, "matches": clap_sha == expected["clap_checkpoint"]},
    }
    summary["pretrained_fingerprints"] = identities
    summary["initial_artifact_hashes"] = {
        str(beats): beats_sha,
        str(clap): clap_sha,
        str(finetuned): file_sha256(finetuned),
    }
    mismatches = [name for name, value in identities.items() if not value["matches"]]
    summary["identity_failures"] = {
        "pretrained_checkpoint_mismatches": mismatches,
        "unilm_source_mismatch": not summary["unilm_source_identity"]["matches"],
    }
    return {"beats": beats, "clap": clap, "finetuned": finetuned, "unilm": unilm, "training_config": config, "bundle": bundle}


def verify_pretrained_tensor_values(
    model,
    beats_path: Path,
    clap_path: Path,
    fine_tuned_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare base tensors, excluding only keys verified as fine-tuned."""
    beats_ckpt = torch.load(beats_path, map_location="cpu", weights_only=True, mmap=True)
    beats_state = beats_ckpt.get("model") if isinstance(beats_ckpt, dict) else None
    if not isinstance(beats_state, dict):
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "beats_state", "BEATs checkpoint has no model state mapping")
    restored_beats = model.audio_encoder.model.state_dict()
    beats_prefix = "audio_encoder.model."
    beats_overlay = {
        name[len(beats_prefix):]: tensor
        for name, tensor in fine_tuned_state.items()
        if name.startswith(beats_prefix)
    }
    beats_comparison = compare_state_after_finetuned_overlay(restored_beats, beats_state, beats_overlay)
    beats_result = {
        "loader": "original AudioEncoder uses load_state_dict(checkpoint['model'], strict=True)",
        "strict_loader_succeeded": True,
        "checkpoint_key_count": len(beats_state),
        "model_state_key_count": len(restored_beats),
        **beats_comparison,
        "restored_values_exact_for_unmodified_pretrained_keys": beats_comparison["passed"],
        "encoder_layers": len(model.audio_encoder.model.encoder.layers),
    }
    del beats_ckpt, beats_state, restored_beats

    # The original CLAP loader intentionally strict-loads only the required
    # text branch/projection subset with strict=False and rejects missing keys
    # for those components. Verify every matching tensor against that subset.
    try:
        raw = torch.load(clap_path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError) as exc:
        if "mmap" not in str(exc).lower():
            raise
        raw = torch.load(clap_path, map_location="cpu", weights_only=False)
    clap_state = raw.get("state_dict", raw.get("model", raw))
    if not isinstance(clap_state, dict):
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "clap_state", "LAION-CLAP checkpoint state is not a mapping")
    cleaned = {name.removeprefix("module."): value for name, value in clap_state.items()}
    target_state: dict[str, torch.Tensor] = {}
    target_state.update({f"text_branch.{name}": value for name, value in model.text_encoder.branch.state_dict().items()})
    target_state.update({f"text_projection.{name}": value for name, value in model.text_encoder.projection.state_dict().items()})
    relevant_checkpoint_keys = {key for key in cleaned if key.startswith(("text_branch.", "text_projection."))}
    checkpoint_only_keys = relevant_checkpoint_keys - set(target_state)
    # Transformers versions differ on whether RoBERTa's deterministic
    # position_ids buffer is persistent in state_dict(). The source loader
    # intentionally uses strict=False and rejects missing text parameters;
    # this one non-parameter buffer is the only accepted checkpoint-only key.
    allowed_compatibility_buffers = {"text_branch.embeddings.position_ids"}
    compatibility_buffers = sorted(checkpoint_only_keys & allowed_compatibility_buffers)
    invalid_checkpoint_only_keys = sorted(checkpoint_only_keys - allowed_compatibility_buffers)
    for key in compatibility_buffers:
        value = cleaned[key]
        if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
            invalid_checkpoint_only_keys.append(key)
    clap_overlay: dict[str, torch.Tensor] = {}
    for name, tensor in fine_tuned_state.items():
        if name.startswith("text_encoder.branch."):
            clap_overlay["text_branch." + name[len("text_encoder.branch."):]] = tensor
        elif name.startswith("text_encoder.projection."):
            clap_overlay["text_projection." + name[len("text_encoder.projection."):]] = tensor
    clap_base = {key: cleaned[key] for key in relevant_checkpoint_keys if key in target_state}
    clap_comparison = compare_state_after_finetuned_overlay(target_state, clap_base, clap_overlay)
    clap_result = {
        "loader": "original TextEncoder validates RoBERTa/projection geometry and loads LAION CLAP with strict=False; missing text/projection keys are rejected",
        "source_required_subset_keys": len(target_state),
        "checkpoint_relevant_keys": len(relevant_checkpoint_keys),
        "missing_required_keys": len(clap_comparison["missing_keys"]),
        "unexpected_text_or_projection_keys": len(invalid_checkpoint_only_keys),
        "allowed_compatibility_buffers": [
            {"key": key, "shape": list(cleaned[key].shape), "dtype": str(cleaned[key].dtype)}
            for key in compatibility_buffers
        ],
        **clap_comparison,
        "unexpected_keys": invalid_checkpoint_only_keys,
        "restored_values_exact_for_unmodified_pretrained_keys": clap_comparison["passed"] and not invalid_checkpoint_only_keys,
        "source_loader_reports_exhaustive_other_component_key_diagnostics": False,
    }
    del raw, clap_state, cleaned, target_state
    if not beats_result["restored_values_exact_for_unmodified_pretrained_keys"]:
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "beats_tensor_parity", f"BEATs restored state differs: {beats_result}")
    if not clap_result["restored_values_exact_for_unmodified_pretrained_keys"]:
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "clap_tensor_parity", f"CLAP text/projection restored state differs: {clap_result}")
    return {"beats": beats_result, "clap": clap_result}


def verify_finetuned_tensor_values(model, checkpoint_state: dict[str, Any]) -> dict[str, Any]:
    trained = checkpoint_state["model_state_dict"]
    parameters = dict(model.named_parameters())
    missing = sorted(set(trained) - set(parameters))
    shape_mismatches = [name for name in set(trained) & set(parameters) if tuple(trained[name].shape) != tuple(parameters[name].shape)]
    mismatches = []
    if not missing and not shape_mismatches:
        for name, tensor in trained.items():
            if not torch.equal(parameters[name].detach().cpu(), tensor.detach().cpu().to(dtype=parameters[name].dtype)):
                mismatches.append(name)
                if len(mismatches) >= 8:
                    break
    # trainable names are now frozen by the adapter; the source loader and the
    # adapter factory checked exact trainable-key equality before freezing.
    if missing or shape_mismatches or mismatches:
        raise VerificationFailure("FAIL — FINE-TUNED PARTIAL RESTORATION", "trained_tensor_restore", f"Fine-tuned tensors differ after original restoration: missing={missing[:8]}, shapes={shape_mismatches[:8]}, values={mismatches}")
    return {
        "expected_trainable_keys": len(trained),
        "actual_checkpoint_keys": len(trained),
        "missing_required_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
        "non_tensor_values": 0,
        "non_finite_tensors": 0,
        "trained_tensors_compared_after_original_loader": len(trained),
        "exact_tensor_matches": len(trained),
        "tensor_mismatches": 0,
        "loader": "expert4_finetune.inference.load_finetuned -> trainer.load_trainable_state_dict",
        "strict": "source-defined strict=False for partial state; exact trainable key set and shapes checked by source, actual values checked by adapter factory and verifier",
    }


def make_synthetic(batch_size: int, lengths: list[int], *, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed + batch_size + sum(lengths))
    wavs = torch.randn(batch_size, 1, MAX_SAMPLES, generator=generator, dtype=torch.float32)
    for index, valid in enumerate(lengths):
        wavs[index, :, valid:] = 0.0
    return wavs.to(device), torch.tensor(lengths, dtype=torch.int64, device=device)


def source_and_loaded_module_origins(model, unilm: Path) -> dict[str, Any]:
    from model.expert_adapters.expert4_adapter import validate_expert4_module_origins

    beats_dir = unilm if (unilm / "BEATs.py").is_file() else unilm / "beats"
    module_name = type(model.audio_encoder.model).__module__
    prefix = module_name.rsplit(".", 1)[0]
    names = [f"{prefix}.BEATs", f"{prefix}.backbone", f"{prefix}.modules"]
    origins = {}
    for name in names:
        module = sys.modules.get(name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None) if module is not None else None
        path = getattr(module, "__file__", None) if module is not None else None
        origins[name] = {"loaded": module is not None, "file": path, "spec_origin": origin}
    expected = {
        names[0]: str((beats_dir / "BEATs.py").resolve()),
        names[1]: str((beats_dir / "backbone.py").resolve()),
        names[2]: str((beats_dir / "modules.py").resolve()),
    }
    for name, target in expected.items():
        record = origins[name]
        if not record["loaded"] or Path(record["file"]).resolve() != Path(target):
            raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "module_origins", f"M2D/BEATs module origin mismatch for {name}: {record}, expected={target}")
    validate_expert4_module_origins(unilm)
    return {
        "private_package_modules": origins,
        "expected_source_files": expected,
        "top_level_BEATs_alias_cleaned": "BEATs" not in sys.modules,
        "audio_encoder_model_class_module": module_name,
        "expert4_source_modules": {
            name: getattr(sys.modules.get(name), "__file__", None)
            for name in ("expert4_finetune.model", "expert4_finetune.audio_encoder", "expert4_finetune.text_encoder", "expert4_finetune.seq_coattn", "expert4_finetune.inference", "expert4_finetune.trainer")
        },
    }


def verify_real_beats_package_import(unilm: Path) -> dict[str, Any]:
    """Exercise the scoped import bridge on vendored BEATs source only.

    This deliberately imports the source modules but does not construct/load
    the pretrained BEATs model. It remains safe to run before artifact identity
    validation permits the heavyweight factory.
    """
    from model.expert_adapters.expert4_adapter import _scoped_beats_import

    from model.expert_adapters.expert4_adapter import _beats_dir
    beats_dir = _beats_dir(unilm)
    aliases_before = {name: sys.modules.get(name) for name in ("BEATs", "backbone", "modules")}
    path_before = list(sys.path)
    with _scoped_beats_import(unilm) as module:
        package = module.__package__
        backbone = sys.modules.get("backbone")
        modules = sys.modules.get("modules")
        origins = {
            "BEATs": getattr(module, "__file__", None),
            "backbone": getattr(backbone, "__file__", None),
            "modules": getattr(modules, "__file__", None),
            "alias_is_exact_module": sys.modules.get("BEATs") is module,
            "backbone_alias_is_private_module": backbone is sys.modules.get(f"{package}.backbone"),
            "modules_alias_is_private_module": modules is sys.modules.get(f"{package}.modules"),
            "package": package,
        }
    path_restored = sys.path == path_before
    aliases_restored = all(sys.modules.get(name) is previous for name, previous in aliases_before.items())
    expected = {
        "BEATs": str((beats_dir / "BEATs.py").resolve()),
        "backbone": str((beats_dir / "backbone.py").resolve()),
        "modules": str((beats_dir / "modules.py").resolve()),
    }
    origins_match = all(Path(origins[name]).resolve() == Path(path) for name, path in expected.items())
    if not origins_match or not aliases_restored or not path_restored or not all((origins["alias_is_exact_module"], origins["backbone_alias_is_private_module"], origins["modules_alias_is_private_module"])):
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "beats_import_bridge", f"Scoped BEATs source import/origin/cleanup check failed: {origins}")
    return {
        "status": "PASS_SOURCE_IMPORT_ONLY",
        "module_origins": origins,
        "expected_origins": expected,
        "temporary_aliases_restored": aliases_restored,
        "sys_path_fully_restored": path_restored,
        "real_factory_or_checkpoint_load": "NOT_PERFORMED by this lightweight preflight; separately verified in the main factory stage",
    }


def native_validation_case(csv_path: Path, audio_root: Path, count: int, preferred: list[str]):
    from scipy.io import wavfile
    from expert4_finetune.dataset import Expert4Dataset, collate_expert4

    dataset = Expert4Dataset(csv_path, audio_root, require_score=True)
    selected: list[int] = []
    used_files: set[str] = set()
    for wanted in preferred:
        found = next((i for i, row in enumerate(dataset.rows) if Path(row["sample_id"]).name == wanted and row["sample_id"] not in used_files), None)
        if found is not None and found not in selected:
            selected.append(found)
            used_files.add(dataset.rows[found]["sample_id"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index, row in enumerate(dataset.rows):
            if row["sample_id"] not in used_files:
                selected.append(index)
                used_files.add(row["sample_id"])
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise ValueError(f"Validation dataset has only {len(selected)} distinct audio files; requested {count}")
    items = [dataset[index] for index in selected]
    batch = collate_expert4(items)
    wav_metadata = []
    for row in (dataset.rows[index] for index in selected):
        sample_rate, raw_audio = wavfile.read(row["path"])
        wav_metadata.append({
            "sample_id": row["sample_id"],
            "path": str(row["path"]),
            "original_sample_rate_hz": int(sample_rate),
            "original_frame_count": int(raw_audio.shape[0]),
            "original_channels": 1 if raw_audio.ndim == 1 else int(raw_audio.shape[1]),
            "post_resample_crop_valid_samples": int(next(item["wav"].numel() for item in items if item["sample_id"] == row["sample_id"])),
            "caption": row["caption"],
        })
    return batch, wav_metadata


def run_verification(args, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {} if summary is None else summary
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    summary.update({
        "status": "IN_PROGRESS",
        "requested_device": args.device,
        "device": str(device),
        "offline": True,
        "offline_environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
        "seed": args.seed,
        "precision": "torch.float32; no autocast or quantization",
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": package_version("transformers"),
            "laion_clap": package_version("laion-clap") or package_version("laion_clap"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "cuda_memory_before_load": cuda_memory(device),
        "cases": [],
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    summary["stage"] = "resource_and_checkpoint_preflight"
    paths = validate_checkpoint_resources(args, summary)
    stage = "roberta_offline_preflight"
    summary["stage"] = stage
    emit("ROBERTA_OFFLINE_PREFLIGHT_START")
    try:
        summary["text_encoder_cache"] = preflight_text_cache()
    except Exception as exc:
        summary["text_encoder_cache"] = {"status": "FAIL", "identifier": "roberta-base", "error": f"{type(exc).__name__}: {exc}"}
        emit(f"ROBERTA_OFFLINE_PREFLIGHT_FAILED: {type(exc).__name__}: {exc}")
    else:
        emit("ROBERTA_OFFLINE_PREFLIGHT_PASS")

    stage = "identity_validation"
    summary["stage"] = stage
    identity_failures = summary.get("identity_failures", {})
    if identity_failures.get("pretrained_checkpoint_mismatches") or identity_failures.get("unilm_source_mismatch"):
        try:
            summary["beats_import_compatibility"] = verify_real_beats_package_import(paths["unilm"])
        except Exception as exc:
            summary["beats_import_compatibility"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        summary["original_factory"] = {
            "construction_attempted": False,
            "reason": "Original loader rejects the current UniLM source fingerprint before model construction; no fingerprint bypass was used.",
        }
        if args.validation_csv and args.audio_root:
            try:
                csv_path, audio_root = repo_path(args.validation_csv), repo_path(args.audio_root)
                if csv_path.is_file() and audio_root.is_dir():
                    batch, samples = native_validation_case(csv_path, audio_root, args.num_real_samples, args.preferred_sample_ids)
                    summary["real_validation"] = {
                        "status": "NOT_VERIFIED",
                        "native_preprocessing_status": "PASS",
                        "inference_status": "NOT_RUN due source fingerprint mismatch",
                        "csv": str(csv_path),
                        "audio_root": str(audio_root),
                        "samples": samples,
                        "sample_ids": batch["sample_ids"],
                        "waveform_shape": list(batch["wavs"].shape),
                        "waveform_dtype": str(batch["wavs"].dtype),
                        "valid_lengths": batch["wav_lens"].tolist(),
                        "captions": batch["captions"],
                        "native_preprocessing": "Expert4Dataset.__getitem__ + collate_expert4 from expert4_finetune/dataset.py",
                    }
            except Exception as exc:
                summary["real_validation"] = {"status": "NOT_VERIFIED", "native_preprocessing_status": "FAIL", "inference_status": "NOT_RUN", "error": f"{type(exc).__name__}: {exc}"}
        raise VerificationFailure(
            "FAIL — PRETRAINED IDENTITY OR RESTORATION",
            "pretrained_and_source_identity",
            "Artifact identity checks failed; original Expert4 factory was intentionally not called. "
            f"details={identity_failures}",
        )

    stage = "original_factory_construction_and_restoration"
    summary["stage"] = stage
    from model.expert_adapters.expert4_adapter import (
        Expert4Adapter,
        _scoped_beats_import,
        validate_expert4_module_origins,
    )

    sys_path_before = list(sys.path)
    aliases_before = {name: sys.modules.get(name) for name in ("BEATs", "backbone", "modules")}
    beats_dir = paths["unilm"] if (paths["unilm"] / "BEATs.py").is_file() else paths["unilm"] / "beats"
    beats_path_count_before = sum(bool(entry) and Path(entry).resolve(strict=False) == beats_dir.resolve() for entry in sys.path)
    emit("RESTORATION_START: original Expert4Adapter.from_finetuned -> original load_finetuned")
    with _scoped_beats_import(paths["unilm"]) as beats_module:
        adapter = Expert4Adapter.from_finetuned(
            beats_checkpoint=paths["beats"],
            clap_checkpoint=paths["clap"],
            unilm_source_dir=paths["unilm"],
            finetuned_checkpoint=paths["finetuned"],
            device=device,
            offline=True,
        )
        aliases_during_factory = {
            name: {
                "file": getattr(sys.modules.get(name), "__file__", None),
                "same_private_module": sys.modules.get(name) is sys.modules.get(f"{beats_module.__package__}.{name}")
                if name != "BEATs" else sys.modules.get(name) is beats_module,
            }
            for name in ("BEATs", "backbone", "modules")
        }
    model = adapter.model
    if type(model).__module__ != "expert4_finetune.model" or type(model).__name__ != "Expert4":
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", stage, f"Adapter wrapped unexpected class: {type(model).__module__}.{type(model).__name__}")
    beats_path_count_after = sum(bool(entry) and Path(entry).resolve(strict=False) == beats_dir.resolve() for entry in sys.path)
    aliases_restored = all(sys.modules.get(name) is previous for name, previous in aliases_before.items())
    summary["beats_import_compatibility"] = {
        "factory_construction_passed": True,
        "temporary_aliases_during_factory": aliases_during_factory,
        "temporary_aliases_restored_after_factory": aliases_restored,
        "top_level_aliases_after_factory": {name: sys.modules.get(name) is not None for name in aliases_before},
        "beats_sys_path_count_before": beats_path_count_before,
        "beats_sys_path_count_after": beats_path_count_after,
        "beats_sys_path_restored": beats_path_count_before == beats_path_count_after,
        "sys_path_full_list_unchanged": sys.path == sys_path_before,
        "source_added_other_sys_path_entries": [entry for entry in sys.path if entry not in sys_path_before],
    }
    if not aliases_restored or not summary["beats_import_compatibility"]["beats_sys_path_restored"]:
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "beats_import_cleanup", "BEATs temporary aliases or source path were not restored after construction")
    validate_expert4_module_origins(paths["unilm"], require_beats=False)
    summary["module_origins"] = source_and_loaded_module_origins(model, paths["unilm"])
    summary["original_factory"] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "constructor": "Expert4.from_pretrained(beats_checkpoint, clap_checkpoint, unilm_source_dir) via original inference.load_finetuned",
        "restoration": "PASS",
        "partial_best_loader": "PASS",
    }
    summary["pretrained_tensor_restoration"] = verify_pretrained_tensor_values(
        model, paths["beats"], paths["clap"], paths["bundle"]["model_state_dict"]
    )
    summary["finetuned_tensor_restoration"] = verify_finetuned_tensor_values(model, paths["bundle"])
    summary["fine_tuned_checkpoint_schema"]["status"] = "PASS_STRICT_TRAINABLE_SET_AND_VALUES"

    seq_dim = getattr(model.seq_coattn, "dim", None)
    head_dim = getattr(model.head.net[0], "in_features", None)
    feature_dim = adapter.feature_dim
    if type(seq_dim) is not int or type(head_dim) is not int or feature_dim != 2 * seq_dim or head_dim != feature_dim:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "dimensions", f"Inconsistent feature metadata: seq_dim={seq_dim}, head_dim={head_dim}, adapter.feature_dim={feature_dim}")
    summary["runtime_dimensions"] = {
        "seqcoattn_audio_query_width": seq_dim,
        "seqcoattn_text_query_width": seq_dim,
        "regression_head_input_width": head_dim,
        "adapter_feature_dim": feature_dim,
        "actual_runtime_audio_feature_width": None,
        "actual_runtime_text_feature_width": None,
        "actual_fused_feature_width": feature_dim,
    }

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapter_only = sum(parameter.numel() for name, parameter in adapter.named_parameters() if not name.startswith("model."))
    all_frozen_before = all(not parameter.requires_grad for parameter in model.parameters())
    adapter.train()
    eval_after_train = all(not module.training for module in model.modules())
    frozen_after_train = all(not parameter.requires_grad for parameter in model.parameters())
    adapter.train(False)
    eval_after_train_false = all(not module.training for module in model.modules())
    adapter.eval()
    summary["frozen_eval"] = {
        "total_expert_parameters": total_params,
        "trainable_expert_parameters_after_restore": trainable_params,
        "adapter_only_trainable_parameters": adapter_only,
        "all_expert_parameters_frozen_before_mode_test": all_frozen_before,
        "all_expert_modules_eval_after_adapter_train": eval_after_train,
        "all_expert_parameters_frozen_after_adapter_train": frozen_after_train,
        "all_expert_modules_eval_after_adapter_train_false": eval_after_train_false,
    }
    if not all((all_frozen_before, eval_after_train, frozen_after_train, eval_after_train_false)) or trainable_params or adapter_only:
        raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "freeze_eval", "Expert4Adapter did not preserve frozen/eval behavior")

    probe = ForwardProbe(model)
    try:
        synthetic_cases = [
            ("synthetic_B1_full", [MAX_SAMPLES], ["A close recording of a hand drum."], args.seed),
            ("synthetic_B1_short", [80_000], ["A bird sings beside a river."], args.seed + 1),
            ("synthetic_B2_mixed_lengths", [80_000, MAX_SAMPLES], ["A hand drum plays a short rhythm.", "A bird sings beside a flowing river."], args.seed + 2),
        ]
        for label, lens, captions, seed in synthetic_cases:
            wavs, lengths = make_synthetic(len(lens), lens, seed=seed, device=device)
            case = run_parity_case(model, adapter, probe, label=label, wavs=wavs, lengths=lengths, captions=captions, atol=args.atol, rtol=args.rtol)
            summary["cases"].append(case)
            summary["runtime_dimensions"]["actual_runtime_audio_feature_width"] = case["standalone_masking_and_counts"]["released_fused_blocks"][0]["shape"][1]
            summary["runtime_dimensions"]["actual_runtime_text_feature_width"] = case["standalone_masking_and_counts"]["released_fused_blocks"][1]["shape"][1]
            emit(f"CASE_PASS: {label} score_max_abs_diff={case['score_parity']['max_abs_diff']} feature_max_abs_diff={case['feature_parity']['max_abs_diff']}")

        # Deterministic repeated evaluation check on a full-length B=1 sample.
        repeat_wavs, repeat_lengths = make_synthetic(1, [MAX_SAMPLES], seed=args.seed, device=device)
        first = run_one_path(model, adapter, probe, repeat_wavs, repeat_lengths, ["A close recording of a hand drum."], use_adapter=True, label="repeat_1", atol=args.atol, rtol=args.rtol)
        second = run_one_path(model, adapter, probe, repeat_wavs, repeat_lengths, ["A close recording of a hand drum."], use_adapter=True, label="repeat_2", atol=args.atol, rtol=args.rtol)
        repeated_score = compare_tensors(first["score"], second["score"], atol=args.atol, rtol=args.rtol)
        repeated_features = compare_tensors(first["features"], second["features"], atol=args.atol, rtol=args.rtol)
        summary["repeated_evaluation"] = {"score": repeated_score, "features": repeated_features, "passed": repeated_score["passed"] and repeated_features["passed"]}
        if not summary["repeated_evaluation"]["passed"]:
            raise VerificationFailure("FAIL — NUMERICAL PARITY OR MASKING", "repeated_eval", "Repeated evaluation was not deterministic within tolerance")
    finally:
        probe.close()

    summary["real_validation"] = {"status": "NOT_VERIFIED", "reason": "validation CSV/audio root not supplied"}
    if args.validation_csv or args.audio_root:
        if not args.validation_csv or not args.audio_root:
            summary["real_validation"] = {"status": "NOT_VERIFIED", "reason": "both --validation-csv and --audio-root are required"}
        else:
            csv_path, audio_root = repo_path(args.validation_csv), repo_path(args.audio_root)
            if not csv_path.is_file() or not audio_root.is_dir():
                summary["real_validation"] = {"status": "NOT_VERIFIED", "reason": f"validation resources missing: csv={csv_path}, audio_root={audio_root}"}
            else:
                stage = "native_validation_inference"
                summary["stage"] = stage
                real_probe = ForwardProbe(model)
                try:
                    real_batch, wav_metadata = native_validation_case(csv_path, audio_root, args.num_real_samples, args.preferred_sample_ids)
                    wavs, lengths = real_batch["wavs"].to(device), real_batch["wav_lens"].to(device)
                    captions = real_batch["captions"]
                    real_case = run_parity_case(model, adapter, real_probe, label="native_validation_subset", wavs=wavs, lengths=lengths, captions=captions, atol=args.atol, rtol=args.rtol)
                    summary["real_validation"] = {
                        "status": "PASS",
                        "csv": str(csv_path),
                        "audio_root": str(audio_root),
                        "samples": wav_metadata,
                        "sample_ids": real_batch["sample_ids"],
                        "waveform_shape": list(wavs.shape),
                        "waveform_dtype": str(wavs.dtype),
                        "valid_lengths": lengths.detach().cpu().tolist(),
                        "captions": captions,
                        "native_preprocessing": "Expert4Dataset.__getitem__ + collate_expert4 from expert4_finetune/dataset.py",
                        "parity": real_case,
                    }
                    summary["cases"].append(real_case)
                except Exception as exc:
                    summary["real_validation"] = {
                        "status": "NOT_VERIFIED",
                        "stage": stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "note": "Core restoration and synthetic parity remain separately reported; the real-data subset did not pass.",
                    }
                    emit(f"REAL_VALIDATION_NOT_VERIFIED: {type(exc).__name__}: {exc}")
                finally:
                    real_probe.close()

    summary["cuda_memory_after_verification"] = cuda_memory(device)
    summary["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    summary["tested_batch_sizes"] = sorted({case["batch_size"] for case in summary["cases"]})
    summary["single_pass_verified"] = all(
        case["adapter_masking_and_counts"]["module_invocation_counts"] == {
            "expert4": 1, "audio_encoder": 1, "text_encoder": 1, "seq_coattn": 1,
            "audio_to_text_attn": 1, "text_to_audio_attn": 1, "pool": 2, "head": 1,
        }
        for case in summary["cases"]
    )
    summary["base_and_finetuned_restore"] = "PASS"
    summary["synthetic_parity"] = "PASS"
    summary["core_runtime_verification"] = "PASS"
    # Re-hash the trusted model artifacts after inference so the report can
    # confirm that verification did not modify checkpoint contents.
    after_hashes = {
        str(paths["beats"]): file_sha256(paths["beats"]),
        str(paths["clap"]): file_sha256(paths["clap"]),
        str(paths["finetuned"]): file_sha256(paths["finetuned"]),
    }
    summary["artifacts_unchanged"] = {
        "before_sha256": summary["initial_artifact_hashes"],
        "after_sha256": after_hashes,
        "all_contents_unchanged": after_hashes == summary["initial_artifact_hashes"],
        "all_sizes_and_mtimes_unchanged": all(
            file_record(Path(name))["size_bytes"] == summary["resources"][key]["size_bytes"]
            and file_record(Path(name))["mtime_ns"] == summary["resources"][key]["mtime_ns"]
            for key, name in (("beats_checkpoint", str(paths["beats"])), ("clap_checkpoint", str(paths["clap"])), ("finetuned_checkpoint", str(paths["finetuned"])))
        ),
    }
    if not summary["artifacts_unchanged"]["all_contents_unchanged"]:
        raise VerificationFailure("FAIL — PRETRAINED IDENTITY OR RESTORATION", "artifact_integrity", "Checkpoint contents changed during verification")
    summary["status"] = "PASS — INCLUDING REAL VALIDATION INFERENCE" if summary["real_validation"]["status"] == "PASS" else "PASS — REAL CHECKPOINT & NUMERICAL PARITY VERIFIED"
    summary.pop("stage", None)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-checkpoint", default="weights/BEATs_iter3_plus_AS2M.pt")
    parser.add_argument("--clap-checkpoint", default="weights/laion-clap.pt")
    parser.add_argument("--finetuned-checkpoint", default="weights/finetuned_weights/expert4_best.pt")
    parser.add_argument("--unilm-source-dir", default="expert4_finetune/unilm")
    parser.add_argument("--training-config", default="expert4_finetune/expert4_colab.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--offline", action="store_true", default=True, help="offline mode is always enforced")
    parser.add_argument("--validation-csv")
    parser.add_argument("--audio-root")
    parser.add_argument("--num-real-samples", type=int, default=2)
    parser.add_argument("--preferred-sample-ids", nargs="*", default=["07500.wav", "07501.wav"])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--report-json", help="optional path for a machine-readable summary copy")
    args = parser.parse_args(argv)
    if args.num_real_samples <= 0:
        parser.error("--num-real-samples must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    summary: dict[str, Any] = {"status": "IN_PROGRESS", "stage": "startup", "errors": []}
    try:
        run_verification(args, summary)
    except VerificationFailure as exc:
        summary.update({"status": exc.status, "stage": exc.stage, "errors": [f"{type(exc).__name__}: {exc}"]})
        traceback.print_exc(file=sys.stderr)
    except Exception as exc:
        cause = exc
        cause_chain = []
        while cause is not None:
            cause_chain.append(f"{type(cause).__name__}: {cause}")
            cause = cause.__cause__
        message = " <- ".join(cause_chain)
        text = message.lower()
        if "out of memory" in text or isinstance(exc, (ImportError, ModuleNotFoundError, OSError)):
            status = "BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION"
        elif "fingerprint" in text or "checkpoint" in text or "state_dict" in text:
            status = "FAIL — PRETRAINED IDENTITY OR RESTORATION"
        else:
            status = "FAIL — NUMERICAL PARITY OR MASKING"
        summary.update({"status": status, "stage": summary.get("stage", "runtime"), "errors": cause_chain or [message]})
        traceback.print_exc(file=sys.stderr)
    if "initial_artifact_hashes" in summary and "artifacts_unchanged" not in summary:
        try:
            before = summary["initial_artifact_hashes"]
            after = {name: file_sha256(Path(name)) for name in before}
            summary["artifacts_unchanged"] = {
                "before_sha256": before,
                "after_sha256": after,
                "all_contents_unchanged": before == after,
            }
        except Exception as integrity_exc:
            summary["artifacts_unchanged"] = {"status": "NOT_VERIFIED", "error": f"{type(integrity_exc).__name__}: {integrity_exc}"}
    if summary.get("status") == "IN_PROGRESS":
        summary["status"] = "BLOCKED — MISSING RESOURCE OR ENVIRONMENT LIMITATION"
    summary.setdefault("errors", [])
    summary["verification_command"] = "python3 scripts/verify_expert4_runtime.py --beats-checkpoint ... --clap-checkpoint ... --finetuned-checkpoint ... --unilm-source-dir ... --device auto --offline"
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    print("FINAL_JSON_START")
    print(rendered)
    print("FINAL_JSON_END")
    if args.report_json:
        report_path = repo_path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
