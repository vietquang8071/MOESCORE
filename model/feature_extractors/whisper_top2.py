"""Source-faithful Whisper-large-v2 feature extraction from Top 2 XACLE.

The decoder is conditioned on the raw caption tokens and the encoder output
from the same waveform. Both sequence outputs use plain mean pooling, exactly
as in the source implementation (including decoder positions for padding).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from model.expert_adapters.top2_common import adapter_config, target_device


MODEL_ID = "openai/whisper-large-v2"
SAMPLE_RATE = 16_000
WAVEFORM_SAMPLES = 160_000
MEL_BINS = 80
MEL_FRAMES = 3_000
HIDDEN_SIZE = 1_280
FEATURE_EXTRACTION_VERSION = "top2-whisper-large-v2-meanpool-v1"
_NAME = "WhisperTop2FeatureExtractor"

_PREPROCESSING_IDENTITY = {
    "sample_rate": SAMPLE_RATE,
    "waveform_shape": [1, WAVEFORM_SAMPLES],
    "audio_batch": "waveform.cpu().numpy().flatten()",
    "feature_extractor_call": {
        "sampling_rate": SAMPLE_RATE,
        "return_tensors": "pt",
        "extra_arguments": [],
    },
    "input_features_shape": [MEL_BINS, MEL_FRAMES],
    "tokenizer_call": {"return_tensors": "pt", "padding": True},
    "caption_source": "captions_raw, passed without normalization",
    "decoder_conditioning": "tokenized.input_ids passed as decoder_input_ids",
    "decoder_attention_mask": "not passed",
    "pooling": "arithmetic mean over complete sequence dimension, including padding",
    "output_dimensions": {"audio": HIDDEN_SIZE, "text": HIDDEN_SIZE},
}


def _json_identity(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite JSON-compatible values") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_model_files(directory: Path) -> list[Path]:
    """Validate a complete large-v2 HF snapshot and return its identity files."""
    config_path = directory / "config.json"
    preprocessor_path = directory / "preprocessor_config.json"
    tokenizer_config_path = directory / "tokenizer_config.json"
    for required in (config_path, preprocessor_path, tokenizer_config_path):
        if not required.is_file() or not os.access(required, os.R_OK):
            raise FileNotFoundError(f"Whisper-large-v2 checkpoint is missing {required.name}: {directory}")

    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        preprocessor_config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Whisper-large-v2 checkpoint has unreadable configuration: {directory}") from exc
    if model_config.get("model_type") != "whisper" or model_config.get("d_model") != HIDDEN_SIZE:
        raise ValueError("local checkpoint is not Whisper-large-v2 compatible (expected Whisper d_model=1280)")
    declared_identity = model_config.get("_name_or_path")
    declared_identity = declared_identity.lower() if isinstance(declared_identity, str) else ""
    if "whisper-" in declared_identity and "whisper-large-v2" not in declared_identity:
        raise ValueError(
            f"local checkpoint explicitly identifies a different Whisper variant: {declared_identity!r}"
        )
    if "whisper-large-v2" not in declared_identity:
        raise ValueError(
            "local checkpoint config must identify openai/whisper-large-v2; refusing an unverified variant"
        )
    if preprocessor_config.get("feature_size") != MEL_BINS:
        raise ValueError("local Whisper feature extractor must use 80 Mel bins")
    if preprocessor_config.get("sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
        raise ValueError("local Whisper feature extractor must use a 16000 Hz sample rate")

    vocab_path = directory / "vocab.json"
    merges_path = directory / "merges.txt"
    for required in (vocab_path, merges_path):
        if not required.is_file() or not os.access(required, os.R_OK):
            raise FileNotFoundError(f"Whisper-large-v2 checkpoint is missing {required.name}: {directory}")

    files = [config_path, preprocessor_path, tokenizer_config_path, vocab_path, merges_path]
    for optional_tokenizer_asset in ("tokenizer.json", "special_tokens_map.json", "added_tokens.json"):
        optional_path = directory / optional_tokenizer_asset
        if optional_path.is_file():
            files.append(optional_path)

    weight_files: list[Path] = []
    for standalone in ("model.safetensors", "pytorch_model.bin"):
        candidate = directory / standalone
        if candidate.is_file():
            weight_files.append(candidate)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = directory / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Whisper checkpoint weight index: {index_path}") from exc
        weight_files.append(index_path)
        for shard_name in shard_names:
            shard_path = directory / shard_name
            if not shard_path.is_file() or not os.access(shard_path, os.R_OK):
                raise FileNotFoundError(f"Whisper checkpoint is missing weight shard: {shard_path}")
            weight_files.append(shard_path)
    if not weight_files:
        raise FileNotFoundError(f"Whisper-large-v2 checkpoint has no model weights: {directory}")
    files.extend(weight_files)
    return sorted(set(files))


def _checkpoint_identity(
    model_name_or_path: str,
    *,
    cache_dir: Any = None,
    revision: Optional[str] = None,
) -> dict[str, Any]:
    reference = Path(model_name_or_path).expanduser()
    if reference.exists():
        directory = reference.resolve(strict=True)
        if not directory.is_dir():
            raise NotADirectoryError(f"Whisper model path must be a directory: {directory}")
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to identify the cached Whisper checkpoint") from exc
        cache_options: dict[str, Any] = {
            "cache_dir": None if cache_dir is None else str(cache_dir),
        }
        if revision is not None:
            cache_options["revision"] = revision
        config_file = try_to_load_from_cache(MODEL_ID, "config.json", **cache_options)
        if not isinstance(config_file, str) or not Path(config_file).is_file():
            raise RuntimeError(
                "could not resolve the loaded Whisper-large-v2 snapshot locally to fingerprint its checkpoint"
            )
        # Keep the snapshot symlink path: resolving the file first moves into
        # Hugging Face's blobs/ directory, which has no snapshot filenames.
        directory = Path(config_file).expanduser().absolute().parent

    files = _local_model_files(directory.resolve(strict=True))
    return {
        "model_id": MODEL_ID,
        "files": {path.name: _sha256_file(path) for path in files},
    }


def _model_reference(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise TypeError("whisper_top2.model_name_or_path must be a model ID or local model directory")
    reference = str(Path(value).expanduser()) if Path(value).expanduser().exists() else str(value)
    if reference == MODEL_ID:
        return reference
    path = Path(reference).expanduser()
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Whisper model path must be a directory: {path}")
        _local_model_files(path.resolve(strict=True))
        return str(path.resolve(strict=True))
    raise ValueError(
        f"{_NAME} only supports {MODEL_ID} or a complete local Whisper-large-v2 directory; got {reference!r}"
    )


def _validate_batch(batch: Any) -> tuple[torch.Tensor, list[str]]:
    if not isinstance(batch, Mapping):
        raise TypeError(f"{_NAME} batch must be a mapping")
    if "expert3_wavs" not in batch:
        raise KeyError(f"{_NAME} batch is missing required field 'expert3_wavs'")
    if "captions_raw" not in batch:
        raise KeyError(f"{_NAME} batch is missing required field 'captions_raw'")
    waveforms = batch["expert3_wavs"]
    if not isinstance(waveforms, torch.Tensor):
        raise TypeError(f"{_NAME} expert3_wavs must be a torch.Tensor")
    if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1, WAVEFORM_SAMPLES):
        raise ValueError(
            f"{_NAME} expert3_wavs must have shape [B,1,{WAVEFORM_SAMPLES}]; got {tuple(waveforms.shape)}"
        )
    if waveforms.shape[0] <= 0:
        raise ValueError(f"{_NAME} batch must contain at least one sample")
    if waveforms.dtype != torch.float32:
        raise TypeError(f"{_NAME} expert3_wavs must use torch.float32; got {waveforms.dtype}")
    if not torch.isfinite(waveforms).all().item():
        raise ValueError(f"{_NAME} expert3_wavs contains non-finite values")

    sample_rate = batch.get("sample_rate")
    if sample_rate is not None:
        if isinstance(sample_rate, torch.Tensor):
            integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
            valid_rate = (
                sample_rate.dtype in integer_dtypes
                and sample_rate.numel() in {1, waveforms.shape[0]}
                and bool((sample_rate == SAMPLE_RATE).all().item())
            )
        elif type(sample_rate) is int:
            valid_rate = sample_rate == SAMPLE_RATE
        elif isinstance(sample_rate, Sequence) and not isinstance(sample_rate, (str, bytes)):
            valid_rate = len(sample_rate) == waveforms.shape[0] and all(
                type(rate) is int and rate == SAMPLE_RATE for rate in sample_rate
            )
        else:
            valid_rate = False
        if not valid_rate:
            raise ValueError(f"{_NAME} batch sample_rate must be {SAMPLE_RATE} Hz")

    captions = batch["captions_raw"]
    if isinstance(captions, (str, bytes, torch.Tensor)) or not isinstance(captions, Sequence):
        raise TypeError(f"{_NAME} captions_raw must be a sequence of strings")
    if len(captions) != waveforms.shape[0]:
        raise ValueError(
            f"{_NAME} needs one caption per audio item; got {len(captions)} for batch size {waveforms.shape[0]}"
        )
    if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
        raise ValueError(f"{_NAME} captions_raw must contain non-empty strings")
    return waveforms, list(captions)


class WhisperTop2FeatureExtractor(nn.Module):
    """Return pooled Whisper encoder audio and caption-conditioned decoder vectors.

    Stable E6 import path::

        from model.feature_extractors import WhisperTop2FeatureExtractor

    The input uses MOESCORE's unified data batch keys. The class deliberately
    does not return a score or concatenate its two 1280-dimensional vectors.
    """

    def __init__(
        self,
        feature_extractor: Any,
        tokenizer: Any,
        model: nn.Module,
        *,
        device: Any = "cpu",
        sample_rate: int = SAMPLE_RATE,
        model_name_or_path: str = MODEL_ID,
        checkpoint_identity: Optional[Mapping[str, Any]] = None,
        preprocessing_identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
            raise ValueError(f"{_NAME} requires the MOESCORE 16000 Hz input contract; got {sample_rate!r}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"{_NAME} model must be a torch.nn.Module")
        if not callable(feature_extractor) or not callable(tokenizer):
            raise TypeError(f"{_NAME} feature_extractor and tokenizer must be callable")
        self.device = target_device(device, adapter_name=_NAME)
        self.sample_rate = sample_rate
        self.model_name_or_path = model_name_or_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.model: Optional[nn.Module] = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self._closed = False
        self.feature_extraction_version = FEATURE_EXTRACTION_VERSION
        self.checkpoint_identity = _json_identity(
            checkpoint_identity or {"model_id": model_name_or_path, "identity": "injected-model"},
            field="checkpoint_identity",
        )
        self.preprocessing_identity = _json_identity(
            preprocessing_identity or _PREPROCESSING_IDENTITY,
            field="preprocessing_identity",
        )
        self.eval()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, device: Any = None) -> "WhisperTop2FeatureExtractor":
        section, runtime = adapter_config(config, "whisper_top2")
        unknown = set(section) - {"model_name_or_path", "sample_rate"}
        if unknown:
            raise ValueError(f"whisper_top2 config contains unknown fields: {', '.join(sorted(unknown))}")
        model_name_or_path = _model_reference(section.get("model_name_or_path", MODEL_ID))
        sample_rate = section.get("sample_rate", SAMPLE_RATE)
        if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
            raise ValueError(f"whisper_top2.sample_rate must be {SAMPLE_RATE}")
        offline = runtime.get("offline", True)
        if type(offline) is not bool:
            raise TypeError("runtime.offline must be a boolean")
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_NAME,
        )
        cache_dir = runtime.get("cache_dir")
        if cache_dir is not None and not isinstance(cache_dir, (str, os.PathLike)):
            raise TypeError("runtime.cache_dir must be a filesystem path")
        loading_options: dict[str, Any] = {"local_files_only": offline}
        if cache_dir is not None:
            loading_options["cache_dir"] = str(cache_dir)
        try:
            from transformers import AutoFeatureExtractor, WhisperModel, WhisperTokenizer
        except ImportError as exc:
            raise RuntimeError(f"{_NAME} requires the transformers package") from exc

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path, **loading_options)
            tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, **loading_options)
            model = WhisperModel.from_pretrained(model_name_or_path, **loading_options)
        except Exception as exc:
            if offline:
                raise RuntimeError(
                    f"{_NAME} could not load the exact local {MODEL_ID} resources in offline mode: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise RuntimeError(
                f"{_NAME} failed to load {model_name_or_path}: {type(exc).__name__}: {exc}"
            ) from exc

        model_config = getattr(model, "config", None)
        model_width = getattr(model_config, "d_model", HIDDEN_SIZE)
        model_type = getattr(model_config, "model_type", "whisper")
        if model_width != HIDDEN_SIZE or model_type != "whisper":
            raise ValueError(
                f"loaded checkpoint is not Whisper-large-v2 compatible: model_type={model_type!r}, d_model={model_width!r}"
            )
        checkpoint_identity = _checkpoint_identity(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=getattr(model_config, "_commit_hash", None),
        )
        preprocessing_identity = {
            **_PREPROCESSING_IDENTITY,
            "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        }
        return cls(
            feature_extractor,
            tokenizer,
            model,
            device=selected_device,
            sample_rate=sample_rate,
            model_name_or_path=model_name_or_path,
            checkpoint_identity=checkpoint_identity,
            preprocessing_identity=preprocessing_identity,
        ).eval()

    def train(self, mode: bool = True) -> "WhisperTop2FeatureExtractor":
        # The wrapper and backbone are permanently inference-only.
        super().train(False)
        if self.model is not None:
            self.model.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed or self.model is None:
            raise RuntimeError(f"{_NAME} is closed and no longer owns a Whisper model")
        waveforms, captions = _validate_batch(batch)
        audio_batch = [waveform.cpu().numpy().flatten() for waveform in waveforms]

        with torch.inference_mode():
            inputs = self.feature_extractor(
                audio_batch,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            input_features = getattr(inputs, "input_features", None)
            if not isinstance(input_features, torch.Tensor):
                raise TypeError(f"{_NAME} feature extractor must return tensor input_features")
            expected_features = (waveforms.shape[0], MEL_BINS, MEL_FRAMES)
            if tuple(input_features.shape) != expected_features:
                raise ValueError(
                    f"{_NAME} input_features must have shape {expected_features}; got {tuple(input_features.shape)}"
                )
            if not torch.isfinite(input_features).all().item():
                raise ValueError(f"{_NAME} input_features contains non-finite values")

            tokenized = self.tokenizer(captions, return_tensors="pt", padding=True)
            decoder_input_ids = getattr(tokenized, "input_ids", None)
            if not isinstance(decoder_input_ids, torch.Tensor):
                raise TypeError(f"{_NAME} tokenizer must return tensor input_ids")
            if (
                decoder_input_ids.ndim != 2
                or decoder_input_ids.shape[0] != waveforms.shape[0]
                or decoder_input_ids.shape[1] <= 0
            ):
                raise ValueError(f"{_NAME} decoder_input_ids must have shape [B,S] with matching non-empty batch")

            outputs = self.model(
                input_features.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
            )
            encoder_hidden = getattr(outputs, "encoder_last_hidden_state", None)
            decoder_hidden = getattr(outputs, "last_hidden_state", None)
            audio_features = self._pool_hidden(
                encoder_hidden, label="encoder", batch_size=waveforms.shape[0]
            )
            text_features = self._pool_hidden(
                decoder_hidden, label="decoder", batch_size=waveforms.shape[0]
            )
            audio_features = audio_features.to(device=self.device, dtype=torch.float32)
            text_features = text_features.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(audio_features).all().item():
                raise ValueError(f"{_NAME} audio features contain non-finite values")
            if not torch.isfinite(text_features).all().item():
                raise ValueError(f"{_NAME} text features contain non-finite values")
        return audio_features, text_features

    @staticmethod
    def _pool_hidden(hidden: Any, *, label: str, batch_size: int) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError(f"{_NAME} {label} hidden state must have shape [B,S,{HIDDEN_SIZE}]")
        if hidden.shape[0] != batch_size or hidden.shape[1] <= 0 or hidden.shape[2] != HIDDEN_SIZE:
            raise ValueError(
                f"{_NAME} {label} hidden state must have shape [B,S,{HIDDEN_SIZE}] with B={batch_size}; "
                f"got {tuple(hidden.shape)}"
            )
        if not torch.isfinite(hidden).all().item():
            raise ValueError(f"{_NAME} {label} hidden state contains non-finite values")
        return hidden.mean(dim=1)

    def close(self) -> None:
        """Release the frozen backbone, tokenizer, and CUDA allocator cache."""
        if self._closed:
            return
        if self.model is not None:
            self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self._closed = True
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "FEATURE_EXTRACTION_VERSION",
    "HIDDEN_SIZE",
    "MODEL_ID",
    "SAMPLE_RATE",
    "WAVEFORM_SAMPLES",
    "WhisperTop2FeatureExtractor",
]
