"""Pretrained Microsoft MS-CLAP 2023 adapter with verified file inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter
from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.top2_common import (
    adapter_config,
    freeze_module,
    huggingface_environment,
    model_device,
    pack_embeddings,
    resolved_resource,
    target_device,
    validate_audio_batch,
)


_AUDIO_SAMPLES = 320_000
_SPLITS = frozenset({"train", "validation", "test"})
_ADAPTER_NAME = "Top2MSCLAPAdapter"
_MSCLAP_2023_RATE_HZ = 44_100
_MSCLAP_2023_DURATION_SECONDS = 7


def _center_crop_or_repeat(audio_time_series: Any, sample_rate: int, audio_duration: float) -> torch.Tensor:
    """Apply MS-CLAP's original repeat/truncate policy and deterministic center crop.

    WAV decoding, optional resampling, flattening and channel handling remain
    owned by the installed MS-CLAP ``read_audio`` implementation.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{_ADAPTER_NAME} received an invalid decoded sample rate: {sample_rate!r}")
    if isinstance(audio_duration, bool) or not isinstance(audio_duration, (int, float)):
        raise ValueError(f"{_ADAPTER_NAME} received an invalid configured duration: {audio_duration!r}")
    if not math.isfinite(float(audio_duration)) or audio_duration <= 0:
        raise ValueError(f"{_ADAPTER_NAME} audio duration must be positive and finite")
    audio = torch.as_tensor(audio_time_series).reshape(-1)
    if audio.numel() == 0:
        raise ValueError(f"{_ADAPTER_NAME} cannot preprocess an empty or unreadable waveform")
    target_samples = int(float(audio_duration) * sample_rate)
    if target_samples <= 0:
        raise ValueError(f"{_ADAPTER_NAME} configured duration produces an empty target waveform")
    if target_samples >= audio.numel():
        repeat_factor = math.ceil(target_samples / audio.numel())
        audio = audio.repeat(repeat_factor)[:target_samples]
    else:
        start = (audio.numel() - target_samples) // 2
        audio = audio[start : start + target_samples]
    # This matches upstream load_audio_into_tensor's torch.FloatTensor output.
    return audio.to(device="cpu", dtype=torch.float32).contiguous()


def _deterministic_load_audio_into_tensor(
    self: Any, audio_path: str, audio_duration: float, resample: bool = False
) -> torch.Tensor:
    """Override only the random segment-selection stage of MS-CLAP loading."""
    try:
        waveform, sample_rate = self.read_audio(audio_path, resample=resample)
    except Exception as exc:
        raise RuntimeError(f"{_ADAPTER_NAME} failed to read WAV {audio_path!r}: {exc}") from exc
    return _center_crop_or_repeat(waveform, sample_rate, audio_duration)


def _deterministic_clap_class(clap_class: type) -> type:
    """Create an adapter-owned subclass; never mutate the installed package."""
    if not isinstance(clap_class, type):
        raise TypeError(f"{_ADAPTER_NAME} expected the installed CLAP class, got {clap_class!r}")

    class DeterministicCenterCropCLAP(clap_class):
        load_audio_into_tensor = _deterministic_load_audio_into_tensor

    DeterministicCenterCropCLAP.__name__ = "DeterministicCenterCropCLAP"
    DeterministicCenterCropCLAP.__qualname__ = "DeterministicCenterCropCLAP"
    return DeterministicCenterCropCLAP


def _dimension_from_api(api: Any, explicit: Optional[int]) -> int:
    args = getattr(api, "args", None)
    value = getattr(args, "d_proj", None)
    if value is None and isinstance(args, Mapping):
        value = args.get("d_proj")
    if value is None:
        value = explicit
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{_ADAPTER_NAME} cannot determine the projected embedding width; "
            "the MS-CLAP 2023 config must expose d_proj or config must set embedding_dim"
        )
    if explicit is not None and explicit != value:
        raise ValueError(
            f"{_ADAPTER_NAME} embedding_dim conflicts with the model d_proj: "
            f"{explicit} != {value}"
        )
    return value


def _audio_roots(value: Any) -> dict[str, Path]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{_ADAPTER_NAME} audio_roots must map split names to directories")
    roots: dict[str, Path] = {}
    for split, path in value.items():
        if split not in _SPLITS:
            raise ValueError(f"{_ADAPTER_NAME} audio_roots has unknown split {split!r}")
        roots[split] = resolved_resource(
            path,
            field=f"audio_roots.{split}",
            adapter_name=_ADAPTER_NAME,
            directory=True,
        )
    return roots


class Top2MSCLAPAdapter(ExpertAdapter):
    """Expose Microsoft CLAP 2023 audio/text embeddings and their cosine.

    MS-CLAP's source API reads audio files. A batch may supply verified
    ``wav_paths`` directly, or the adapter resolves ``wave_file_names`` under
    the configured ``audio_roots[split]`` directory. It never guesses a path.
    """

    def __init__(
        self,
        clap_api: Any,
        *,
        embedding_dim: Optional[int] = None,
        audio_roots: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not callable(getattr(clap_api, "get_audio_embeddings", None)) or not callable(
            getattr(clap_api, "get_text_embeddings", None)
        ):
            raise TypeError(f"{_ADAPTER_NAME} requires the original MS-CLAP embedding API")
        clap_module = getattr(clap_api, "clap", None)
        if not isinstance(clap_module, nn.Module):
            raise TypeError(f"{_ADAPTER_NAME} MS-CLAP API must expose its nn.Module as .clap")
        resolved_dim = _dimension_from_api(clap_api, embedding_dim)
        super().__init__(feature_dim=5 * resolved_dim)
        self.embedding_dim = resolved_dim
        self.clap = clap_module
        self.clap_api = clap_api
        self.audio_roots = _audio_roots(audio_roots)
        self._freeze_model()

    @classmethod
    def from_config(cls, config: Mapping, *, device: Any = None) -> "Top2MSCLAPAdapter":
        section, runtime = adapter_config(config, "top2_msclap")
        if "checkpoint" not in section:
            raise ValueError("top2_msclap config is missing checkpoint")
        checkpoint = resolved_resource(
            section["checkpoint"], field="checkpoint", adapter_name=_ADAPTER_NAME
        )
        selected_device = target_device(
            device if device is not None else runtime.get("device", "auto"),
            adapter_name=_ADAPTER_NAME,
        )
        offline = runtime.get("offline", True)
        cache_dir = runtime.get("cache_dir")
        selected_cuda = torch.cuda.device(selected_device) if selected_device.type == "cuda" else nullcontext()
        try:
            with huggingface_environment(offline=offline, cache_dir=cache_dir), selected_cuda:
                from msclap import CLAP

                deterministic_clap = _deterministic_clap_class(CLAP)
                clap_api = deterministic_clap(
                    model_fp=str(checkpoint),
                    version="2023",
                    use_cuda=selected_device.type == "cuda",
                )
                args = getattr(clap_api, "args", None)
                configured_rate = getattr(args, "sampling_rate", None)
                configured_duration = getattr(args, "duration", None)
                if configured_rate != _MSCLAP_2023_RATE_HZ or configured_duration != _MSCLAP_2023_DURATION_SECONDS:
                    raise ValueError(
                        "MS-CLAP 2023 preprocessing configuration differs from the verified "
                        f"44.1 kHz/7 s policy: rate={configured_rate!r}, duration={configured_duration!r}"
                    )
                clap_api.moescore_audio_policy = {
                    "audio_crop_policy": "center",
                    "audio_duration_seconds": configured_duration,
                    "sample_rate_hz": configured_rate,
                    "short_audio_policy": "repeat_then_truncate",
                }
            clap_api.clap.to(selected_device)
        except ImportError as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} requires the msclap package; install the Microsoft MS-CLAP dependency"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{_ADAPTER_NAME} failed to restore the local MS-CLAP 2023 checkpoint "
                f"{checkpoint}; verify tokenizer resources and the checkpoint format. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        adapter = cls(
            clap_api,
            embedding_dim=section.get("embedding_dim"),
            audio_roots=section.get("audio_roots"),
        )
        return adapter.eval()

    def _freeze_model(self) -> None:
        freeze_module(self.clap)

    def train(self, mode: bool = True) -> "Top2MSCLAPAdapter":
        super().train(mode)
        self._freeze_model()
        return self

    def _resolve_wav_paths(self, batch: Mapping[str, Any], batch_size: int) -> list[str]:
        supplied = batch.get("wav_paths")
        if supplied is not None:
            if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
                raise TypeError(f"{_ADAPTER_NAME} wav_paths must be a sequence of paths")
            if len(supplied) != batch_size:
                raise ValueError(
                    f"{_ADAPTER_NAME} needs {batch_size} wav_paths; got {len(supplied)}"
                )
            paths = []
            for index, value in enumerate(supplied):
                if not isinstance(value, (str, os.PathLike)):
                    raise TypeError(f"{_ADAPTER_NAME} wav_paths[{index}] is not a filesystem path")
                explicit_path = Path(value).expanduser().resolve(strict=False)
                paths.append(
                    str(
                        resolved_resource(
                            explicit_path,
                            field=f"wav_paths[{index}]",
                            adapter_name=_ADAPTER_NAME,
                        )
                    )
                )
            return paths

        split = batch.get("split")
        if split not in _SPLITS:
            raise ValueError(f"{_ADAPTER_NAME} batch split must be one of {sorted(_SPLITS)}")
        if split not in self.audio_roots:
            raise ValueError(
                f"{_ADAPTER_NAME} needs wav_paths in the batch or audio_roots[{split!r}] in config"
            )
        names = batch.get("wave_file_names")
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise TypeError(f"{_ADAPTER_NAME} wave_file_names must be a sequence")
        if len(names) != batch_size:
            raise ValueError(
                f"{_ADAPTER_NAME} needs {batch_size} wave_file_names; got {len(names)}"
            )

        root = self.audio_roots[split]
        paths = []
        for index, name in enumerate(names):
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or Path(name).name != name
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(
                    f"{_ADAPTER_NAME} wave_file_names[{index}] must be a basename; got {name!r}"
                )
            path = (root / name).resolve(strict=False)
            if not path.is_relative_to(root):
                raise ValueError(f"{_ADAPTER_NAME} resolved audio path escapes configured root: {path}")
            paths.append(
                str(
                    resolved_resource(
                        path,
                        field=f"audio_roots.{split}/{name}",
                        adapter_name=_ADAPTER_NAME,
                    )
                )
            )
        return paths

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = validate_audio_batch(
            batch,
            audio_key="expert2_wavs",
            sample_count=_AUDIO_SAMPLES,
            adapter_name=_ADAPTER_NAME,
            model=self.clap,
            require_model_device=False,
        )
        paths = self._resolve_wav_paths(batch, audio.shape[0])
        selected_device = model_device(self.clap) or audio.device
        cuda_context = (
            torch.cuda.device(selected_device) if selected_device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), cuda_context:
            audio_embedding = self.clap_api.get_audio_embeddings(paths)
            text_embedding = self.clap_api.get_text_embeddings(captions)
        score, original_features = pack_embeddings(
            audio_embedding,
            text_embedding,
            batch_size=audio.shape[0],
            embedding_dim=self.embedding_dim,
            output_device=selected_device,
            adapter_name=_ADAPTER_NAME,
        )
        audio_embedding, text_embedding = original_features.chunk(2, dim=-1)
        return score, build_paired_cross_features(audio_embedding, text_embedding)


__all__ = ["Top2MSCLAPAdapter"]
