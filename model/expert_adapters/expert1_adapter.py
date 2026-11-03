"""Adapter for the fine-tuned LAION-CLAP Expert 1 implementation.

The LAION package and the standalone training module are imported lazily by
``from_finetuned``. Importing this module therefore does not initialize an
expert, load a checkpoint, or contact a model hub.
"""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

import torch
from torch import nn

from model.expert_adapters.base import ExpertAdapter


PathValue = Union[str, os.PathLike]
_AUDIO_SAMPLES = 48_000 * 10


def _resolved_resource_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expert 1 {field} must be a resolved filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Expert 1 {field} must already be resolved to an absolute path; got {path}"
        )
    path = path.resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(
            f"Expert 1 {field} expected a checkpoint file at {path}; file does not exist"
        )
    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"Expert 1 {field} expected a readable checkpoint file at {path}"
        )
    return path


def _target_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Expert 1 requested device {target}, but CUDA is unavailable")
    return target


@contextmanager
def _huggingface_offline(offline: bool):
    """Temporarily enforce the installed Hugging Face client's offline mode."""
    if type(offline) is not bool:
        raise TypeError(f"Expert 1 offline must be a boolean; got {offline!r}")
    if not offline:
        yield
        return

    environment_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    for key in environment_keys:
        os.environ[key] = "1"

    hub_constants = sys.modules.get("huggingface_hub.constants")
    previous_hub_offline = getattr(hub_constants, "HF_HUB_OFFLINE", None)
    if hub_constants is not None:
        hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = previous_hub_offline
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Expert1Adapter(ExpertAdapter):
    """Expose the original Expert 1 score and its audio/text CLAP features.

    Args:
        model: An already constructed ``expert1_finetune.Train_Expert1.Expert1``
            (or a compatible lightweight ``nn.Module`` for isolated tests).
        embedding_dim: Optional trusted embedding width. The production
            Expert 1 model exposes this as the LAION CLAP core's
            ``joint_embed_shape``; an explicit value is useful for injected
            models and must agree with that metadata when it is available.

    The adapter owns ``model`` as a registered child module, adds no trainable
    layers, freezes its parameters, and keeps the child in evaluation mode even
    when a parent calls ``train()``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        embedding_dim: Optional[int] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expert 1 model must be an nn.Module; got {type(model).__name__}")

        model_embedding_dim = self._model_embedding_dim(model)
        if embedding_dim is None:
            embedding_dim = model_embedding_dim
        elif type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(
                f"Expert 1 embedding_dim must be a positive integer; got {embedding_dim!r}"
            )
        elif model_embedding_dim is not None and embedding_dim != model_embedding_dim:
            raise ValueError(
                "Expert 1 explicit embedding_dim does not match LAION model metadata: "
                f"{embedding_dim} != {model_embedding_dim}"
            )

        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError(
                "Expert 1 embedding_dim is unavailable; construct the original model "
                "or provide a positive dimension from trusted model metadata"
            )

        super().__init__(feature_dim=2 * embedding_dim)
        self.embedding_dim = embedding_dim
        self.model = model
        self._freeze_model()

    @staticmethod
    def _model_embedding_dim(model: nn.Module) -> Optional[int]:
        """Read the LAION CLAP projection width exposed by the constructed model."""
        encoder = getattr(model, "model", None)
        clap_module = getattr(encoder, "model", None)
        clap_core = getattr(clap_module, "model", None)
        value = getattr(clap_core, "joint_embed_shape", None)
        if type(value) is int and value > 0:
            return value
        return None

    @staticmethod
    def _restore_finetuned_state_dict(model: nn.Module, checkpoint: Path) -> None:
        """Strictly load the raw full Expert 1 state dict using safe loading."""
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Expert 1 could not safely read fine-tuned checkpoint {checkpoint}: {exc}"
            ) from exc

        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                f"Expert 1 fine-tuned checkpoint {checkpoint} must be a non-empty raw tensor state_dict"
            )
        if any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError(
                f"Expert 1 fine-tuned checkpoint {checkpoint} must be a raw tensor state_dict, "
                "not a structured training bundle"
            )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Expert 1 strict fine-tuned state restoration failed for {checkpoint}: {exc}"
            ) from exc

    @classmethod
    def from_finetuned(
        cls,
        pretrained_checkpoint: PathValue,
        finetuned_checkpoint: PathValue,
        *,
        device: Union[str, torch.device] = "cpu",
        embedding_dim: Optional[int] = None,
        offline: bool = True,
    ) -> "Expert1Adapter":
        """Construct the original Expert 1, restore its raw full fine-tuned state.

        Both paths must already be resolved absolute paths. The original
        ``Expert1`` constructor loads the pretrained LAION checkpoint once,
        with ``enable_fusion=False`` as defined by its source. The fine-tuned
        state is then loaded strictly. There is no pretrained-only fallback.
        The model is initialized and restored on CPU before moving to ``device``.
        With ``offline=True``, Hub access is disabled during source-model
        construction; required tokenizer/model assets must already be cached.
        """
        if embedding_dim is not None and (
            type(embedding_dim) is not int or embedding_dim <= 0
        ):
            raise ValueError(
                f"Expert 1 embedding_dim must be a positive integer; got {embedding_dim!r}"
            )
        if type(offline) is not bool:
            raise TypeError(f"Expert 1 offline must be a boolean; got {offline!r}")
        pretrained_path = _resolved_resource_path(
            pretrained_checkpoint, field="pretrained_checkpoint"
        )
        finetuned_path = _resolved_resource_path(
            finetuned_checkpoint, field="finetuned_checkpoint"
        )

        # Import only when restoration is explicitly requested. This module
        # import does not instantiate models; Expert1's constructor does.
        with _huggingface_offline(offline):
            try:
                from expert1_finetune.Train_Expert1 import Expert1
            except Exception as exc:
                raise RuntimeError(
                    "Expert 1 could not import expert1_finetune.Train_Expert1.Expert1"
                ) from exc

            model = Expert1(pretrained_path, torch.device("cpu"))
        cls._restore_finetuned_state_dict(model, finetuned_path)
        resolved_embedding_dim = (
            cls._model_embedding_dim(model) if embedding_dim is None else embedding_dim
        )
        if resolved_embedding_dim is None:
            raise ValueError(
                "Expert 1 could not determine the CLAP embedding width from the "
                "constructed model; provide a trusted embedding_dim"
            )

        adapter = cls(model, embedding_dim=resolved_embedding_dim)
        return adapter.to(_target_device(device)).eval()

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Expert1Adapter":
        """Build Expert 1 from a validated, path-resolved LOCAL-03 config.

        Accepts either the full resolved MoE mapping or its ``expert1``
        section. It validates only Expert 1 resources; unavailable Expert 2–4
        files do not block this scoped factory. Relative paths are rejected so
        path precedence remains owned by ``resolve_moe_paths``.
        """
        if not isinstance(config, Mapping):
            raise TypeError("Expert 1 config must be a mapping")
        if "experts" in config:
            try:
                expert_config = config["experts"]["expert1"]
            except (KeyError, TypeError) as exc:
                raise ValueError("Expert 1 config is missing experts.expert1") from exc
            runtime_config = config.get("runtime", {})
        else:
            expert_config = config
            runtime_config = {}
        if not isinstance(expert_config, Mapping):
            raise TypeError("Expert 1 config section must be a mapping")
        if expert_config.get("enable_fusion") is not False:
            raise ValueError("Expert 1 experts.expert1.enable_fusion must be false")
        try:
            pretrained_checkpoint = expert_config["pretrained_checkpoint"]
            finetuned_checkpoint = expert_config["finetuned_checkpoint"]
        except KeyError as exc:
            raise ValueError(f"Expert 1 config is missing {exc.args[0]}") from exc
        configured_device = device
        if configured_device is None:
            configured_device = runtime_config.get("device", "cpu")
        offline = runtime_config.get("offline", True)
        if type(offline) is not bool:
            raise TypeError(f"Expert 1 runtime.offline must be a boolean; got {offline!r}")
        return cls.from_finetuned(
            pretrained_checkpoint,
            finetuned_checkpoint,
            device=configured_device,
            offline=offline,
        )

    def _freeze_model(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "Expert1Adapter":
        """Set adapter mode while forcing every wrapped expert module to eval."""
        super().train(mode)
        self.model.eval()
        return self

    def _validate_batch(self, batch: Any) -> tuple[torch.Tensor, Sequence[str]]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expert 1 batch must be a mapping; got {type(batch).__name__}")
        if "expert1_wavs" not in batch:
            raise KeyError("Expert 1 batch is missing required field 'expert1_wavs'")
        if "captions_raw" not in batch:
            raise KeyError("Expert 1 batch is missing required field 'captions_raw'")

        audio = batch["expert1_wavs"]
        if not isinstance(audio, torch.Tensor):
            raise TypeError(
                f"Expert 1 expert1_wavs must be a torch.Tensor; got {type(audio).__name__}"
            )
        if audio.ndim != 3:
            raise ValueError(
                f"Expert 1 expert1_wavs must have rank-3 shape [B, 1, 480000]; "
                f"got {tuple(audio.shape)}"
            )
        if audio.shape[0] <= 0:
            raise ValueError("Expert 1 expert1_wavs batch must be non-empty")
        if audio.shape[1] != 1:
            raise ValueError(
                f"Expert 1 expert1_wavs must have one channel [B, 1, 480000]; "
                f"got {tuple(audio.shape)}"
            )
        if audio.shape[2] != _AUDIO_SAMPLES:
            raise ValueError(
                "Expert 1 expert1_wavs must contain 480000 samples per item "
                f"(48 kHz, 10 seconds); got {audio.shape[2]}"
            )
        if audio.dtype != torch.float32:
            raise TypeError(
                f"Expert 1 expert1_wavs must use torch.float32; got {audio.dtype}"
            )
        if not torch.isfinite(audio).all().item():
            raise ValueError("Expert 1 expert1_wavs must contain only finite values")

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = None
        if model_device is not None and audio.device != model_device:
            raise ValueError(
                "Expert 1 expert1_wavs must be on model device "
                f"{model_device}; got {audio.device}"
            )

        captions = batch["captions_raw"]
        if isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence):
            raise TypeError(
                "Expert 1 captions_raw must be a Python sequence of strings; "
                f"got {type(captions).__name__}"
            )
        if len(captions) != audio.shape[0]:
            raise ValueError(
                "Expert 1 captions_raw must have the same batch size as expert1_wavs; "
                f"got {len(captions)} captions for batch {audio.shape[0]}"
            )
        for index, caption in enumerate(captions):
            if not isinstance(caption, str):
                raise TypeError(
                    f"Expert 1 captions_raw[{index}] must be a string; "
                    f"got {type(caption).__name__}"
                )
            if not caption.strip():
                raise ValueError(f"Expert 1 captions_raw[{index}] is an empty caption")
        return audio, captions

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        audio, captions = self._validate_batch(batch)

        # The source Expert1.forward calls its CLAPAudioTextEncoder once and
        # computes this exact dot product. Calling the encoder directly lets
        # the adapter reuse both outputs without a second large encoder pass.
        audio_embedding, text_embedding = self.model.model(audio, captions)
        self._validate_embedding(audio_embedding, "audio_embedding", audio.shape[0])
        self._validate_embedding(text_embedding, "text_embedding", audio.shape[0])
        if audio_embedding.device != text_embedding.device:
            raise ValueError("Expert 1 audio/text embeddings must be on the same device")
        if audio_embedding.dtype != text_embedding.dtype:
            raise TypeError("Expert 1 audio/text embeddings must use the same dtype")

        score = (audio_embedding * text_embedding).sum(dim=-1)
        features = torch.cat([audio_embedding, text_embedding], dim=-1)
        return score.unsqueeze(-1), features

    def _validate_embedding(self, embedding: Any, name: str, batch_size: int) -> None:
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"Expert 1 {name} must be a torch.Tensor; got {type(embedding).__name__}"
            )
        if embedding.ndim != 2:
            raise ValueError(
                f"Expert 1 {name} must have rank-2 shape [B, D]; "
                f"got {tuple(embedding.shape)}"
            )
        if embedding.shape[0] != batch_size:
            raise ValueError(
                f"Expert 1 {name} batch dimension mismatch: "
                f"expected {batch_size}, got {embedding.shape[0]}"
            )
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expert 1 {name} embedding width mismatch: "
                f"expected {self.embedding_dim}, got {embedding.shape[1]}"
            )
        if not torch.is_floating_point(embedding):
            raise TypeError(f"Expert 1 {name} must be floating point; got {embedding.dtype}")
