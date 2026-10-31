"""M2D-CLAP Expert 3 adapted from the released MOESCORE implementation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


M2D_2025_MODEL_DIR = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"


def _load_portable_m2d(weight_file: str, flat_features: bool):
    # Keep third-party imports lazy so CPU unit tests do not need timm/nnAudio.
    from .m2d.vendor.portable_m2d import PortableM2D

    return PortableM2D(weight_file=weight_file, flat_features=flat_features)


def prepare_m2d_checkpoint(checkpoint_path, layout_root):
    """Give PortableM2D the model-directory name it parses, without copying weights."""
    source = Path(checkpoint_path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"M2D-CLAP checkpoint not found: {source}")
    if source.parent.name == M2D_2025_MODEL_DIR:
        return source

    expected_dir = Path(layout_root).expanduser() / M2D_2025_MODEL_DIR
    expected_dir.mkdir(parents=True, exist_ok=True)
    target = expected_dir / "checkpoint-30.pth"
    resolved_source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != resolved_source:
            raise FileExistsError(
                f"Refusing to replace existing M2D checkpoint layout {target}; "
                f"it points to {target.resolve()}, requested {resolved_source}"
            )
        return target
    try:
        target.symlink_to(resolved_source)
    except OSError as exc:
        raise OSError(
            f"Could not create a checkpoint symlink at {target}; "
            "choose a writable cache/layout directory"
        ) from exc
    return target


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_output_dim(module):
    if hasattr(module, "out_features"):
        return int(module.out_features)
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if linears:
        return int(linears[-1].out_features)
    norm = getattr(module, "norm", None)
    if norm is not None and hasattr(norm, "normalized_shape"):
        shape = norm.normalized_shape
        return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    raise ValueError(f"Cannot infer projection output dimension for {type(module).__name__}")


def _text_encoder_output_dim(text_encoder):
    """Infer text embedding size when the M2D checkpoint has no text_proj."""
    for encoder in (getattr(text_encoder, "text_encoder", None), getattr(text_encoder, "model", None)):
        if encoder is None:
            continue
        config = getattr(encoder, "config", None)
        for key in ("projection_dim", "hidden_size", "d_model"):
            value = getattr(config, key, None)
            if value is not None:
                return int(value)
        get_dim = getattr(encoder, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            value = get_dim()
            if value is not None:
                return int(value)
    raise ValueError(
        "M2D checkpoint has no backbone.text_proj and the text encoder does not expose its output dimension"
    )


class MoescoreExpert3(nn.Module):
    """Partial fine-tuning of M2D-CLAP with the MOESCORE Expert 3 head."""

    def __init__(
        self,
        checkpoint_path,
        cache_dir="./hf_cache",
        activation_checkpointing=True,
        concat_assignment="released",
        checkpoint_layout_root=None,
        expected_checkpoint_sha256=None,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        cache_dir = Path(cache_dir).expanduser().absolute()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"M2D-CLAP checkpoint not found: {checkpoint_path}")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if expected_checkpoint_sha256 and checkpoint_sha256.lower() != expected_checkpoint_sha256.lower():
            raise ValueError(
                "M2D pretrained checkpoint SHA-256 does not match the expected M2D-CLAP 2025 artifact: "
                f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        layout_root = Path(checkpoint_layout_root).expanduser() if checkpoint_layout_root else cache_dir / "m2d_layout"
        compatible_checkpoint = prepare_m2d_checkpoint(checkpoint_path, layout_root)

        self.m2d = _load_portable_m2d(str(compatible_checkpoint), flat_features=True)
        # PortableM2D constructs the BERT module lazily; initialize it before freezing.
        self.m2d.get_clap_text_encoder()
        self.concat_assignment = concat_assignment
        if concat_assignment not in {"released", "semantic"}:
            raise ValueError("concat_assignment must be 'released' or 'semantic'")
        self.activation_checkpointing = bool(activation_checkpointing)
        self._configure_partial_finetuning()

        audio_projector = getattr(self.m2d.backbone, "audio_proj", None)
        text_projector = getattr(self.m2d.backbone, "text_proj", None)
        if audio_projector is None:
            raise ValueError("Expected M2D-CLAP backbone.audio_proj module")
        audio_dim = _linear_output_dim(audio_projector)
        # Official M2D-CLAP checkpoints may omit text_proj. PortableM2D then
        # returns the text encoder embedding directly from encode_clap_text().
        text_dim = (
            _linear_output_dim(text_projector)
            if text_projector is not None
            else _text_encoder_output_dim(self.m2d.text_encoder)
        )
        if audio_dim != text_dim:
            raise ValueError(f"Audio/text embedding dimensions differ: {audio_dim} vs {text_dim}")
        self.embedding_dim = audio_dim
        self.cross_feature_dim = 5 * self.embedding_dim
        self.alignment_head = nn.Sequential(
            nn.Linear(self.cross_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.alignment_head.apply(self._initialize_head)
        self.model_config = {
            "embedding_dim": self.embedding_dim,
            "cross_feature_dim": self.cross_feature_dim,
            "concat_assignment": self.concat_assignment,
            "activation_checkpointing": self.activation_checkpointing,
            "trainable_audio_blocks": [-2, -1],
            "trainable_text_layers": [-2, -1],
            "has_text_projector": text_projector is not None,
        }
        self.base_checkpoint_identity = {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "model_directory": M2D_2025_MODEL_DIR,
        }

    @staticmethod
    def _initialize_head(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=2e-5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _configure_partial_finetuning(self):
        for parameter in self.m2d.parameters():
            parameter.requires_grad_(False)

        audio_blocks = getattr(self.m2d.backbone, "blocks", None)
        text_model = getattr(self.m2d, "text_encoder", None)
        text_layers = getattr(
            getattr(getattr(text_model, "text_encoder", None), "encoder", None), "layer", None
        )
        if audio_blocks is None or len(audio_blocks) < 2:
            raise ValueError("Expected a ViT backbone with at least two Transformer blocks at backbone.blocks")
        if text_layers is None or len(text_layers) < 2:
            raise ValueError(
                "Expected BERT layers at text_encoder.text_encoder.encoder.layer; "
                "the lazy text encoder may not have been initialized"
            )

        for block in audio_blocks[-2:]:
            block.requires_grad_(True)
        for layer in text_layers[-2:]:
            layer.requires_grad_(True)
        self.m2d.backbone.audio_proj.requires_grad_(True)
        text_projector = getattr(self.m2d.backbone, "text_proj", None)
        if text_projector is not None:
            text_projector.requires_grad_(True)

    def trainable_parameter_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def encode_modalities(self, wavs, captions):
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs = wavs.squeeze(1)
        if wavs.ndim != 2:
            raise ValueError(f"Expected waveform shape [B,T] or [B,1,T], got {tuple(wavs.shape)}")
        if not isinstance(captions, (list, tuple)) or not all(isinstance(x, str) for x in captions):
            raise TypeError("captions must be a list of raw strings")
        if wavs.size(0) != len(captions):
            raise ValueError(f"Audio/text batch sizes differ: {wavs.size(0)} vs {len(captions)}")

        checkpoint_activations = self.activation_checkpointing and self.training and torch.is_grad_enabled()
        if checkpoint_activations:
            audio_emb = checkpoint(self.m2d.encode_clap_audio, wavs, use_reentrant=False)
            anchor = wavs.new_zeros((), requires_grad=True)
            text_emb = checkpoint(
                lambda _anchor: self.m2d.encode_clap_text(captions), anchor, use_reentrant=False
            )
        else:
            audio_emb = self.m2d.encode_clap_audio(wavs)
            text_emb = self.m2d.encode_clap_text(captions)

        if audio_emb.ndim != 2 or text_emb.ndim != 2:
            raise ValueError(
                f"Expected clip embeddings [B,D], got audio={tuple(audio_emb.shape)} "
                f"text={tuple(text_emb.shape)}"
            )
        expected = (wavs.size(0), self.embedding_dim)
        if tuple(audio_emb.shape) != expected or tuple(text_emb.shape) != expected:
            raise ValueError(
                f"Expected audio/text embeddings {expected}, got "
                f"{tuple(audio_emb.shape)} and {tuple(text_emb.shape)}"
            )
        return F.normalize(text_emb.float(), dim=-1), F.normalize(audio_emb.float(), dim=-1)

    def forward(self, batch):
        wavs = batch.get("wavs", batch.get("audio"))
        captions = batch.get("captions", batch.get("caption_tokens"))
        if wavs is None or captions is None:
            raise KeyError("batch must include 'wavs' and raw-string 'captions'")
        text_emb, audio_emb = self.encode_modalities(wavs, captions)
        # Released MOESCORE returns (text, audio) but binds it as (audio, text).
        if self.concat_assignment == "released":
            audio_feats, text_feats = text_emb, audio_emb
        else:
            audio_feats, text_feats = audio_emb, text_emb
        cos = (audio_feats * text_feats).sum(dim=-1, keepdim=True)
        diff = torch.abs(audio_feats - text_feats)
        prod = audio_feats * text_feats
        cross_features = torch.cat(
            [audio_feats, text_feats, diff, prod, cos.expand(-1, self.embedding_dim)], dim=-1
        )
        # Keep the small-initialized regression head in FP32. Under CUDA FP16
        # autocast, the scaled batch-reduced gradient can overflow while
        # accumulating into the final scalar bias for skewed score batches.
        with torch.autocast(device_type=cross_features.device.type, enabled=False):
            predicted_score = self.alignment_head(cross_features.float())
        return predicted_score, cross_features
