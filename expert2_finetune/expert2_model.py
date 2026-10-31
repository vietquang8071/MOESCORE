"""Cosine MGA-CLAP Expert 2 with MoE-compatible state-dict names."""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class MGAEncoder(nn.Module):
    def __init__(self, ase):
        super().__init__()
        self.model = ase

    def forward(self, audio, text):
        ase = self.model
        _, word, mask = ase.encode_text(text)
        text_embed = F.normalize(ase.msc(word, ase.codebook, mask), dim=-1)
        _, frame = ase.encode_audio(audio.squeeze(1))
        audio_embed = F.normalize(ase.msc(frame, ase.codebook), dim=-1)
        return audio_embed, text_embed


class Expert2(nn.Module):
    def __init__(self, ase, freeze_policy="partial"):
        super().__init__()
        if freeze_policy not in ("partial", "full"):
            raise ValueError("freeze_policy must be partial or full")
        self.model = MGAEncoder(ase)
        self.freeze_policy = freeze_policy
        self._set_trainable()

    def _set_trainable(self):
        for name, param in self.model.model.named_parameters():
            if self.freeze_policy == "full":
                trainable = name.startswith(("audio_encoder.audio_enc.layers.", "audio_encoder.audio_enc.norm.", "text_encoder.text_encoder.", "frame_proj.", "word_proj.")) or name == "codebook"
            else:
                trainable = name.startswith((
                    "audio_encoder.audio_enc.layers.3.blocks.0.",
                    "audio_encoder.audio_enc.layers.3.blocks.1.",
                    "audio_encoder.audio_enc.norm.",
                    "text_encoder.text_encoder.encoder.layer.10.",
                    "text_encoder.text_encoder.encoder.layer.11.",
                    "frame_proj.", "word_proj.",
                )) or name == "codebook"
            param.requires_grad_(trainable)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            ase = self.model.model
            ase.audio_encoder.audio_enc.eval()
            ase.text_encoder.text_encoder.eval()
            if self.freeze_policy == "full":
                for layer in ase.audio_encoder.audio_enc.layers:
                    layer.train()
                ase.audio_encoder.audio_enc.norm.train()
                ase.text_encoder.text_encoder.train()
            else:
                for block in ase.audio_encoder.audio_enc.layers[3].blocks[-2:]:
                    block.train()
                for layer in ase.text_encoder.text_encoder.encoder.layer[-2:]:
                    layer.train()
        return self

    def forward(self, batch):
        audio, text = self.model(batch["mga_wavs"], batch["caption_tokens"])
        score = (audio * text).sum(dim=-1, keepdim=True)
        return score, torch.cat((audio, text), dim=-1)


def load_mga_state(ase, state):
    state = dict(state)
    legacy_key = "text_encoder.text_encoder.embeddings.position_ids"
    if legacy_key in state and legacy_key not in ase.state_dict():
        positions = state.pop(legacy_key).cpu()
        if positions.ndim != 2 or positions.shape[0] != 1 or not torch.equal(
            positions, torch.arange(positions.shape[1]).unsqueeze(0)
        ):
            raise RuntimeError("Unexpected legacy BERT position_ids contents")
    ase.load_state_dict(state, strict=True)


def create_expert2(mga_source, architecture_config, pretrained_path, device, freeze_policy="partial"):
    source = str(Path(mga_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.ase_model import ASE

    config = copy.deepcopy(architecture_config)
    config["audio_encoder_args"]["pretrained"] = False
    config["text_encoder_args"]["init_from_config"] = True
    config["training"]["spec_augmentation"] = False
    ase = ASE(config)
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    load_mga_state(ase, checkpoint["model"])
    return Expert2(ase, freeze_policy=freeze_policy).to(device)
