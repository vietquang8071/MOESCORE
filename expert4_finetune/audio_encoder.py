"""BEATs token representations with the released layer aggregation."""
import importlib
import sys
from pathlib import Path

import torch
from torch import nn


def aggregate_layer_outputs(layer_results, layer_weights):
    if len(layer_results) != layer_weights.numel() + 1:
        raise ValueError(f"Expected initial state plus {layer_weights.numel()} layer outputs, got {len(layer_results)}")
    hidden = torch.stack([result[0].transpose(0, 1) for result in layer_results[1:]])
    weights = torch.softmax(layer_weights, dim=0)
    if not torch.allclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6):
        raise ValueError('BEATs aggregation weights do not sum to one')
    return (hidden * weights[:, None, None, None]).sum(dim=0)


class AudioEncoder(nn.Module):
    def __init__(self, checkpoint_path, unilm_source_dir):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        source = Path(unilm_source_dir)
        beats_dir = source if (source / 'BEATs.py').is_file() else source / 'beats'
        if not checkpoint_path.is_file() or not (beats_dir / 'BEATs.py').is_file():
            raise FileNotFoundError(f"BEATs checkpoint or source missing: {checkpoint_path}, {beats_dir}")
        sys.path.insert(0, str(beats_dir.resolve()))
        try:
            module = importlib.import_module('BEATs')
            if Path(module.__file__).resolve() != (beats_dir / 'BEATs.py').resolve():
                raise RuntimeError('A different BEATs module was already imported; restart the Python runtime')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            cfg = module.BEATsConfig(checkpoint['cfg'])
            self.model = module.BEATs(cfg)
            self.model.load_state_dict(checkpoint['model'], strict=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize BEATs from {checkpoint_path}: {exc}") from exc
        self.num_layers = len(self.model.encoder.layers)
        if self.num_layers != 12 or cfg.encoder_layers != 12:
            raise ValueError(f"Expected 12 BEATs encoder layers, got {self.num_layers} (config {cfg.encoder_layers})")
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        self.audio_feats_proj = nn.Sequential(nn.LayerNorm(cfg.encoder_embed_dim), nn.Linear(cfg.encoder_embed_dim, 512), nn.GELU(), nn.LayerNorm(512))
        for param in self.model.parameters():
            param.requires_grad = True
        for layer in self.model.encoder.layers[:-2]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, wavs, lengths):
        if wavs.ndim != 3 or wavs.shape[1] != 1 or lengths.shape != (wavs.shape[0],):
            raise ValueError('Expected wavs [B,1,N] and lengths [B]')
        if (lengths <= 0).any() or (lengths > wavs.shape[-1]).any():
            raise ValueError('Waveform lengths must be positive and at most the padded length')
        batch = wavs.squeeze(1)
        batch = (batch - batch.mean(dim=-1, keepdim=True)) / (batch.std(dim=-1, keepdim=True) + 1e-8)
        mask = torch.arange(batch.shape[-1], device=batch.device)[None, :] >= lengths[:, None]
        # Kaldi fbank is kept in FP32 even when the transformer uses CUDA AMP.
        with torch.autocast(device_type=batch.device.type, enabled=False):
            x = self.model.preprocess(batch.float())
        mask = self.model.forward_padding_mask(x, mask)
        x = self.model.patch_embedding(x.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.model.layer_norm(x)
        mask = self.model.forward_padding_mask(x, mask).bool()
        if self.model.post_extract_proj is not None:
            x = self.model.post_extract_proj(x)
        x = self.model.dropout_input(x)
        _, results = self.model.encoder(x, padding_mask=mask, layer=12)
        aggregated = aggregate_layer_outputs(results, self.layer_weights)
        features = self.audio_feats_proj(aggregated).float()
        if features.ndim != 3 or features.shape[-1] != 512 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('Invalid BEATs feature or padding-mask dimensions')
        if not torch.isfinite(features).all() or mask.all(dim=1).any():
            raise ValueError('BEATs produced non-finite features or an entirely padded sequence')
        return features, mask
