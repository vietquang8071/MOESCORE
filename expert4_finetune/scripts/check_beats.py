"""Validate an official BEATs checkpoint, aggregation, masks and gradients."""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert4_finetune.audio_encoder import AudioEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--unilm-source-dir', required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = AudioEncoder(args.checkpoint, args.unilm_source_dir).to(device)
    # One full-length sample and one half-length sample test both masks.
    wavs = torch.randn(2, 1, 160000, device=device) * 0.01
    wavs[1, :, 80000:] = 0
    lengths = torch.tensor([160000, 80000], device=device)
    features, mask = encoder(wavs, lengths)
    if encoder.num_layers != 12 or features.shape[:2] != mask.shape or features.shape[-1] != 512 or mask.dtype != torch.bool:
        raise RuntimeError('BEATs checkpoint/aggregation validation failed')
    features.square().mean().backward()
    for name, parameter in [('aggregation', encoder.layer_weights),
                             ('audio projection', encoder.audio_feats_proj[1].weight),
                             ('last BEATs layer', next(encoder.model.encoder.layers[-1].parameters()))]:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f'{name} did not receive a finite gradient')
    print(f'BEATs checks passed: {encoder.num_layers} layers, features={tuple(features.shape)}, mask={tuple(mask.shape)}')


if __name__ == '__main__':
    main()
