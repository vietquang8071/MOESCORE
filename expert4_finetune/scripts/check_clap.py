"""Validate the supplied LAION-CLAP text checkpoint and token projection."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert4_finetune.text_encoder import TextEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TextEncoder.load_pretrained(args.checkpoint).to(device)
    features, mask = encoder(['A dog barks.', 'Wind blows past a microphone.'])
    if features.shape[0] != 2 or features.shape[-1] != 512 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise RuntimeError('Invalid CLAP projected token or mask shape')
    features.square().mean().backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.branch.encoder.layer[-1].parameters()):
        raise RuntimeError('Final RoBERTa layer received no finite gradient')
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.projection.parameters()):
        raise RuntimeError('Pretrained text projection received no finite gradient')
    if any(p.requires_grad or p.grad is not None for p in encoder.branch.encoder.layer[0].parameters()):
        raise RuntimeError('Early RoBERTa layer is not frozen')
    print(f'CLAP checks passed: RoBERTa hidden=768, projected={tuple(features.shape)}, mask={tuple(mask.shape)}')


if __name__ == '__main__':
    main()
