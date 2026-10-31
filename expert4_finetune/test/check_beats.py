
import argparse
from types import SimpleNamespace

import torch

from model.wrapper import AudioEncoder


def check_beats(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SimpleNamespace(
        device=device,
        audio_wrapper=SimpleNamespace(
            beats_ckpt_path=checkpoint_path,
            proj_output_dim=512,
        ),
    )

    # 1. Load pretrained checkpoint
    # AudioEncoder internally uses strict=True by default.
    encoder = AudioEncoder(cfg).to(device)
    encoder.eval()

    n_layers = len(encoder.model.encoder.layers)
    hidden_dim = encoder.model.cfg.encoder_embed_dim
    n_weights = encoder.layer_weights.numel()

    print("BEATs layers:", n_layers)
    print("Hidden dimension:", hidden_dim)
    print("Learnable layer weights:", n_weights)

    assert n_layers == 12, (
        f"Expected 12 BEATs layers, found {n_layers}"
    )
    assert n_weights == n_layers

    # 2. Create synthetic audio:
    #    Sample 1: 1 second
    #    Sample 2: 0.5 second + padding
    sr = 16000
    B = 2

    wavs = torch.randn(B, 1, sr, device=device)
    wavs[1, :, sr // 2:] = 0.0

    wav_lens = torch.tensor(
        [sr, sr // 2],
        device=device,
        dtype=torch.long,
    )

    # 3. Capture BEATs encoder outputs
    captured = {}

    def inspect_layers(module, inputs, output):
        _, layer_results = output

        captured["total_entries"] = len(layer_results)
        captured["layer_shapes"] = [
            tuple(item[0].shape)
            for item in layer_results[1:]
        ]

    handle = encoder.model.encoder.register_forward_hook(
        inspect_layers
    )

    try:
        with torch.no_grad():
            audio_seq, audio_mask = encoder.get_audio_sequence(
                wavs, wav_lens
            )
    finally:
        handle.remove()

    # 4. Check layer aggregation
    n_hidden = len(captured["layer_shapes"])

    print("Layer results entries:", captured["total_entries"])
    print("Aggregated hidden states:", n_hidden)
    print("Hidden state shapes:", captured["layer_shapes"])

    assert captured["total_entries"] == n_layers + 1
    assert n_hidden == n_layers

    # 5. Check final projected features
    print("Audio sequence shape:", tuple(audio_seq.shape))
    print("Padding mask shape:", tuple(audio_mask.shape))

    assert audio_seq.ndim == 3
    assert audio_seq.shape[0] == B
    assert audio_seq.shape[-1] == 512
    assert audio_mask.shape == audio_seq.shape[:2]
    assert audio_mask.dtype == torch.bool
    assert torch.isfinite(audio_seq).all().item()

    # Second sample should have some padding.
    assert audio_mask[1].any().item()

    # 6. Check layer weights
    alpha = torch.softmax(encoder.layer_weights, dim=0)
    assert torch.allclose(
        alpha.sum(),
        torch.tensor(1.0, device=device),
        atol=1e-6,
    )

    print("Layer weights sum:", alpha.sum().item())

    # 7. Check freeze policy
    for i, layer in enumerate(encoder.model.encoder.layers):
        expected_trainable = i >= n_layers - 2
        actual = [p.requires_grad for p in layer.parameters()]

        assert all(v == expected_trainable for v in actual), (
            f"Unexpected trainability in BEATs layer {i}"
        )

    assert encoder.layer_weights.requires_grad
    assert all(
        p.requires_grad
        for p in encoder.audio_feats_proj.parameters()
    )

    print("\nALL BEATs CHECKS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    check_beats(args.checkpoint)
