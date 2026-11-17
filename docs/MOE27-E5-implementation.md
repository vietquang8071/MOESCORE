# MOE27-E5 implementation

## Architecture

E5 combines three frozen Set A components in this fixed order:

1. `top2_m2d`: 768-dimensional audio and text embeddings, 3,073 features.
2. `top2_msclap`: 1,024-dimensional audio and text embeddings, 4,097 features.
3. `top2_mga`: 1,024-dimensional audio and text embeddings, 4,097 features.

Each component uses the shared MOESCORE `build_paired_cross_features` helper. Its exact five blocks are `[audio, text, abs(normalize(audio) - normalize(text)), normalize(audio) * normalize(text), cosine]`. The original audio and text values are retained; the cosine occupies one scalar. The component widths add to 11,267.

`SetAResidualHead` computes `scale * mean(cosines) + bias + residual_mlp(features)`. The mean is the unweighted arithmetic mean. Scale and bias start at 1 and 0. The residual MLP is `Linear(11267,256) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(256,64) -> GELU -> Linear(64,1)` by default; its last layer starts at zero. Its hidden activations do not replace or alter the returned feature vector.

`SetAResidualAdapter` extends the existing `ExpertAdapter` and implements `_forward`. It accepts the verified component outputs in `batch["component_outputs"]` and returns exactly `(score5, features5)`, shaped `[B,1]` and `[B,11267]`. The adapter validates component names, widths, dtype, device, finite values, five-block ordering, and agreement between each score and its cosine block. It detaches the frozen component outputs before regression. No registry or shared Top2 adapter was changed.

## Component input schema

The Full7 Core assembly can pass already-computed component outputs without loading their models during E5 training or cached inference:

```python
batch = {
    "component_outputs": {
        "top2_m2d": {"score": score_m2d, "features": features_m2d},
        "top2_msclap": {"score": score_msclap, "features": features_msclap},
        "top2_mga": {"score": score_mga, "features": features_mga},
    }
}
score5, features5 = expert5(batch)
```

Every score is float32 `[B,1]`; component features are float32 `[B,3073]`, `[B,4097]`, and `[B,4097]`. All six tensors must have the same device and batch size. Concatenation is M2D, then MS-CLAP, then MGA. The returned `features5` is the complete concatenation used by the future Global Gating Network.

The required Full7 integration work is to add E5 to the internal expert construction/configuration and feature-store assembly, feed this nested input schema, and include E5 in the agreed global order and dimensions. The resulting expert output dimension is 11,267. E5 is intentionally not registered globally by this change.

## Source and checkpoint findings

The authoritative source inspected was `/home/vitquay1708/Working_Space/Research/2027/code/EnsembleSVR-XACLE`, including `train.py`, `features/extract_features.py`, `features/all_feature_dict.py`, `features/proximity_features.py`, `load_pretrained_models/load_model.py`, `datasets/fetch_data.py`, and `config_submission1.json`. The source loader uses `torchaudio.load`, then pads or truncates waveforms to 160,000 samples with the configured 16 kHz rate. Its MGA extraction runs `encode_audio`, codebook `msc`, and `F.normalize`; text extraction uses `encode_text`, `msc` with the attention mask, then normalization.

The source repository does not include its referenced `m2d`, `MGA-CLAP`, or `examples` vendor directories in this environment, so an independent instantiation of the source M2D/MGA implementations was not possible. The MOESCORE adapters use the corresponding locally vendored implementations and checkpoints. Numerical equality against an independently loaded EnsembleSVR-XACLE model was not measured; no exact upstream parity claim is made.

The current MOESCORE Top2 adapters were reused where their extraction path matches the source:

| Component | Source-faithful path used by E5 | Observed difference or limit |
|---|---|---|
| M2D-CLAP | Existing `Top2M2DAdapter`; `PortableM2D`, `flat_features=True`, `encode_clap_audio` / `encode_clap_text`, 768 dimensions. | The source vendor tree is unavailable for a paired numeric comparison. The checkpoint loader expects its model-ID directory layout; the existing smoke helper uses a temporary symlink to the actual checkpoint rather than copying it. |
| MS-CLAP 2023 | Existing `Top2MSCLAPAdapter`; `get_audio_embeddings(wav_paths)`, `get_text_embeddings(captions)`, 1,024 dimensions and the 2023 local checkpoint. | The original source uses path-based audio embeddings with a random seven-second segment; the MOESCORE adapter uses a deterministic centered seven-second crop. This is a preprocessing difference, so exact audio-embedding parity is not claimed. |
| MGA-CLAP | E5-only `SetAMGASourceAdapter`; existing ASE model restoration, `encode_audio`, codebook `msc`, `encode_text`, and source normalization; 1,024 dimensions. | The shared MOESCORE Top2 adapter consumes the 32 kHz `expert2_wavs` input, while the Top2 source uses the 16 kHz, 160,000-sample waveform. E5 uses `expert3_wavs` to preserve the source contract. The restored checkpoint matched 415/415 model tensors and reported one unexpected checkpoint key. |

Real-checkpoint smoke checks ran against one official training sample (`train:00000.wav`) on CPU using the local checkpoints. Existing MOESCORE Top2 outputs were finite and had the agreed output widths: M2D score `0.35162234`, features `[1,3073]`; MS-CLAP score `0.64994228`, features `[1,4097]`; shared 32 kHz MGA score `0.47984895`, features `[1,4097]`. The E5 source-compatible 16 kHz MGA path returned score `0.30485559`, features `[1,4097]`. The MGA score difference is consistent with a different input-rate/sample preprocessing path; it is not a source-vs-MOESCORE embedding parity measurement.

The local checkpoint files are in the main MOESCORE repository and are not copied into this worktree:

```text
/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/checkpoint-30.pth
/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/CLAP_weights_2023.pth
/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/mga-clap.pt
```

The local config records their resolved paths in `configs/set_a_residual.local.yaml`. The portable config takes a resource root from environment variables. These resource paths identify the caches/backbones; residual-head checkpoints never save backbone state.

## Caches and data safeguards

`SetACachedDataset` reads either a verified `full_9` store or a standalone E5 cache. It checks the current `paired_cross_4d_plus_1_v2` representation, widths, actual five feature blocks, cosine consistency, cache and checkpoint identities, source-manifest hash, sample IDs/order, WAV names, and labels. Training accepts only the `train` split and transforms raw targets with `(target_raw - 5.0) / 5.0`. Test caches must not contain labels. It does not load a pretrained model.

`write_set_a_cache_from_full9` reuses verified M2D and MS-CLAP rows from a current full9 store and overlays separately extracted source-compatible 16 kHz MGA rows. It intentionally ignores the full9 MGA feature rows because the generic MOESCORE MGA path uses 32 kHz. Component sample IDs must exactly match the source manifest in order. `write_set_a_cache` can also write a cache from three individually verified output shards. Both writers require current cache identities and refuse old 5D representations.

No feature-store manifest or cache was present in this E5 worktree during implementation. The local config therefore points to `/tmp/moe27-e5/full9_feature_store` as a placeholder. Full-dataset extraction and training were not started. A caller must first provide a verified full9 store and source-compatible 16 kHz MGA outputs, assemble the standalone E5 cache, then set `feature_store_root` to that cache. The training CLI is runnable against that prepared cache.

## Training and inference

From this worktree, after preparing the cache and updating `feature_store_root` in the local config:

```bash
python scripts/train_set_a_residual.py --config configs/set_a_residual.local.yaml
```

The trainer uses AdamW, MSE on normalized training targets, deterministic seeding, configurable learning rate, weight decay, batch size, epochs, gradient clipping, and resume. It writes `expert5_latest.pt` and `expert5_final.pt` under the configured output directory. Resume with:

```bash
python scripts/train_set_a_residual.py \
  --config configs/set_a_residual.local.yaml \
  --resume /path/to/expert5_latest.pt
```

Inference from a prepared validation or test cache is:

```bash
python scripts/infer_set_a_residual.py \
  --config configs/set_a_residual.local.yaml \
  --checkpoint /path/to/expert5_final.pt \
  --split test \
  --output /path/to/expert5_test_scores.csv
```

The inference script returns normalized scores and their inverse transform `score * 5 + 5`; test labels are not loaded. Checkpoints contain the E5 state, optimizer, architecture and feature contracts, epoch/step, seed, split/cache and component identities, data provenance, and normalization metadata. Loading rejects architecture, feature-order/width, score-normalization, and provenance mismatches.

## Verification

Synthetic CPU tests cover the five-block representation, widths/order, arithmetic mean, affine calibration, residual gradients and zero initialization, `ExpertAdapter` output, invalid tensors/devices, cache alignment/provenance, training/resume/checkpoint restoration, and refusal to train validation/test splits. A bounded real-checkpoint check was completed for the current Top2 adapters and E5's source-compatible MGA path on one sample. CUDA was unavailable, so CUDA training was not exercised. There was no full real feature extraction, full-dataset training, or paired numerical comparison against the unavailable upstream M2D/MGA vendor trees.
