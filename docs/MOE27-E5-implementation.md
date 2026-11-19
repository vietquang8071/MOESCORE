# MOE27-E5: Set A residual expert

## Architecture and Full9 input

E5 reuses the three frozen component outputs already stored by MOESCORE's verified `full_9` feature-store pipeline, in this order:

1. `top2_m2d`: embedding `D=768`, paired feature width `4D+1=3073`.
2. `top2_msclap`: embedding `D=1024`, paired feature width `4097`.
3. `top2_mga`: embedding `D=1024`, paired feature width `4097`.

All component rows come from the **same Full9 store identity**, source manifest, sample order, and waveform identities. MGA uses the Full9 input `expert2_wavs`, `[B,1,320000]`, 32 kHz, maximum 320,000 samples. E5 does not run a separate MGA extraction path.

Each feature row uses the shared MOESCORE five-block contract `[audio, text, abs(normalize(audio)-normalize(text)), normalize(audio)*normalize(text), cosine_scalar]`. Audio and text values remain unnormalized in their original blocks. The current `4D+1` helper is reused; old `5D` stores are rejected. The E5 feature order is M2D, MS-CLAP, MGA and the unchanged output width is `3073 + 4097 + 4097 = 11267`.

The score is exactly:

```text
mean_cosine = (m2d_score + msclap_score + mga_score) / 3
score5 = scale * mean_cosine + bias + residual_mlp(features5)
```

Scale and bias initialize to 1 and 0. The final residual layer is zero-initialized. The default residual architecture is `Linear(11267,256) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(256,64) -> GELU -> Linear(64,1)`. There is no internal gating, SVR, backbone fine-tuning, or feature compression. The adapter returns `(score5, features5)` with shapes `[B,1]` and `[B,11267]`.

## Data contract and provenance

The preferred API is:

```python
train_dataset = SetACachedDataset(
    full9_root, split="train", require_labels=True
)
validation_dataset = SetACachedDataset(
    full9_root, split="validation", require_labels=True
)
```

The dataset reads the M2D, MS-CLAP, and MGA score/feature slices from a single verified Full9 store. It validates store and source-manifest hashes, split and ordered sample IDs, WAV identities, component checkpoint/cache identities, preprocessing identity, feature representation and dimensions, finite values, and labels. It rejects obsolete 5D representations or an MGA contract other than the Full9 32 kHz input above. Standalone precomputed E5 caches remain supported as an optional generic path, but are not required for Full9 training or inference.

Raw labels are checked against the source manifest after conversion to the cache's float32 representation, with numeric, finite, and `[0,10]` validation. Labeled official train and validation rows expose `label_raw` and `target_normalized = (target_raw - 5.0) / 5.0`; test labels remain hidden.

E5 fits on 100% of the official train split and evaluates on 100% of official validation. There is no internal train/holdout split. Validation runs after every epoch in evaluation mode under `torch.inference_mode()` and never participates in backward or optimizer updates.

## Preprocessing and source fidelity

E5 consumes the actual cached outputs produced by the existing Full9 adapters; it does not claim numerical equality with the original Top 2 extraction pipeline. In particular, the original Top 2 MGA loader uses a 16 kHz, 160,000-sample waveform, while this design deliberately adopts Full9 MGA at 32 kHz and 320,000 samples. Full9 MS-CLAP's audio crop behavior also follows the existing MOESCORE adapter rather than the original source's random seven-second selection. The source repository's M2D/MGA vendor dependencies were not available for an independent paired numerical comparison. No standalone E5 MGA adapter or real-checkpoint extraction is needed for this design.

The verified Full9 store is the runtime resource. No E5 checkpoint contains pretrained backbone weights. Local machine paths and checkpoints are not embedded in production configuration or copied into this worktree.

## Configuration and commands

Set the store and output roots, then train:

```bash
export MOESCORE_FULL9_FEATURE_STORE=/path/to/verified/full9_feature_store
export MOESCORE_E5_OUTPUT=/path/to/e5_runs
python scripts/train_set_a_residual.py --config configs/set_a_residual.example.yaml
```

The selection metric is `official_validation_mse`. The trainer records `train_mse`, `validation_mse`, `best_validation_mse`, epoch, and learning rate after each epoch. Strict improvement by more than `min_delta` selects `expert5_best.pt`; early stopping uses the same validation MSE and preserves the existing patience behavior. `expert5_latest.pt` is the resumable last epoch, and `expert5_final.pt` copies the best official-validation checkpoint. Patience defaults to five epochs; set it to `null` to disable early stopping.

Resume in the same output directory with the latest checkpoint:

```bash
python scripts/train_set_a_residual.py \
  --config configs/set_a_residual.example.yaml \
  --resume "$MOESCORE_E5_OUTPUT/expert5_latest.pt"
```

Resume restores optimizer, epoch, global step, seed, best validation score/epoch, patience progress, metric history, and both split memberships. It verifies official train and validation provenance, selection policy, training configuration, and the sibling best-checkpoint hash. Training hyperparameters and batch size must match; `epochs` is the requested total and can be extended. Legacy format V2 holdout checkpoints remain loadable for inference and legacy Full7 membership checks, but cannot be resumed under the new policy.

Run inference on a verified Full9 validation or test split (runtime device accepts `auto`, `cpu`, or `cuda`):

```bash
python scripts/infer_set_a_residual.py \
  --config configs/set_a_residual.example.yaml \
  --checkpoint "$MOESCORE_E5_OUTPUT/expert5_best.pt" \
  --split test \
  --output "$MOESCORE_E5_OUTPUT/test_scores.csv"
```

Both `best` and `final` are suitable for inference; prefer `best` for downstream evaluation and Full7 integration. Format V3 checkpoints contain the head, optimizer snapshot, architecture and feature contracts, score normalization, epoch/step, component model/cache identities, official train and validation provenance, exact sample membership, selection policy, and metrics. They do not contain CLAP backbone tensors or fabricate legacy holdout fields.

## Full7 integration contract

The Full7 core passes:

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

Each score is float32 `[B,1]`; feature shapes are `[B,3073]`, `[B,4097]`, `[B,4097]`. All tensors share batch size and device. E5 returns float32 score `[B,1]` and the full ordered concatenation `[B,11267]`. Full7 verifies the checkpoint's official train/validation source identity and exact membership, then records train rows as `IN_SAMPLE_FIT`, validation rows as `OUT_OF_SAMPLE_FIT_VALIDATION_SELECTED`, and test rows as `OUT_OF_SAMPLE`. Legacy V2 checkpoints retain their internal holdout membership classification.

## Verification status

CPU synthetic Full9 stores exercise direct component reuse, MGA's 32 kHz contract, label conversion and source-manifest checks for validation, cache provenance, output dimensions, full-train optimization, no-gradient validation, early stopping, validation-best/final selection, resume checks, `device: auto`, checkpoint inference, and Full7 membership. The local environment has no official Full9 data, so no real-cache smoke test or full-size training was run. The exact test commands and results are recorded in `MOE27-E5-report.md`.
