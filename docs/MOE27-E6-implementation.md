# MOE27-E6: Set B residual expert

## Architecture and contract

E6 consumes frozen Full9 `top2_laion` outputs and frozen `openai/whisper-large-v2` cache vectors. The existing LAION adapter produces a 2049-wide `4D+1` paired vector from `[audio512, text512, absolute normalized difference512, normalized product512, cosine1]`. The existing Whisper extractor produces separate pooled audio and caption decoder vectors, each width 1280. E6 joins them in the exact order `[LAION2049, Whisper audio1280, Whisper text1280]`, returning float32 `[B,4609]`. It adds no Whisper cosine, compression, ASR, gating, SVR or backbone parameters.

`SetBResidualAdapter` accepts exactly `component_outputs.top2_laion.{score,features}`, `whisper_audio` and `whisper_text`. It validates names, widths, dtypes, devices, finite values, the complete LAION paired representation and cosine consistency. It detaches all frozen inputs. `SetBResidualHead` computes `scale * laion_cosine + bias + residual_mlp(features6)`. Scale/bias initialize to 1/0; the final residual linear layer initializes to zero. The MLP is `4609→256→64→1`, with LayerNorm, GELU and dropout 0.2 in the agreed order. The adapter returns score `[B,1]` and features `[B,4609]`.

## Verified cache join

`SetBCachedDataset(full9_root, whisper_root, split=...)` reads the existing verified Full9 feature store, its hashed source manifest and the existing hashed Whisper shards. It reads only the LAION score and feature slice from Full9. The source manifest supplies ordered sample IDs, source WAV names, raw captions and WAV byte hashes. E6 reconstructs Whisper's ordered dataset fingerprint with the exact canonical per-row `{sample_id, caption_sha256, audio_sha256}` JSON serialization and newline separator used by `build_whisper_dataset_identity`; it compares that fingerprint through the Whisper cache reader. Optional `source_audio_dir` verifies the original WAV bytes against the Full9 source manifest. Cache order and count must match exactly. The reader rejects incomplete, corrupt or incompatible caches and checks Whisper's extraction batch-padding identity plus unified preprocessing source hashes.

Official `train` and `validation` labels stay raw `[0,10]` in Full9. E6 emits `target_normalized=(label_raw-5)/5` at the dataset boundary, preserving float32 precision. Test stores must be label-free. Training rejects train/validation sample overlap and mismatched LAION checkpoint, Whisper checkpoint or Whisper preprocessing identities. No extra E6 cache or model-ID logic is used.

## Training and checkpoints

The policy is fixed at **30 completed epochs**, all official train samples, AdamW, MSE, batch size 8, learning rate 0.001, weight decay 0.0001, seed 42, optional gradient clipping 1.0. There is no internal holdout or early stopping. After every epoch, the trainer evaluates the complete official validation split with `eval()` and `inference_mode()`. MSE is sum of squared errors divided by sample count. Strictly lower validation MSE replaces `expert6_best.pt`; ties preserve the earlier best. `expert6_latest.pt` stores the latest optimizer and resume state. `expert6_final.pt` is always the epoch-30 model, even when an earlier epoch was best. All writes are atomic. Checkpoints contain only E6 head tensors, optimizer state, architecture/feature/normalization contracts, train and validation provenance, identities, hyperparameters, metric history, epoch/step and selection policy `official_validation_mse_min_v1`.

Resume uses only `expert6_latest.pt` in the configured output directory. It verifies architecture, optimizer settings, both dataset provenances and the sibling best checkpoint SHA-256, then resumes from the next epoch with deterministic per-epoch seed/order. The configured 30 epochs are the total. A completed run cannot be resumed.

## CLI

Set `MOESCORE_FULL9_FEATURE_STORE`, `MOESCORE_WHISPER_FEATURE_CACHE` and `MOESCORE_E6_OUTPUT`, then run:

```bash
python scripts/train_set_b_residual.py --config configs/set_b_residual.example.yaml
python scripts/train_set_b_residual.py --config configs/set_b_residual.example.yaml --resume "$MOESCORE_E6_OUTPUT/expert6_latest.pt"
python scripts/infer_set_b_residual.py --config configs/set_b_residual.example.yaml --checkpoint "$MOESCORE_E6_OUTPUT/expert6_best.pt" --split test --output "$MOESCORE_E6_OUTPUT/test_scores.csv"
```

Inference verifies the joined cache and checkpoint, preserves sample order, and writes normalized and raw (`5*normalized+5`) scores without clamping. Runtime device accepts `auto`, `cpu`, or available `cuda`. Optional config maps `source_manifest_paths` and `source_audio_dirs` by split.

## Full7 interface and competition caveat

Full7 should instantiate `SetBResidualAdapter` with a `SetBResidualHead`, load `expert6_best.pt` after provenance checks, pass the exact batch schema above, and add the returned `[B,1]` score plus `[B,4609]` feature vector at E6's agreed assembly position. This branch does not alter the global registry or gating dimensions.

The chosen best checkpoint uses official validation labels for **model selection**; validation receives no gradient updates. The project rules text cited in `e6-overview.md` does not explicitly settle whether this checkpoint selection is allowed. Obtain written organizer clarification before competition submission. Pretrained LAION and Whisper checkpoints and external data also require the project's disclosure/approval checks. Synthetic cache tests do not establish real-backbone numerical parity.
