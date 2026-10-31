# MOESCORE Expert 3 reproduction

This pipeline fine-tunes only MOESCORE Expert 3 (M2D-CLAP), using the XACLE 2027 baseline repository for the vendored M2D runtime, metric definitions, dataset conventions, and hybrid loss. It does not train the MoE gate or the other experts.

## Audited source and artifacts

Primary references: [XACLE 2027 baseline](https://github.com/XACLE-Challenge/XACLE_2027_baselinemodel), [MOESCORE paper](https://arxiv.org/abs/2601.06829), [MOESCORE source](https://github.com/S-Orion/MOESCORE), and [official M2D implementation/release](https://github.com/nttcslab/m2d/releases/tag/v0.5.0). The TASALIGN paper is cited by name because the task materials provide its training settings, but this repository contains no verified copy or stable public URL; the provenance rows below follow the task-provided citation.

- The vendored `PortableM2D` parses model/input/text-encoder metadata from `Path(weight_file).parent.name`. The checkpoint in the shared workspace was outside that expected directory, so `prepare_m2d_checkpoint()` creates a symlink under `m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/`. It never copies or overwrites the pretrained file.
- The supplied checkpoint is 1,694,677,655 bytes and its SHA-256 is `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`, matching the digest recorded in the XACLE baseline's M2D runtime README. Its directory name identifies the standard M2D-CLAP 2025 model configured for 16 kHz input, with BERT-base text weights included (`B`, `TI`). Training checks the digest before constructing the model.
- `PortableM2D.get_clap_text_encoder()` creates the text model lazily. Expert 3 invokes it before applying the freeze map or creating the optimizer. The implementation uses `backbone.blocks[-2:]` and `text_encoder.text_encoder.encoder.layer[-2:]`; the patch embedding and all other pretrained parameters remain frozen.
- The supplied checkpoint has `audio_proj` weights but no `text_proj.weight` (the Colab loader reports no included text-projector parameters). The vendored `make_it_CLAP()` only creates `backbone.text_proj` when that checkpoint key exists, and `encode_clap_text()` then returns BERT embeddings directly. Expert 3 therefore fine-tunes the final two BERT layers and the audio projector, without inventing a randomly initialized text projector. If a different checkpoint includes `text_proj`, the code detects and unfreezes it.
- The baseline invokes the checkpoint with `flat_features=True`. The M2D-CLAP projections produce 768-dimensional audio and text embeddings. The expected parameter hierarchy comes from the vendored source, rather than the names in MOESCORE's original substring checks.
- The current local dataset root is `/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset`. Its average-score CSVs have columns `wav_file_name,text,average_score,model_ID,model_name`, with 7,500 train, 3,000 validation, and 3,000 test rows. The checked sample WAV is mono, 16 kHz. The XACLE baseline README says its 2027 development set is inherited from the 2026 challenge dataset; counts match the published split sizes. The code checks all WAV paths and split disjointness. It trains on `train_average.csv` only.

## Model and feature construction

1. Load `PortableM2D(weight_file=..., flat_features=True)`.
2. Initialize the lazy BERT encoder and then freeze all M2D parameters.
3. Unfreeze the last two ViT blocks, last two BERT layers, and both M2D projection modules. Keep the patch embedding frozen.
4. Call `encode_clap_audio(wavs)` and `encode_clap_text(captions)` on 16 kHz mono waveforms and raw caption strings. Normalize both outputs with L2 normalization.
5. Preserve MOESCORE's released tuple binding by default: its encoder returns `(text_embeds, audio_embeds)`, then `Expert3.forward()` assigns those values to `(audio_feats, text_feats)`. `concat_assignment="semantic"` is available as an explicit comparison option; the reproduction default is `released`.
6. Concatenate `[audio, text, abs(audio-text), audio*text, cosine-expanded]`. With 768-dimensional embeddings, this is 3,840 dimensions. The released code hardcodes 3,072, which is inconsistent with its five 768-dimensional components; the head input dimension is computed as `5 * embedding_dim`.
7. Use the released MLP: `3840 -> 512 -> 128 -> 1`, ReLU and dropout 0.3 after the first two linear layers. Linear weights use truncated normal with standard deviation `2e-5`; biases are zero.

## Objective and training decisions

The baseline hybrid loss is reused: clipped MSE with `tau=0.25`, plus half-weighted pairwise contrastive loss with `margin=0.1`; both use `CBL=false`. Scores are normalized as `(score - 5) / 5`. Prediction and target vectors are flattened to `[B]` before loss evaluation. The baseline's `tau` tensor was changed to a registered buffer so it follows `.to(cuda)`; this fixes the baseline loss's CPU/GPU device mismatch without changing its formula.

| Setting | Value | Provenance |
|---|---:|---|
| Physical training batch | 8, `drop_last=true` | TASALIGN paper; intentionally overrides MOESCORE YAML's 16 |
| Gradient accumulation | 1 | Required so every pairwise loss sees the same eight samples in one forward pass |
| Validation batch | 1 | MOESCORE `config.yaml` |
| Learning rate | `1e-5` | MOESCORE YAML and TASALIGN paper |
| Optimizer | Adam, betas `(0.9, 0.999)` | TASALIGN paper; not verified as Expert-3-specific in released training code |
| Scheduler | Linear warm-up then linear decay | TASALIGN paper; exact warm-up fraction is not disclosed |
| Warm-up fraction | 10% default, configurable | Explicit implementation assumption |
| Epochs / early stop / seed | 50 / 50 / 42 | MOESCORE YAML |
| Audio | mono, 16 kHz, pad/trim to 10 s | XACLE baseline data path and MOESCORE config |
| AMP / activation checkpointing | FP16 AMP / enabled | Encoder uses FP16 autocast; the small-initialized regression head runs FP32 to prevent scaled output-bias gradient overflow. Activation checkpointing recomputes the full audio/text encoder forward to reduce saved activations. |

The DataLoader uses `batch_size=8`, `drop_last=True`, and zero workers. The trainer rejects another batch size or accumulation setting, asserts eight examples at each train step, and stops with an actionable error on CUDA OOM. It never changes the batch or freezes intended layers as a fallback. The warm-up ratio is an assumption because the paper does not specify one.

## Known reproduction differences

- `encode_clap_audio_multiscale()` is called in released MOESCORE Expert 3 but is absent from the vendored official M2D runtime. This implementation uses `encode_clap_audio()` as the agreed compatibility substitute; equivalence is not established.
- MOESCORE's released layer-name patterns do not match the official M2D module tree. Direct module references select the intended final two ViT and BERT layers.
- Text-encoder lazy initialization is completed before freezing and optimizer construction.
- This checkpoint contains no text projection module; its official `encode_clap_text()` path returns BERT embeddings directly. The implementation handles that checkpoint faithfully and only unfreezes `text_proj` when the checkpoint actually provides it. No untrained replacement projection is added.
- Regression input dimension is corrected from the released hardcoded 3,072 to the shape-consistent 3,840 (computed from runtime embedding size).
- The released encoder returns text then audio, and the Expert3 caller binds them in the opposite variable order. The default retains that literal concatenation order.
- This task sets physical batch size 8 from TASALIGN instead of MOESCORE's released YAML batch size 16.
- Adam betas and linear warm-up/decay are paper-level settings, not a verified Expert-3-only optimizer recipe from `Train_MOE.py`; that script trains the gate. The 10% warm-up ratio is not published.
- The local dataset is the XACLE 2027 development distribution described by the baseline as inherited from 2026. Its rows/counts match the challenge split sizes, but this implementation report does not claim a byte-for-byte comparison with an archived XACLE 2026 dataset.

## Checkpointing and inference

`best_model.pt` stores every trainable M2D/projection/head parameter plus the model configuration, pretrained checkpoint identity, epoch, and validation metrics. Frozen weights are reconstructed from the verified pretrained file. `last_checkpoint.pt` additionally stores optimizer, scheduler, scaler, early-stop state, and Python/NumPy/PyTorch/DataLoader RNG state. Both files are written atomically. Resume checks the pretrained digest, training hyperparameters, CSV byte hashes, and path-independent WAV name/size manifests. WAV contents are not fully hashed so Drive staging does not require reading all audio twice; if files are replaced with different content of identical byte size, the manifest will not detect that change.

Inference reconstructs the same model and loads `best_model.pt`; it writes original-scale scores (`normalized * 5 + 5`) and computes SRCC, LCC, KTAU, and MSE for labeled validation/test CSVs. Validation SRCC selects the checkpoint. Test metrics are for reporting only and must not be used to select a checkpoint.

Custom inference is also supported with a metadata CSV containing the configured filename and caption columns, plus `--wav-dir`; the score column is optional. If labels are present for at least two pairs, metrics are included. A single unlabeled pair produces predictions without target fields or metrics. Resume checks reject changes to the train/validation CSV selectors, audio directory selector, column mapping, expected split counts, and early-stop patience; the dataset root itself may change to support Colab staging.

## Run locally

From the baseline repository root, install dependencies without replacing the existing CUDA PyTorch build:

```bash
python -m pip install -r requirements-expert3.txt
python train_expert3.py --check-data
python train_expert3.py --smoke-test
python train_expert3.py
```

The smoke test reads one genuine training batch of eight and performs forward, backward, finite-gradient checks, and one optimizer step; it does not start epochs. Override paths for a different machine:

```bash
python train_expert3.py \
  --data-root /path/to/XACLE_Challenge_2027_development_dataset \
  --m2d-checkpoint /path/to/checkpoint-30.pth \
  --output-dir /path/to/expert3_runs \
  --cache-dir /path/to/huggingface_cache
```

Resume from a run's last completed epoch:

```bash
python train_expert3.py --resume runs/expert3_moescore/<run_id>/last_checkpoint.pt
```

Evaluate the best validation-selected checkpoint or test split:

```bash
python inference_expert3.py \
  --config runs/expert3_moescore/<run_id>/config.json \
  --checkpoint runs/expert3_moescore/<run_id>/best_model.pt \
  --dataset validation

python inference_expert3.py \
  --config runs/expert3_moescore/<run_id>/config.json \
  --checkpoint runs/expert3_moescore/<run_id>/best_model.pt \
  --dataset test
```

## Google Colab

Open `notebooks/finetune_expert3_colab.ipynb` from the project in Colab. Mount Drive, edit the project/data/checkpoint paths, optionally stage the data/checkpoint to `/content`, and run cells in order. The notebook checks GPU and paths, validates the dataset, performs the batch-8 real-model smoke test, trains or resumes, and evaluates validation/test. Output directory remains on Drive so completed checkpoints and logs persist.

## Verification status and benchmark

The published MOESCORE Expert 3 validation reference is SRCC 0.6257, LCC 0.6438, KTAU 0.4496, MSE 3.7369. These are targets, not guaranteed outcomes, especially because the documented `encode_clap_audio()` substitution is unverified.

CPU unit tests cover module freezing, feature shapes/order, batch-eight loss/gradients, dataset resampling/padding, optimizer membership, and best/resume checkpoint round trips. This machine exposes a GTX 1650 with 4 GB VRAM, not a Colab T4; the vendored runtime dependencies are not installed in the local environment. Therefore no real M2D forward/backward, T4 batch-eight smoke, or full training run is claimed here. Run the notebook smoke test on the requested T4 before launching epochs.
