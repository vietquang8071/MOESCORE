# MOE27-E6 — Set B: LAION-CLAP + Whisper Residual Expert (Overview v2)

**Project:** XACLE 2027 / MOESCORE seven-expert MoE  
**Target development branch:** `dev/moe27-e6`  
**Integration branch:** `feature/xacle2027-7expert-moe`  
**Document type:** Architecture and implementation overview v2 — **not** an implementation/completion report  
**Reviewed against:** MOESCORE integration source as of 2026-11-18; XACLE 2027 task, dataset description, and rules.

**v2 decision (2026-11-18):** No `model_id` requirement. Train on **100% official train for exactly 30 epochs**; evaluate **official validation MSE after each epoch** and save the lowest-validation-MSE checkpoint. **No internal holdout and no early stopping.** Official test is excluded from checkpoint selection. The competition-rule caveat about using validation for model selection is documented below.

## 1. Purpose and scope

E6 is the **Set B residual expert**. It predicts the semantic alignment of an audio–caption pair by combining:

1. A frozen **Top2 LAION-CLAP** component, which provides a paired audio/text feature vector and an audio–text cosine score.
2. A frozen **Whisper-large-v2** feature extractor, which provides separate audio-encoder and caption-conditioned decoder representations.
3. A trainable **calibration + residual MLP** that predicts the final E6 score from those frozen features.

The architecture must remain one internal expert of the seven-expert system. **Do not train or fine-tune either pretrained backbone, introduce an internal gate, use an SVR, or substitute an ensemble of independent prediction models.** E6 returns a score and its complete, uncompressed feature vector; the **Global Gating Network** combines expert outputs later.

The original Top 2 Set B system used an SVR-style pathway. **MOE27-E6 intentionally replaces that downstream regression mechanism with a residual head**, while reusing the frozen feature-extraction components where their identity/contracts match.

### Status at time of writing

| Dependency or component | Status |
|---|---|
| `model/expert_adapters/top2_laion_adapter.py` | Already implemented for the Full9 pipeline |
| Full9 `top2_laion` output contract (`4D+1`) | Implemented; verified in existing synthetic/regression tests |
| `model/feature_extractors/whisper_top2.py` | Implemented and merged into Integration |
| `data/whisper_feature_cache.py` and extraction CLI | Implemented; Whisper tests previously passed 26/26 |
| E6 residual head, E6 joined dataset, E6 trainer, E6 adapter | **Not implemented as E6-specific modules yet** |
| Real Whisper-large-v2 checkpoint/extraction | **Not verified in the reported local environment** |
| Verified, aligned real LAION/Whisper caches for E6 | **Not yet demonstrated** |
| Seven-expert registration/gating integration for E6 | **Pending Core integration** |

The `dev/moe27-e6` worktree must first be confirmed to contain the current Integration changes. Synchronizing branch histories is a prerequisite, not proof of E6 implementation.

## 2. Exact feature architecture

### 2.1 Frozen inputs

| Component | Embedding / output | E6 contribution | Source |
|---|---|---:|---|
| Top2 LAION-CLAP | Audio embedding `a_L` and text embedding `t_L`, each `[B,512]` | `[B,2049]` | Existing Full9 `top2_laion` output |
| Whisper audio encoder | Sequence mean of `encoder_last_hidden_state` | `[B,1280]` | `WhisperTop2FeatureExtractor` / verified Whisper cache |
| Whisper caption-conditioned decoder | Sequence mean of decoder `last_hidden_state` | `[B,1280]` | Same Whisper forward/cache as audio |
| **Concatenation** | LAION → Whisper audio → Whisper text | **`[B,4609]`** | E6 feature assembly |

LAION uses the shared MOESCORE five-block `4D+1` contract, with `D=512`:

```python
laion_features = concat(
    laion_audio,                       # [B,512] original audio embeddings
    laion_text,                        # [B,512] original text embeddings
    abs(l2norm(laion_audio) - l2norm(laion_text)),
    l2norm(laion_audio) * l2norm(laion_text),
    cosine(laion_audio, laion_text),   # [B,1]
)                                        # [B,2049]

features6 = concat(
    laion_features,                    # [B,2049]
    whisper_audio,                     # [B,1280]
    whisper_text,                      # [B,1280]
)                                        # [B,4609]
```

**Preserve the first two LAION blocks as emitted by the shared 4D+1 helper; normalize only the interaction blocks. Do not invent an audio–text cosine between Whisper encoder and decoder vectors.** They are distinct Whisper representations rather than a verified contrastively aligned pair. Do not add a learned feature projection or compression before returning `features6`.

### 2.2 Score equation

Let `s_L: [B,1]` be the cosine score returned by `Top2LAIONAdapter` (not a score already converted to the raw 0–10 scale). Then:

```text
z6 = C_B(s_L) + R_B(features6)
   = scale_B * s_L + bias_B + residual_mlp_B(features6)
```

- `z6` is the **normalized E6 prediction** on the challenge's centered target scale; the train target is `z_target = (target_raw - 5.0) / 5.0`.
- At the model boundary, return `score6 = z6` with shape `[B,1]`, float32. If a raw 0–10 score is needed, use `score_raw = 5.0 * score6 + 5.0` at the explicit conversion boundary, consistent with the chosen training target. Do not silently mix raw and normalized scores in the gate.
- **Recommended E5-consistent initialization:** `scale_B=1`, `bias_B=0`, last residual layer weights/biases initialized to zero, so the initial prediction is exactly LAION cosine. This is a proposed E6 implementation default, not yet verified E6 code.
- **Proposed configurable residual architecture:** `Linear(4609,256) → LayerNorm(256) → GELU → Dropout(0.2) → Linear(256,64) → GELU → Linear(64,1)`. Confirm through implementation tests; no E6-specific hidden-size code exists yet.

E6 output contract:

```python
score6, features6 = expert6(batch)
assert score6.shape == (B, 1)
assert features6.shape == (B, 4609)
```

## 3. Audio and caption preprocessing

### LAION-CLAP (`top2_laion`)

- Input key: `expert1_wavs`.
- Input waveform: `float32 [B,1,480000]`, **48 kHz**, maximum 10 s, via the unified dataset preprocessing.
- Model-provided projected audio/text embeddings have dimension 512 in the current Full9 feature contract.
- Use **the identical checkpoint, model variant, processor/native API choice, preprocessing and cache identity** already used to construct the accepted Full9 `top2_laion` output. The adapter supports different model-loading paths; a matching vector shape alone does not establish model parity.
- The existing adapter uses `build_paired_cross_features`; **reuse that output**, rather than reconstructing LAION features in a new extraction pipeline.

### Whisper (`openai/whisper-large-v2`)

- Input key: `expert3_wavs`.
- Waveform: `float32 [B,1,160000]`, **16 kHz**, maximum 10 s.
- Original caption field: `captions_raw`. Preserve source caption text, including whitespace, as required by the current extractor.
- Processor: `AutoFeatureExtractor`; tokenizer: `WhisperTokenizer`; model: `WhisperModel`.
- Feature extractor produces `[B,80,3000]` log-Mel input features.
- Tokenizer uses `padding=True`; caption token IDs are fed to `decoder_input_ids` with the audio input, without transcription or `generate()`.
- Return **unmasked sequence mean pooling**, including padding positions, from both encoder and decoder. These are 1,280-dimensional each.
- The backbone remains frozen and in `eval()` under inference mode.
- **Important:** decoder mean depends on batch-local token padding. For a reproducible first real extraction, prefer batch size **1**; cache preprocessing identity binds batch size, device, dtype, model snapshot, and source preprocessing. Inference must use consistent cached features or a contractually equivalent extraction policy.

**Source fidelity caveat:** The implemented MOESCORE LAION path may differ from the original Top 2 LAION source's feature selection/model variant. E6 deliberately consumes the verified Full9 LAION output and must not claim exact numerical parity with the original Top 2 implementation without a same-checkpoint, same-input numerical comparison.

## 4. Cache and data provenance

### 4.1 Intended source of each feature

- **LAION:** use `top2_laion` score `[B,1]` and features `[B,2049]` from an existing **verified Full9 feature store** when compatible with the required LAION checkpoint/preprocessing identities. Source row IDs, original captions, WAV hashes, and component cache identity are already recorded in the Full9 provenance.
- **Whisper:** use the separate verified cache under `<whisper_cache_root>/<split>/`, with `whisper_audio_features [N,1280]`, `whisper_text_features [N,1280]`, ordered `sample_ids`, source checkpoint identity, preprocessing/dataset identity, and shard SHA-256 hashes.
- **E6 joined dataset:** construct a read-only view over both sources; do not duplicate backbone weights or re-extract a frozen component when compatible features are available.

### 4.2 Fail-closed join (mandatory)

For each split (`train`, `validation`, `test`), the E6 reader must:

1. Validate the Full9 store using the existing store validator; require `full_9`, current `4D+1` contract and `top2_laion` width 2049.
2. Validate the Whisper cache manifest and all shards; require Whisper-large-v2, float32 finite `[N,1280]` audio/text, complete split and expected ordered IDs.
3. Require **exact** split and ordered `sample_id` equality. No silent dropping, filling, sorting, deduplication, or cross-split matching.
4. Cross-check source WAV SHA-256, exact caption identity, and row order between the underlying official metadata/Full9 source manifest and the Whisper dataset identity. Because the Whisper dataset digest is built from WAV **bytes** and caption **bytes**, validate it from the same source records/files or other cryptographically equivalent verified evidence, not by comparing unrelated digests by string alone.
5. Check LAION and Whisper checkpoint, preprocessing and cache identities; reject stale, partially extracted, wrong-version, corrupt or mixed-identity caches.
6. Read no validation/test labels for fitting; omit test labels entirely from prediction records.
7. Preserve the native source split and waveform identity in every output row and the resulting E6 checkpoint.

**Current integration gap:** The existing Whisper cache has a `WhisperFeatureCacheWriter.load()` interface, not a dedicated E6 joined dataset. Full9 already has `MoeFeatureStoreDataset`; implement an E6-specific reader/view and avoid changing existing Full9 or Whisper identities in place.

### 4.3 Official split policy (v2: no internal holdout)

**Explicit project decision:** For E6 v2, do not use `model_id`/`model_name` as grouping or training-selection keys. The source dataset may contain those fields; E6 does not need to read, join, validate, or persist them. **This does not remove mandatory sample, audio, caption, checkpoint, or cache provenance validation.**

- **Official train:** all **7,500 audio-text pairs** are eligible for optimizer updates; no 85:15 train/holdout split, no SHA256-based sample-level split, no train-data exclusion.
- **Official validation:** all **3,000 pairs** are evaluated after each epoch to compute the predeclared checkpoint-selection metric, validation MSE. The validation set is **never backpropagated**, placed in the optimizer's DataLoader, or used for feature/backbone fine-tuning.
- **Official test:** reserved for final, independent evaluation/inference after selection. Never use test labels for a loss, checkpoint selection, early stopping, or training adjustments.
- Keep all three splits distinct. Reject duplicate, reordered or mismatched IDs **within each joined Full9/Whisper split**, plus any train/validation/test identity overlap. Cache validation and checksums remain required even though no holdout is created.

The train and validation caches must be produced from their corresponding official splits with consistent feature algorithms/identities, but **their dataset/store identities will naturally differ because they are different splits**. Only corresponding LAION/Whisper rows *within* the same split must align exactly.

## 5. Training, validation and checkpoint selection — chosen v2 protocol

### 5.1 Fixed training configuration

| Field | v2 contract | Notes |
|---|---|---|
| Training split | **100% official train** | No internal holdout |
| Validation split | **Official validation** | Eval only; has labels, no gradients |
| Epoch count | **30 completed epochs** | No early stopping; `epochs=30` means total, not 30 additional on resume |
| Backbone policy | Frozen LAION and frozen Whisper | Prefer verified caches; neither backbone is optimized |
| Trainable weights | `scale_B`, `bias_B`, residual MLP | E6 head only |
| Target transform | `z_target = (label_raw - 5.0) / 5.0` | From [0,10] to [-1,1] |
| Training loss | **MSE** on normalized target | Mean over the current training batch |
| Optimizer | **AdamW** | Proposed `lr=1e-3`, `weight_decay=1e-4` |
| Example batch size | **8** | E6 head/cache training; distinct from Whisper extraction batch size |
| Selection metric | **Official validation MSE** on normalized targets | Evaluate after every epoch |
| Best rule | Strictly lower validation MSE | On ties, keep the earlier checkpoint; no `min_delta` |
| Early stopping | **Disabled** | Do not stop before epoch 30 |
| Seed | **42** | Proposed reproducibility default; record in checkpoint |
| Model-ID grouping | **Not used** | Deferred |

These hyperparameters are proposed E6 defaults unless the implementation deliberately changes them with a documented, approved rationale. The architecture and 30-epoch/validation-selection protocol are agreed; an E6 training implementation has **not yet been verified**.

### 5.2 Mathematical objective

For training example `i`, normalized target and prediction are:

```text
z_target_i = (MOS_raw_i - 5.0) / 5.0
z_pred_i   = scale_B * laion_cosine_i + bias_B + residual_mlp_B(features6_i)
```

The optimization objective for an E6 training minibatch of size `B` is:

```text
L_train = (1/B) * sum_i (z_pred_i - z_target_i)^2
```

**Validation MSE**, computed *only in evaluation mode* over all `N_val` validation samples, is:

```text
MSE_val(epoch) = (1/N_val) * sum_i (z_pred_i(epoch) - z_target_i)^2
best_epoch = argmin_{epoch = 1,...,30} MSE_val(epoch)
```

Implementation: accumulate **sum of squared errors and total sample count** across validation batches (`reduction='sum'`) then divide once by `N_val`; this avoids giving a short final batch the same weight as a full batch. Do not compute the average of per-batch mean MSE unless weighted by batch sizes. Use `model.eval()`, `torch.inference_mode()`, a deterministic, non-shuffled validation loader, and no optimizer actions in validation. Evaluate train MSE separately in eval mode if reporting a comparable train/validation curve; batch training losses are not equivalent to epoch-end full-train MSE.

A lower validation MSE selects `expert6_best.pt` even if training MSE rises or falls differently. This is **model selection based on validation labels**, not merely a passive test report; see the rules caveat in §5.5.

### 5.3 Required 30-epoch lifecycle

```python
best_val_mse = float('inf')
best_epoch = None

for epoch in range(resume_epoch, 30):
    train_head_one_epoch(train_loader)       # official train only
    train_mse = evaluate_mse(train_loader)   # eval-mode metric; optional but recommended
    val_mse = evaluate_mse(validation_loader)  # NO gradients, NO optimizer

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_epoch = epoch + 1
        save_best_checkpoint(epoch + 1, best_val_mse)

    save_latest_checkpoint(epoch + 1, best_val_mse, best_epoch)

save_final_checkpoint(epoch=30)               # last-epoch weights, NOT a copy of best
```

Above is **illustrative pseudocode**, not a runnable implementation. Training must complete 30 epochs unless interrupted by an error/user cancellation. For a clean run, a non-finite train or validation MSE is a hard failure, never treated as a valid best value. All reported metrics must specify the normalized target scale (and any optional raw-scale conversion).

### 5.4 Checkpoint roles, provenance and resume

| Filename | Required contents/meaning | Normal use |
|---|---|---|
| `expert6_best.pt` | E6 head weights at the **lowest official validation MSE** epoch, with `best_epoch`, `best_val_mse`, selection criterion and provenance | **Preferred for Full7 E6 component/inference**, subject to rules confirmation |
| `expert6_latest.pt` | E6 head + optimizer and full resumable state at **most recently completed** epoch; best-state metadata, progress and metrics | Resume only |
| `expert6_final.pt` | E6 head weights from **epoch 30** whether or not it is best; explicitly marked `final_epoch_30` | Comparison/reproducibility |

Each checkpoint must record the E6 feature/architecture contract (`[B,4609]`), score scale, LAION/Whisper checkpoint and cache identities, train/validation dataset provenance, seed, optimizer/config, epoch/global step, train/validation metrics, and selection policy (`official_validation_mse_min_v1`). **No LAION or Whisper backbone weights belong inside an E6 checkpoint.** If the checkpoint contains the optimizer state for reproducibility, that is an E6-head optimizer only.

Resume must restore E6 model, optimizer, progress, seed, best metric/epoch and the provenance/identity of the sibling `best` checkpoint; it must reject changed training hyperparameters, an incompatible E6 architecture, altered train or validation dataset/cache identities, or a changed checkpoint-selection metric. `epochs=30` is the total target; resuming after epoch 12 performs epochs 13–30. Do not overwrite the best checkpoint based on a different validation dataset or a different Whisper batch-padding policy. Preserve deterministic ordering/seeds where practical and document any hardware-level nondeterminism.

**Unlike E5:** E5's current `expert5_final.pt` is a copy of *best-on-internal-holdout*. For **E6 v2**, `expert6_final.pt` is intentionally **epoch 30**, and `expert6_best.pt` is **best-on-official-validation**. Do not reuse the E5 trainer unchanged or conflate the meanings.

### 5.5 Competition-rule interpretation and validation-selection caveat

The project-supplied `Task-rules.txt` states:

> 1) Only the training data can be used for training models.  
> 2) Only the validation and test data can be used for evaluating models.

The project has **chosen** official validation MSE for E6 checkpoint selection. This satisfies the technical distinction that **gradient updates occur only on official train** and official validation runs only in `eval()` without gradients. **However, selecting the best checkpoint using validation labels is itself model selection**; the supplied rules do **not explicitly clarify whether that selection is permitted** under the phrase "only ... for evaluating models." Therefore, **obtain written organizer confirmation before using the validation-selected `expert6_best.pt` in a competition submission**, or adopt a compliant alternative if instructed. Do not claim that absence of gradients alone resolves this ambiguity.

The same task rules also require organizer approval/disclosure for external data and pretrained models; existing frozen checkpoints are not automatically exempt. The external-model request deadline in the supplied rules is **2026-10-14**, so confirm what approvals have already been obtained rather than assuming new approval is available. Do not use test labels to select checkpoints. No new audio generation or separate prediction-model ensembling is introduced.

## 6. E6 adapter and Full7 integration

A proposed E6 adapter input schema is:

```python
batch = {
    "component_outputs": {
        "top2_laion": {
            "score": laion_cosine,       # float32 [B,1]
            "features": laion_features,   # float32 [B,2049]
        },
    },
    "whisper_audio": whisper_audio,     # float32 [B,1280]
    "whisper_text": whisper_text,       # float32 [B,1280]
}
score6, features6 = expert6(batch)      # [B,1], [B,4609]
```

This **E6 schema is a recommended future interface**, not an already registered or verified API. The implementation should reject incorrect shapes, mismatched batch size/device/dtype, nonfinite tensors, unsupported LAION feature representation, and inconsistent LAION cosine scalar. Its component inputs are detached; gradients are restricted to the E6 residual/calibration head.

Full7 Core must later:

1. Construct E6 from its selected trained checkpoint and provenance.
2. Provide a coherent LAION + Whisper feature batch (cached or extracted with compatible identities).
3. Insert the E6 score and full feature vector in the **agreed E1→E7 order**, at width 4609.
4. Extend expert registry/assembly/gating dimensions once as a **separate integration change**, rather than modifying the legacy `top1_4` or `full_9` feature-store contracts prematurely.

There is **no internal E6 gate, no E6 SVR, and no Whisper cosine scoring head**. The single-model Global Gating Network remains the sole place that combines E1–E7 predictions.

## 7. Proposed implementation breakdown

These paths are **proposed files**, not proof they already exist:

| Stage | Suggested work | Acceptance criteria |
|---|---|---|
| **E6.1 — Contracts/head** | `model/composite_experts/set_b_residual.py`, `model/expert_adapters/set_b_residual_adapter.py` | Exact score equation; 2049+1280+1280=4609; zero residual initialization; strict float32/device/shape/finite checks |
| **E6.2 — Aligned cache reader** | `data/set_b_cached_dataset.py` with verified Full9 + Whisper alignment | Full9 LAION and Whisper caches join fail-closed; no accidental train/validation/test leakage; sample-ID alignment and split integrity |
| **E6.3 — Training/checkpoints** | `training/set_b_config.py`, `training/set_b_trainer.py`, `scripts/train_set_b_residual.py`, example YAML | 100% official-train MSE for 30 epochs; official-validation MSE best selection; no holdout/early stopping; best/latest/final and strict resume |
| **E6.4 — Inference/verification** | `scripts/infer_set_b_residual.py` and E6 tests/docs | CPU synthetic end-to-end, decimal labels, split/cache mismatch, 30-epoch validation selection and resume, `device: auto` |
| **E6.5 — Real-cache gate** | Real Whisper-large-v2 smoke/extraction plus verified matching Full9 LAION cache | End-to-end read-only join on real samples; exact sample/hash/identity consistency and inference checks |
| **E6.6 — Full7 wiring** | Dedicated Core integration change | E6 `[B,1] + [B,4609]` wired into E1→E7 without breaking legacy Full9/4-expert code |

### Minimum automated test matrix

- Feature order, dimensions, raw LAION blocks and helper-derived `4D+1` interactions; identical LAION cosine and final block.
- Whisper is taken as two independent 1280-D feature vectors; no fabricated cross-cosine, transcription or ASR loss.
- Baseline exactly equals LAION cosine at initialization; calibrator and residual receive gradients while component tensors/backbones stay frozen.
- Batch sizes 1 and >1; invalid widths, dtypes, nonfinite values, mixed devices and missing/extra keys fail.
- Reject misordered/missing/duplicated sample IDs, mismatched captions/WAV hashes, split, checkpoint identity, stale Whisper batch-preprocessing identity and incomplete caches.
- Use **all official train** rows for optimizer updates; use **official validation** labels only in evaluation-mode MSE; enforce no validation optimizer/backward calls and no test leakage.
- Exactly 30 epochs without early stopping; checkpoint selection by minimum official-validation MSE (including unequal final batch sizes); best/latest/final-epoch separation, tie handling, exact resume protections, and normalized↔raw score conversion.
- CPU synthetic Full9 + Whisper cache round trip, train, resume, inference; then real-cache smoke when resources exist.

## 8. Preconditions, blockers and non-goals

**Before implementation:** confirm the E6 branch has merged the current Integration HEAD and passes the applicable existing Whisper/Full9/E5 regressions. Never develop E6 against stale `4D+1` or pre-Whisper contracts.

**Before real extraction:** obtain a complete, permitted `openai/whisper-large-v2` checkpoint and matching tokenizer/feature-extractor assets; have a verified Full9 LAION cache of the appropriate split, official metadata, and enough storage/VRAM. Start with batch size 1 for Whisper and preserve its extraction identity.

**Before declaring E6 ready for Full7 runtime:** require a successful real-cache alignment smoke test, completed 30-epoch E6 training on official train, and a verified official-validation selection record; no synthetic-only test can establish this. For competition submission, also resolve the rules caveat in §5.5.

**Non-goals:** training either backbone; creating a new LAION variant to mimic source embeddings by width alone; recalculating Whisper cosine; standalone SVR; internal E6 gating; changing E5/E7 or global registry in the E6 feature branch; using official validation examples for gradient updates; using test labels for selection; implementing any model-ID split; copying large pretrained weights into Git.

## 9. Source anchors

- `model/expert_adapters/top2_laion_adapter.py` — existing LAION model and `4D+1` feature creation.
- `model/expert_adapters/paired_cross_features.py` — shared LAION five-block contract.
- `model/expert_adapters/registry.py` — `full_9` `top2_laion=2049`, 48 kHz waveform contract.
- `model/feature_extractors/whisper_top2.py` — exact Whisper forward and `[1280,1280]` pooling contract.
- `data/whisper_feature_cache.py` and `scripts/extract_whisper_features.py` — cache identity, resumability and CLI.
- `docs/MOE27-Whisper-implementation.md` — extracted Whisper status and source fidelity notes.
- `data/moe_feature_store.py` and `scripts/extract_moe_feature_store.py` — Full9 store and source-manifest provenance.
- `data/unified_dataset.py` — source sample IDs, captions and waveform preprocessing for cache alignment.
- `training/set_a_trainer.py`, `docs/MOE27-E5-implementation.md` — implementation reference for head-only MSE, checkpoint metadata and resume; **E6 uses different official-validation selection and final-checkpoint semantics**.
- Project-provided `Data-2027-description.txt`, `Task-description.txt`, `Task-rules.txt` — official dataset framing, unseen-generator objective and competition constraints as provided.

**Interpretation boundary:** Sections 1–3 summarize the agreed E6 direction and observed component implementations; Sections 4–8 specify the **chosen v2 protocol** and a **not-yet-implemented** E6 design/acceptance plan. This file does not claim that E6 training, real-cache verification or Full7 wiring already exists. Written clarification on validation-based selection is still recommended before competition submission.
