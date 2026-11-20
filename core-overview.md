# MOE27-Core — Full7 Global Gating Integration

**Project:** XACLE Challenge 2027 / MOESCORE  
**Document:** `core-overview.md`  
**Status:** **Core Design V1 — design frozen; Core.1–Core.5 implementation synthetically verified; real-data verification pending**
**Implementation annotation (2026-11-19):** Core.1–Core.5 are implemented and synthetically verified in this worktree; real-data verification, official training, measured validation results, and competition approvals remain pending. See `docs/MOE27-Core-implementation.md` and `MOE27-Core-report.md`.
**Date:** 2026-11-19  
**Integration baseline:** `vietquang8071/MOESCORE`, branch `feature/xacle2027-7expert-moe`, reviewed at commit `834222e2fcde378db51fc902270bd9235553ab72`  
**Intended implementation worktree:** `MOESCORE-Core`, intended feature branch `dev/moe27-core` (remote branch creation/push **not yet verified**).  
**Scope:** Integrate the seven already specified experts behind **one** Global Gating Network, without retraining or redesigning the underlying experts.

> **Status convention:** **FROZEN** = a V1 implementation decision; **EXISTING** = observed behavior in the reviewed repository; **PLANNED** = not yet implemented in Full7; **BLOCKED/VERIFY** = evidence or external approval required. This document is an implementation specification, **not** a claim that Full7 has been trained or evaluated on real data.

---

## 1. Purpose, scope, and non-goals

### 1.1 Objective — FROZEN

Produce an auditable, offline-cache-based **seven-expert** MOESCORE model. For an audio–text pair, each frozen expert returns a single scalar score and an uncompressed feature vector. A single trainable Global Gating Network outputs seven sample-dependent mixture weights and produces a final prediction by their weighted sum.

The V1 design must:

1. Preserve the seven experts' released/fine-tuned architectures and checkpoint/feature contracts.
2. Keep **all** expert/backbone and E5/E6 residual-head parameters frozen during Global Gate training.
3. Consume verified, precomputed feature stores, not run seven large backbones inside the gate's training loop.
4. Add a new, explicit `full_7` profile without changing behavior of `top1_4` or `full_9`.
5. Preserve the existing MOESCORE gate architecture, configured losses, optimizer, scheduler, validation metrics, and checkpoint semantics, except for explicit, tested Full7 schema extensions.
6. Reuse existing strict SHA-256/provenance, atomic-write, resume, and label-boundary mechanisms.
7. Keep audio generation `model_id` / `model_name` **out of the gate inputs and routing** in V1, to emphasize generalization to unseen generators.
8. Report unresolved data, score-scale, training-leakage, and competition-rule risks honestly.

### 1.2 Non-goals — FROZEN

- No new expert, auxiliary scoring head, SVR, per-expert post-hoc calibrator, or second global ensemble.
- No backbone fine-tuning or training of E5/E6 inside the gate's optimizer.
- No feature compression/PCA or extra feature normalization in V1; reuse the exact frozen adapter features.
- No model-ID-conditioned routing, leakage through labels/test metadata, or automatic remapping of official splits.
- No re-extraction of Full9 or Whisper features during the gate training loop.
- No use of official validation/test examples for gradient updates.
- No assumption that synthetic tests validate actual backbone numerical parity or that pretrained model use has been approved.

---

## 2. Seven-expert feature contract — FROZEN

Every Full7 component has the interface:

```python
score_i:    torch.Tensor  # float32 [B, 1], normalized-target-compatible score contract
features_i: torch.Tensor  # float32 [B, D_i]
```

For all components, `B > 0`, tensors are finite and share a device, dtype, sample membership, and row ordering.

| Position | Full7 name | Source / construction | Feature formula | Dimension |
|---:|---|---|---|---:|
| 1 | `expert1` (E1) | LAION-CLAP fine-tuned Top1 adapter | `4 × 512 + 1` | **2,049** |
| 2 | `expert2` (E2) | MGA-CLAP Top1 adapter, post-MSC | `4 × 1024 + 1` | **4,097** |
| 3 | `expert3` (E3) | M2D-CLAP fine-tuned Top1 adapter, released 5D feature contract | `5 × 768` | **3,840** |
| 4 | `expert4` (E4) | BEATs / SeqCoAttn bidirectional pooled fusion | `512 + 512` | **1,024** |
| 5 | `expert5` (E5) | Set A frozen residual head over Top2 M2D/MS-CLAP/MGA | `3,073 + 4,097 + 4,097` | **11,267** |
| 6 | `expert6` (E6) | Set B frozen residual head over Top2 LAION and Whisper audio/text | `2,049 + 1,280 + 1,280` | **4,609** |
| 7 | `expert7` (E7) | Existing `m2d_lora` output, relabeled at the Full7 assembly boundary | `4 × 768 + 1` | **3,073** |
| | **TOTAL** | | | **29,959** |

The Full7 registry key for E7 is **`expert7`**; its existing **Full9 source key** remains **`m2d_lora`**. This name mapping must be explicit and deterministic; do not rename the existing Full9 expert or change its source extraction contract.

**Exact fixed assembly order:**

```python
FULL7_EXPERT_ORDER = (
    "expert1", "expert2", "expert3", "expert4",
    "expert5", "expert6", "expert7",
)
FULL7_FEATURE_DIMS = {
    "expert1": 2049, "expert2": 4097, "expert3": 3840,
    "expert4": 1024, "expert5": 11267, "expert6": 4609,
    "expert7": 3073,
}
FULL7_FEATURE_WIDTH = 29959
```

**Implementation note:** these constants are an illustrative specification, not yet checked-in code. Where possible, derive and assert E5/E6 dimensions against `feature_order_contract()` in their source modules to avoid drifting magic numbers.

### 2.1 Why these dimensions are authoritative — EXISTING

- **E1/E2:** `build_paired_cross_features()` concatenates `[audio, text, |L2(audio)-L2(text)|, L2(audio) * L2(text), cosine]` for `4D+1`; the first two blocks are retained in their original float32 value scale. E2 preserves its original MGA model's score rather than automatically replacing it with the appended cosine feature.
- **E3:** source forward concatenates five 768-wide blocks, with the final scalar similarity **repeated to 768 positions**; the released `concat_assignment="released"` ordering is part of its checkpoint contract. E3 is **not** interchangeable with a `4D+1` M2D representation.
- **E4:** `SeqCoAttn(dim=512)` pools bidirectional cross-attention outputs and concatenates two 512-wide vectors.
- **E5:** `SetAResidualAdapter` concatenates verified Full9 Top2 `top2_m2d`, `top2_msclap`, `top2_mga` feature vectors. Its head predicts `scale × mean(component_scores) + bias + residual_mlp(features5)`; the gate consumes the full 11,267 features, **not** a residual MLP hidden layer.
- **E6:** `SetBResidualAdapter` concatenates Full9 `top2_laion [2049]`, Whisper audio `[1280]`, and Whisper caption/text `[1280]`; its head predicts `scale × laion_score + bias + residual_mlp(features6)`. There is **no** Whisper cosine head.
- **E7:** the existing M2D-LoRA adapter uses a standardized 768-dimensional audio/text interaction `4D+1` vector. Its final interaction scalar is a standardized dot-product, **not automatically an L2 cosine**; its score is converted explicitly from raw to MOESCORE target scale.

All expert scores are `[B,1]`; the **dimension column describes only the feature vectors**.

---

## 3. Global Gating Network — FROZEN

### 3.1 Inputs, network, and mixture

```text
Verified frozen E1 ... E7 outputs (same sample order)
        ├── scores   S: float32 [B, 7]
        └── features F: float32 [B, 29959]
                         │
                     Linear(29959, 512)
                         │
                     LayerNorm(512)
                         │
                         ReLU
                         │
                     Dropout(0.1)
                         │
                     Linear(512, 7)
                         │
                     Softmax(dim=-1)
                         │
                mixture weights W: [B, 7]
                         │
           prediction = (S * W).sum(dim=1, keepdim=True)
                         │
                score_normalized: [B,1]
                         │
             score_raw = 5 * score_normalized + 5
```

Mathematically:

\[
F = [F_1;\ldots;F_7]\in\mathbb{R}^{B\times29959},\quad
S = [s_1,\ldots,s_7]\in\mathbb{R}^{B\times7},
\]
\[
W=\operatorname{softmax}(\operatorname{Gate}(F)),\quad
\hat y_{\rm norm}=\sum_{i=1}^7 W_i s_i,\quad
\hat y_{\rm raw}=5\hat y_{\rm norm}+5.
\]

The **gate sees only feature vectors**; scores are used in the final convex combination. Do not append model ID, labels, seven scores, or hidden metadata to `F`.

### 3.2 Invariants

- `weights.shape == (B,7)`, `weights >= 0`, and `weights.sum(-1) ≈ 1` for every row.
- `pred_normalized.shape == (B,1)` and `torch.isfinite(pred_normalized).all()`.
- Output features/scores are detached from every frozen expert head/backbone; gate optimizer has **only** gate parameters.
- `eval()`/`train()` mode affects the gate as intended; no expert is switched to train mode during the gate training path.
- Preserve the existing gate's `Linear → LayerNorm → ReLU → Dropout → Linear → Softmax` mathematics; no hidden gating modifications.
- **Do not clamp** normalized predictions to `[-1,1]` during training, validation, or test. Enforce any official output formatting/range requirements only at a separately documented submission boundary if required; do not silently change the learned prediction.

---

## 4. Score-scale audit and contract — FROZEN POLICY; REAL DATA VERIFY

### 4.1 Target and output scale

Official average human score is on the raw **0–10** scale. The established gate training target transform is:

```python
y_normalized = (y_raw - 5.0) / 5.0
y_raw = 5.0 * y_normalized + 5.0
```

**Do not conflate score semantics and numeric range.** A cosine score may be approximately within `[-1,1]` without being calibrated to the human target; a regression head may output values outside `[-1,1]` even if trained on normalized labels. Do not apply a blind MinMaxScaler, StandardScaler, sigmoid, tanh, clipping, or second calibrator to expert scores in V1.

### 4.2 Score-origin ledger

| Expert | Current score source | V1 action | Required audit evidence |
|---|---|---|---|
| E1 | CLAP audio/text dot-product from original fine-tuned encoder | Preserve adapter output | Verify embedding normalization and resulting distribution |
| E2 | MGA-CLAP original source cosine/post-MSC score | Preserve adapter output | Verify consistency with original score semantics and feature blocks |
| E3 | Fine-tuned M2D regression-head score | Preserve **only after scale is established** | Inspect training target transform/checkpoint contract; test actual outputs |
| E4 | SeqCoAttn regression head using `tanh` | Preserve adapter output | Verify target semantics, finite outputs |
| E5 | Set A affine mean-cosine + residual MLP trained with normalized targets | Preserve trained normalized output | Verify E5 checkpoint, no double conversion |
| E6 | Set B affine LAION cosine + residual MLP trained with normalized targets | Preserve trained normalized output | Verify E6 checkpoint, no double conversion |
| E7 | M2D-LoRA regression/CORAL raw prediction converted with `(score_raw-5)/5` | Preserve converted output | Verify checkpoint and score selection mode |

**Compatibility rule:** the Full7 builder must record an explicit per-expert score contract (`source`, `target_scale`, `conversion`, `checkpoint/cache fingerprint`, `audit_status`). It must reject unknown, contradictory, raw-vs-normalized, or missing scale metadata **instead of guessing from observed min/max**.

- If source evidence proves an expert output is **raw 0–10**, convert **exactly once** at the named source/assembly boundary and record the transformation in provenance. This is a contract correction, not a learned calibrator.
- If source evidence proves that an expert output is already on the normalized prediction scale, preserve it.
- If an expert yields cosine similarity, preserve its declared cosine semantics but record that it is an **uncalibrated similarity score**; the audit must explicitly establish whether the existing gate baseline can use it as a score proxy. Equal numeric range does **not** certify equivalent calibration.
- If E3's actual score target cannot be established, mark real Full7 training **BLOCKED**, do not silently transform E3.

### 4.3 Preflight diagnostics (training split only for distribution analysis)

For each expert on **official train**, record: sample count, min/max, mean/std, quantiles (e.g. p01/p50/p99), non-finite count, proportion outside `[-1,1]`, and (when labels available) prediction/target diagnostics. This is an **audit**, not a fitted scaler or a new model. Record configuration, source revision, expert checkpoint SHA-256, cache identity, and output hash.

Validation/test must **not** be used to fit any calibrator or change scale rules. Validation can be used only for authorized evaluation under competition rules. Real-score acceptance is contingent on documented score semantics, not on passing a `[-1,1]` range check.

---

## 5. Offline Full7 data construction — FROZEN

### 5.1 Inputs and reuse

**Required read-only input artifacts:**

1. Verified **Full9** feature-store shards and source manifest, split by `train`, `validation`, `test`.
2. Verified **Whisper large-v2** feature-cache shards and manifests for matching splits.
3. A trusted, compatible **E5** head checkpoint (default intended: `expert5_best.pt`).
4. A trusted, compatible **E6** head checkpoint (default intended: `expert6_best.pt`).
5. The verified source manifest/label provenance and exact Full7 code/score-contract version.

The Full9 store supplies all required raw component outputs except Whisper:

```text
Full9 source key    -> Full7 destination
expert1             -> expert1
expert2             -> expert2
expert3             -> expert3
expert4             -> expert4
m2d_lora            -> expert7

top2_m2d           -┐
top2_msclap          ├-> SetAResidualAdapter + E5 checkpoint -> expert5
top2_mga            -┘

top2_laion          -┐
Whisper audio        ├-> SetBResidualAdapter + E6 checkpoint -> expert6
Whisper caption text-┘
```

**Do not** substitute the Top1 E1 LAION vector for the `top2_laion` E6 input, or Top1 E3 M2D for the `top2_m2d` E5 input. Same-family naming does not imply interchangeable checkpoint or preprocessing provenance.

### 5.2 Per-split build algorithm

For each split, with **no gradient** and all frozen heads in `eval()`:

1. Verify the Full9 store and `full_9` contract using existing validators. Read its verified source-manifest SHA-256, ordered IDs, filenames, WAV byte hashes, and original captions.
2. Verify the Whisper cache via existing reader and reconstruct/compare its dataset identity against the Full9 source manifest. Validate Whisper checkpoint, preprocessing, extraction-batch-padding identity, and shard hashes.
3. Verify the E5/E6 checkpoint **formats, architecture, feature order, normalization, SHA-256, and component model/cache identities** using existing loaders. Do not deserialize checkpoints from untrusted paths. Check compatibility with the target split's features.
4. Keep original row order. Reject missing, duplicated, extra, or reordered samples, plus mismatched caption/audio fingerprints. **Do not align by row index alone** if IDs differ and do not silently inner-join/drop samples.
5. Slice the Full9 score/feature matrices by the verified `FULL9_EXPERT_ORDER` and corresponding dimensions, not hard-coded byte offsets.
6. Construct E5 component batch using precisely `top2_m2d`, `top2_msclap`, and `top2_mga` `(score, features)` mappings. Compute `(score5, features5)` with the validated E5 adapter; do not recompute embeddings.
7. Construct E6 component batch using `top2_laion` `(score, features)`, Whisper audio `[B,1280]`, and Whisper text `[B,1280]`. Compute `(score6, features6)` with the validated E6 adapter.
8. Take E1–E4 and E7 `(score, features)` directly from Full9; assemble E1→E7 into `scores: [B,7]`, `features: [B,29959]`.
9. Run shape/dtype/device/finite, score-semantic, ordering, provenance, and feature-contract checks. Detach tensors and move arrays to CPU float32 before saving.
10. Write to a **new immutable Full7 store**, atomically, in resumable shards, recording per-shard hashes, source identities, code revision, E5/E6 checkpoint SHA-256 and provenance, and the score-contract manifest.
11. Independently re-open the store through its reader and verify every shard and identity before declaring success.

**Labels:** train and validation store the original raw `[0,10]` label in float32 and a `labels_scale="raw_0_to_10"` marker. The **test store must be label-free**, regardless of which metadata files are present in the dataset distribution. Do not use test scores for tuning or auditing performance.

### 5.3 Fail-closed integrity checks

- Split name, sample count, ordered sample IDs, ordered filenames, source WAV SHA-256, and raw-caption SHA-256 must agree where the joined sources are expected to agree.
- Full9 and Whisper row provenance must be consistent; E5/E6 checkpoint component identities must match the Full9/Whisper cache identities.
- Cross-split **extraction** identities may match; **split-specific** Full9 store/source-manifest hashes must not be required equal.
- E5/E6 checkpoint versions, expected architecture settings, and feature order must match the instantiated heads.
- Reject corrupted shard payloads, stale/mismatched caches, missing or wrong split labels, duplicated IDs, non-finite data, score-scale ambiguity, and unapproved schema changes.
- Preserve the E6 P1 checks for complete LAION cache identity when creating a Full7 store. WAV SHA-256 values are validated per source row, without comparing audio content across official splits.
- Write immutable outputs keyed by source/checkpoint/contract fingerprints; changed input identity means a **new output identity/path**, not silent reuse.

### 5.4 Stacking leakage — explicitly accepted V1 limitation

The E5/E6 heads may generate **in-sample predictions on official train** that was also used to optimize those heads. The gate trained on those predictions may overfit to train-specific residual patterns. V1 uses this straightforward frozen-prediction approach as **agreed**, with the following safeguards:

- Log which E5/E6 checkpoints were selected, their train membership, and whether each Full7 row is in-sample or out-of-sample with respect to the composite head.
- Do not present the V1 approach as leakage-free or claim cross-validation/OOF provenance.
- Never use validation/test labels as gate inputs.
- Reserve true **out-of-fold (OOF)** composite predictions and comparison against the V1 approach for a later, explicit ablation; **do not** automatically introduce OOF folds or retraining of E5/E6 in Core V1.

---

## 6. Full7 feature-store and registry schema — PLANNED

### 6.1 Registry compatibility

Add explicit profile `full_7` to `model/expert_adapters/registry.py` (or a tightly scoped companion contract module used by the registry) with exactly the order and widths in §2.

**Do not change:**

- `EXPERT_ORDER` / `top1_4` (four-expert legacy), its existing widths or default profile behavior.
- `FULL9_EXPERT_ORDER` / `full_9`, the nine original source-adapter names, source checkpoint identities, or extraction semantics.
- Existing `validate_and_assemble_expert_outputs()` behavior for `top1_4` and `full_9`.
- Any original training, checkpoint, or inference behavior for those legacy profiles.

The seven final outputs are **already computed** offline, so a full_7 assembly helper may accept a strict mapping of seven `(score, feature)` tuples and call a shared `validate_and_assemble_expert_outputs(profile="full_7")`. Do not require factories for `expert5`, `expert6`, or `expert7` unless the chosen API explicitly needs them; do not reinterpret the Full9 extractor as a Full7 extractor.

**Important current code constraint:** `data/moe_feature_store.py` currently has schema v1 for `top1_4` and v2 specifically for `full_9`. `training/checkpoint_manager.py` similarly has a v1/`full_9` v2 split. **Simply adding `full_7` to `EXPERT_PROFILES` would not produce valid Full7 caches/checkpoints.** Extend the schemas/version routing and verification deliberately (e.g., **v3** for `full_7`) or create a separate strongly validated Full7 cache format. Neither approach may weaken the existing v1/v2 checks.

### 6.2 Proposed Full7 store manifest fields

An example **conceptual schema**, not yet existing code:

```yaml
schema_version: 3
expert_profile: full_7
split: train  # train | validation | test
expert_order: [expert1, expert2, expert3, expert4, expert5, expert6, expert7]
feature_dimensions: {expert1: 2049, expert2: 4097, expert3: 3840, expert4: 1024,
                     expert5: 11267, expert6: 4609, expert7: 3073}
score_width: 7
feature_width: 29959
score_dtype: float32
feature_dtype: float32
labels_scale: raw_0_to_10  # absent on test
source_full9_store_identity: '<verified per-split identity>'
source_full9_manifest_sha256: '<per-split sha256>'
source_whisper_manifest_sha256: '<per-split sha256>'
source_sample_manifest_sha256: '<per-split sha256>'
full7_component_identities: '<E1-E7 checkpoint/cache identity mappings>'
score_contracts: '<seven declared score semantics and conversion policies>'
core_assembly_contract_sha256: '<canonical sha256>'
shards: '<ordered shard metadata and file sha256 values>'
manifest_sha256: '<canonical payload sha256>'
```

Requirements: make each recorded identity independently checkable; store actual canonical mappings/hashes, not placeholders. The E5/E6 checkpoint hash, architecture, and source/cache identity must be transitively bound to the Full7 store. **Per-split source identities differ; shared extraction/score policies must agree.**

### 6.3 Expected dataset batch contract

```python
batch = {
    "sample_ids": list[str],
    "wave_file_names": list[str],
    "split": "train",  # or validation / test
    "expert_profile": "full_7",
    "scores": torch.float32,   # [B, 7], ordered E1...E7
    "features": torch.float32, # [B, 29959], same order
    "labels_raw": torch.float32,  # [B], train and validation ONLY
    "labels_scale": "raw_0_to_10",  # train and validation ONLY
}
```

Optionally preserve validated `num_class` metadata for existing losses, but **do not enable class-balanced loss paths (`cbl`)** unless their shape/broadcast correctness is explicitly fixed and tested. V1 baseline config keeps `cbl: false`.

---

## 7. Training protocol — FROZEN

### 7.1 Split and parameter policy

- **Train:** official train only, using verified Full7 cached scores/features with raw train labels. Never update gate/expert parameters from official validation/test.
- **Validation:** official validation for evaluation; the existing gate's best-checkpoint policy is maximum SRCC. This policy, any validation-driven early stopping, and scheduler usage remain **competition-approval caveats** (see §11).
- **Test:** label-free inference only.
- **Frozen:** all E1–E7 scores/features generated from fixed, fingerprinted expert state; during gate training, no expert model is constructed or updated.
- **Trainable:** parameters belonging to the single `GatingNetwork` only; optimizer parameter-ID audit must confirm this.
- **Sampler / resume:** retain the current `StatefulRandomSampler`, seed and deterministic resume behavior. Current baseline uses `drop_last=True` in `scripts/train_moe_gating.py`; preserve this in V1 unless deliberately changed with a separate documented test/decision. Report how many train rows are actually used each epoch rather than implying `drop_last=True` consumes every row per epoch.

### 7.2 Baseline hyperparameters (from `configs/moe_cloud.example.yaml`)

| Item | Core V1 value / behavior |
|---|---|
| Gate topology | `29959 → 512 → LayerNorm → ReLU → Dropout(0.1) → 7 → Softmax` |
| Optimizer | `AdamW`, only gate parameters |
| Learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Batch size | `8` train, `8` validation; `64` inference |
| Maximum epochs | `50` |
| Seed | `2027` |
| Early-stop patience | `50` (existing gate baseline behavior) |
| Checkpoint interval | `1000` optimizer steps |
| Scheduler | `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)` on validation regression loss |
| Best checkpoint | highest **official validation raw-scale SRCC**, only if allowed by organizer |
| Resume | latest gate, includes optimizer, scheduler, sampler and RNG states |

Any hyperparameter ablation is a **separate experiment**, never an unrecorded implementation change. All values must be recorded in checkpoint/config provenance. The E5/E6 residual-head training protocols **do not** overwrite Core gating settings.

### 7.3 Objective

Match the existing `MoeGateTrainer` and loss configuration:

\[
\mathcal L_{\rm reg} = 1.0\,\mathcal L_{\rm ClippedMSE}
                    +0.5\,\mathcal L_{\rm Contrastive},
\]
\[
\mathcal L_{\rm total}=\mathcal L_{\rm reg}
       +0.2\,\mathcal L_{\rm diversity}.
\]

Baseline loss parameters:

```yaml
loss:
  loss_weights: [1.0, 0.5]
  clipped_mse: {tau: 0.25, mode: utt, beta: 0.99, cbl: false}
  contrastive: {margin: 0.1, beta: 0.99, cbl: false}
```

`L_diversity` is the existing normalized mixture-entropy penalty (`mean(1 - H(weights)/log(7))`), **not** a new auxiliary expert model. Compute regression targets only from `(labels_raw - 5)/5`. Preserve existing shape safeguards preventing `[B,1] × [B]` unintended broadcasting.

### 7.4 Evaluation and checkpoint selection

Evaluate without gradients over official validation in original row order. Report:

- SRCC (primary checkpoint selection metric **subject to approval**),
- LCC / Pearson correlation,
- KTAU / Kendall tau-b,
- MSE and MAE on raw `0–10` scores,
- optional loss breakdown and gate-weight diagnostics.

For undefined correlations (constant target/prediction), use the existing explicit undefined status instead of fabricating a score.

`best_gate.pt` follows a **strictly improving SRCC** policy, with ties retaining the earlier checkpoint. `latest_gate.pt` is resumable and stores actual training step/epoch, optimizer/scheduler/sampler/RNG state, training config, source identities, best metric, and selection history. Do not silently substitute an E5/E6 checkpoint or save backbone parameters in a gate checkpoint.

**Competition note:** The current gating implementation uses validation regression loss for scheduler stepping and validation SRCC for best selection/early stop. Both are validation-driven training-policy decisions. The submitted system must not assume those operations are permitted merely because validation does not backpropagate gradients; see §11.

### 7.5 Core checkpoint format — PLANNED

Use a **new explicit Full7 checkpoint schema version** (proposed v3), preserving v1 (`top1_4`) and v2 (`full_9`). Include:

```text
format_version / expert_profile=full_7 / E1..E7 order / feature dims
29959 input width / 7 expert count / gate architecture
only gate state_dict; optimizer + scheduler + sampler + RNG states for latest
training settings, config SHA256, epoch/global_step, best validation SRCC
train/validation Full7 store identities + verified score/preprocessing contract
E1..E7 source checkpoint identities + E5/E6 checkpoint file SHA256
source Full9/Whisper cache identity and manifest fingerprints
schema/source-code/score-transform fingerprints, selection policy ID
```

Validation/resume must reject wrong feature width, order, scale, source identities, checkpoint SHA-256, split, config, or architecture. Inference must verify **split-independent** extraction and score contracts against the test Full7 store, while allowing genuinely different train/validation/test **sample-specific** source-manifest hashes. Use atomic writes and existing safe loading patterns (`weights_only=True` where compatible). If trusted E5/E6 loaders use broader deserialization, make their trusted-local-artifact requirement explicit.

---

## 8. Inference, submission, and reproducibility — FROZEN

### 8.1 Inference steps

1. Open and fully validate the label-free `full_7/test` store.
2. Verify the gate checkpoint profile, feature width, expert order, training source identities, split-independent source/cache/score compatibility, and state-dict shape.
3. Run `gate.eval()` under `torch.inference_mode()`, preserving source order.
4. Compute normalized gate output, convert **once**: `score_raw = 5 * score_normalized + 5`.
5. Export raw prediction and required filename using the existing `write_submission_csv` convention (`wave_file_name`, `pred_score`). Preserve the exact competition submission schema after validating it separately.
6. Optionally produce a **separate diagnostic artifact** with ordered `sample_id`, normalized/raw prediction, seven expert scores, seven gate weights, and manifest/checkpoint fingerprints; diagnostic columns must not leak into the official submission file.

Keep existing gate inference rounding behavior (currently two decimal places by default) for compatibility, but preserve full precision in the optional audit artifact. Never clamp/round **before** gate fusion or validation metrics.

### 8.2 Reproducibility outputs

Write a run manifest containing: Git commit, runtime versions, config fingerprint, input store identities and sample counts, expert checkpoint SHA-256, feature and score contract hashes, gate checkpoint hash, seed, training/evaluation split policy, model-selection policy, device/dtype, commands and measured results. Separate **synthetic tests**, **real-cache smoke checks**, **actual training**, and **official-validation metrics**.

### 8.3 No implicit end-to-end raw inference

V1's guaranteed production path is **cached** Full7 build → gate training/inference. A raw-audio+caption inference facade may be added later if it reproduces the same seven verified frozen expert outputs and provenance. Do not promise an end-to-end raw-audio CLI until that separate runtime has been implemented and numerically checked.

---

## 9. Implementation work packages — PLANNED

The following paths describe the **preferred change locations**, not files already implemented for Full7. Favor extending existing modules only where compatibility can be proven.

| Step | Existing or proposed modules | Work | Exit gate |
|---|---|---|---|
| **Core.0** | `core-overview.md` | Design freeze; contracts, risks and source status | This document reviewed |
| **Core.1** | `model/expert_adapters/registry.py`, `model/moe_gating_core.py`, `model/moe_forward_integration.py` | Add isolated `full_7` profile and seven-output validation; preserve old profiles | Synthetic `[B,7]` / `[B,29959]` and softmax tests |
| **Core.2** | New `data/full7_assembler.py` (suggested); `data/moe_feature_store.py` or separate versioned Full7 store; new `scripts/build_full7_feature_store.py` | Verify Full9/Whisper/E5/E6, construct/serialize seven outputs, SHA-256 and immutable provenance | Full7 train/validation/test synthetic stores re-open and validate |
| **Core.3** | `data/moe_training_dataset.py`, `training/config.py`, `training/moe_trainer.py`, `training/moe_evaluator.py`, `training/checkpoint_manager.py` | Full7-aware batches, gate-only training, evaluation, v3 checkpoint and resume | Contract, optimizer freeze, resume and identity tests |
| **Core.4** | `scripts/train_moe_gating.py`, `scripts/evaluate_moe_gating.py`, `scripts/infer_moe_gating.py`; new `configs/moe_full7.example.yaml` | CLI integration and label-free submission | Synthetic end-to-end training, validation and test inference |
| **Core.5** | New `tests/test_full7_*.py` (suggested), existing legacy regression tests | Fault injection, negative tests, reproducibility and backward compatibility | All focused/related tests pass; no new legacy failures |
| **Core.6** | Run manifests, verified external artifacts | Real split alignment, checkpoint audit, actual gate training and evaluation | Real metrics reported with approval status |

Do not modify pretrained expert source trees, original Full9 extraction behavior, E5/E6 training protocols, or their checkpoint formats just to make the new Core interface fit. Keep Full7-specific logic at the assembly/builder boundary.

---

## 10. Acceptance tests and release checklist

### 10.1 Unit and contract tests (synthetic)

- [ ] `profile_contract("full_7")` gives exactly E1→E7 and 29,959 features; `top1_4`/`full_9` unchanged.
- [ ] Seven scores concatenate to `[B,7]`; features to `[B,29959]`, each exact slice tested with recognizable sentinel values.
- [ ] Incorrect expert order, name (`m2d_lora` vs Full7 `expert7`), missing/extra outputs, shape, dtype, mixed-device and non-finite cases all fail closed.
- [ ] Gate weights `[B,7]` are nonnegative, finite, normalized; final `[B,1]` equals a direct weighted sum numerically.
- [ ] No expert (including E5/E6 heads) receives gradient updates; only gate weights change after a train step.
- [ ] E5/E6 input composition uses their **Top2** cache keys and complete feature vectors, not Top1 substitutes or compressed versions.
- [ ] Synthetic Full9 + Whisper caches produce a valid Full7 store for all three splits; test remains label-free; source order is preserved.
- [ ] Corrupt/missing/duplicate/reordered WAV/caption/sample identity, wrong E5/E6 checkpoint or LAION/Whisper identity, and changed score scale all fail before optimization/output.
- [ ] Train and validation differ in legitimate sample-specific fingerprints but agree on required extraction/score contracts.
- [ ] Resume from latest is bitwise/numerically deterministic under controlled CPU fixtures, rejects altered source/config/checkpoint, and preserves optimizer/scheduler/sampler/RNG states.
- [ ] Test inference preserves sample order and writes exactly the expected submission schema; optional diagnostic weights sum to one.
- [ ] No validation/test label ever enters training updates; no model ID is present in gate tensors.
- [ ] Legacy `top1_4` and `full_9` store, gate, inference and checkpoint regression tests retain their behavior.

### 10.2 Score audit tests

- [ ] Seven explicit score contracts, with source and transform provenance, exist and are complete.
- [ ] E3 training-label scale is independently established from code/checkpoint evidence.
- [ ] No extra normalization/clamp/MinMax/StandardScaler is silently inserted.
- [ ] Raw↔normalized mapping is applied exactly once at documented boundaries; MSE evaluates against raw labels after output conversion.
- [ ] Training-split score statistics are recorded; anomalies/unknown semantics block real Full7 training pending review.

### 10.3 Real-data exit criteria

- [ ] Complete official source manifests and Full9 cache for all required splits have been verified.
- [ ] Whisper large-v2 cache has a verified one-to-one join with Full9 for every split.
- [ ] Trusted, real E5 and E6 checkpoints are available and compatible with all source cache identities.
- [ ] E5/E6 real predictions and cross-split identity checks pass; the previous **synthetic-only** E6 status is not misreported as real verification.
- [ ] Full7 cache for train, validation, test validates independently.
- [ ] Full7 gate actually trains on cached official train and evaluates under the approved split policy.
- [ ] Submission CSV matches the official format, expected count and source ordering.
- [ ] Competition approvals are documented for external pretrained models/data and validation-driven model selection, or the process is revised in an explicitly approved way.

---

## 11. Competition constraints and unresolved approvals — BLOCKED/VERIFY

Source documents supplied for this project: `Task-rules.txt`, `Task-description.txt`, `Data-2027-description.txt`.

**Dataset characteristics:** official development data have 7,500 train, 3,000 validation and 3,000 test audio–text pairs, average human semantic-alignment ratings on 0–10, and generation-model identifiers. The 2027 challenge emphasizes **generalization to previously unseen generation models**. V1 never feeds model IDs into the Gate.

Relevant organizer rules (paraphrased):

1. **Only training data may be used to train models.**
2. **Validation/test data are designated for evaluation.**
3. External datasets and pretrained models require prior organizer approval, disclosure of the pretrained model's training data, and open-access eligibility. The stated email deadline was **October 14, 2026**.
4. Gathering or generating novel training/evaluation audio, text or scores is prohibited, subject to the published allowance for transformations of existing public sources.
5. Ensembling independent model results is prohibited; **internal ensembling within one model**, such as an MoE, is permitted.
6. Each participant may submit up to four systems.

**V1 implementation decisions versus organizer approval:**

- The existing Core gate uses validation regression loss to drive `ReduceLROnPlateau` and validation SRCC to choose `best_gate.pt` / early stopping. The E6 training code also uses validation MSE for best E6 checkpoint selection. **These are validation-guided training/model-selection decisions even without validation gradients.** The provided rules do **not explicitly settle** their permissibility. Obtain written organizer guidance **before competition submission**; do not claim compliance from an absence of gradients alone.
- The external LAION, Whisper, CLAP, BEATs, M2D and other pretrained resources must have their approval and data-disclosure status verified independently. Do not interpret the fact that code references them as permission to submit with them.
- The single global gating model over seven internal experts is **intended** to satisfy the internal-MoE exception; final organizer interpretation is authoritative.
- Since the external-model request deadline is in the past as of this document date, **do not assume late approval is possible**. Escalate using the organizer's official contact/process rather than representing approval as already granted.
- Nothing in this document authorizes training on official validation/test or using their labels to fit score transformations.

**Implementation gating:** synthetic development can proceed without real model weights, provided every artifact is clearly synthetic. Real-cache construction and gate training are blocked on compatible real checkpoints/caches and a defensible score audit. Competition submission is blocked on rule/approval confirmation.

---

## 12. Source map and evidence status

### Reviewed source files (7expert branch, commit `834222e`)

- `model/expert_adapters/registry.py` — existing `top1_4`, `full_9` order and feature widths; **no Full7 profile yet**.
- `model/moe_gating_core.py` — gate topology, strict feature validation.
- `model/moe_forward_integration.py` — frozen adapter integration and weighted-sum semantics.
- `model/expert_adapters/paired_cross_features.py` — canonical `4D+1` feature blocks.
- `model/expert_adapters/expert{1,2,3,4}_adapter.py`; E3/E4 source models — E1–E4 scores and feature contracts.
- `model/expert_adapters/m2d_lora_runtime.py`, `model/expert_adapters/m2d_lora_adapter.py` — E7 feature and score-scale transforms.
- `model/composite_experts/set_a_residual.py`, `model/expert_adapters/set_a_residual_adapter.py`, `training/set_a_trainer.py` — E5 feature, head, training/identity rules.
- `model/composite_experts/set_b_residual.py`, `model/expert_adapters/set_b_residual_adapter.py`, `training/set_b_trainer.py` — E6 feature, head, P1 fixes, identity rules.
- `data/moe_feature_store.py`, `data/moe_training_dataset.py`, `data/moe_expert_cache.py`, `data/whisper_feature_cache.py` — storage, strict provenance and datasets.
- `training/moe_trainer.py`, `training/moe_evaluator.py`, `training/checkpoint_manager.py`, `training/config.py` — current gate training, metrics, checkpoint versions.
- `scripts/train_moe_gating.py`, `scripts/infer_moe_gating.py`, `configs/moe_cloud.example.yaml` — baseline operational policy and config.
- `docs/MOE27-E5-implementation.md`, `docs/MOE27-E6-implementation.md`, `MOE27-E6-report.md`, `e6-overview.md` — expert-to-Core interface and known real-cache caveats.

### Still not established by this document

- A pushed Core implementation branch, real Full7 store, completed gate training, measured real-data SRCC/LCC/KTAU/MSE, numerical parity against true E1–E7 checkpoints, real E5/E6 checkpoint availability, or organizer permissions.
- The definitive empirical scale compatibility of all seven expert scores (especially E3) with normalized gate targets.
- That using official validation labels for best-checkpoint selection/scheduler control is allowed for submission.

---

## 13. Implementation directive for Codex

Treat this document as the **single V1 design specification**. First inspect the live Core worktree and 7expert baseline, verify branch/commit ancestry, and preserve unrelated changes. Implement Core.1 through Core.5 with explicit backward-compatible schemas and synthetic tests, then report test commands, pass/fail counts, source identities, files changed and outstanding BLOCKED/VERIFY conditions. **Do not fabricate real checkpoints, silently downgrade identity checks, train on validation/test, push/merge without authorization, or declare the real-data phase complete based on synthetic tests.**

**Design-freeze summary:** 7 experts; fixed feature order and **29,959** total width; 7 score columns; frozen E1–E7; cached Full9+Whisper source; E5/E6 frozen heads; one `29959→512→7` gate; original ClippedMSE + 0.5 Contrastive + 0.2 Diversity loss; baseline AdamW/scheduler; official train gradients only; provisional validation-SRCC best selection subject to organizer approval; SHA-256 provenance; label-free test; no model-ID routing; backward compatibility for `top1_4` and `full_9`.
