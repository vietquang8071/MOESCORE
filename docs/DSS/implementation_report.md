# MOE8-DSS implementation report

## Architecture and provenance

Branch: `experiment/moe8-domain-wise-spearman`, based on Full8 `01c57ff`; DSS contract: `d517001`. The paired frozen Full8 gate run manifest records source `68124d0e57005d29149f659780ab749d14e09532`. The current gate, feature-store reader, Full8 contract, expert registry, target transform and evaluator source hashes match the historical run manifest. Other historical source files changed after the cache was made; DSS does not rewrite their cache provenance. The baseline configuration and gate weights remain read-only and are never used to initialize DSS.

The existing gate is `Linear(33032, 512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512, 8) → Softmax`. Eight cached expert scores are fused by a weighted sum. Generator ID never enters the gate. Raw labels are mapped with `(raw-5)/5`; raw predictions use `5*normalized+5` without clipping.

The actual Kaggle datasets are `qnv8071/xacle-2027-data` and `qnv8071/moescore-full8-artifacts`. Small official CSVs and manifests were inspected locally. They contain exactly 7,500/3,000/3,000 official rows, and every domain count matches §3.1 of the overview. All 13,500 official metadata rows align by split-qualified WAV identity and caption SHA-256 with the Full8 manifests. Store identities: train `1c6c0e58f9892360828a4adc3d1c5722cc7af6e69b1212b6a48688523d7c2346`, validation `e805db01827ff0b2720b4c5680cb7088fe2d28950e3f0664c41e8dc307d3a75c`, test `6be115987b34b076343458ae4ffc3ae2cb3dcdc260ab2963bf704c9b341043cf`. Manifest WAV hashes show five train/validation and one train/test overlaps; no validation/test overlap. Official splits are retained. Kaggle CLI exposes last-update timestamps but no version number in the mounted-file contract; run fingerprints are recorded instead.

## Implementation

| Area | DSS file | Contract |
|---|---|---|
| Exact L2 soft rank | `loss/l2_soft_rank.py` | PyTorch pool-adjacent-violators projection onto the permutahedron |
| Ranking objective | `loss/domain_spearman.py` | Ascending prediction soft ranks, average target ties, 0.5 global + 0.5 valid-domain mean |
| Sampling | `training/domain_balanced_sampler.py` | 59 deterministic 4×32 batches; shuffled per-domain queues; epoch-boundary state and coverage |
| Official metadata | `data/dss_dataset.py` | Stable ID/caption join, domain counts, cached-label agreement, read-only Full8 store |
| Training | `training/dss_runner.py`, `scripts/train_moe8_dss.py` | Fresh gate, AdamW 1e-5/1e-4, 30×59 steps, full-validation hard SRCC selection, safe resume |
| Evaluation | `training/dss_evaluation.py` | Best-checkpoint reload, complete three-split inference, raw metrics, domain summaries, diagnostics and plots |
| Kaggle | `notebooks/MOESCORE_Full8_DSS_Kaggle.ipynb`, `kaggle/dss/` | Dedicated private GPU kernel, pinned source, automatic mount discovery and path overrides |
| Config | `configs/moe_full8_dss.example.yaml` | DSS overrides separate from immutable historical Full8 config |

The local `torchsort==0.1.10` build failed because the host toolkit is CUDA 12.0 while local PyTorch was compiled for CUDA 13.0. Instead, DSS uses the **same L2 permutahedron projection** with a pure PyTorch isotonic PAV implementation. This is not a sigmoid approximation or a different objective. Tests compare its forward values with SciPy's independent isotonic solver and check finite-difference gradients. Kaggle does not need a compiled rank extension. The reference method is Blondel et al., [Fast Differentiable Sorting and Ranking](https://proceedings.mlr.press/v119/blondel20a.html); the source formulation was cross-checked against the [official fast-soft-sort implementation](https://github.com/google-research/fast-soft-sort/blob/master/fast_soft_sort/numpy_ops.py).

## Effective protocol

Full run: one fresh seed-2027 gate, all nine domains, batch 128, four domains × 32, 59 steps per epoch, 30 epochs, 1,770 updates, fixed AdamW learning rate 1e-5 and weight decay 1e-4, no scheduler or early stop. L2 strength is 0.1; groups below 8 or with constant target ranks are excluded from the domain mean, and an invalid global/all-domain batch fails. Validation uses all 3,000 samples and strict finite hard SRCC improvement; ties retain the earlier epoch. Smoke runs 2 epochs and 118 steps in a separate directory and never runs public-test inference. Full evaluation reloads `best.pt`, then writes 7,500/3,000/3,000 predictions, metrics, diagnostics and plots. `last.pt` stores optimizer, sampler and RNG states for verified same-run resume.

## Verification and limitations

Targeted DSS plus Full8/D01 regression suite: 85 passed, 17 subtests passed, three expected overlap warnings. A full repository suite run with an import alias yielded 623 passed, one skipped, 12 failed in unrelated Expert2/3/4 and pre-existing Full9 notebook tests. Those components were not modified. The Full9 failure is a pre-existing notebook Python-version assertion (`3.13.16` versus `3.12`). The final targeted suite is rerun before push and recorded in the review report.

The preflight also hashes every official WAV and checks it against the cache fingerprint before creating the gate. The full 1.8 GB frozen cache and 3.6 GB official audio are available only on Kaggle; local checks used downloaded manifests/metadata, a small WAV fixture, and synthetic Full8 shards. Real shard and full WAV validation, CUDA execution, and all measured training/evaluation results remain unverified until the submitted Kaggle job runs. The first notebook submission is deliberately `RUN_MODE="smoke"` to honor the required stage gate, even though the notebook supports `RUN_MODE="full"` after smoke verification.

## Compliance matrix

| Requirement | Implementation status |
|---|---|
| No LOGO, all nine domains, official splits | Implemented; metadata counts verified |
| Reuse frozen Full8 features and scores, no baseline warm start | Implemented; baseline file is hashed only |
| Exact L2 hybrid ranking objective | Implemented; independent forward and gradient tests |
| Balanced deterministic sampler and coverage | Implemented; sampler tests |
| Metadata/cache integrity and overlap report | Implemented; real metadata/manifest joins verified; full shard pass awaits Kaggle |
| 30 epochs, 1,770 updates, hard validation selection | Implemented; real execution awaits smoke |
| Best-checkpoint final three-split evaluation | Implemented; synthetic 13,500-row artifact test |
| Baseline comparison | Paired validation metrics available; public-test evidence checked at runtime |
| Kaggle GPU smoke and full result | Pending submission and execution |
