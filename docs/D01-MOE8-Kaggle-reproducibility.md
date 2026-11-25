# XACLE 2027 D01 — frozen MOE8 reproduction

## Inputs

The private Kaggle kernel is `qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics`. It attaches only:

- `qnv8071/moescore-full8-artifacts` — paired format-7 Gate checkpoint, configuration, run manifest and Full8 store.
- `qnv8071/xacle-2027-data` — official public train, validation and test metadata.

The notebook locates mounts by these verified slugs, scoped to the corresponding dataset mount. It rejects missing or ambiguous checkpoints, run/config files, stores or metadata roots. It does not scan unrelated dataset mounts. The private kernel has CPU only; Internet is enabled only to fetch pinned repository commits. No pretrained weights or backbones are fetched.

## Runtime workflow

1. Checkout the notebook's pinned source commit and fetch the Gate's recorded compatibility commit.
2. Verify the Gate file SHA against `run_manifest.json`, safely load the format-7 payload, and verify its complete 33-file source hash map against the recorded source commit.
3. Check the active Gate, score-transform, feature-store and Full8-contract source files against the checkpoint source. Use the explicit D01 format-7 adapter to validate fingerprints and load the frozen state dict with `strict=True`.
4. Read train, validation and test directly through the existing shard-backed Full8 dataset reader. Its validator checks split manifests, identities, expert order/dimensions, every shard hash, array shapes/dtypes, finiteness, and label policy.
5. Compare train/validation store identities to the paired Gate, and compare checkpoint provenance and preprocessing/score contracts across all splits.
6. Run the gate on CPU in `eval()` and `torch.inference_mode()`. The notebook loads no expert checkpoint and does not reconstruct, extract, assemble, calibrate or train anything.
7. Write label-free public-test predictions, record their CSV hash, and only then inspect public test metadata for optional descriptive labels.
8. Generate D01 score-envelope metrics, bins/tails, generator summaries, overlap notes and plots.

Split-specific source-manifest hashes are validated separately by the Full8 manifest validator and are not required to match across splits. Sample IDs remain in the recorded shard order. The gate and store identities must match exactly; mismatches stop the run with the observed split and identity values.

## WAV overlap and evaluation

Cross-split SHA-256 matches are descriptive metadata, not a precondition for inference. D01 keeps all 7,500 train, 3,000 validation and 3,000 public test samples. It reports duplicate hashes/sample IDs and a possible evaluation-bias warning. Optional overlap sensitivity excludes affected rows only while aggregating metrics; prediction files and feature stores retain every row.

Train results are descriptive and may be in-sample. Validation results disclose the checkpoint's selection policy. The run is not a LOGO experiment. Public test labels are not part of inference and are read only after the test prediction CSV is frozen; if labels are absent, supervised test metrics remain unavailable. No private evaluation data is used.

The score envelope is computed on the normalized expert scores. Raw values are interpreted as `5 * normalized + 5`, with no clipping or calibration. Raw MSE is 25 times normalized MSE. E1/E2 similarity scores retain their original semantics and are not treated as calibrated MOS. The oracle projection and MSE lower bound are theoretical sample-wise diagnostics, not a gate result that is guaranteed to be learnable.

## Outputs

The notebook checks out D01 runtime source commit `8010c4708a023e032ace397e68deb6afa01f56d5`. The paired Gate compatibility source remains pinned to the artifact-recorded commit `68124d0e57005d29149f659780ab749d14e09532`. The final deployment commit contains the synchronized notebook, metadata and documentation; runtime helper code is loaded from the stable D01 source commit.

The notebook writes to `/kaggle/working/d01_moe8/`:

- `predictions/{train,validation,test}_d01.csv` — sample ID, WAV filename, eight scores, eight gate weights, prediction and envelope; test labels remain blank.
- `reports/summary.json`, `global_metrics.csv`, `per_bin_metrics.csv`, `generator_metrics.csv`, `structural_vs_gating.json`, `provenance.json`, `artifact_validation.json`, `audio_overlap.json`, and `D01-MOE8-report.md`.
- `reports/overlap_metric_sensitivity.json` — metrics with/without overlapping IDs where labels exist; no rows are removed from stored data or predictions.
- `plots/*.png` — matched-split histograms, envelope endpoint/scatter plots, per-bin errors, width distributions, tails and sufficiently populated generator groups.

The provenance report records both dataset IDs, Gate SHA/format/source commit, Full8 store identities by split, preprocessing contracts, expert checkpoint provenance, counts, validation status, overlap policy and explicit zero-training/zero-extraction status. Kaggle's mounted input does not expose dataset version numbers to the notebook; the dataset IDs are recorded.

## Reproduce and inspect

Push the existing private kernel with:

```bash
kaggle kernels push -p kaggle/d01_moe8
```

Inspect status and logs:

```bash
kaggle kernels status qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics
kaggle kernels logs qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics
```

After the job completes, download outputs:

```bash
kaggle kernels output qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics -p d01_moe8_output
```

An accepted `kernels push` confirms submission only; it does not verify that the Kaggle run finished successfully.
