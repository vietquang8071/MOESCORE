# MOE8 LOGO-Lite implementation report

## Protocol and provenance

This is `MOE8-LOGO-Lite`, profile `full_8`, protocol `gate_only_target_selection`, benchmark type `GATE_ONLY_PROXY_NOT_STRICT_DG`. The experiment uses cached Full8 features and scores, trains a fresh gate for each eligible generator, selects by SRCC on that generator's official train subset, and evaluates only its train holdout, official validation, and labeled public-test subsets. No baseline gate weights, expert models, expert extraction, expert retraining, baseline training, or baseline inference are used.

- Experiment branch: `experiment/moe8-logo-lite`.
- Baseline-compatible source commit recorded in the supplied gate manifest: `68124d0e57005d29149f659780ab749d14e09532` on the Full8 source history. The existing experiment branch was preserved; its previous tip was `8953689` and its common ancestor with the current Full8 branch was `a10648a`.
- Implementation commit: `3780a3c0519ce9be00a36d5883d71f2b85ffeeff`.
- Baseline gate SHA-256: `ea49e31e5e2fb638871ca57d1d5d9e6ab5970b6d86cdf82176138b4a5c07e9a7`, checked only for provenance.
- The supplied baseline YAML reproduces the baseline run manifest's effective configuration SHA-256 `cf2018d31e481d63e1a1f1803b55a75b14b9be10757b6535634f83ce50d44b81` after removing its device/output paths and recording its score-review policy. The runner verifies this relationship before training.

The effective baseline uses seed 2027, batch size 8, 50 maximum epochs, AdamW at `1e-5` with weight decay `1e-4`, diversity weight `0.2`, clipped MSE tau `0.25`, contrastive margin `0.1`, loss weights `[1.0, 0.5]`, early-stop patience 50, and inference batch size 64. The existing trainer supplies the same `ReduceLROnPlateau` scheduler. The actual gate has eight experts, 33,032 input features, eight expert score columns, and a 512-unit hidden layer.

## Reused components and files

The runner reuses `MoeTrainingDataset`, the validated memory-mapped feature store, `collate_moe_features`, `StatefulRandomSampler`, `GatingNetwork`, `MoeGateTrainer`, `MoeGateEvaluator`, `MoeCheckpointManager`, `compute_metrics`, and the existing raw/normalized target transform. The checkpoint manager received one optional Full8 LOGO selection-policy value; existing calls retain their default.

Created: `training/logo_protocol.py`, `training/logo_runner.py`, `training/logo_plots.py`, `scripts/train_moe8_logo.py`, `configs/moe_full8_logo.example.yaml`, `notebooks/MOESCORE_Full8_LOGO_Lite_Kaggle.ipynb`, `kaggle/logo_lite/kernel-metadata.json`, and three `tests/test_logo_*.py` files. Modified: `training/checkpoint_manager.py`. This report is the sole documentation addition. Pre-existing uncommitted D01/G05 work was not staged or modified.

## Dataset, cache, and integrity contracts

The named Kaggle datasets resolve to `qnv8071/xacle-2027-data` and `qnv8071/moescore-full8-artifacts`. The dataset's official score column is `average_score`, generator ID column is `model_ID`, and sample file column is `wav_file_name`. The Full8 cache consists of `train`, `validation`, and `test` stores with 7,500, 3,000, and 3,000 rows. Public-test labels are absent from the cache and are attached by stable `test:<wav_file_name>` identity from official labeled metadata. Train and validation cache labels are checked against metadata. Metadata text is checked against cached caption SHA-256; expert order, dimensions, score contracts, checkpoint identities, manifests, shards, and array hashes are validated before training.

The source cache has expert score columns in its recorded native scales. Experts 1–2 are `unitless_similarity_proxy`; experts 3–8 are `normalized_minus1_to1`. Prediction CSVs preserve those eight original cached score values and export the fused gate prediction after the existing inverse transform on the raw 0–10 challenge scale. Labels never enter gate inputs.

The official metadata contains one confirmed non-generator control, `audiocaps`, configured by name and retained in every optimization set. Eight generator IDs are eligible from official train. No IDs are missing or ambiguous, and no generator appears only in evaluation splits:

| ID | Name | Train | Validation | Public test |
|---:|---|---:|---:|---:|
| 0 | audiocaps (control) | 2,500 | 1,000 | 1,000 |
| 1 | audiogen | 750 | 250 | 250 |
| 2 | audioldm | 500 | 250 | 250 |
| 3 | audioldm2 | 500 | 250 | 250 |
| 4 | make-an-audio2 | 750 | 250 | 250 |
| 5 | stable_audio | 750 | 250 | 250 |
| 6 | tango | 500 | 250 | 250 |
| 7 | tango2 | 500 | 250 | 250 |
| 8 | tangoflux | 750 | 250 | 250 |

These counts and the complete metadata-to-manifest ID/caption alignment were verified locally from the small official CSVs and remote cache manifests. Full shard-byte verification is deferred to the Kaggle preflight because the 1.8 GB feature store is mounted there.

Cached WAV hashes reveal five train/validation overlap groups and one train/public-test overlap group; validation/test has none. Both overlap allowances are explicit in the research configuration and the preflight report. These overlaps can bias evaluation; they do not authorize use of official validation or public-test labels for training decisions.

Each fold's optimization and holdout IDs are disjoint and reconstruct official train. Fold views preserve `source_split` separately from the trainer-facing `split` and `fold_role`. The trainer's split checks remain active. The best finite complete-holdout SRCC wins, with strict improvement selecting the earliest epoch on ties. Undefined SRCC is never converted to zero. Resume is bound to configuration, fold membership, source metadata, cache identities, and checkpoint hashes. Completed folds are skipped only after required output files and their hashes are checked.

For each nonempty held-out subset, the same best gate produces raw-scale predictions, ground truth, eight expert scores, eight finite gate weights, and normalized entropy. `compute_metrics` supplies SRCC, LCC, Kendall tau-b, and raw-scale MSE across the whole subset. Aggregate LOGO macro, weighted, pooled, worst, and population standard deviation are calculated independently per split. Missing/failed folds remain visible in coverage. All aggregate values are independently recalculated from saved prediction CSVs.

## Verification

- Local CPU integration smoke: one synthetic Full8 fold trained, selected and reloaded a checkpoint, exported all three held-out CSVs, built partial aggregate outputs, and validated resume. No synthetic records are used for real XACLE results.
- `tests/test_logo_protocol.py tests/test_logo_runner.py tests/test_logo_integration.py tests/test_moe_checkpoint_resume.py`: 12 passed, 3 subtests passed.
- Broader test run excluding pre-existing dirty D01 tests: 536 passed, 1 skipped, 1 unrelated failure, 1 deselected, 265 subtests passed. The unrelated failure is `test_kaggle_full9_dual_gpu.py::DualGpuTests::test_notebook_has_executable_stages_and_single_editable_configuration`: an existing Full9 notebook records Python `3.13.16` while that test expects `3.12`. A separate D01 test added by pre-existing uncommitted work fails because its fixture creates the same directory twice. Neither is caused by the LOGO files.
- Python compilation, notebook cell compilation, and `git diff --check` passed.

## Kaggle execution

Attach both datasets listed above, enable a T4 GPU and Internet for the GitHub checkout, then Save & Run All [the notebook](../../notebooks/MOESCORE_Full8_LOGO_Lite_Kaggle.ipynb). The default notebook performs full-cache preflight, generator statistics, a separate one-epoch smoke fold, then all eight full folds. `SOURCE_COMMIT` can pin an exact implementation commit. `RESUME_ARTIFACTS_OVERRIDE` can restore a previously exported LOGO artifact directory into a fresh Kaggle session. The notebook writes final outputs to `/kaggle/working/moescore_full8_logo_lite_artifacts/`; smoke outputs remain under its `smoke/` subdirectory.

CLI equivalents inside Kaggle:

```bash
python scripts/train_moe8_logo.py --config configs/moe_full8_logo.example.yaml --dry-run
python scripts/train_moe8_logo.py --config configs/moe_full8_logo.example.yaml --generator-id 1 --smoke-epochs 1 --resume
python scripts/train_moe8_logo.py --config configs/moe_full8_logo.example.yaml --all-generators --resume
python scripts/train_moe8_logo.py --config configs/moe_full8_logo.example.yaml --aggregate-only
```

The output files are intentionally outside Git and the supplied Full8 artifact bundle remains read-only.

## Research limitations and execution status

Frozen experts may already have encountered held-out generators during prior XACLE fine-tuning. Official train holdout labels select gate checkpoints, so train-holdout metrics are selection-dependent rather than unbiased test estimates. LOGO-Lite evaluates gate-level transfer, not strict whole-model unseen-generator generalization. Concatenated fold predictions are research evaluations, not an ensemble submission system. The source baseline used `EXPERIMENTAL_SKIPPED` human score-review policy, which is recorded in LOGO checkpoint provenance. Results are research-only and are not official XACLE metrics.

At this report's creation, local software tests and manifest/metadata preflight are complete. Real cache shard validation, real single-fold smoke, all-fold GPU training, and empirical metrics require the Kaggle execution. No empirical result is claimed here.
