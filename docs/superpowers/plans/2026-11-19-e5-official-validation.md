# E5 official validation refactor plan

## Goal and constraints

Train E5 on every labeled official train row and evaluate/select/early-stop on every labeled official validation row. Preserve the residual model, normalized MSE, optimizer and clipping. Keep legacy E5 inference loading where valid, reject legacy resume, and retain Full7 integrity checks and E6 behavior.

## Files and interfaces

- `data/set_a_cached_dataset.py`: emit raw and normalized labels for labeled train/validation rows; require labels when requested and retain manifest consistency checks.
- `training/set_a_config.py`, `configs/set_a_residual*.yaml`, and `scripts/train_set_a_residual.py`: use `training.selection_metric: official_validation_mse`, load both official splits with labels, print their counts, and pass both to `fit`.
- `training/set_a_trainer.py`: train on the full train dataset; evaluate validation under inference mode each epoch; save versioned policy/provenance/membership/progress and select best/final by validation MSE. Keep V2 checkpoint loading, reject V2 resume.
- `data/full7_assembler.py`: verify official-policy train/validation membership and preserve legacy holdout classification.
- `notebooks/MOESCORE_Full7_T4x2_Kaggle.ipynb`, E5 docs, and targeted tests: use the new configuration and behavior without manual checkpoint cleanup.

## Approach and decisions

- Use Full9 `SetACachedDataset` instances for train and validation; no internal split is created during fitting.
- Save exact ordered train and validation sample IDs alongside their complete per-split provenance so membership can be checked against a Full9 split manifest.
- Treat V2 holdout checkpoints as inference-compatible legacy artifacts, but never resume them under the official-validation policy.
- Run CPU synthetic tests only; leave E6 and Global Gate duplicate-WAV rules unchanged.

## Execution tasks

1. Add failing dataset/config tests, then implement validation-label and selection-metric contracts.
2. Add failing trainer/checkpoint tests for full train coverage, validation-only selection, no validation gradients, early stopping, resume provenance, inference and legacy behavior; implement the E5 policy.
3. Add failing Full7 membership tests for official train/validation/test rows and legacy holdout rows; implement policy-aware membership verification.
4. Update the Kaggle E5 cell and E5 documentation, run relevant test suites and holdout-path searches, review the diff, and commit on the current branch.

## Verification

Run targeted E5 dataset/trainer and Full7 tests, then the repository test suite if runtime permits. Verify the official-policy path has no `holdout_fraction` dependency, no train/validation gradient path, checkpoint loaders accept valid V2 and V3 artifacts, and Full7 labels the official splits with the required membership values.
