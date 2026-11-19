# E5 official validation refactor report

## Training policy

E5 reads M2D, MS-CLAP, and MGA outputs in the existing Full9 feature order and trains the unchanged residual adapter on the complete official train split. Every epoch evaluates the complete official validation split under `model.eval()` and `torch.inference_mode()`; validation targets never enter the backward or optimizer-update path.

The loss remains normalized-target MSE with `target_normalized = (target_raw - 5.0) / 5.0`. `training.selection_metric` is `official_validation_mse`; the same validation MSE drives best-checkpoint selection and early stopping with the existing patience and `min_delta` semantics. `expert5_final.pt` is the best official-validation checkpoint, and `expert5_latest.pt` remains the resumable latest epoch.

## Provenance and compatibility

Format V3 checkpoints record the official train and validation provenance, exact ordered sample IDs, selection policy, optimizer state, progress, and per-epoch metrics. The loader continues to accept valid V2 holdout checkpoints for inference. V2 checkpoints cannot resume under the new policy. Full7 preserves legacy V2 holdout membership classification and labels V3 rows as `IN_SAMPLE_FIT`, `OUT_OF_SAMPLE_FIT_VALIDATION_SELECTED`, or `OUT_OF_SAMPLE` for train, validation, and test respectively.

The Full9 feature contracts, component order, 11267-wide E5 feature output, residual MLP, frozen component behavior, AdamW settings, MSE, and gradient clipping are unchanged. E6 training and cross-split duplicate-WAV checks are unchanged.

## Kaggle behavior

The Full7 notebook generates its E5 YAML from `configs/set_a_residual.example.yaml`, uses the official-validation metric, loads labeled train and validation splits, and reports both counts. If old holdout-policy outputs exist, the notebook detects them and starts the new policy without passing the legacy checkpoint to `--resume`.

## Verification

Synthetic CPU verification:

- `/home/vitquay1708/miniconda3/envs/nlp/bin/python -m pytest -q tests/test_set_a_training.py tests/test_set_a_residual.py tests/test_full7_pipeline.py` — **35 passed, 4 subtests passed**.
- Full notebook JSON validation and Python compilation — **19 code cells compiled**.
- `/home/vitquay1708/miniconda3/envs/nlp/bin/python -m pytest -q` — **481 passed, 28 failed, 1 skipped, 260 subtests passed**. No E5 or Full7 test failed. The failures were in unrelated E2/E3/E4 and Kaggle Full9 tests: this repository checkout lacks the expected `MOESCORE` import namespace for several tests, and a Full9 notebook metadata test expects Python 3.12 while the notebook declares 3.13.16.

No pretrained feature extraction or GPU training was run. Real 7,500/3,000-row Full9 provenance and runtime remain unverified here.
