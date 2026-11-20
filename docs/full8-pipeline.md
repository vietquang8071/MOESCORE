# Full8 MOESCORE pipeline

Full8 is a separate profile that preserves the ordered Full7 experts and appends frozen Expert 8, the M2D-CLAP audio/text LoRA CORAL model. It uses 33,032 features, eight score columns, and the existing 512-unit Global Gate. Full7 stores and checkpoints remain independent.

## Expert 8 inputs

The authoritative LoRA checkpoint is `m2d-lora-Nam.pt`. Local extraction configs can use `weights/m2d-lora-Nam.pt`; the notebook discovers that exact filename under `/kaggle/input` unless `E8_LORA_CHECKPOINT_OVERRIDE` is set. The compatible base is `checkpoint-30.pth` with SHA-256 `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`. The E8 checkpoint SHA-256 is stored in its cache identity and Full8 store/Gate provenance. The adapter validates the ten-logit head and all 24 ordered LoRA tensors and runs frozen inference only.

E8 requires 16 kHz source WAVs. It does not resample. Extraction uses batch size one by default and writes resumable, content-addressed atomic shards with source IDs, filenames, WAV hashes, caption hashes, and checkpoint identities.

## Local commands

Prepare `configs/expert8_extraction.example.yaml` with the exact base and LoRA paths, then extract and assemble:

```bash
python scripts/extract_expert8_features.py \
  --config configs/expert8_extraction.example.yaml \
  --dataset-root /path/to/XACLE_Challenge_2027_development_dataset \
  --output-root runs/full8_expert_cache --device cuda:0 --batch-size 1

python scripts/build_full8_feature_store.py \
  --full7-root /path/to/verified_full7_store \
  --expert8-cache-root runs/full8_expert_cache \
  --output-root runs/full8_store
```

Train the Full8 Gate from scratch using `configs/moe_full8.example.yaml`, setting its feature-store and output paths. Train, validation and test are validated independently against their own source manifests; WAV hashes are not compared across splits. Set `MOESCORE_EXPERIMENTAL_SKIP_SCORE_REVIEW=1` only for research runs that intentionally skip human score review. The train-only score audit still runs and the checkpoint is marked `EXPERIMENTAL_SKIPPED`.

## Kaggle

Use [MOESCORE_Full8_T4x2_Kaggle.ipynb](../notebooks/MOESCORE_Full8_T4x2_Kaggle.ipynb), attach the XACLE data and the Full9 cache or a verified Full7 store, and attach the E8 LoRA and base M2D checkpoints. The notebook preflights both checkpoint files before expensive stages, reuses existing feature caches, executes E8 on one T4 at a time, and trains a distinct Full8 Gate. `Save & Run All` evaluates official validation and performs label-free test inference before any optional post-inference test-label metrics.

The current Full7/Full8 implementation does not compare WAV hashes across official splits. Human review is not claimed when the experimental score-review bypass is selected. Post-inference test metrics are research-only and do not establish competition compliance.
