# Expert 3 implementation report

## Delivered

- `models/moescore_expert3.py`: verified M2D checkpoint path layout, lazy BERT initialization, partial fine-tuning, released-order cross-modal features, and the 3,840-input alignment head. The regression MLP runs in FP32 inside the AMP step to avoid FP16 accumulation overflow in its final bias gradient.
- `datasets/moescore_expert3_dataset.py`: configurable CSV columns, audio existence/read checks, mono conversion, resampling, 10-second pad/trim, and score normalization.
- `train_expert3.py`: strict physical batch eight, validation metrics/SRCC selection, AMP, activation checkpointing, early stopping, smoke test, and resume CLI.
- `utils/expert3_checkpoint.py`: versioned atomic best/resume checkpoints with M2D identity and RNG/training state; resume checks include a portable split manifest.
- `inference_expert3.py`: validation/test/custom prediction CSV, with metrics when at least two labels are present.
- `configs/expert3_moescore.json`, `requirements-expert3.txt`, `notebooks/finetune_expert3_colab.ipynb`, and this reproduction document.
- Minimal baseline fix in `losses/loss_function.py`: make clipped-MSE `tau` a device-aware registered buffer.

## Audit findings

- The local 1.6 GB pretrained checkpoint SHA-256 matches the digest in the XACLE baseline's M2D README and the intended M2D-CLAP 2025 model directory identity.
- Its original file remains untouched; the runtime creates a symlink under a temporary directory with the expected parent name because the vendored loader derives architecture from that directory name.
- Local metadata contains 7,500/3,000/3,000 average-score rows with the required `wav_file_name`, `text`, `average_score` columns. The dataset directory is the XACLE 2027 development set; the baseline describes it as inherited from XACLE 2026.
- Source audit confirmed the final blocks live at `backbone.blocks[-2:]` and BERT layers at `text_encoder.text_encoder.encoder.layer[-2:]`. The supplied checkpoint has an audio projector but no `text_proj`; the trainer fine-tunes the final BERT layers and conditionally unfreezes `text_proj` only for checkpoints that contain one.
- The baseline hybrid loss matches the specified clipped-MSE + pairwise objective when predictions/targets are flattened and CBL is off. The `tau` buffer fix is required for CUDA execution.

## Verification performed

- `python -m unittest discover -s tests -p test_moescore_expert3.py -v`: 13 tests passed. Tests include checkpoint-without-`text_proj` and autocast-head regressions, lightweight M2D-shaped model/checkpoint behavior, synthetic WAV handling, unlabeled inference, resume-config integrity, and split-overlap rejection.
- A local GTX 1650 diagnostic using the exact batch-eight hybrid loss and a score-skewed batch confirmed finite trainable gradients after the FP32 regression-head change.
- Python compilation, notebook JSON/Python-cell parsing, inference CLI help, and `git diff --check` passed.
- `train_expert3.py --check-data` validated all three split CSVs, WAV path inventories, counts (7,500 train, 3,000 validation, 3,000 test), and a sample `[1,160000]` waveform per split; the sample is mono 16 kHz.
- Pretrained checkpoint digest was computed and matched the repository-recorded SHA-256.
- No complete 50-epoch training run was started.

## Remaining hardware check

The available GPU is a GTX 1650 with 4 GB VRAM, and the local Python environment lacks `timm` and other vendored M2D dependencies. The synthetic CUDA diagnostic passed, but the actual pretrained M2D batch-eight forward/backward has not been executed locally. The Colab notebook runs that smoke test first with physical batch size 8, AMP, and activation checkpointing. If the T4 still OOMs, the trainer stops and reports the memory constraint; it does not substitute microbatch accumulation or a smaller batch.

## Commands

```bash
python -m pip install -r requirements-expert3.txt
python train_expert3.py --check-data
python train_expert3.py --smoke-test
python train_expert3.py
python train_expert3.py --resume runs/expert3_moescore/<run_id>/last_checkpoint.pt
python inference_expert3.py --config runs/expert3_moescore/<run_id>/config.json --checkpoint runs/expert3_moescore/<run_id>/best_model.pt --dataset validation
python inference_expert3.py --config runs/expert3_moescore/<run_id>/config.json --checkpoint runs/expert3_moescore/<run_id>/best_model.pt --dataset test
```
