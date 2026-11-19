# Train/validation WAV overlap experiment

The official data contains the known pair `train:03768.wav` and `validation:07603.wav`, with shared WAV SHA-256 `0de242afa78769b6f6baf5b552a4d80fbea05fb5c615f1868a695e57b91eb1c2`. The Full7 workflow can now run this research experiment when the opt-in is set in all applicable stages:

```yaml
training:
  allow_train_validation_audio_overlap: true # E6
```

```yaml
data:
  allow_train_validation_audio_overlap: true # Full7 Global Gate
```

The Full7 builder accepts `--allow-train-validation-audio-overlap`. Every flag defaults to `false`; without an explicit opt-in, shared train/validation SHA-256 values still raise an error. The builder always rejects a duplicate involving the official test split. The exception does not delete or rewrite rows, labels, filenames, feature caches, source manifests, or hashes. E6 and Global Gate continue to use the complete official train and validation splits, including the final partial training batch.

Each stage records a deterministic diagnostic containing every overlapping hash, the associated train and validation sample IDs, affected row counts, the selected policy, and a SHA-256 identity. Console warnings show at most five examples; JSON artifacts and checkpoint metadata retain the complete membership. E6 writes `expert6_audio_overlap_diagnostic.json`. Full7 builder writes the same diagnostic to its console report and optional `--report-json` output. Global Gate includes it in `run_manifest.json`, the checkpoint's feature-store identity, and training/evaluation reports.

**Validation results may be optimistically biased because identical WAV audio appears in both training and validation.** The experimental override does not remove this leakage, make validation independent, or establish competition compliance. Existing score review and test-split protections remain in effect.

E6 checkpoint format 2 records the explicit policy, source provenance, train/validation membership identities, optimizer state, and overlap diagnostic. Format 1 remains loadable for inference, but cannot resume training because it does not identify the policy. Full7 Gate checkpoint format 4 records the overlap policy and source membership; legacy format 3 remains loadable for inference when its pinned source-code identity matches, but cannot resume under the new policy.
