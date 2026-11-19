# Train/Validation Audio Overlap Experiment Plan

## Objective

Add a default-off, explicit experimental exception for identical WAV hashes shared by official train and validation splits. Keep all test-split, feature, sample identity, score, checkpoint, and provenance checks active. Persist complete overlap diagnostics and disclose potential validation bias.

## Work sequence

1. Add focused tests for E6 strict/experimental behavior and resume metadata, Full7 builder train/validation exceptions with test-overlap rejection, and Global Gate strict/experimental provenance plus checkpoint metadata.
2. Add a shared validator/diagnostic builder for WAV identities, including complete per-hash sample membership and a stable SHA-256 diagnostic identity.
3. Thread the default-false E6 option through config, trainer, CLI, artifacts, and versioned checkpoints. Preserve legacy inference loading and reject unprovable legacy resume state clearly.
4. Add the Full7 builder CLI option and permit only train/validation intersections. Continue rejecting every pair that includes test.
5. Thread the default-false Global Gate data option through config, provenance validation, run diagnostics, and versioned checkpoints. Keep resume checks exact and score-review gates unchanged.
6. Update the Full7 Kaggle notebook to opt in centrally and propagate the flag to E6 YAML, builder CLI, and Gate YAML. Keep cache reuse and existing E5/E6 policies intact.
7. Update documentation, run the requested targeted and relevant integration tests, validate syntax/YAML/notebook cells, then commit and push the branch.

## Verification

- Run focused E6, Full7 builder, Global Gate, and checkpoint tests with synthetic fixtures only.
- Run relevant integration tests for checkpoint loading, Full7 assembly, score audit, and inference.
- Compile changed Python files, parse affected YAML and notebook JSON, and compile every notebook code cell.
- Inspect the final diff and search for flag propagation and overlap checks before delivery.
