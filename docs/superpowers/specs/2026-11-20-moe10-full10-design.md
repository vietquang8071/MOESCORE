# MOE-10 Full10 Integration Design

## Intent

Add the existing M2D-CLAP audio/text LoRA CORAL checkpoint as Expert 10 and support an independently trained `full_10` gate. Experts 1–9, Full9 artifacts, and the existing loss remain unchanged.

## Goals and constraints

- Restore the `head`/ordered `lora`/`cfg` checkpoint with `weights_only=True`, strict validation, and E31's verified `model.parameters()` ordering.
- Emit normalized CORAL score `(sigmoid(logits).sum(-1) - 5) / 5` and the original standardized 3073-D E31 interaction vector.
- Add profile-aware config, extraction/cache reuse, schema-v2 store/checkpoint support, gate training/evaluation, Kaggle dual-GPU scheduling, tests, and a Vietnamese report.
- Keep the source checkpoint local and unchanged; use no test labels as model inputs; do not launch remote jobs.

## Approaches

1. Duplicate the Full9 paths and fork all assumptions for ten experts. This is simple locally but risks drift and duplicated execution logic.
2. Generalize existing profile-driven contracts and queues while keeping Full9's public profile, checkpoint schema, and defaults intact. This reuses the current cache and store design and is the selected approach.
3. Rebuild the gate/extraction architecture around a new abstraction. This adds unnecessary migration risk and does not fit the existing schema-v2 system.

## Design calls

- The E31 notebook is authoritative for the list checkpoint order: audio LoRA is injected first (ascending ViT blocks, `qkv`, `proj`); text LoRA follows (ascending BERT layers, `query`, `value`); serialization uses the model parameter traversal filtered by `requires_grad`.
- Full10 shares existing `_expert3_audio`, M2D runtime, text resource, and M2D dependencies; it has a distinct adapter/runtime provenance and checkpoint identity.
- Schema-v2 is shared by Full9 and Full10, with the profile and exact order/dimensions checked before reuse or load.
- Kaggle execution takes an explicit profile and maintains one sequential queue per GPU. Full9 remains the default for its existing entrypoint.

## Acceptance

Focused TDD tests, the existing regression suite, safe real-checkpoint restoration/forward if local text resources permit, profile/cache/store/gate/Kaggle tests, and final diff review. GPU extraction/training/evaluation are reported only if actually run.
