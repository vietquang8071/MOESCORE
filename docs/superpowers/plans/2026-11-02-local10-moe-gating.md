# LOCAL-10 Trainable MoE Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight, trainable four-expert gating core and forward wrapper that preserves the local MOESCORE gating mathematics while consuming the registry-derived 7936-wide feature vector.

**Architecture:** Put the original Linear–LayerNorm–ReLU–Dropout–Linear–Softmax network in an import-light module, deriving feature width and expert order from `model.expert_adapters.registry`. Add an orchestration module that invokes frozen adapters under `no_grad`, calls the existing output assembler, computes gating weights and applies the original four-term weighted sum. Keep heavyweight `model/MOE_model.py`, expert sources, checkpoints and loss code unchanged.

**Tech Stack:** Python 3.10, PyTorch, existing `ExpertAdapter`/registry contracts, `unittest`.

**Spec:** User-provided LOCAL-10 specification in the current task.

## Global Constraints

- Gating input dimension is `sum(EXPERT_FEATURE_DIMS[name] for name in EXPERT_ORDER)` = 7936.
- Preserve hidden width 512, `LayerNorm(512)`, in-place ReLU, dropout 0.1, four logits and softmax on the last dimension.
- Preserve original four-term weighted-score formula; do not normalize or calibrate expert scores.
- Call the existing output assembler and preserve `EXPERT_ORDER`.
- Freeze experts and keep them in eval mode after parent `.train()` calls; gating remains trainable and follows parent mode.
- Do not construct real experts, deserialize checkpoints, use real XACLE data, or change loss/training/model-source files.
- Preserve entropy regularization coefficient 0.2 and document the raw-label versus normalized-loss boundary.

## Review Focus

- Incorrect feature ordering or stale 7168 gate width — test exact concatenation order and derived `in_features=7936`.
- Expert dropout reactivated by parent `.train()` — test all expert submodules remain eval while the gate enters train mode.
- Gradients leaking through expert outputs — test integration inference is under `no_grad` and only gating parameters update.
- Batch-dependent gating caused by normalization — test sample outputs remain stable alone and in mixed/reordered eval batches.
- Scores accidentally normalized or clamped — test exact weighted-sum parity, input immutability and convex-combination bounds without imposing `[-1,1]`.

---

### Task 1: Lightweight Gating Network

**Files:**
- Create: `model/moe_gating_core.py`
- Test: `tests/test_moe_gating.py`

**Interfaces:**
- Consumes `EXPERT_ORDER`, `EXPERT_FEATURE_DIMS`, and `EXPECTED_FEATURE_WIDTH` from the existing registry.
- Produces `GatingNetwork(input_feature_dim=EXPECTED_FEATURE_WIDTH, num_experts=len(EXPERT_ORDER))` and returns weights `[B,4]` from features `[B,7936]`.

- [ ] Write tests for exact layer sequence/dimensions, softmax properties, sample dependence, eval batch-independence, B=1/2/4, state-dict round-trip, invalid input, and gating gradients.
- [ ] Run the new gating tests and confirm expected RED failures before implementation.
- [ ] Implement only the original gating sequence; reject incompatible dimensions/expert count and malformed/nonfinite features.
- [ ] Re-run tests and confirm all gating behaviors pass.

### Task 2: Frozen Expert Forward Integration

**Files:**
- Create: `model/moe_forward_integration.py`
- Test: `tests/test_moe_forward_integration.py`

**Interfaces:**
- Consumes a mapping from each registry name to an existing `ExpertAdapter`, a unified batch, and the existing output assembler.
- Produces `(final_score[B,1], weights[B,4])`; experts run once each under `torch.no_grad()` in stable registry order.

- [ ] Write deterministic adapter-stub tests for B=1/2/4, routing/order, single calls, forward parity, input immutability, no-grad experts, gradients to gate, frozen/eval state and error propagation.
- [ ] Run focused tests and confirm expected RED failures.
- [ ] Implement the minimal wrapper; override `train()` so experts stay frozen/eval while the gate follows the requested mode.
- [ ] Re-run integration tests, including a synthetic optimizer step that updates gate parameters only.

### Task 3: Diagnostics, Audits and Regression

**Files:**
- Create: `scripts/verify_moe_gating.py`
- Create: `docs/LOCAL-10/architecture-audit.md`
- Create: `docs/LOCAL-10/implementation-report.md`

**Interfaces:**
- CLI runs only synthetic CPU checks and writes JSON; it never imports/constructs pretrained experts or loads checkpoints.
- Reports cite local source files and distinguish source-proven score domains from unmeasured real distributions.

- [ ] Implement the diagnostic CLI and test its JSON/status behavior.
- [ ] Document original gate/loss/label semantics, 7168-to-7936 migration incompatibility and LOCAL-10.1 score-distribution plan.
- [ ] Run focused tests, syntax compilation, the diagnostic CLI and the complete lightweight unittest suite.
- [ ] Record exact counts, failures/warnings and files changed in the Vietnamese implementation report.

