# MOE27 Whisper Top 2 Feature Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the original Top 2 Whisper-large-v2 extraction path as a reusable frozen E6 component with a resumable split-aware feature cache.

**Architecture:** Add a standalone `nn.Module` wrapper around the exact Hugging Face feature extractor, tokenizer and `WhisperModel`; validate unified batches and return two pooled vectors. Store CPU float32 outputs in atomic NPZ shards with a checksummed manifest, then expose extraction and one-sample smoke modes through a CLI.

**Tech Stack:** Python, PyTorch, Transformers, NumPy, pytest, PyYAML.

**Spec:** User request MOE27-02, 2026-11-18.

## Global Constraints

- Use `openai/whisper-large-v2` with `AutoFeatureExtractor`, `WhisperTokenizer`, and `WhisperModel`.
- Input is `expert3_wavs: Tensor[B,1,160000]` and aligned raw `captions_raw` at 16 kHz.
- Preserve direct caption tokenization, decoder conditioning, and unmasked mean pooling.
- Return finite float32 `(B,1280)` audio and text tensors; keep the model frozen and in eval mode.
- Do not add a score head, similarity, registry entry, or global feature-store changes.
- Default to offline loading; do not download weights or commit model resources.
- Keep train, validation, and test caches isolated and reject identity mismatches/duplicate IDs.

## Review Focus

- Raw caption whitespace and decoder padding positions must remain represented — tests pin exact tokenizer inputs and ordinary sequence means.
- A wrong Whisper variant or incomplete local checkpoint must fail clearly — tests pin loading arguments and local resource validation.
- A cache crash or changed identity must not cause stale or duplicate rows to be reused — tests pin metadata, checksums, and duplicate detection.
- Batch/sample ordering must survive resume — tests pin ID-to-feature row order and split isolation.
- Feature extractor output must follow the 80x3000 frontend contract and finite f32 output contract — tests pin shape, dtype, and non-finite rejection.

---

### Task 1: Whisper-large-v2 component and mock tests

**Files:**
- Create: `model/feature_extractors/__init__.py`
- Create: `model/feature_extractors/whisper_top2.py`
- Test: `tests/test_whisper_top2.py`

**Interfaces:**
- Consumes: unified batch keys `expert3_wavs`, `captions_raw`.
- Produces: `WhisperTop2FeatureExtractor.from_config(mapping)` and callable output `(audio[B,1280], text[B,1280])`; expose checkpoint and preprocessing identities.

- [x] Write mock tests for input validation, exact preprocessing/tokenizer/model calls, mean pooling, frozen/eval state, deterministic outputs, and offline loading.
- [ ] Run the tests; collection is blocked by missing PyTorch (`ModuleNotFoundError: torch`).
- [x] Implement the wrapper and loading/resource validation.
- [ ] Extractor test execution remains blocked; focused Python syntax check passes.

### Task 2: Dedicated split-aware feature cache

**Files:**
- Create: `data/whisper_feature_cache.py`
- Test: `tests/test_whisper_feature_cache.py`

**Interfaces:**
- Consumes: sample IDs, audio/text matrices, split, checkpoint/preprocessing identities.
- Produces: `WhisperFeatureCacheWriter.append`, `.sample_ids`, `.load`; arrays are CPU NumPy float32 with width 1280 each.

- [x] Write tests for identity validation, duplicate IDs, finite features, atomic shard/manifest save and load, and split separation.
- [ ] No pre-implementation RED result was recorded; the initial system pytest command could not start because pytest was absent, and the cached Python 3.12 runner was discovered after implementation.
- [x] Implement atomic NPZ shards and checksummed JSON metadata.
- [x] Run cache tests (8 passed) and syntax checks.

### Task 3: Extraction CLI, example config, and implementation report

**Files:**
- Create: `scripts/extract_whisper_features.py`
- Create: `configs/whisper_top2.example.yaml`
- Create: `docs/MOE27-Whisper-implementation.md`

**Interfaces:**
- Consumes: one existing MOESCORE dataset split through `UnifiedXACLEDataset` and `collate_unified`.
- Produces: `result/whisper_feature_cache/<split>/` and a bounded `--smoke-test` path.

- [x] Implement sequential extraction with configurable batch size defaulting to 1 and resumable sample-ID filtering.
- [x] Document commands, cache identities/schema, unavailable checkpoint/dependencies, verification limits, and E6 import/feature contract.
- [x] Run script help and syntax checks.

### Task 4: Repository verification and commit

**Files:**
- Review only: Whisper-owned files from Tasks 1–3.

- [x] Run new tests, relevant existing tests, and the repository suite where runtime dependencies permit; cache tests and one existing test pass, full collection stops on missing PyTorch.
- [x] Check local resources; exact Whisper-large-v2 weights and runtime dependencies are absent, so bounded real-model parity cannot run.
- [ ] Review the diff and stage only Whisper-related files.
- [ ] Commit on `dev/moe27-whisper` with `feat(whisper): port Top2 Whisper feature extractor for E6`.
