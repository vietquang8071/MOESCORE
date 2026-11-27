# MOE8-DSS code review report

## Scope and method

Reviewed the DSS implementation on `experiment/moe8-domain-wise-spearman` against all sections of `DSS-overview.md`, the connected Full8 gate/cache/metric code, the official CSV and downloaded manifest contracts, the Kaggle notebook, and the full Git change list. A separate reviewer inspected the untracked DSS files and reported findings; the implementation was then reviewed and retested after fixes. No baseline, LOGO, expert, or cache source files were changed.

## Findings and fixes

| Severity | Finding | Evidence and resolution |
|---|---|---|
| P1 | Kaggle notebook initially had a source-commit placeholder. | Both notebook copies now pin reviewed implementation commit `00ec238d6e280cfd319fc3317f3bc02bd2942732`; notebook test parses and compiles every code cell. |
| P2 | Runner resume and final export paths lacked integration checks. | Added a two-epoch synthetic runner test with forced interruption and exact gate-state comparison after resume; added a 13,500-row artifact/plot test. |
| P2 | Baseline public-test availability was asserted without searching. | Preflight now searches the mounted bundle for candidate saved prediction/metric files. Only the paired baseline validation result is compared when provenance permits; public-test deltas are explicitly unavailable if no compatible data exist. |
| P2 | Exceptions left only `PARTIAL` metadata. | Runner now writes `FAILED` and the exception reason while preserving an already complete artifact. |
| P2 | Git HEAD alone could misidentify uncommitted source. | Preflight rejects dirty source checkouts. Kaggle checks out the reviewed implementation commit exactly. |
| P2 | Official WAV bytes were not compared directly with cache fingerprints. | Preflight now hashes every official WAV and rejects mismatches; a synthetic mismatch test verifies the failure. |
| P2 | Local torchsort build failed because local CUDA toolkit 12.0 differs from PyTorch CUDA 13.0. | DSS uses the exact L2 isotonic PAV operator in pure PyTorch; tests compare against an independent SciPy isotonic solver and finite-difference gradients. GPU runtime remains a smoke-test gate. |

## Critical protocol checks

- Prediction ranks use ascending exact L2 permutahedron projection; target ranks use exact average ties. No prediction tensor is detached or converted to NumPy in the differentiable path.
- The optimized objective contains only equal-weight global and valid-domain Soft Spearman terms. Original MSE, contrastive and diversity functions are not called by the DSS trainer.
- Every batch has 4 distinct domains × 32 distinct samples; 59 batches per epoch, all nine domains, 26 or 27 appearances per domain. Sampler queues and RNG survive epoch-boundary resume.
- The gate is freshly constructed from the existing `GatingNetwork`; baseline gate bytes are hashed for provenance and never loaded into its parameters.
- Cache alignment uses split-qualified identity, caption SHA, official WAV SHA, label agreement, ordered eight-score/33,032-feature contracts and the existing full shard validator. Test labels do not enter the sampler, optimizer or selection metric.
- Validation selection uses hard SRCC on all 3,000 official validation samples, strict finite improvement and earliest ties. Final inference reloads `best.pt` and uses ordinary non-replacement loaders.
- Raw MSE, Spearman, Pearson and Kendall tau-b, nine-domain aggregates, score/routing diagnostics, 13,500 prediction records, and required plots are exported by the final path.

## Verification

- DSS plus Full8/D01 regression suite including the pinned notebook: **86 passed**, 17 subtests passed, three expected synthetic-overlap warnings.
- Synthetic runner interruption/resume produced identical selected epoch, validation SRCC, and gate tensors to an uninterrupted run.
- A full repository suite with a temporary `MOESCORE` import alias yielded **623 passed, 1 skipped, 12 failed**. All 12 failures are in unchanged Expert2/3/4 or existing Full9 notebook tests; the Full9 failure asserts Python 3.12 metadata while its checked-in notebook says 3.13.16. No DSS or Full8 regression test failed.
- The full frozen shard payload, all official WAV bytes, CUDA/PAV performance, and Kaggle resource limits cannot be verified locally and are required evidence from the submitted GPU smoke run.

## Verdict

No unresolved P0 or P1 finding remains in the reviewed source and notebook. The first Kaggle submission is a two-epoch smoke run. A successful `kaggle kernels push` confirms submission only; it does not establish GPU smoke success or authorize the full run until logs and artifacts are checked.
