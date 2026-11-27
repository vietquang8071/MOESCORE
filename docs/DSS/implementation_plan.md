# MOE8-DSS implementation plan

The approved contract is [DSS-overview.md](DSS-overview.md). This branch is already isolated at `d517001` on Full8 `01c57ff`. The frozen cache was made from `68124d0`; the Full8 store validator, gate, registry, and target transform have no source diff between those commits. The historical cache and baseline configuration remain read-only.

1. Add ranking-loss tests, then implement L2 `torchsort` Soft Spearman, exact average target ranks, invalid-group accounting, and finite-value checks in `loss/domain_spearman.py`.
2. Add sampler tests, then implement deterministic 4 × 32 scheduling, per-domain shuffled queues, exposure reporting, and epoch-boundary state in `training/domain_balanced_sampler.py`.
3. Add data/training/evaluation integration tests, then implement a dedicated DSS runner and CLI. Use the existing Full8 gate, cache validator, metric function, and score transform. Join official metadata to cache sample IDs; keep all nine domains; select by full-validation hard SRCC.
4. Add an explicit DSS configuration and a Save & Run All Kaggle notebook with a pinned source commit, input discovery, preflight, separate smoke/full outputs, and required artifact exports.
5. Run targeted and Full8 regression tests; review the whole diff against the contract, fix P0/P1 issues, and write the implementation and review reports.
6. Commit and push this branch, then push the dedicated Kaggle kernel. Per the latest user instruction, stop immediately after a successful Kaggle push and provide the kernel link and log command. A push confirms submission, not GPU success.

Scope: DSS-only code, tests, config, notebook, and reports. No baseline, LOGO, expert, or historical cache changes. Kaggle outputs must stay outside tracked source.
