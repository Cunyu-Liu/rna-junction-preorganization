# RNA Junction Preorganization v1.2 — Reviewer Attack Matrix

**Run**: `v1_2_tecto_qmap_20260803`

## Anticipated attacks & controls

| Attack | Response | Evidence |
|---|---|---|
| "Transfer claim is restricted to 98 mutants" | Owned: restricted to mttr6 TTR set; no extrapolation to arbitrary TLR; explicitly stated in claim bounds | Q4 selection_boundary; Q5 restrictions |
| "Unbalanced fold sizes (83/11/2/2)" | Mutation graph structure dictates K=4 max without leakage; small folds contribute high-variance estimates, reported honestly | Q4 mutation_graph + fold_assignment |
| "B4 depends on tecto model" | True — B4 uses tecto old_dg as input; transfer claim presumes pre-existing tecto model; not a standalone predictor | Q5 B4 definition; model_card |
| "Why does B2 (mg_1_2) perform so badly?" | 11 right-censored variants have unreliable mg_1_2 (>40 or unstable); B2 RMSE=1.463 degraded by extreme values; this is a known weakness of the published univariate relationship | Q2 attrition; Q5 B2 |
| "Data leakage / split overlap" | Mutation-graph Hamming-1 adjacency; connected components assigned atomically; leakage_violations=0 | Q4 freeze_summary |
| "Trend-only or correlation-only replay" | Q3 requires per-variant per-endpoint comparison; no trend-only pass; 1,666 records, 0 FAIL | Q3 replay_summary |
| "Tecto model doesn't beat baseline" | Owned as tecto-only limitation; T3 proper score 41.8 vs 27.0; the transfer claim does not depend on tecto model beating baseline, only on old_dg being informative input | T3 results; Q5 B4 |
| "1 kcal precision not met" | Tecto-only interval width 1.35 > 1.0; Q5 B4 RMSE=0.1951 < 1.0 but that is transfer RMSE, not tecto precision; both reported honestly | T2/T3; Q5 |
| "Are Bonilla/Shin/Yesselman independent?" | No — single RNA-MaP/tectoRNA platform cluster; explicitly disclaimed | dataset_card |
| "qMaPseq independently reproduced junction preorganization?" | No — Q5 is a restricted transfer test, not reproduction; transfer ≠ mechanism | claim_adjudication §4 |
| "DMS proves it?" | DMS permanently NOT_ADMITTED; reactivity ≠ ΔG; no crosswalk | estimand_spec; data_availability |
| "Not reproducible / no provenance" | Full manifest + 12 sentinels + SHA-256 + Q3 tolerances frozen + Q4 freeze locked + Q5 locked before run | reproducibility |
| "Overclaimed publication" | Claim caps enforced; STRONG_CROSS_SYSTEM_RESTRICTED not full cross-system; mechanism declined | claim_adjudication |

## One-line defensibility
Every claim is bounded: transfer is restricted to 98 mttr6 TTR mutants with 4-fold mutation-graph CV; no mechanism, no arbitrary-TLR generalization, no DMS, no 1-kcal tecto precision, no tecto-baseline-beating claim is made.
