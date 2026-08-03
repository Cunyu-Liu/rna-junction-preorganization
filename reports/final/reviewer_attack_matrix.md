# RNA Junction Preorganization v1.2 — Reviewer Attack Matrix

**Run**: `v1_2_tecto_qmap_20260803`

## Anticipated attacks & controls

| Attack | Response | Evidence |
|---|---|---|
| "Interval is too wide / 1 kcal not met" | Honest: partial-ID functional; median 1.35 kcal/mol; coverage 0.957; sensitivity shown | T2/T3 results |
| "Model doesn't beat a simple baseline" | Owned as a limitation; frame as the honest benchmark + interval methodology | T3 proper score 41.8 vs 27.0 |
| "Data leakage / split overlap" | motif-family disjoint holdout; mmseqs split; homolog leakage n_overlap=0 | T1/T2 result |
| "Are Bonilla/Shin/Yesselman independent validations?" | No — single RNA-MaP/tectoRNA platform cluster; explicitly disclaimed | dataset_card |
| "Why not qMaPseq cross-validation?" | Q0 NOT_ADMITTED (Figshare 403 / Zenodo refused); cross-system claim closed per contract | Q0 finalizer |
| "DMS proves it?" | DMS permanently NOT_ADMITTED; reactivity ≠ ΔG; no crosswalk | estimand_spec |
| "Synthetic validation is not biological" | Synthetic is operator/model calibration only (M0), never a biology claim | M0 finalizer |
| "Not reproducible / no provenance" | Full manifest + sentinels + SHA-256 + replay path | reproducibility |
| "Overclaimed publication" | Claim caps enforced; mechanism position declined | claim_adjudication |

## One-line defensibility
Every claim is bounded by the identified-set/coverage framing; no mechanism, no cross-system, no DMS, no 1-kcal precision, no baseline-beating claim is made.