# Final Claim Adjudication — Q3-Q5 Complete

**Date**: 2026-08-04
**Branch**: v1.2/tecto-qmap
**Commit**: 0874c88
**Terminal State**: `QMAP_TRANSFER_SUPPORTED`

## All Gates

| Gate | Status | Summary |
|------|--------|---------|
| T0-S0-T1-M0-T2-T3 | PASS | Tecto reanalysis complete |
| Q0 | PASS | Provenance verification (Figshare + Zenodo) |
| Q1 | PASS | 99-variant registry (1 ref + 98 mutants) |
| Q2 | PASS | Attrition: 84 fitted + 11 censored + 2 closing-pair + 1 alt-structure = 98 |
| Q3 | PASS | Endpoint replay: 1600 PASS + 66 NA + 0 FAIL (tolerances frozen) |
| Q4 | PASS | Selection/split freeze: 4 folds, 0 leakage, 12 items locked |
| Q5 | PASS | Transfer test: B4 RMSE=0.195, gain=0.51, perm p=0 |

## Q3 Endpoint Replay

- 8 endpoints × 98 variants, tolerances frozen BEFORE run
- Numerical (mg_1_2, errs, gaaa_avg, rna_map_dg): abs=1e-6 OR rel=1e-4
- Categorical (failure/censoring/structural-QC reason): exact string match
- 11 right-censored variants: NOT_APPLICABLE for E2/E3 (authoritative = E7)
- Result: 1600 PASS + 66 NOT_APPLICABLE + 0 FAIL

## Q4 Selection/Split Freeze

- 98 variants from mttr6 TTR mutant set (no extrapolation to arbitrary TLR)
- Mutation graph: 193 edges, 4 connected components (sizes 83/11/2/2)
- K=4 folds (determined by graph structure), 0 leakage violations
- All 12 freeze items locked before viewing transfer outcome

## Q5 Locked Transfer Test

### Baselines
| Baseline | RMSE (kcal/mol) | NLPD | Spearman | 68% Coverage |
|----------|----------------|------|----------|-------------|
| B1 (mean) | 0.706 | 1.092 | — | 0.831 |
| B2 (mg_1_2) | 1.463 | 6.124 | 0.768 | 0.769 |
| B3 (mutations) | 0.739 | 1.166 | -0.175 | 0.798 |
| B4 (tecto-dG cal) | **0.195** | **0.533** | **0.947** | **0.682** |

### Adjudication Criteria
| Criterion | Result |
|-----------|--------|
| (a) B4 RMSE < 1.0 kcal/mol | PASS (0.195) |
| (b) Gain over B1 > 0.3, 95%CI excludes 0 | PASS (0.51, CI [0.40, 0.62]) |
| (c) 68% coverage in [0.55, 0.80] | PASS (0.682) |
| (d) Label permutation p < 0.05 | PASS (p=0) |

### Additional Checks
- Mutation-class bootstrap 95%CI: [0.36, 0.63] (excludes 0)
- All 4 criteria pass → **QMAP_TRANSFER_SUPPORTED**

## Maximum Allowable Claim

**STRONG_CROSS_SYSTEM (restricted)**: The tecto model (old_dg) with isotonic calibration supports a restricted cross measurement-system transfer claim: held-out RNA-MaP reference ΔG can be predicted to within 0.195 kcal/mol RMSE (95% CI of gain over mean baseline: [0.40, 0.62] kcal/mol), within the 98-variant mttr6 TTR mutant set, using 4-fold mutation-graph-aware cross-validation.

## Limitations
1. Restricted to 98 mttr6 TTR mutants — no extrapolation to arbitrary TLR families
2. 4-fold CV with unbalanced fold sizes (83/11/2/2) due to mutation graph structure
3. B4 uses tecto old_dg as input — requires a pre-existing tecto model
4. 11 censored variants enter likelihood but their mg_1_2 is unreliable
5. B2 (mg_1_2 univariate) performs poorly due to censored variants with extreme mg_1_2 values
