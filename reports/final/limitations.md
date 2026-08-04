# RNA Junction Preorganization v1.2 — Limitations

**Run**: `v1_2_tecto_qmap_20260803`

## Cross-system transfer (Q5) limitations
1. **Restricted variant scope**: transfer claim restricted to 98 mttr6 TTR mutants. No extrapolation to arbitrary TLR families, other tetraloop/receptor combinations, or non-mttr6 scaffolds.
2. **Unbalanced fold sizes**: 4-fold CV with fold sizes [83, 11, 2, 2] dictated by mutation graph structure (giant component of 83). K=4 is maximum without leakage; small folds (size 2) contribute high-variance fold-level estimates.
3. **B4 depends on pre-existing tecto model**: B4 uses tecto old_dg as input feature; the transfer claim presumes a tecto model has already been fit. Not a standalone RNA-MaP→ΔG predictor.
4. **Censored variants**: 11 right-censored variants enter the likelihood but their mg_1_2 is unreliable (mg_1_2 > 40 or unstable fit). Their numerical mg_1_2 is NOT a valid replay endpoint; the censoring reason (E7) is authoritative.
5. **B2 (mg_1_2 univariate) poor**: B2 RMSE=1.463 kcal/mol, degraded by censored variants with extreme mg_1_2 values. This is a known weakness of the published univariate relationship, not of the transfer test.

## Tecto-only (T2/T3) limitations
6. **Precision**: tecto-only identified-set median interval width 1.35 kcal/mol (p90 1.66); only 11.2% ≤ 1.0 kcal/mol. Does not reach 1 kcal/mol precision.
7. **Model performance**: hierarchical (motif+scaffold) model does **not** beat motif-mean baseline on frozen motif-family holdout (proper score 41.8 vs 27.0; τ=0.06, ρ=0.07).
8. **Platform specificity**: tecto results within RNA-MaP/tectoRNA platform cluster (Denny/Bonilla/Shin/Yesselman). Not independent external measurement systems.
9. **Censoring**: 1,932 tecto rows left-censored at −7.1 kcal/mol floor; inference relies on censored likelihood.
10. **Effective-N**: tecto groups are junctions (N=1,336), scaffolds (N=9), motifs (N=15). Group-adjusted effective-N is modest.

## Conservative boundary (fail-closed)
- Transfer claim is **restricted cross-system**, not full cross-system generalization.
- Transfer ≠ mechanism; transfer ≠ reproduction of junction preorganization.
- No absolute free energy independent of platform/scaffold is claimed.
- DMS reactivity / geometric state / sequence embedding are distinct objects, not the same latent truth as ΔG.
- current 7,500-construct DMS permanently NOT_ADMITTED_FINAL_V1_2.

## Condition controls (Q5)
- Closing-pair-only mutants (n=18): mean residual = -0.0156 (near zero, as expected — near-wild-type ΔG).
