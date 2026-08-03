# RNA Junction Preorganization v1.2 — Limitations

**Run**: `v1_2_tecto_qmap_20260803`

## Scientific limitations
1. **Precision**: The target-specific functional is only identified as an interval; median interval width 1.35 kcal/mol (p90 1.66) does not reach 1 kcal/mol precision. Only 11.2% of junction intervals ≤ 1.0 kcal/mol.
2. **Model performance**: The hierarchical (motif+scaffold) model does **not** beat a simple motif-mean baseline on the frozen motif-family holdout (proper score 41.8 vs 27.0; held-out ranking τ=0.06, ρ=0.07). Gains to date are not recoverable at the current effective-N.
3. **Platform specificity**: All results are within the RNA-MaP/tectoRNA platform cluster (Denny/Bonilla/Shin/Yesselman). They do not constitute independent external measurement systems.
4. **Cross-system evidence**: The strong cross-measurement-system (qMaPseq) claim is closed because the canonical labeled dataset is not verifiable from the host (Figshare HTTP 403, Zenodo connection refused). Q0 = QMAP_NOT_ADMITTED.
5. **Censoring**: 1,932 rows are left-censored at the −7.1 kcal/mol floor; inference relies on a censored likelihood, and the identified set is wider where censoring dominates.
6. **Effective-N**: Reported groups are junctions (N=1,336), scaffolds (N=9), motifs (N=15). Reads/rows must not be treated as independent N; group-adjusted effective-N is modest.

## Conservative boundary (fail-closed)
- No absolute free energy independent of platform/scaffold is claimed.
- DMS reactivity / geometric state / sequence embedding are treated as distinct objects, not the same latent truth as ΔG.
- No qMaPseq reproduction, no cross-junction generalization, no DMS-validation claim.

## Not addressed
- No new wet-lab data; no qMaPseq Q1–Q5 (blocked by Q0 NOT_ADMITTED).
- No current-DMS crosswalk (permanently closed under v1.2).