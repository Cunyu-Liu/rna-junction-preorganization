# RNA Junction Preorganization v1.2 — Model Card

**Run**: `v1_2_tecto_qmap_20260803`

## T2 — censored-likelihood partial-ID estimator
- **Task**: recover the target-specific thermodynamic functional (ΔG, kcal/mol) for two-way junction insertion from left-censored (at −7.1 kcal/mol) tectoRNA measurements.
- **Model**: censored-likelihood (Tobit-type) estimator with junction/motif/scaffold structure; outputs an identified set/interval, not a pseudo-exact point.
- **Stage**: T2 (tecto-only inference). Device: `cuda` (A100 MIG 1g.5gb).
- **Inputs**: 11,893 rows (9,961 measured, 1,932 censored), 1,336 junctions, 15 motifs, 9 scaffolds.
- **Calibration**: interval coverage 0.957 on held-out; synthetic coverage validated in M0 (0.9–1.0).
- **Limitations**: interval width median 1.35 kcal/mol (>1.0); does not reach 1 kcal precision.

## T3 — hierarchical target-specific functional
- **Model**: hierarchical (motif + scaffold random effects) predictive functional over the censored-likelihood identified-set targets.
- **Comparison**: matched simple baselines (motif-mean, scaffold-mean) on frozen motif-family holdout (n=392 rows).
- **Result**: T3 proper score 41.8 vs motif-mean 27.0; relative gain −0.55; boots. pos frac 0.0 → **does not beat baseline**.
- **Ranking**: held-out Kendall τ=0.06, Spearman ρ=0.07 (weak).
- **Operator sensitivity**: dg11 width median 0.55, dg10_5mM 0.50 kcal/mol (tighter but sensitivity-only).

## Intended use
- Conditional thermodynamic preference / partial-ID functional within the tectoRNA platform (10 mM Mg²⁺).
- **Not** for absolute ΔG, cross-system claims, DMS equivalence, or all-junction generalization.

## Ethical/safety
- Computationally derived; no new wet-lab or human data. Data provenance audited; no fabricated labels.