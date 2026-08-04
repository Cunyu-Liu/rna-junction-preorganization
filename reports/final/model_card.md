# RNA Junction Preorganization v1.2 — Model Card

**Run**: `v1_2_tecto_qmap_20260803`

## Q5 B4 — locked partial-ID calibration (cross-system transfer)
- **Task**: predict held-out RNA-MaP reference `rna_map_dg` (kcal/mol) for mttr6 TTR mutants from tecto `old_dg` via isotonic calibration.
- **Model**: linear calibration `rna_map_dg ~ old_dg` fit on training fold, then isotonic regression on (predicted, true) training pairs; frozen before viewing transfer outcome.
- **Stage**: Q5 (locked transfer test). Device: CPU (isotonic regression + 4-fold CV).
- **Inputs**: 98 mttr6 TTR variants; tecto `old_dg` from T2/T3 tecto pipeline; RNA-MaP `rna_map_dg` from Zenodo `rna_map_dg.csv`.
- **Outputs**: point prediction + Gaussian prediction interval (68%/95% PI).
- **Performance**: held-out RMSE = 0.1951 kcal/mol; NLPD = 0.5334; Spearman = 0.947; 68% coverage = 0.682; 95% coverage = 0.884; 68% PI width = 0.3518; calibration slope = 0.850, intercept = -1.4674.
- **Gain over B1 (mean)**: 0.511 kcal/mol (95%CI [0.403, 0.618], p=0.0006355).
- **Limitations**: restricted to 98 mttr6 TTR mutants; requires pre-existing tecto model; unbalanced fold sizes [83,11,2,2].

## Q5 baselines (for comparison)
- **B1 (intercept/mean)**: predict training-set mean `rna_map_dg`. RMSE=0.7056.
- **B2 (published univariate mg_1_2)**: linear `rna_map_dg ~ mg_1_2`. RMSE=1.4633 (degraded by censored variants).
- **B3 (sequence/mutation)**: linear `rna_map_dg ~ (bp_muts one-hot + mutation_count)`. RMSE=0.7391.

## T2 — censored-likelihood partial-ID estimator (tecto-only, context)
- **Task**: recover target-specific thermodynamic functional (ΔG, kcal/mol) for two-way junction insertion from left-censored (−7.1 kcal/mol) tectoRNA measurements.
- **Model**: censored-likelihood (Tobit-type) estimator with junction/motif/scaffold structure; outputs identified set/interval.
- **Stage**: T2. Device: `cuda` (A100 MIG 1g.5gb).
- **Inputs**: 11,893 rows (9,961 measured, 1,932 censored), 1,336 junctions, 15 motifs, 9 scaffolds.
- **Calibration**: interval coverage 0.957 on held-out; M0 synthetic coverage 0.9–1.0.
- **Limitations**: interval width median 1.35 kcal/mol (>1.0); does not reach 1 kcal precision.

## T3 — hierarchical target-specific functional (tecto-only, context)
- **Model**: hierarchical (motif + scaffold random effects) over censored-likelihood identified-set targets.
- **Comparison**: matched baselines on frozen motif-family holdout (n=392).
- **Result**: proper score 41.8 vs motif_mean 27.0; `t3_beats_baseline = false`. Honest negative.
