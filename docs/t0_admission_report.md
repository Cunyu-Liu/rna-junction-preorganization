# T0 tecto Data Admission Report

run_id: v1_2_tecto_qmap_20260803
generated_at_utc: 2026-08-03T15:52:52.587735+00:00
canonical sha256: 0989ddc00bb230fdb00bbc65433c943a0419e35c3d0799b481e741c4a24defe2

## Censoring semantics (-7.1 kcal/mol)
- direction: left-censored (very stable end; more negative than measurable floor)
- basis: Kd <= 5000 nM measurable range; cap is the most stable measurable value
- rows at cap: 5865
- constructs exclusively censored: 6
- likelihood: censored likelihood for rows at cap; do not treat as exact point values

## Replicate / bootstrap / covariance semantics
- per_replicate_rows_present: False
- rows_are_aggregated: True
- bootstrap_ci_columns: ['err10', 'err9', 'err11', 'err10_5mM']
- bootstrap_ci_meaning: 95% CI from bootstrapped cluster fluorescence (paper Figure 1H)
- two_replicate_experiments: reported in paper Figure 1G; not present as row-level replicates
- replicate_variance_recoverable: False
- covariance_default: NOT independent; same construct/scaffold shared across rows
- scaffold_context: chip_scaffold (9 values) reused across constructs; context must enter grouping/hierarchical model
- note: Treat per-row err as measurement uncertainty, not independent replicate noise

## Attrition
- raw_rows: 28935
- with_sublibrary_rows: 28935
- measured_rows: 28157
- noncensored_rows: 22292
- censored_rows_at_cap: 5865
- interpolated_only_rows: 216
- missing_rows: 562
- distinct_constructs: 1713
- distinct_junctionmat_constructs: 1636
- distinct_measured_constructs: 1713
- distinct_noncensored_constructs: 1706
- distinct_exclusively_censored_constructs: 6

## Effective N
- constructs: 1713
- motifs: 60
- scaffolds: 9
- studies: 1
- junctionmat_constructs: 1636
- measured_constructs: 1713
- noncensored_constructs: 1706
- exclusively_censored_constructs: 6
- independent_scaffold_groups: 9
- independent_study_groups: 1
- connected_components: 1
- giant_component_size: 1713

## Motif-construct-scaffold-study graph
- levels: {'constructs': 1713, 'scaffolds': 9, 'motifs': 60, 'studies': 1}
- connected components: 1
- giant component size: 1713
- component size distribution: [1713]

## Outer holdout feasibility
- construct_holdout: {'feasible': True, 'note': 'hold out whole junction_ids; leakage via shared scaffold/motif must be blocked'}
- motif_family_holdout: {'feasible': True, 'n_motifs': 60, 'note': 'hold out whole motif families; construct->scaffold reuse must be blocked'}
- scaffold_holdout: {'feasible': True, 'n_scaffolds': 9, 'note': 'only 9 scaffolds; high risk of scaffold-level confounding'}
- study_holdout: {'feasible': False, 'n_studies': 1, 'note': 'single study (Denny 2018); cross-study generalization requires qMaPseq or other platform'}
- giant_component_rule: {'random_row_split': 'FORBIDDEN', 'same_construct_cross_fold': 'FORBIDDEN', 'required': 'pre-registered multi-axis blocked generalization'}

## Provenance
- source_is_paper_supplementary_workbook: True
- source_path: 261_SI.xlsx (Denny et al. 2018 Cell, supplementary)
- row_level_source_row_present: True
- row_level_checksum: per-row SHA-256 of serialized canonical JSON computed at build
- complete: True

## License
- status: CELL_PAPER_SUPPLEMENTARY_DATA
- note: NIH open-access author manuscript (PMC6053692); Cell article data. Verify the specific Cell license (Elsevier) before any redistribution; analysis/reproduction use is standard but redistribution terms must be confirmed.
- allowed_for_analysis: True
- redistribution_confirmed: False

## Gate status
- T0 is RUNNING, NOT PASS. The finalizer must confirm all 18 T0 admission items.

