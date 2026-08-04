# RNA Junction Preorganization v1.2 — Final Delivery Report

**Run**: `v1_2_tecto_qmap_20260803` | **Contract**: `rna 三级.md` (SHA-256 `32d09729638b7681...`)
**Date**: 2026-08-04 UTC

## 1. Preflight summary
- Host: `bms-18937653-012` (A100, 8× GPU). Worktree `/home/cunyuliu/rna_junction_preorganization_v1_2_20260803` (branch `v1.2/tecto-qmap`), clean.
- **All 12 gates PASS**: T0–S0–T1–M0–T2–T3–Q0–Q1–Q2–Q3–Q4–Q5.
- qMaP terminal state: `QMAP_TRANSFER_SUPPORTED`. Max allowable claim: `STRONG_CROSS_SYSTEM_RESTRICTED`.
- Raw data / artifacts under `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`; code in isolated worktree; no unrelated processes killed.

## 2. What was executed end-to-end
- **T0** data admission: Denny sources/versions fixed; 1,687/1,713/1,636 sets reconstructed.
- **S0** estimand/operator/symmetry freeze.
- **T1** cleaning/QC/effective-N/split freeze; motif-family holdout; leakage audit.
- **M0** synthetic & operator-identification tests (interval coverage calibration).
- **T2** tecto-only inference: censored-likelihood partial-ID; coverage 0.957; `signal_detected=true`; `INCONCLUSIVE_FOR_1_KCAL_PRECISION`.
- **T3** target-specific functional: hierarchical vs matched baselines; does **not** beat motif-mean (41.8 vs 27.0); operator sensitivity.
- **Q0** integrity & license: ENA 16 FASTQ SHA-256 + rna_map clone + Figshare + Zenodo all admitted (recovered after v1.1 network failure).
- **Q1** registry: 99 variants (1 ref + 98 mutants).
- **Q2** attrition: 84 fitted + 11 censored + 2 closing-pair + 1 alt-structure = 98.
- **Q3** endpoint replay: 1600 PASS + 66 NA + 0 FAIL (tolerances frozen).
- **Q4** selection/split freeze: 98 variants, 4 folds, 0 leakage, 12 items locked.
- **Q5** locked transfer test: B4 RMSE=0.1951, gain=0.511 (CI [0.403,0.618]), cov68=0.682, perm p=0.0 → `QMAP_TRANSFER_SUPPORTED`.

## 3. Claim adjudication (contract §24)
- **Combination**: Tecto PASS + QMAP_TRANSFER_SUPPORTED
- **Allowed max claim**: STRONG_CROSS_SYSTEM_RESTRICTED (restricted cross measurement-system transfer).
- **Restrictions**: 98 mttr6 TTR mutants only; no extrapolation to arbitrary TLR; B4 requires tecto model.

## 4. Completion status
```
IMPLEMENTATION_COMPLETE = true
ALL_12_GATES_PASS = true
QMAP_TERMINAL_STATE = QMAP_TRANSFER_SUPPORTED
MAX_ALLOWABLE_CLAIM = STRONG_CROSS_SYSTEM_RESTRICTED
```

## 5. Deliverables generated
- `claim_adjudication.md`, `claim_matrix.json`, `limitations.md`, `data_availability.md`, `reproducibility.md`, `model_card.md`, `dataset_card.md`, `reviewer_attack_matrix.md`, `final_delivery_report.md`
- Canonical manifest with all 12 `gate_decisions` populated; sentinels for all gates.
- Q3–Q5 artifacts: `q3_replay_comparison.jsonl` (1,666 records), `q4_mutation_graph.json` + `q4_fold_assignment.json`, `q5_transfer_summary.json` + 4 baseline fold_results.
