# RNA Junction Preorganization v1.2 — Final Delivery Report

**Run**: `v1_2_tecto_qmap_20260803` | **Contract**: `rna 三级.md` (SHA-256 `32d09729...`)
**Date**: 2026-08-04 UTC

## 1. Preflight summary
- Host: `bms-18937653-012` (A100, 8× GPU). Worktree `/home/cunyuliu/rna_junction_preorganization_v1_2_20260803` (branch `v1.2/tecto-qmap`), clean.
- Tecto gates T0–S0–T1–M0–T2–T3 = **PASS**; Q0 = **NOT_ADMITTED**.
- Raw data / artifacts under `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`; code in isolated worktree; main checkout untouched; no unrelated processes killed.

## 2. What was executed end-to-end
- **T0** data admission: fixed Denny sources/versions, reconstructed 1,687/1,713/1,636 sets, per-difference evidence.
- **S0** estimand/operator/symmetry freeze (see `specs/estimand_spec.json`, `symmetry_frame_spec.json`, `operator_uncertainty_spec.json`).
- **T1** cleaning/QC/effective-N/split freeze; motif-family holdout; leakage audit.
- **M0** synthetic & operator-identification tests (interval coverage calibration on synthetic fixtures).
- **T2** tecto-only inference: censored-likelihood partial-ID; coverage 0.957; permutation `signal_detected=true`; disposition `INCONCLUSIVE_FOR_1_KCAL_PRECISION`.
- **T3** target-specific functional: hierarchical vs matched baselines; does **not** beat motif-mean (proper score 41.8 vs 27.0); operator sensitivity; group bootstrap.
- **Q0** integrity & license freeze: ENA 16 FASTQ SHA-256 verified + rna_map clone (Apache-2.0); Figshare 403 / Zenodo refused → `QMAP_NOT_ADMITTED`.

## 3. Claim adjudication (contract §24)
- **Combination**: Tecto PASS + QMAP_NOT_ADMITTED
- **Allowed max claim**: tecto-specific reanalysis, benchmark, or partial-ID case study.
- **Claim class**: `TECTO_REANALYSIS_AND_PARTIAL_ID_CASE_STUDY`
- **Mechanism / cross-system / 1-kcal / baseline-beating claims**: NOT supported; permanently prohibited list enforced.

## 4. Completion status
```
IMPLEMENTATION_COMPLETE = true
SCIENTIFIC_SUCCESS = inconclusive
PUBLICATION_ROUTE = TECTO_REANALYSIS_AND_PARTIAL_ID_CASE_STUDY
```

## 5. Deliverables generated
- `claim_adjudication.md`, `claim_matrix.json`, `limitations.md`, `data_availability.md`, `reproducibility.md`, `model_card.md`, `dataset_card.md`, `reviewer_attack_matrix.md`
- Canonical manifest updated with Q0 + claim_class; sentinels for all gates.

## 6. Recommended paper route
A **tecto-specific partial-identification case study / reanalysis**:
1. Auditable data census & provenance (T0/S0/T1).
2. Synthetic operator/model calibration (M0) — coverage honesty.
3. Censored-likelihood partial-ID inference on the tectoRNA platform (T2) with intervals, coverage, negative controls.
4. Honest model-vs-baseline comparison (T3) reporting the negative gain as a platform limitation.

Mechanism title and cross-system claims must be dropped (not supported).