# RNA Junction Preorganization v1.2 — Reproducibility

**Run**: `v1_2_tecto_qmap_20260803`

## Provenance & checksums
- Contract: `rna 三级.md` SHA-256 `32d09729638b7681b6efcfdf8b2addc3c7f83060e37ce5ef3dd5c5a051702252`.
- Canonical manifest: `manifests/canonical_manifest_v1_2_20260803.json` (schema 1.0.0), single source of execution state. All 12 gates PASS; `gate_decisions` populated for all 12 gates.
- Gate sentinels: `manifests/sentinel_{T0,S0,T1,M0,T2,T3,Q0,Q1,Q2}.txt`, `Sentinel_{Q3,Q4,Q5}.txt`.
- All artifacts recorded with absolute paths and SHA-256; raw data read-only.

## Environment
- Host: `bms-18937653-012` (Ubuntu 5.15.0-173), conda env `rna_junction_preorganization_v1_1`, Python 3.10.20, numpy 2.2.6, pandas 2.3.3, scipy 1.15.2, biopython 1.87.
- Q3–Q5 are CPU-only (replay + statistics + isotonic calibration); no GPU Gate required. T2/T3 used GPU (A100 MIG 1g.5gb), verified in `reports/gpu_preflight_20260801.md`.

## Replay path
1. Recreate isolated worktree from commit `0874c88` (Q3–Q5 code+results).
2. Re-run each stage finalizer in order; each re-verifies contract hash, commit, artifacts, checksums, tests.
3. Q3: `np.random.seed(42)` before `compute_all_mg_1_2`; tolerances frozen in `specs/q3_endpoint_replay_spec.json` (numerical abs=1e-6 OR rel=1e-4; categorical exact match).
4. Q4: mutation graph from `aligned_seq` Hamming-1; connected components assigned atomically to folds (no leakage).
5. Q5: 4 baselines (B1 mean, B2 mg_1_2 univariate, B3 mutations, B4 tecto old_dg + isotonic calibration); 4-fold CV; label permutation n=100.

## Key results to reproduce
- Q3: 1600 PASS + 66 NA + 0 FAIL (98 variants × 8 endpoints).
- Q4: 193 edges, 4 components, 0 leakage.
- Q5: B4 RMSE=0.1951, gain=0.511 (CI [0.403,0.618]), cov68=0.682, perm p=0.0.
- T2: interval coverage 0.957, width median 1.35 kcal/mol, `signal_detected=true`.
- T3: model vs motif_mean 41.8 vs 27.0 (negative gain).

## Audits
- Schema validation, gate verifier, test suite (`tests/`), leakage audit (`homolog_leakage n_overlap=0` for tecto; mutation-graph `leakage_violations=0` for qMaP), negative controls (T2 permutation, calibration drift, out-of-range; Q5 label permutation, condition controls, negative-nucleotide controls).
