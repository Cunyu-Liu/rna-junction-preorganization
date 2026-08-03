# RNA Junction Preorganization v1.2 — Reproducibility

**Run**: `v1_2_tecto_qmap_20260803`

## Provenance & checksums
- Contract: `rna 三级.md` SHA-256 `32d09729638b7681b6efcfdf8b2addc3c7f83060e37ce5ef3dd5c5a051702252`.
- Canonical manifest: `manifests/canonical_manifest_v1_2_20260803.json` (schema 1.0.0), the single source of execution state.
- Gate sentinels: `sentinel_T0/S0/T1/M0/T2/T3/Q0.txt`.
- All artifacts recorded with absolute paths and SHA-256; raw data read-only.

## Environment
- Host: `bms-18937653-012` (Ubuntu 5.15.0-173), conda env (`rna_junction_preorganization_v1_1`), Python 3.10.20, torch 2.9.0+cu126, numpy 2.2.6, pandas 2.3.3, scipy 1.15.2, biopython 1.87.
- GPU execution verified per stage via `cuda_probe` (A100 MIG 1g.5gb for T2/T3); no CPU silent downgrade.

## Replay path
1. Recreate isolated worktree from commit of the final manifest.
2. Re-run each stage finalizer in order; each re-verifies contract hash, commit, artifacts, checksums, and tests.
3. T2/T3 require GPU; use `CUDA_VISIBLE_DEVICES` per `reports/gpu_preflight_20260801.md`.

## Key results to reproduce
- T2: interval coverage 0.957, width median 1.35 kcal/mol, permutation `signal_detected=true`.
- T3: model vs motif-mean baseline proper score 41.8 vs 27.0 (negative gain), group bootstrap pos frac 0.0.
- Q0: `QMAP_NOT_ADMITTED` (Figshare 403 / Zenodo refused).

## Audits
- Schema validation, `AcceptanceManifest`/gate verifier, test suite, leakage audit (`homolog_leakage n_overlap=0`), negative controls (permutation, calibration drift, out-of-range).