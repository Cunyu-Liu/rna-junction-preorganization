# Reproduction Guide

## Scope
Reproduces the grouped, right-censor-aware benchmark and the fail-closed
identifiability-boundary conclusion. There is NO scientific claim to submit
(SOTA_NOT_ADJUDICATED; NO_SUBMISSION_AUTHORIZATION).

## Requirements
- Remote server path `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z`
  containing data/, protocol/, p1_full/, p3_full_v2/, p4_final/, p5_diagnostics/.
- Conda env `rna_junction_preorganization_v1_1` (numpy 2.2.6, scipy 1.15.2,
  pandas 2.3.3, scikit-learn 1.7.2) - see environment.lock.
- Code: git repo `rna-junction-preorganization` at commit `8df99c154c07c0eade92c9bfe9287d2f541944bf` (audit/ tree).

## Steps
1. `conda activate rna_junction_preorganization_v1_1`
2. Run each phase runner in order (see audit/p0..p5):
   - P0: audit/provenance, audit/data, audit/numerics, audit/benchmark (P0.5)
   - P1: audit/benchmark baselines -> p1_full/Predictions.jsonl
   - P2: audit/p2 hypothesis/nulls/bootstrap
   - P3: audit/p3 p3_run.py -> nested-CV gates
   - P4: audit/p4 p4_run.py -> coverage-matched comparison
   - P5: audit/p5 p5_run.py -> identifiability-boundary diagnostics
3. Verify artifacts against checksums.sha256.
4. Read ReleaseManifest.json for the sealed phase statuses.

## Expected outcome
The candidate is NOT_PROMOTED; the surviving narrative is a benchmark /
identifiability-boundary. Re-running reproduces the same fail-closed verdict.
