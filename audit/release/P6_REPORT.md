# Phase 6 Report — Reproduction & Release Preparation (contract Phase 6)

- **Phase**: P6 (reproduction, code cleanup, release prep)
- **Date**: 2026-08-09
- **Run root**: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/`
- **Evidence class**: `DEVELOPMENT_ONLY` — engineering release, no scientific claim
- **SOTA status**: `SOTA_NOT_ADJUDICATED`; **NO_SUBMISSION_AUTHORIZATION**
- **scientific_claim_authorized = false**

---

## 1. Purpose

Phase 6 proves the benchmark numbers are reproducible from frozen code + legal
data and that the (negative) conclusion is properly scoped. Because the P0–P5
audit sealed a fail-closed, benchmark/identifiability-boundary result with no
promotable mechanism, there is **no submission** to prepare; the release artifacts
lock down the reproducible benchmark and the scoped negative result.

## 2. Deliverables

Generated in `audit/release/` by `p6_finalize.py` (computes SHA-256 at runtime):

- `environment.lock` — conda env `rna_junction_preorganization_v1_1`
  (numpy 2.2.6, scipy 1.15.2, pandas 2.3.3, scikit-learn 1.7.2)
- `REPRODUCE.md` — phase-by-phase rerun instructions and expected fail-closed outcome
- `ReleaseManifest.json` — run id, git commit, phase statuses, artifact hashes
- `checksums.sha256` — SHA-256 of 16 key data/prediction/decision artifacts
- `DataCard.md` — data universe, censoring, panel structure, leakage control
- `ModelCard.md` — benchmark models, candidate, metric, status
- `LicenseLedger.csv` — data/code/env licensing status (data license needs legal review)
- `SubmissionClaimMatrix.csv` — claim→status mapping (no authorized scientific claim)
- `STATUS.json`

## 3. Verification

16/16 artifacts hashed and present (no missing). All artifacts are on `/mnt`
(code/commits on `/home`), per the storage policy. The release manifest is pinned
to git commit `8df99c1`.

## 4. Overall audit outcome (P0–P6)

- P0 `P0_PASS_COMPARISON_ELIGIBLE` (14/14 gates, fail-closed)
- P1 baselines complete
- P2 `CONDITIONAL_KNOWN_OPERATOR_SIGNAL`
- P3 candidate retained as targeted extrapolation fix
- P4 `NOT_PROMOTED` (fail-closed, coverage-matched)
- P5 `IDENTIFIABILITY_BOUNDARY_NARRATIVE` (no mechanism claim)
- P6 release prepared

**Final scientific posture:** junction sequence provides no incremental
supported-NLL beyond motif/context/scaffold/nearest-neighbour and censored-marginal
baselines; the surviving contribution is a strict grouped/right-censored benchmark
and a negative identifiability-boundary result. `SOTA_NOT_ADJUDICATED`,
`NO_SUBMISSION_AUTHORIZATION`, `scientific_claim_authorized = false`.
