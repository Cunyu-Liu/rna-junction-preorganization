# RNA Junction Preorganization v1.1

## Project Status: CLOSED / FINALIZED

This project has reached its final state and is **no longer being actively
pursued** (finalized 2026-08-19 UTC). The repository on this branch
(`r0_audit_repair_20260811`) is the isolated execution root for the contract
in `contract/1.1.docx` (v1.1, dated 2026-08-01).

### Final scientific conclusion

A right-censoring-aware evaluation framework for RNA junction preorganization
was built and validated on a single study (Denny et al., Cell 2018; 11,893
rows × 1,336 junctions × 9 operators). The frozen method (7-member ensemble +
two-layer mu EB correction + decoupled sigma calibration) reaches **0.7243
pooled junction-macro NLL, +27.86% vs the linear baseline** (edit-cluster CI
[0.2416, 0.3794], 37 folds stable).

The sequence-signal hypothesis was **NOT supported at the pre-registered
gate** (P0.6 verdict `NOT_SUPPORTED_AT_PRE_REGISTERED_GATE`, D1 =
`TRACK_A_LOCKED`): the 63-D sequence map gives a negative gain, and 75+
model-level improvement routes were systematically closed, with post-hoc
calibration being the only positive family. The transferable-mechanism claim
is not authorized.

### Final status flags

- SOTA: `SOTA_NOT_ADJUDICATED`
- Submission / release authorization: `NO_SUBMISSION_AUTHORIZATION`
- `scientific_claim_authorized = false`
- Seal: 12/12 checksums pass (including `NullArtifact.json`), dual-env
  raw→final replay = 0.724302 / 0.724302

See **`reports/project_final_summary_20260819.md`** for the full final-state
summary and key metrics.

## Background

The contract is the only scientific and engineering authority for this
project. During execution the project went through the R0–R6 repair cycle and
the 2026-08-11 strict post-execution audit; the surviving contribution is the
**benchmark / identifiability-boundary track** (evaluation methodology +
boundary closure), not a base-model breakthrough.

## Non-negotiable boundaries

- DMS and tectoRNA are complementary cross-assay observations, not presumed to
  observe one shared latent ensemble.
- The normalized partition-function forward operator is mandatory.
- `Omega*` or `U*` must be frozen before any target-specific preorganization
  claim.
- Operator, transport, and symmetry uncertainty are part of the primary
  problem, not optional nuisance annotations.
- Random row splits, interpolated primary test labels, test posterior labels,
  PDB/MD as error-free truth, and conditional results written as robust
  results are prohibited.
- A failed or unsupported gate is preserved with its evidence; thresholds and
  split definitions are not changed to obtain a positive result.

## Data / licensing note

The dataset is restricted for redistribution and is not distributed with this
repository; access path + checksums are recorded in the release ledger.
