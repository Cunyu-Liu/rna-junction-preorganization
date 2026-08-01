# RNA Junction Preorganization v1.1

This repository is the isolated execution root for the contract in
`contract/1.1.docx` (v1.1, dated 2026-08-01).

The contract is the only scientific and engineering authority for this
project. The current decision is **Conditional GO after Phase 0.5**. The
project is currently authorized only for Phase 0 data/provenance/semantic
audit and the Phase 0.5 specification-and-feasibility gate.

Current state is intentionally fail-closed:

- Phase 0: `IN_PROGRESS`
- Phase 0.5: `LOCKED_UNTIL_PHASE_0_PASS`
- Phase 1 and later: `LOCKED`
- sequence model training: `FORBIDDEN_UNTIL_L3`
- GPU training/validation: not started; existing jobs are protected

Large data, caches, weights, and run artifacts belong under the external
artifact root recorded in `manifests/project_manifest.json`; they must not be
placed in this Git repository.

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

See `contract/CONTRACT_SCOPE.md` for the execution summary and
`reports/remote_preflight_20260801.md` for the initial read-only preflight.
