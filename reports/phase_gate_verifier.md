# Phase gate verifier

`scripts/verify_phase_gates.py` is a read-only governance verifier. It does
not read scientific payloads, change manifests, change phase state, allocate a
GPU, or infer scientific validity.

It verifies that:

- the remote contract source is present and has the expected SHA-256;
- all contract hash fields in the project manifests agree;
- Phase 0.5 and later phases retain their required locked states;
- the Phase 0 acceptance record remains false while the blocking prerequisite
  is unresolved;
- the source registry remains `NOT_EXECUTED` until payload provenance and
  acceptance evidence exist; and
- a metadata-only source audit is not treated as Phase 0 PASS.

Exit semantics:

- `0`: governance invariants verified and no blocking violation observed;
- `2`: fail-closed blocking result, with machine-readable violations on stdout;
- any other nonzero code: execution or environment error.

The verifier must be run before any future phase transition. A `0` result is
necessary but not sufficient for scientific gate passage; it does not replace
the contract's data, matching, manual-audit, transport, operator, or symmetry
acceptance evidence.
