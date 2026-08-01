# Phase 0 manual matching audit schema

This table is a component-level acceptance artifact. It does not itself
unlock Phase 0, admit DMS labels, or establish a scientific result. The
project-level gate remains closed until the complete primary-payload,
provenance, censoring, count hierarchy, matching, and manual-review evidence
specified by `1.1.docx` is present.

The input is UTF-8 tab-separated text with one row per manually reviewed
record. It must contain these columns:

```text
audit_id
source_record_ref
candidate_record_ref
automated_decision
manual_decision
reason_code
evidence_ref
evidence_sha256
```

`automated_decision` and `manual_decision` must be one of `matched`,
`rejected`, or `ambiguous`. Record references are opaque identifiers or hashes;
sequences, reactivities, counts, effect values, and reviewer notes must not be
written into this audit table or its aggregate output. `evidence_ref` points to
an auditable source/provenance artifact and `evidence_sha256` fixes its exact
content.

The validator accepts the component only when all of the following hold:

- at least 50 manually adjudicated `matched` rows;
- at least 30 manually adjudicated `rejected` or `ambiguous` rows; and
- automated/manual decision agreement is at least 0.95 across reviewed rows.

Any missing column, duplicate `audit_id`, missing evidence reference/hash,
invalid decision, hash mismatch, or unmet threshold is recorded as
`BLOCKED_MANUAL_MATCHING_AUDIT`. A component pass still has
`scientific_gate_effect=NO_PHASE_0_PASS` and must be incorporated into the
contract-level Phase 0 verifier.
