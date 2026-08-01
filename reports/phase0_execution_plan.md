# Phase 0 execution plan — data/provenance/semantic audit

This is a plan and acceptance template, not a result. It is intentionally
blocked until the remote project can verify the exact contract source bytes.

## Planned work

1. Register each public source with DOI/URL, download date, license, absolute
   raw-file path, SHA-256, processing commit, and source-specific status.
2. Reproduce the Denny workbook schema and reconcile the contract's reported
   counts. Preserve raw, interpolated, censored, replicate, and covariance
   semantics as separate fields.
3. Confirm the `-7.1 kcal/mol` censoring direction and write interval-likelihood
   tests before any model fitting.
4. Recover the DMS JSON, sequence, construct, raw-count, background, and
   read-depth hierarchy; define connected-component split groups.
5. Build a three-state matched/ambiguous/rejected motif table with boundary,
   strand transform, flanks, construct/context, study source, and audit status.
6. Manually audit at least 50 matched records and 30 rejected/ambiguous records.

## Phase 0 stop rules

- Any fatal field ambiguity or untraceable primary label stops modeling.
- Matching accuracy below 95% stops modeling.
- Interpolated values are excluded from the primary test.
- Random row splits, test posterior labels, or silent missing-as-zero handling
  are prohibited.
- A completed smoke/proxy test cannot create a scientific PASS marker.

## Acceptance record

The formal record is `manifests/acceptance_phase0.json`. It must be replaced by
a new run-specific record, never overwritten, once the source deployment gate
is resolved. A PASS requires complete evidence and a terminal marker; absence
of a terminal marker is not PASS.
