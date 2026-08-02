# Phase 0 crosswalk review intake

## Current state

The project is fail-closed at Phase 0. The official processed DMS payload has been acquired, its archive integrity has been verified, and the six official processed condition files have an equal construct-condition identity set. The remaining blocking state is:

`BLOCKED_RAW_PROCESSED_CROSSWALK_AND_MANUAL_REVIEW_PENDING`

This state means that the evidence currently available does not bind the raw ENA/SRA FASTQ run manifest to the official processed condition/construct namespace. The Denny subset aggregate audit also remains unresolved: integer presence in a candidate identifier set is not accepted as count or identity mapping evidence.

No primary labels are admitted, no Phase 0 pass is declared, and no GPU scientific validation or training may start from this intake alone.

## Evidence required to unblock the crosswalk gate

A reviewer must provide an auditable row-level crosswalk using the empty TSV template in this directory. Each row must bind a source record to a candidate processed record and cite immutable evidence.

Required evidence for each accepted binding:

- source record reference containing the ENA/SRA run accession or an equivalent immutable source identifier;
- candidate processed condition/construct reference using the official payload namespace;
- the automated candidate decision and the human decision;
- a reason code explaining the decision;
- an evidence path or stable source reference;
- the SHA256 of the cited evidence object or evidence snapshot;
- reviewer identity and review timestamp recorded in the project acceptance record, without placing private credentials or controlled labels in this table.

The binding rule, source version, and reviewer procedure must be recorded in a companion acceptance/audit record before the rows are considered scientific evidence. Existing source-code semantics and payload member metadata establish processing provenance, but they do not establish a run-accession crosswalk by themselves.

## Manual-review acceptance thresholds

The existing manual-matching validator remains authoritative. The submitted review table must satisfy all of the following:

- at least 50 rows with manual decision `matched`;
- at least 30 rows with manual decision `rejected` or `ambiguous`;
- automated/manual agreement at least 0.95;
- unique, non-empty audit identifiers;
- non-empty evidence reference and SHA256 for every row;
- no fabricated, inferred, or backfilled rows;
- no raw sequences, labels, effect values, or other controlled payload values copied into the intake artifact.

The table is not a substitute for the exact raw-to-processed provenance mapping. If the reviewer cannot resolve a candidate, the row must be recorded as `rejected` or `ambiguous` with evidence and reason; it must not be silently omitted.

## Gate semantics

Until the crosswalk and manual-review acceptance record are independently verified:

- `phase0_gate_effect` remains `NO_PHASE_0_PASS`;
- `scientific_gate_effect` remains `NO_UNLOCK`;
- `primary_labels_admitted` remains `false`;
- training, model comparison, and scientific claims remain prohibited.

A failed or incomplete review is preserved as evidence and does not justify lowering thresholds, changing split semantics, normalizing unmatched namespaces, or treating aggregate counts as exact mappings.

## Files

- `manual_matching_review_template.tsv`: empty schema-only intake table; no review rows are present.
- `manifests/matching_audit.json`: current fail-closed gate state.
- `manifests/data_registry.json`: official processed payload provenance and admission state.
- `manifests/phase0_payload_inventory.json`: processed payload identity audit and gate effect.
