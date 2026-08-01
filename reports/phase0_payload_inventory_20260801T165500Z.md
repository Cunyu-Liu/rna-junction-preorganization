# Phase 0 payload inventory refresh (20260801T165500Z)

This refresh is governance evidence only. It does not unlock Phase 0 or admit primary labels.

- Public ENA file-level inventory: 15 runs, 30 paired FASTQ files, 95,123,388,656 compressed bytes.
- One small paired run passed hash/gzip/record/pair-ID audit; sequence content was not emitted.
- Main DMS/nomod/denature/37C download is still in progress under the recorded PID/log.
- Denny subset audit found an explicit variant-count sum of 24,073 and a candidate field with 1,713 distinct values; the contract's 1,687/1,713/1,636 mapping is unresolved.
- Figshare HTTP 403, OUP access challenge, and PMC package 404 are preserved as access evidence; no access control was bypassed.

- The official PMC media-1.docx route returned HTTP 200 text/html with a POW challenge rather than a DOCX; the Zenodo route probe returned curl exit 7/HTTP 000 and remains unverified.

- A dependency-free Denny XLSX OOXML structure audit completed without decoding any cell values; semantic count/censor/matching evidence remains unresolved.

- Official DMS processing source code was pinned to a public Git commit and its field semantics were registered; no source-code field was admitted as a primary label.

- One selected main-library paired FASTQ run passed hash/gzip/record/pair-ID audit; this is file-integrity evidence only and does not establish construct-level DMS labels or QC hierarchy.

- A resume-wrapper process-detection incident was preserved with no final-file overwrite or partial deletion; the corrected wrapper now blocks while the original downloader is active. The affected partial remains unverified.

- Dependency-free Denny semantic evidence extracted a 24,073 variant-count sum, a 1,713 numeric cardinality candidate, and measured/interpolated 9/10/11-bp ΔG headers; exact subset mapping, censor direction, and accepted raw/interpolated semantics remain unresolved.

- Download failure evidence: SRR31402663 returncode=56 partial preserved; SRR31402664 returncode=56 partial preserved. Safe resume is required.
## Gate

`PHASE_0 = IN_PROGRESS`; `scientific_gate_effect = NO_PHASE_0_PASS`; `primary_labels_admitted = false`.
