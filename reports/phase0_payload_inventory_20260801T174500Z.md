# Phase 0 payload inventory refresh (20260801T174500Z)

This refresh is governance evidence only. It does not unlock Phase 0 or admit primary labels.

- Public ENA file-level inventory: 15 runs, 30 paired FASTQ files, 95,123,388,656 compressed bytes.
- One small paired run passed hash/gzip/record/pair-ID audit; sequence content was not emitted.
- Main DMS/nomod/denature/37C download is still in progress under the recorded PID/log.
- Denny subset audit found an explicit variant-count sum of 24,073 and a candidate field with 1,713 distinct values; the contract's 1,687/1,713/1,636 mapping is unresolved.
- Figshare HTTP 403, OUP access challenge, and PMC package 404 are preserved as access evidence; no access control was bypassed.

- The official PMC media-1.docx route returned HTTP 200 text/html with a POW challenge rather than a DOCX; the Zenodo route probe returned curl exit 7/HTTP 000 and remains unverified.

- A dependency-free Denny XLSX OOXML structure audit completed without decoding any cell values; semantic count/censor/matching evidence remains unresolved.

## Gate

`PHASE_0 = IN_PROGRESS`; `scientific_gate_effect = NO_PHASE_0_PASS`; `primary_labels_admitted = false`.
