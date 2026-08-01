# Phase 0 payload inventory — interim record

Date: 2026-08-01 (Asia/Shanghai)

Contract SHA-256: `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`

## Current evidence class

This is an interim Phase 0 inventory, not an acceptance record. It records
public metadata, one downloaded supplementary workbook, checksums, and a
documented access failure. No primary label has been admitted to a modeling
table, and no raw values are emitted in the metadata audit JSON.

## Denny / tectoRNA

Downloaded from the author laboratory's Supporting Info link:

<https://herschlaglab.stanford.edu/s/261_SI.xlsx>

Artifact:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/denny/261_SI.xlsx`

- Size: `5692851` bytes
- SHA-256: `f68ed6049b31750ad044a303fdb157ec64fb1e565cbf56fad247e0bbc1deda20`
- Workbook sheets: 2
- `library_annotations`: 28,938 non-empty rows, 27 reported columns
- `sublibrary_descriptions`: 8 non-empty rows, 3 reported columns
- Candidate fields observed: measured ΔG, measurement error, measured and interpolated ΔG
- Candidate fields not confirmed by the metadata audit: raw, censor, covariance, replicate

The absent candidate terms are an unresolved semantic issue, not evidence that
the underlying workbook lacks the concepts. The next audit must inspect the
workbook documentation and the paper's definitions, reconcile the contract's
reported counts, and determine whether censoring is encoded outside the
workbook or through a convention not present in the header.

Metadata audit JSON:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/denny_workbook_metadata.json`

### Aggregate semantics audit

The protected aggregate audit was run against the hash-verified workbook. It
reported the following evidence without exporting row-level sequences, labels,
or effect values:

- the header row contains measured ΔG, measurement error, and measured-plus-
  interpolated ΔG fields;
- raw/censor/replicate/covariance fields were not confirmed by the header
  audit;
- the value used by the contract as the censoring boundary appears as an upper
  numeric-boundary candidate in the measured and measured-plus-interpolated
  columns, but the censoring mechanism and direction are not yet accepted;
- the `Number of variants` sublibrary values sum to 24,073, with six reported
  sublibrary counts; this is not the same unit as the contract's 1,687/1,713/
  1,636 reconciliation target;
- one anonymous library column has 1,713 distinct integer values, but its
  source-defined meaning and relationship to 1,687 and 1,636 are not proven.

Evidence:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/denny_semantics_20260801T065000Z.json`

SHA-256: `ee434df573ea76879404bb084854033ec82b3b26038c06270da466791756a32f`

The audit log contains the openpyxl warning that an unknown workbook
extension was removed while reading in read-only mode. The warning is
preserved and does not count as a scientific pass.

## Deenalattha / DMS

Downloaded metadata:

- NCBI BioProject metadata for `PRJNA1188187`.
- Crossref article metadata for DOI `10.1093/nar/gkag672`.
- Public code repository refs for `YesselmanLabPublications/2025_char_3d_struct_features`.

The Figshare API endpoint for article `27880434` returned HTTP 403 from the
remote host. No Figshare payload was downloaded. The 403 response is retained
as an access-probe artifact; it is not treated as a license denial or as proof
that the dataset is unavailable. An approved alternate public route may be
used later, but access controls will not be bypassed.

The official article and its public code identify additional provenance that
must be reconciled before payload admission: 7,500 constructs; a standard
quality filter requiring more than 2,000 aligned reads and signal-to-noise
ratio above 4; a separate 37°C/2-minute protocol with its own exclusions; raw
FASTQ files in SRA project `PRJNA1188187`; and code-produced directories for
mutation histograms, construct JSON, motif JSON, and residue JSON. These are
registered as source-level facts, not as downloaded primary payloads.

## Gate status

The inventory is `IN_PROGRESS_PARTIAL_PUBLIC_PAYLOADS`. Phase 0 remains
unaccepted because the following are still open:

- exact Denny count/censor/raw/interpolated/unit and covariance semantics;
- DMS JSON supplement, treated/background/read-depth hierarchy, and construct
  mapping;
- source-level data terms and licenses for every downloaded payload;
- matched/ambiguous/rejected motif table;
- manual audit of at least 50 matched and 30 rejected/ambiguous cases;
- matching accuracy at least 0.95 and no fatal ambiguity;
- terminal Phase 0 pass/failure marker.

Phase 0.5 and all later phases remain locked. The GPU probe remains a runtime
precondition only and has no effect on this gate.
