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
