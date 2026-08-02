# Author-release crosswalk audit: Zenodo v1.0.0

Run ID: `author_release_crosswalk_zenodo_16884333_20260802`

Contract SHA-256: `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`

Status: `AUTHOR_RELEASE_NO_EXPLICIT_CROSSWALK_OR_BARCODE_PAYLOAD_DMS_PHASE0_REMAINS_BLOCKED`

## Purpose

The official PMC methods refer to a `Sequences.xlsx` supplemental document for both the 7,500-library sequences and barcoded RTB primers, and state that the sequencing data were demultiplexed using those RTB barcodes. The previously audited PMC supplementary listing exposed only `media-1.docx`; the final NAR Supplement S2 exposed the 7,500 construct sequence/structure table but no RTB/sample/run mapping.

This audit therefore checked the author-linked Zenodo software release as a separate public provenance route. It is a source-availability audit only. A code release, parser, example command, or generic variable named `condition` cannot be promoted to a construct-level scientific crosswalk.

## Source and transfer

Source: `https://zenodo.org/records/16884333`

DOI: `10.5281/zenodo.16884333`

Release: `v1.0.0`

The archive was downloaded as one chunk because its size is below the 128 MiB transfer limit. It is preserved at:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/author_release/zenodo_16884333/2025_char_3d_struct_features-v1.0.0.zip`

Archive size: `2,271,489` bytes

Archive SHA-256: `93f87d6f450b5650720be38eb6004bdec35ece4a5d9c577acfabcf7156b3cae7`

The release is linked to the author repository `YesselmanLabPublications/2025_char_3d_struct_features`. The repository README directs users to obtain the experimental data from Figshare, rather than from the code release.

## Audit result

The archive contains 40 members. The member-name and text-content audit found:

| Signal | Member-name presence | Text-token count | Interpretation |
|---|---:|---:|---|
| `Sequences.xlsx` | 0 | 0 | Not present |
| barcode payload or RTB payload | 0 | 0 | Not present |
| `trial1` / `trial2` | 0 | 0 | Not present |
| SRA accession token | 0 | 0 | Not present |
| FASTQ payload/name | 0 | 0 | Not present |
| sample-sheet token | 0 | 0 | Not present |
| generic `condition` token | 0 | 2 | Ordinary code context only; not crosswalk evidence |
| `batch` token | 0 | 0 | Not present |

The final machine-readable audit is:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/author_release_crosswalk_zenodo_16884333_20260802.json`

It records `crosswalk_evidence_found=false`, `primary_labels_admitted=false`, `scientific_gate_effect=NO_PHASE_0_PASS`, and `raw_sequences_emitted=false`.

## Development incident and correction

The first implementation classified the two generic `condition` code tokens as positive evidence. That was a false positive. The script was corrected so only explicit source signals can trigger a positive crosswalk classification: `Sequences.xlsx`, RTB/barcode, trial labels, SRA accession, FASTQ, or sample-sheet signals. The corrected script was recompiled and rerun against the same immutable archive; the final status is negative.

The first JSON output was replaced only within its own audit path and is not used as an acceptance result. The development false-positive is retained in the execution history and described here rather than hidden.

## Gate consequence

This route does not resolve the missing provenance edge:

```text
construct/seq ID -> RTB barcode -> condition/batch -> demultiplexed FASTQ -> SRA run -> per-construct filter state
```

The following remain `NOT_AVAILABLE_NOT_ASSERTED`:

- the missing `Sequences.xlsx` barcode/crosswalk payload;
- a direct mapping from `constructN`/`seq_N` to `trial1`/`trial2` and the six processed conditions;
- a per-construct list of the 17 quality-filtered sequences and their reasons;
- an explicit mapping from processed-condition files to raw SRA run accessions.

No raw FASTQ result, read-depth correlation, barcode guess, or stochastic sequence-generation output is admitted as a substitute for the missing source crosswalk.

Therefore:

- the NAR/S2 sequence source remains usable for the sequence-fidelity track;
- the DMS source-transport track remains blocked at Phase 0;
- no Phase 0.5 operator-uncertainty freeze is authorized;
- no GPU validation or sequence-model training is authorized;
- the project remains fail-closed until the author/data publisher provides an explicit crosswalk or the contract’s alternative path is formally satisfied.

## Reproducibility

Audit script:

`/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/scripts/audit_author_release_crosswalk.py`

Audit script SHA-256: `9cbe812ad2f1720460336e691bb4a0623e7351a3ced935efc7e862db5f4baa3f`

The script records member names and token counts only; it does not emit raw sequence or full text payloads.
