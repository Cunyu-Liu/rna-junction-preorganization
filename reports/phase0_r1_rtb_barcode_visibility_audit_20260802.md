# Bounded R1 RTB-barcode visibility audit

Run ID: `r1_rtb_barcode_visibility_20260802`

Contract SHA-256: `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`

Status: `NO_SYSTEMATIC_KNOWN_RTB_BARCODE_SIGNAL_IN_BOUNDED_R1_SAMPLE_CANDIDATE_ONLY`

## Why this audit was run

The official PMC methods describe RTB barcode demultiplexing and provide three example RTB sequences, with the barcode described at the 5′ end of read 1. This audit tests whether those three published examples are visibly present in the current complete R1 FASTQ files. The official method/source context is preserved at:

`https://pmc.ncbi.nlm.nih.gov/articles/PMC11601540/`

This is a bounded visibility test only. It cannot prove the complete barcode set, prove that the files are already demultiplexed, or bind a raw run to an official processed condition.

## Scope and preservation

Inputs were the five currently complete R1 files under:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_sra/main_library/`

For each run, the audit read the first `100000` FASTQ records and searched the first `160` bases of each R1 sequence for the three published example RTB barcodes in both forward and reverse-complement orientation. Only counts, positions, file size, first-1 MiB file hashes, and fingerprints of common prefixes were emitted. No raw sequence or raw header was written.

The existing downloader processes and all R2 files/partials were left untouched.

## Results

| Run | Records | Malformed | Unique first-12 fingerprint count | Dominant first-12 fraction | Header index-like tokens | Published RTB hits |
|---|---:|---:|---:|---:|---:|---:|
| `SRR31402663` | 100,000 | 0 | 97 | 0.986760 | 0 | 0 |
| `SRR31402664` | 100,000 | 0 | 112 | 0.989720 | 0 | 0 |
| `SRR35766784` | 100,000 | 0 | 146 | 0.984160 | 0 | 0 |
| `SRR35766785` | 100,000 | 0 | 55 | 0.993010 | 0 | 1 reverse-complement occurrence |
| `SRR38259812` | 100,000 | 0 | 56 | 0.993070 | 0 | 0 |

The single low-frequency reverse-complement occurrence in `SRR35766785` is not treated as a barcode assignment. All other published-example counts are zero. The dominant first-12 fingerprint is highly concentrated within every run and is shared across runs; it is recorded only as a redacted hash/frequency pattern and is not interpreted as a barcode or condition label.

## Interpretation boundary

The safe statement is:

> In a bounded sample of the five complete R1 FASTQ files, the three published example RTB barcode sequences are not systematically visible, and no header index-like token was observed. The current observation is compatible with barcode removal or with a different/unknown barcode representation, but does not prove either explanation.

The following claims remain prohibited:

- “The files are proven demultiplexed.”
- “The missing `Sequences.xlsx` barcode table has been reconstructed.”
- “`SRR31402663`/`SRR31402664` is definitively trial1/trial2.”
- “A run is definitively `pdb_library_1`, `pdb_library_2`, or `pdb_library_3`.”
- Any primary condition label based on this bounded census.

The raw-to-processed crosswalk gate therefore remains unchanged:

```text
raw FASTQ -> RTB barcode -> condition/batch -> processed condition -> construct namespace
status: unresolved; primary labels not admitted; NO_PHASE_0_PASS
```

## Reproducibility

Audit script:

`/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/scripts/audit_r1_rtb_barcode_visibility.py`

Script SHA-256: `28496c811595f012fe9f15894db623053fe79fcf3e06c709a3001b2c5f913103`

Machine-readable output:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/r1_rtb_barcode_visibility_20260802.json`

Audit JSON SHA-256: `d21f3d957cc9c4ae2981d07e2e9482b237323409fc68376a84c535303e9ad937`

The audit records `barcode_payload_absence_proven=false`, `demultiplexing_status_proven=false`, `raw_to_processed_crosswalk_proven=false`, `primary_labels_admitted=false`, and `scientific_gate_effect=NO_PHASE_0_PASS`.
