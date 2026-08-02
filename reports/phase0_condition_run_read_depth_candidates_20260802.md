# Phase 0 condition/run read-depth candidate audit — 2026-08-02

## Finding

The public SRA run metadata and the processed construct payload produce a
read-depth signal that is useful for triage but does not establish an
accession-preserving raw FASTQ → processed-condition crosswalk. The audit is
therefore explicitly `CANDIDATE_ONLY_RAW_PROCESSED_CROSSWALK_UNRESOLVED`.

The processed payload is the current Figshare `data.zip` archive with SHA-256
`241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`. The SRA
runinfo input is the selected public metadata snapshot with SHA-256
`026ecfda5c90ff034648249fe8eba9ab569a701c9a30a9da7c77088140c49f5e`.

## Candidate signal

| processed construct | processed `num_reads` sum | nearest public run by ratio | ratio | admission |
|---|---:|---|---:|---|
| `pdb_library_1` | 282,955,850 | `SRR31402664` / `rna_library_trial1` | 0.990148497 | NOT_ADMITTED |
| `pdb_library_2` | 265,856,671 | `SRR31402663` / `rna_library_trial2` | 0.988798060 | NOT_ADMITTED |
| `pdb_library_3` | 276,275,687 | `SRR31402663` / `rna_library_trial2` | 1.027549402 | NOT_ADMITTED |
| `pdb_library_37C_2min` | 210,502,435 | `SRR38259812` / `rna_lib_37C_2min` | 0.907758989 | NOT_ADMITTED |
| `pdb_library_denature` | 13,376,235 | `SRR35766785` / `rna_library_denature` | 0.138693929 | NOT_ADMITTED |
| `pdb_library_nomod` | 86,216,011 | `SRR35766784` / `rna_library_nomod` | 0.934122912 | NOT_ADMITTED |

The apparent signal for `pdb_library_1` ↔ `trial1` and `pdb_library_2` ↔
`trial2` is not unique evidence: processed `num_reads` is downstream of
trimming, filtering, pairing, and `rna-map` processing, whereas SRA `spots` is
an archive-level run statistic. The third trial library has no distinct public
SRA run identity in the available metadata. These observations cannot satisfy
the contract's raw/processed crosswalk gate.

## Gate effect

- `raw_processed_crosswalk_gate_effect`: `NO_CHANGE`
- `primary_labels_admitted`: `false`
- `phase0_gate_effect`: `NO_PHASE_0_PASS`
- `scientific_gate_effect`: `NO_UNLOCK`
- `training_started`: `false`

The existing manual-review intake remains the only admissible recovery path:
each reviewed row must carry a real evidence reference and SHA-256, with the
pre-registered agreement and coverage thresholds. No row is created from this
read-depth audit.
