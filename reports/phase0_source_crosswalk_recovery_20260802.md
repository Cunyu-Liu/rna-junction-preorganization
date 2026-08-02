# Phase 0 public-source crosswalk recovery

## Result

The public-source search recovered a partial SRA library metadata relation but did not recover an accepted raw-FASTQ-to-processed-construct crosswalk. The Phase 0 gate therefore remains fail-closed.

## Evidence

- The NCBI SRA runinfo record for PRJNA1188187 contains 15 public rows. Five relevant library names are visible: `rna_lib_37C_2min`, `rna_library_denature`, `rna_library_nomod`, `rna_library_trial1`, and `rna_library_trial2`.
- The exact processed output names are generated from mutation-histogram pickle stems in the official code. The historical Figshare version 1 archive contains `pdb_library_1`, `pdb_library_2`, `pdb_library_3`, and `pdb_library_nomod`, but no run accession or sample manifest.
- The paper text references a `Sequences.xlsx` supplemental document, but the public supplemental DOCX and all eight Figshare article versions checked do not provide that file. The complete public Git history also does not contain it.
- Name similarity is recorded only as a candidate. It is not accepted as provenance because trial1/trial2 cannot be uniquely assigned to the processed standard-library outputs and no reviewer table exists.

## What remains required

A source-level mapping or real reviewer record must bind every selected ENA/SRA run to the exact processed condition/output, cite immutable evidence, and satisfy the existing manual-review thresholds. This audit deliberately does not add fabricated rows or unlock training.

## Gate

`phase0_gate_effect=NO_PHASE_0_PASS`; `scientific_gate_effect=NO_UNLOCK`; `primary_labels_admitted=false`.
