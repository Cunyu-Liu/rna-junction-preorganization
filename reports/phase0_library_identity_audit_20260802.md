# Phase 0 official library identity audit

## Result

The official Figshare library table is now bound to every processed construct identity using the processing semantics published in the authors' code. This is a sub-gate only. It does not bind an SRA/ENA run to a processed condition and does not unlock Phase 0 or scientific work.

## Source-defined rules

- input library identifiers `seq_N` are renamed to processed identifiers `constructN`;
- the common 5-prime sequence is discovered from `data/csvs/p5_sequences.csv` and has length 12 for this payload;
- the processed code trims that 12-nucleotide prefix and a fixed 20-nucleotide 3-prime tail;
- the audit applies the published T-to-U comparison rule and does not emit sequence values.

## Acceptance evidence

The remote audit was run against the verified Figshare archive SHA256 `241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`.

All six official processed condition JSON members passed:

- record count: 7500;
- library/processed identity intersection: 7500;
- sequence matches after source-defined trim: 7500;
- structure matches after source-defined trim: 7500.

Remote artifacts:

- audit JSON: `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/official_library_sequence_identity_20260802T053000Z.json`
- audit JSON SHA256: `1ddd0ef25d33f2def2006baf1377f265fee274f59fb43a2022e01bc92ec0fb71`
- execution log: `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/logs/official_library_sequence_identity_20260802T053000Z.log`
- execution log SHA256: `2df241e09fa5c51395b24c81e18939308aa62781f329ee69c5a798f8dcfd843c`

## Gate effect

`library_identity_gate_effect=PASS` is recorded for this sub-gate.

The following remain unchanged:

- `raw_processed_crosswalk_gate_effect=NO_CHANGE`;
- `phase0_gate_effect=NO_PHASE_0_PASS`;
- `scientific_gate_effect=NO_UNLOCK`;
- `primary_labels_admitted=false`;
- training and GPU scientific validation are not authorized.
