# RNA Junction Preorganization v1.2 — Dataset Card

**Run**: `v1_2_tecto_qmap_20260803`

## Tecto (admitted) dataset
- **Source**: Denny et al. 2018 tectoRNA binding assay (two-way junction tertiary geometry, ΔG) + supplements + public code.
- **Admission**: T0 PASS. Reconstructed 1,687 / 1,713 / 1,636 sets with intersection/difference/exclusion maps and per-difference row-level evidence.
- **Cleaning/Split**: T1 PASS. CleaningLedger, effective-N, motif-family holdout (seed 20260803; holdout motifs `0x1,2x1,2x2`; train `0x2,0x3,1x0,1x1,1x2,1x3,2x0,3x0,3x1,3x3,wc,wc1`).
- **Platform lineage**: RNA-MaP/tectoRNA platform cluster (Denny/Bonilla/Shin/Yesselman). Not independent external systems.
- **Rows**: 11,893 (9,961 measured, 1,932 left-censored at −7.1 kcal/mol); 1,336 junctions; 15 motifs; 9 scaffolds.
- **Condition**: in vitro, 37 °C, 10 mM Mg²⁺ (5 mM variant reported separately).
- **Leakage audit**: junction-level disjoint (n_overlap=0); mmseqs split used to minimize overlap with downstream pretraining.

## qMaPseq (Q0, NOT ADMITTED)
- **ENA PRJNA1086549**: 8 runs / 16 FASTQ, SHA-256 verified — raw, admitted as raw evidence only.
- **GitHub `YesselmanLab/rna_map`** commit `2d7337db`, Apache-2.0 — code, admitted.
- **Canonical labeled dataset**: NOT admitted (Figshare 403 / Zenodo refused) → Q0 `QMAP_NOT_ADMITTED`.

## Not used as labels
- current DMS (permanently `NOT_ADMITTED_FINAL_V1_2`); RMDB/Ribonanza/RNA3DB/Motif Atlas/Rfam/RNAcentral used only for pretraining/operator prior/canonicalization/noise model/exposure audit.