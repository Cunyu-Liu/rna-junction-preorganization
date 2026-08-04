# RNA Junction Preorganization v1.2 — Dataset Card

**Run**: `v1_2_tecto_qmap_20260803`

## Tecto dataset (admitted, T0 PASS)
- **Source**: Denny et al. 2018 tectoRNA binding assay (two-way junction tertiary geometry, ΔG) + supplements + public code.
- **Admission**: T0 PASS. Reconstructed 1,687/1,713/1,636 sets with intersection/difference/exclusion maps.
- **Cleaning/Split**: T1 PASS. CleaningLedger, effective-N, motif-family holdout (seed 20260803; holdout motifs `0x1,2x1,2x2`).
- **Rows**: 11,893 (9,961 measured, 1,932 left-censored at −7.1 kcal/mol); 1,336 junctions; 15 motifs; 9 scaffolds.
- **Condition**: in vitro, 37 °C, 10 mM Mg²⁺ (5 mM variant reported separately).
- **Leakage audit**: junction-level disjoint (n_overlap=0); mmseqs split.

## qMaPseq dataset (admitted, Q0–Q2 PASS)
- **ENA PRJNA1086549**: 8 runs / 16 FASTQ, SHA-256 verified — raw sequencing.
- **GitHub `YesselmanLab/rna_map`** @ `2d7337d`, Apache-2.0 — processing code.
- **Figshare** `10.6084/m9.figshare.25331758` (md5=`7a080dc7...`) — `mttr6_data_full.json` (1,568 rows = 98 variants × 16 Mg concentrations), `mtt6_data_mg_1_2.csv` (98 rows, Hill-equation fits).
- **Zenodo** `10.5281/zenodo.11672684` (md5=`48da131a...`) — `rna_map_dg.csv` (99 variant ΔG labels), `2024_qmap_paper-main.zip` (processing code).
- **Q1 registry**: 99 variants (1 reference + 98 mutants); all have `rna_map_dg` and construct sequences.
- **Q2 attrition**: 84 fitted + 11 right-censored (6 mg_1_2>40, 5 unstable fit) + 2 closing-pair abnormal + 1 alternate-structure = 98. Censored enter likelihood, not deleted.
- **Selection boundary (Q4)**: 98 mttr6 TTR mutants only; no extrapolation to arbitrary TLR families.
- **Split (Q4)**: mutation-graph-aware 4-fold CV (Hamming-1 adjacency); fold sizes [83,11,2,2]; 0 leakage.

## Not used as labels
- current 7,500-construct DMS (permanently `NOT_ADMITTED_FINAL_V1_2`); RMDB/Ribonanza/RNA3DB/Motif Atlas/Rfam/RNAcentral used only for pretraining/operator prior/canonicalization/noise model/exposure audit.
