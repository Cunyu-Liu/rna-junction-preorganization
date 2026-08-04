# RNA Junction Preorganization v1.2 — Data Availability

**Run**: `v1_2_tecto_qmap_20260803`

## Admitted tecto data (raw, read-only)
- Location: `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`
- T0 admission: Denny et al. 2018 tectoRNA source, supplements, public code; 1,687/1,713/1,636 set reconstruction. Manifests: `manifests/t0_admission_analysis.json`, `manifests/t0_source_pin.json`.
- All source files have URL, license, size, download time, SHA-256 in the data registry.

## Admitted qMaPseq data (Q0 PASS)
- **ENA BioProject PRJNA1086549**: 8 runs / 16 FASTQ (`qmap/raw/fastq/`), SHA-256 verified (`qmap/audit/fastq_sha256.txt`), manifest `qmap/raw/ena/PRJNA1086549_read_run_manifest.tsv`.
- **GitHub code** `YesselmanLab/rna_map` @ `2d7337db041497d5707fcc73bd76637896d061a9`, Apache-2.0 (`qmap/raw/code/rna_map/`).
- **Figshare** doi `10.6084/m9.figshare.25331758`: data.zip verified (502,061,658 bytes, md5=`7a080dc74bb3433e57fcdd885b5b7a56`), contains `mttr6_data_full.json` (1,568 rows = 98 variants × 16 Mg concentrations).
- **Zenodo** doi `10.5281/zenodo.11672684`: `2024_qmap_paper-main.zip` verified (md5=`48da131a78f5027d4b1f31a58c08007b`), contains `rna_map_dg.csv` (99 variant ΔG labels).
- All four sources admitted (Q0 PASS) — provenance recovered after v1.1 network failure.

## qMaPseq derived artifacts (Q1–Q5)
- Q1 registry: `qmap/q1/q1_variant_registry.jsonl` (99 entries).
- Q2 attrition: `qmap/q2/q2_attrition.jsonl` (98 entries, 4 categories).
- Q3 replay: `qmap/q3/q3_replay_comparison.jsonl` (1,666 records), `qmap/q3/evidence/` (98 per-variant JSON).
- Q4 freeze: `qmap/q4/q4_fold_assignment.json`, `qmap/q4/q4_mutation_graph.json`.
- Q5 transfer: `qmap/q5/q5_transfer_summary.json`, `qmap/q5/evidence/B(1, 2, 3, 4)_fold_results.json`.

## Status notes
- All raw data is read-only and never overwritten.
- Derived artifacts run-isolated under `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`.
- current 7,500-construct DMS permanently `NOT_ADMITTED_FINAL_V1_2`.
