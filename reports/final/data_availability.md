# RNA Junction Preorganization v1.2 — Data Availability

**Run**: `v1_2_tecto_qmap_20260803`

## Admitted tecto data (raw, read-only)
- Location: `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`
- T0 admission fixed the Denny et al. 2018 tectoRNA source, supplements, and public code; 1,687 / 1,713 / 1,636 set reconstruction and intersection/difference/exclusion maps are in `manifests/t0_admission_analysis.json` and `manifests/t0_source_pin.json`.
- Cleaning / QC / effective-N / split freeze: `manifests/sentinel_T1.txt`, `scripts/t1_build.py`.
- All source files have URL, license, size, download time, and SHA-256 recorded in the data registry.

## qMaPseq raw evidence (Q0)
- ENA BioProject **PRJNA1086549**: 8 runs / 16 FASTQ (`/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap/raw/fastq/`), SHA-256 verified (`qmap/audit/fastq_sha256.txt`), manifest `qmap/raw/ena/PRJNA1086549_read_run_manifest.tsv`.
- GitHub code `YesselmanLab/rna_map` commit `2d7337db041497d5707fcc73bd76637896d061a9`, Apache-2.0 (`qmap/raw/code/rna_map/`).
- **Not admitted**: canonical processed qMaPseq dataset (Figshare doi 10.6084/m9.figshare.25331758 → HTTP 403) and Zenodo archive (connection refused). Q0 = `QMAP_NOT_ADMITTED`.

## Status notes
- All raw data is read-only and never overwritten.
- Derived artifacts are run-isolated under `/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`.
- Any source not found is recorded as `NOT_FOUND_IN_AUDITED_PUBLIC_SOURCES_AS_OF_<DATE>`, never "does not exist".