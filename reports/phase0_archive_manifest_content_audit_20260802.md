# Phase 0 archive manifest/content audit — 2026-08-02

## Purpose

This is a bounded, read-only audit of the verified Figshare `data.zip` payload. It
checks whether the archive itself contains an accession/sample manifest that can
bind source SRA/ENA runs to the processed `pdb_library_*` conditions. It does not
create, infer, or backfill crosswalk rows.

## Authoritative inputs

| Item | Path | Evidence |
|---|---|---|
| Figshare archive | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_figshare/data.zip` | SHA-256 `241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`; ZIP CRC previously passed; 3220 members |
| Processing registry | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json` | `status=PUBLIC_SOURCE_CODE_SEMANTICS_REGISTERED_PRIMARY_PAYLOAD_NOT_ADMITTED`; `source_code_semantics_only=true`; `primary_labels_admitted=false` |
| Public processing source | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/dms_processing_source_20260801T190000Z/` | Registered source code semantics only; raw route is `data/mutation-histograms/*.p`, output route is `data/raw-jsons/constructs/{name}.json` |

## Bounded archive member scan

The member-name scan searched `unzip -Z1` for `manifest`, `metadata`, `sample`,
`accession`, `condition`, `design`, and `provenance`, together with the known
library/sequence/mutation terms. The returned candidate members were construct,
mutation-histogram, sequence-CSV, residue/motif, and revision files. No archive
member named as a sample sheet, accession manifest, condition manifest, design
manifest, or provenance manifest was found in this bounded scan.

Relevant members observed include:

```text
data/csvs/library_sequences.csv
data/csvs/p5_sequences.csv
data/raw-jsons/constructs/pdb_library_1.json
data/raw-jsons/constructs/pdb_library_2.json
data/raw-jsons/constructs/pdb_library_3.json
data/raw-jsons/constructs/pdb_library_nomod.json
data/raw-jsons/constructs/pdb_library_denature.json
data/raw-jsons/constructs/pdb_library_37C_2min.json
data/mutation-histograms/pdb_library_1.p
data/mutation-histograms/pdb_library_2.p
data/mutation-histograms/pdb_library_3.p
data/mutation-histograms/pdb_library_nomod.p
data/mutation-histograms/pdb_library_denature.p
data/mutation-histograms/pdb_library_37C_2min.p
```

## Content-level observations

The first rows of `library_sequences.csv` have the schema
`sequence,name,len,molecular weight,extinction coeff,structure,mfe,ens div` and
use names such as `seq_0`, `seq_1`, etc. The bounded prefix of the construct JSON
contains fields such as `name`, `sequence`, `structure`, `data`, `sn`,
`num_reads`, and `num_aligned`. These fields describe processed sequence and
read-level summaries, but do not contain source accession, SRA run, sample-sheet
row, reviewer identity, or an evidence hash binding a source row to the
processed condition.

The processing registry itself explicitly records that the source-code semantics
are registered but the primary payload is not admitted. Its required manual
review items include source-defined mutation-histogram provenance,
construct-level background/read-depth hierarchy, and reconciliation of raw and
interpolated semantics before matching.

## Decision

```text
archive_direct_source_processed_crosswalk: NOT_FOUND
archive_content_provenance_sufficient_for_row_level_binding: false
new_manual_rows_created: 0
phase0_effect: NO_PHASE_0_PASS
scientific_effect: NO_UNLOCK
```

This audit is negative evidence about the archive's available provenance, not a
claim that the processed conditions are invalid. It does not admit the candidate
relationships `trial1 -> pdb_library_1` or `trial2 -> pdb_library_2`, and it does
not resolve `pdb_library_3`. The contract's row-level manual crosswalk remains
required; no labels, gates, or acceptance thresholds were changed.

## Reproducible read-only probes

```text
unzip -Z1 /mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_figshare/data.zip
unzip -p <archive> data/csvs/library_sequences.csv | head -n 5
unzip -p <archive> data/raw-jsons/constructs/pdb_library_1.json | head -c 1200
unzip -p <archive> data/raw-jsons/constructs/pdb_library_3.json | head -c 1200
```

All probes were read-only and streamed; no archive member was extracted or
modified.
