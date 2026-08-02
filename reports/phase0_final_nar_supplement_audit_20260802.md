# Phase −1 final NAR supplementary-delivery and S2 content audit

Run ID: `nar_gkag672_supplement_20260802T150000Z`

Contract SHA-256: `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`
Status: `PHASE_MINUS1_SUPPLEMENT_AVAILABLE_S2_SEQUENCE_FIDELITY_PASS_DMS_PHASE0_NO_PASS`

## Scope and preservation

This audit was run after the final NAR supplementary package became reachable. The package was transferred in chunks no larger than 128 MiB; because the object is 11,728,925 bytes, the transfer used one chunk. The merged archive passed `unzip -t` and has SHA-256:

`c1231c9facee20f73d97628a7c6c9f981f11d3c4a385a89bcb09cf3e0a8b8e8f`

The official Figshare archive remains preserved at:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_figshare/data.zip`

size `643502044`, SHA-256 `241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`.

The prior failed OUP transfer directory and the separate wrong-root audit attempt were not deleted. They are diagnostic evidence only and are not used as the authoritative result. No user download process, existing untracked file, or previous artifact was killed, overwritten, or staged.

## Source package inventory

The extracted official package contains exactly five files:

| File | Size | SHA-256 | Audit result |
|---|---:|---|---|
| `2025_char_3d_struct_features_supplemental_second_revision_clean.docx` | 10,974,111 | `557330eac3eafe53ba9907d9982de06b1ddddbfdf4a94098a5a25541634d575c` | Document package readable; no row-level source crosswalk fields established |
| `Supplemental_Table_S1.xlsx` | 46,607 | `d63f5867c84a2d3bf1b06bd9e36b8272c10030d7db406ceb3896e8357e129af1` | 714 structural/PDB-related records |
| `Supplemental_Table_S2.xlsx` | 973,355 | `f899f34002fccc863857fada0d9e68a510393216bca4bed5bc626be6c3e28f76` | 7500 construct block audited below |
| `Supplemental_Table_S4.xlsx` | 17,110 | `a516d97fe690365bf088233d636dd2f27e38c11f527d65537f4da1b19fd12476` | 536 motif-token records; no experiment crosswalk |
| `Supplemental_Table_S7.csv` | 238,100 | `81479fb89dd41ebd2f2800d61c95004f546509b3d39e4c78f3966a109eab9f938` | 4,452 PDB/motif-token records; no experiment crosswalk |

The raw article snapshot used for article-level method/quality claims is preserved at:

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/oup_article_minimal_8736262.html`

size `355434`, SHA-256 `1b42af21aeb646d27abb9a9d4217421b1c3c6a457b3526dfded144f271ecf787`.

Article identity: DOI `10.1093/nar/gkag672`, OUP article ID `8736262`.

## S2 result: 7500 construct source is present

`Supplemental_Table_S2.xlsx` is Strict OOXML. The `Sequences` sheet has dimension `A1:E7520`; the actual header is row 8 after seven explanatory/example rows. The audited fields are `sequence_name`, `DNA_sequence`, `RNA_sequence`, and `RNA_secondary_structure`.

The formal construct block is:

- 7,500 rows with IDs `construct0` through `construct7499`;
- 7,500 unique IDs, zero duplicate IDs, zero missing numeric IDs in the range;
- all 7,500 construct rows have non-empty DNA, RNA, and secondary-structure fields;
- 7,512 post-header rows exist in the sheet, leaving 12 non-construct rows (controls/design entries) outside the formal construct block;
- the sheet contains no explicit columns for assay condition, batch, barcode, sample, FASTQ, SRA accession, filter status, or filter reason;
- the `Primers` sheet provides primer information but does not supply the missing construct-to-assay run crosswalk.

This is a positive source-content result, not a DMS scientific gate pass. It establishes that the official sequence/structure design table is available; it does not establish which raw read belongs to which assay condition or run.

## Identity reconciliation with Figshare processed source

The audit compared S2 against `data/csvs/library_sequences.csv` inside the preserved Figshare archive, using uppercase, whitespace removal, and U→T normalization for sequence identity only.

- Figshare library table: 7,500 rows and 7,500 unique sequences.
- S2 RNA sequence ↔ Figshare library sequence: 7,500/7,500 matches in both directions.
- Indexed name relationship: `seq_N` ↔ `constructN`, 7,500/7,500 matches. Exact literal-name equality is intentionally not claimed because the two official tables use different naming namespaces.
- S2 structure ↔ Figshare library structure: 7,497 exact matches; 3 auditable structure discrepancies remain (`construct3362`, `construct6271`, `construct7182`). They have equal lengths but nonzero Hamming differences, which are retained in `s2_figshare_identity.json` by ID, positions, and hashes.

For each of the six processed Figshare construct payloads (`pdb_library_1`, `pdb_library_2`, `pdb_library_3`, `pdb_library_37C_2min`, `pdb_library_denature`, `pdb_library_nomod`), the documented 5′-12 and 3′-20 transport trim gives:

- 7,500/7,500 sequence matches by construct position;
- 7,497/7,500 structure matches by construct position.

Therefore the safe source-fidelity statement is: **the complete 7,500-sequence library is source-identical after the documented transport normalization; structure identity is source-faithful for 7,497/7,500 entries, with three retained discrepancies requiring reconciliation before a full structure identity claim.** No stochastic regeneration was used or admitted as a substitute.

## Article-level quality-filter evidence and remaining limitation

The official article-level text states that the initial dataset contained 7,500 sequences and that sequences with fewer than 2,000 reads or signal-to-noise ratio below 4 were removed, eliminating 17 sequences. It also states that individual reactivity measurements with z-scores above 3 were excluded. The same article reports a distinct shorter-incubation condition with 684 failures versus 17 under the standard protocol.

The supplementary package and S2 audit do **not** provide an explicit per-construct filter column, a 17-row exclusion list, an exclusion reason per construct, or an unambiguous mapping of those exclusions to SRA runs/FASTQ files. Accordingly:

- the article-level rule and aggregate count are `SOURCE_DOCUMENTED`;
- the 17 construct identities and per-row reasons are `NOT_AVAILABLE_NOT_ASSERTED`;
- the standard-versus-shorter-incubation failure count is not a substitute for a run-level crosswalk;
- no filtered row was invented, deleted, relabeled, or backfilled from proxy read-depth correlations.

## Gate decision

| Gate | Decision | Evidence boundary |
|---|---|---|
| Final NAR supplement delivery | PASS | Official package present, SHA-256 recorded, ZIP integrity passed, five members extracted |
| S2 7,500 construct content | PASS | Contiguous unique IDs and non-empty sequence/structure fields audited |
| S2 ↔ Figshare sequence source fidelity | PASS | 7,500/7,500 sequence identity and indexed name correspondence |
| S2 ↔ processed structure identity | CONDITIONAL | 7,497/7,500; three retained mismatches |
| Article-level 17-sequence filter rule | DOCUMENTED | Rule and aggregate count are present in the official article snapshot |
| Per-row 17-sequence exclusion traceability | NOT_ASSERTED | No explicit list/reason/run mapping in available supplement |
| DMS condition/batch/barcode/FASTQ/SRA crosswalk | BLOCKED | Not present in S2/S1/S4/S7; prior bounded public/SRA XML audit remains unresolved |
| Phase 0 DMS scientific unlock | NO-GO / FAIL-CLOSED | Contract-required mapping, censoring, review and manual audit gates remain unmet |
| Sequence-model training | NOT AUTHORIZED | No model training or scientific claim is authorized before Phase 0/0.5 |

## Authoritative next action

Continue the bounded public-data/source-crosswalk recovery path, preserving all unresolved states. If the crosswalk cannot be recovered, keep the DMS track blocked and record the failure as an availability/provenance result. A tectoRNA-only branch may be considered only under the controlling contract’s own stage gates; this audit does not rewrite the contract or authorize a new modeling track.

## Reproducibility pointers

- Code: `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/scripts/audit_nar_supplement.py`
- Code SHA-256: `90897c405269ee38c3c599cd48d9cebc235c6f09cd15c5344f8e3ff45360a9f1`
- Audit outputs: `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/nar_gkag672_supplement_20260802T150000Z/`
  - `supplement_inventory.json`
  - `supplement_sha256.txt`
  - `table_s2_schema.json`
  - `s2_figshare_identity.json`
- Source package: `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/nar_gkag672_20260802T150000Z/`
