# Phase 0 public crosswalk exhaustion search — 2026-08-02

## Result

The bounded search over the current Figshare archive and the pinned current-DMS
PMC/BioC method snapshot completed without finding an accession-preserving
raw-to-processed crosswalk:

`PUBLIC_CROSSWALK_SEARCH_COMPLETE_UNRESOLVED`

This is a negative result over a declared search scope. It is not a proof that
an unpublished author-side mapping does not exist. It preserves the Phase 0
fail-closed state and does not admit any primary labels.

## Inputs and provenance

| Item | Value |
|---|---|
| Contract SHA256 | `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9` |
| Current Figshare archive | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_figshare/data.zip` |
| Current archive size | `643502044` bytes |
| Current archive SHA256 | `241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681` |
| Current-DMS source snapshot | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/pmc_bioc_PMC11601540.json` |
| Source snapshot size | `93087` bytes |
| Source snapshot SHA256 | `5a048b4b4e236426901fb65c357b582979e4e83e1a187025b8cbf91b8c28eb8f` |
| Audit script | `scripts/audit_public_crosswalk_exhaustion.py` |
| Audit script SHA256 | `06726aa48ce7d0dc35559fd630cf33ef39b2323f107d3cb64dcf6dc6bddc6f64` |
| Successful audit JSON | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/public_crosswalk_exhaustion_20260802T151500Z.json` |
| Successful audit JSON SHA256 | `eb57646e58988c12c82ca14a1f6f7287badc7bfcdaf9de671858c93a91d6fabf` |
| Successful audit log | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/public_crosswalk_exhaustion_20260802T151500Z.log` |
| Successful audit log SHA256 | `3f829e66a353f2d91f57dffc6490b423ee8412ad169dd5083f72d4a1fc2dae5c` |

The earlier failed attempts are preserved and are not overwritten:

| Attempt | Status | Evidence |
|---|---|---|
| `20260802T150000Z` | `BLOCKED_INPUT_HASH_MISMATCH` because the Denny source SHA was supplied for the current-DMS snapshot | JSON SHA `ff5e577060c8d4344f39d987ac4b521c7f0550c6ab89ba17592ec1ef4ffb9724`; log SHA `cbd39f61c7bc2063c7adf180e721b1a13fd4306d1a5e83760de585ca8e7d8721` |
| `20260802T150500Z` | empty log after the first inefficient scan was interrupted safely | empty-log SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; the old audit PID was explicitly identified, paused, then terminated |

The first failed run was a validation failure, not evidence about the data.
The second empty log was an execution incident. The successful result below is
the only result used for the scientific reconciliation conclusion.

## Declared search scope

The script searched:

- all current Figshare archive member names;
- text-like candidate members in `data/csvs/`, `metadata/`, and
  `source_metadata/`;
- candidate basenames `README`, `README.md`, and `manifest.txt`;
- candidate basenames containing `sample`, `manifest`, `metadata`, or
  `sequence`;
- candidate text members up to 64 MiB uncompressed;
- the full pinned current-DMS PMC/BioC snapshot bytes.

Large raw-json and revision trees were not decompressed by this particular
content scan. Their member names were still searched. This limitation is
explicit and is why this artifact is a bounded negative search rather than a
universal nonexistence claim.

## Observed result

| Search layer | Observation |
|---|---|
| Archive members | `3220` total; `10` candidate text members scanned; `0` oversized candidate text members skipped |
| Processed namespace in archive member names | 6 main-library names and 6 control-library names, including the expected `pdb_library_*` payload members |
| Raw accession in archive member names | `0` |
| `trial1`/`trial2` in archive member names | `0` |
| `Sequences.xlsx` in archive member names | `0` |
| Sample/manifest token in archive member names | `0` |
| Raw accession in scanned archive content | `0` matching members |
| Processed namespace in scanned archive content | `0` matching members |
| Raw/processed co-occurrence in one archive member | `0` |
| Current-DMS source snapshot | `PRJNA1188187` and `Sequences.xlsx` are mentioned as public-method/source references; concrete raw accession and `pdb_library_*` tokens are absent |

The archive therefore proves the existence and namespace of processed payload
members, but not which SRA/ENA run produced each one. The current paper snapshot
proves the project-level SRA reference and the existence of the referenced
`Sequences.xlsx` concept, but not the row-level file or accession binding.

## Reconciliation conclusion

The existing candidate relationships remain candidates only:

- `SRR31402664` / `rna_library_trial1` → one of `pdb_library_1/2/3`;
- `SRR31402663` / `rna_library_trial2` → one of `pdb_library_1/2/3`;
- controls with matching public library-name fragments remain name-semantic
  candidates, not accepted primary labels.

The bounded search did not change those decisions. The read-depth and raw-prefix
correlations are supporting candidate evidence, not provenance. No candidate is
promoted to a primary label.

## Gate effect and next evidence

- `raw_processed_accession_crosswalk`: **UNRESOLVED**;
- `manual_matching_acceptance`: **PENDING**;
- `primary_labels_admitted`: **false**;
- `phase0_gate_effect`: **NO_PHASE_0_PASS**;
- `scientific_gate_effect`: **NO_UNLOCK**;
- GPU validation/training: **not started**.

The next admissible evidence is an author/source-defined row-level crosswalk,
such as the missing `Sequences.xlsx`/RTB demultiplex table or an equivalent
immutable mapping from raw run to processed condition and construct namespace.
Until then, do not infer labels from member names, read-depth ratios, prefix
correlations, or project-level SRA metadata.
