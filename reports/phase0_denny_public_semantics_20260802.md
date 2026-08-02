# Phase 0 Denny public-semantic binding — 2026-08-02

## Result

The Denny et al. public-method semantics are now bound to the pinned PMC/BioC
source snapshot:

`PUBLIC_SEMANTICS_BOUND_NO_CROSSWALK_UNLOCK`

This is a Phase-0 semantic sub-evidence result only. It does not admit primary
labels, does not establish the raw SRA/ENA-to-processed-library crosswalk, and
does not unlock modeling or GPU training.

## Provenance

| Item | Value |
|---|---|
| Contract | `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/contract/1.1.docx` |
| Contract SHA256 | `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9` |
| Public source | [Denny et al., *High-throughput investigation of diverse junction elements in RNA tertiary folding*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6053692/) |
| DOI | `10.1016/j.cell.2018.05.038` |
| Source snapshot | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/pmc_bioc_PMC6053692.json` |
| Source snapshot SHA256 | `37d03a1b4c9dfe5cfb54f992d7e19a0c22ed0cfb06419e1db101b2309eec9f2a` |
| Source snapshot size | `151811` bytes |
| Snapshot passages audited | `258` |
| Audit script | `scripts/audit_denny_public_semantics.py` |
| Audit script SHA256 | `49e790e5ab6cc8d963de27794ee6fbca68c56c193d80df6949f60fbcd3207b21` |
| Audit JSON | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/denny_public_semantics_20260802T141500Z.json` |
| Audit JSON SHA256 | `b897baadae28c1aae49c28de75f0eafaccad4847e48d1bad3720094a2c860175` |
| Audit log | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/denny_public_semantics_20260802T141500Z.log` |
| Audit log SHA256 | `003db47c298b6ea82936bcdc580e302a1b3aed66fd02e136d292d5390d5ba68e` |

The audit records passage indexes and passage hashes rather than emitting raw
sequences, workbook cell values, or primary labels. All seven required semantic
facts were found exactly once in the pinned snapshot; the JSON contains the
corresponding passage hashes and deterministic match metadata.

## Semantics established by the public source

### 1. `-7.1 kcal/mol` is an upper-bound cap

The source describes `-7.1 kcal/mol` as the upper bound of the measurable
affinity range. Nonbinders are reported on the less-negative side (`ΔG > -7.1`)
and the bound is set using the nonbinder/background distribution. Therefore the
contract interpretation is:

`censor_direction = UPPER_BOUND_CAP_AT_DELTA_G_MINUS_7_1`

This is not a lower-bound floor. Any downstream transformation must preserve
the fact that a value at this boundary is censored/limited rather than an exact
measurement.

### 2. Same-context interpolation

For relatively similar motifs, missing affinity values are interpolated using
the median affinity of other motifs in the same
`chip-scaffold / flow-piece` context.

### 3. Dissimilar-context interpolation

For more disparate motifs, the source first finds the 20 most similar
thermodynamic fingerprints within `0.2 kcal/mol MAD`, using only measured
contexts. Missing values are then filled with the median affinity of those
related motifs in the missing `chip-scaffold / flow-piece` contexts.

### 4. Interpolation is analysis-specific

The source states that clustering/PC analyses may use the interpolated data,
while plots of individual data points use only uninterpolated observations.
The project must not treat an interpolated value as an independent measured
label or as raw experimental evidence.

### 5. The 9/11-bp context is a structural context, not a label rule

The public source explicitly discusses 9-bp versus 10-bp flow pieces, 10-bp
chip pieces, and chip scaffolds spanning 8–12 bp. It also describes nine
scaffolds and 11-bp chip-scaffold contexts. These are assay contexts used in
the thermodynamic fingerprint; they do not identify an SRA run, processed
condition, or junction label.

## Gate effect

This audit changes only the Denny semantic-definition subcomponent:

- Denny semantic-definition subcomponent: **PASS / bound to public source**;
- raw-to-processed accession crosswalk: **still unresolved**;
- manual matching acceptance: **still pending**;
- primary labels admitted: **false**;
- Phase 0 gate: **not passed**;
- scientific/modeling gate: **locked**;
- GPU training: **not started**.

The existing workbook semantic audit remains necessary as a separate artifact:
the public article defines the measurement and interpolation operations, but it
does not by itself bind each raw SRA/ENA run to the processed construct namespace
or satisfy the required manual row review.

## Next admissible action

Proceed with accession-preserving crosswalk reconciliation and the required
manual review. If a candidate cannot be uniquely bound, record it as
`rejected` or `ambiguous` with evidence and hash. Do not use the newly bound
semantic rules to manufacture labels or to bypass the crosswalk/manual-review
gate.
