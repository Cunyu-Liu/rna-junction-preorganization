# Phase 0 source metadata audit

Date: 2026-08-01 (metadata review; no source payload download)

Contract SHA-256: `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`

## Scope and evidence class

This is a narrow provenance reconnaissance pass. It checks official article,
archive, and repository metadata pages for the two primary public observation
sources named by the contract. It does not download raw or processed data, does
not compute source hashes, does not verify licences, and does not construct a
matching table. It is therefore **not** a Phase 0 acceptance record.

## Source A: tectoRNA observation

Registered source: Denny et al., *High-Throughput Investigation of Diverse
Junction Elements in RNA Tertiary Folding*.

Official evidence:

- DOI: <https://doi.org/10.1016/j.cell.2018.05.038>
- Public article record: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6053692/>
- Author/lab publication record listing the article and supporting information:
  <https://herschlaglab.stanford.edu/publications>

The public article metadata and text identify a tectoRNA high-throughput assay.
The reported library includes 1328 junction sequences and 359 sequences with
X-ray crystallographic characterization. The source is relevant to the
contract's two-way-junction observation, but the exact workbook/schema,
raw-versus-interpolated-versus-censored semantics, censor direction, and
replicate/bootstrap/covariance fields remain unverified until the exact source
payload is downloaded and hashed.

## Source B: DMS observation

Registered source: Deenalattha et al., *Characterizing RNA 3D structural
features from DMS reactivity*.

Official evidence:

- Oxford Academic article: <https://academic.oup.com/nar/article/54/14/gkag672/8736262>
- Figshare data record: <https://doi.org/10.6084/m9.figshare.27880434>
- NCBI BioProject: <https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1188187>
- Author code repository: <https://github.com/YesselmanLabPublications/2025_char_3d_struct_features>

The article reports 7500 RNA constructs containing two-way junctions with known
3D structures. It identifies PDB files on Figshare and demultiplexed FASTQ files
under `PRJNA1188187`. It reports removal of sequences with fewer than 2000
reads or signal-to-noise ratio below 4, exclusion of individual measurements
with z-scores above 3, and a canonicalization rule that places the longer motif
strand first and alphabetically orders equal-length strands. It also reports
four reactivity normalization strategies: no-modification control, denatured
control, construct-level average, and common-hairpin reference.

These observations are useful for designing the Phase 0 field audit, but they
are not yet source-level evidence. Treated/background/read-depth hierarchy,
construct IDs, supplement JSON schema, exact PDB/FASTQ file set, checksums,
licence/terms, and split-group reconstruction remain open.

## Gate status

The following contract-required evidence is still absent:

- exact source payloads and SHA-256 values;
- verified license and access terms;
- Denny schema/count reconciliation, including censor direction;
- DMS JSON/raw/background/read-depth reconciliation;
- matched/ambiguous/rejected motif table;
- manual audit of at least 50 matched and 30 rejected/ambiguous cases;
- matching accuracy at least 0.95;
- traceability of all primary labels;
- terminal Phase 0 pass/failure marker.

The contract-source deployment prerequisite is now satisfied by the exact
hash-matched remote copy. The metadata pass itself still has no effect on the
Phase 0 acceptance gate. Phase 0 remains `IN_PROGRESS`, Phase 0.5 and all later
phases remain locked, no primary labels were admitted to modeling, and no GPU
training was started.
