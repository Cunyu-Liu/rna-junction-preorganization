# Phase 0 SRA Full XML metadata audit — 2026-08-02

## Result

The selected public SRA runs were audited against the NCBI E-utilities full XML
records and the raw XML responses were retained with SHA256 hashes:

`SRA_FULL_XML_METADATA_COMPLETE_CANDIDATE_ONLY_NO_CROSSWALK_UNLOCK`

This is provenance evidence only. It does not admit primary labels, does not
establish the raw SRA/ENA-to-processed-library crosswalk, does not satisfy the
required manual review, and does not unlock modeling or GPU training.

## Contract and audit provenance

| Item | Value |
|---|---|
| Contract | `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/contract/1.1.docx` |
| Contract SHA256 | `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9` |
| XML source | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id={RUN}&rettype=full&retmode=xml` |
| Raw XML directory | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/sra_full_xml_20260802T163000Z/` |
| Audit script | `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/scripts/audit_sra_full_xml_metadata.py` |
| Audit script SHA256 | `996c91843a1a687378b2cc4b0610cb54c39a28039c9e3eb07a58075e17b678e3` |
| Completed audit JSON | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/sra_full_xml_metadata_20260802T163500Z.json` |
| Completed audit JSON SHA256 | `280007c8f4ad90598aa2f2d9ffd36892399ada5085d1120e4ecae1487be9b2ca` |
| Completed audit log | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/sra_full_xml_metadata_20260802T163500Z.log` |
| Completed audit log SHA256 | `f0a8f0f538d3c6f7c9da1519aff8ec9413f579a135f5afc3c1bac3b5f0df7cd4` |
| Runs audited | `5` |
| XML parse failures | `0` |

The audit records accession, experiment/sample identity, library metadata,
source-material metadata, original FASTQ file metadata, and processed-namespace
token searches. It does not emit read sequences or infer processed labels.

## Run-level metadata recovered

| SRA run | Experiment | Sample | Library alias/name | `source_material_id` | Processed namespace token in XML |
|---|---|---|---|---:|---|
| `SRR31402663` | `SRX26771789` | `SRS23258854` | `rna_library_trial2` | `2` | none |
| `SRR31402664` | `SRX26771788` | `SRS23258855` | `rna_library_trial1` | `1` | none |
| `SRR35766784` | `SRX30816075` | `SRS26827868` | `rna_library_nomod` | unavailable | none |
| `SRR35766785` | `SRX30816074` | `SRS26827867` | `rna_library_denature` | unavailable | none |
| `SRR38259812` | `SRX33096510` | `SRS28915053` | `rna_lib_37C_2min` | unavailable | none |

The XML records also expose the original FASTQ names, byte sizes, and MD5
values. Those file-level fields are preserved in the completed audit JSON and
are not treated as processed-construct identity evidence.

## Retained raw XML hashes

| Run | Raw XML size | SHA256 |
|---|---:|---|
| `SRR31402663` | `9360` bytes | `a89d5216661845ca6f467195d12b242a7db0dc83b2f76d08561fd74f3b077f8e` |
| `SRR31402664` | `9364` bytes | `6807a637379bf722934af1c8f4e6ad7e09aef68bcab333c28b86ac53d768fba4` |
| `SRR35766784` | `9305` bytes | `242de5d3016f5807c0005a29b46ca55b4334f7630ce0a971d09061d7ff6b76b6` |
| `SRR35766785` | `9344` bytes | `3c1fe1c308d5fb407d674e942c5251b8c68c793ab5bb8f1589b5e9915ac68730` |
| `SRR38259812` | `9315` bytes | `7b1e12934545148f653f85c1c0406b8caf15986977ada59f6dc85b1edf601bfb` |

## Interpretation boundary

The metadata strengthens the existing candidate mapping:

- `rna_library_nomod` → `nomod`, `rna_library_denature` → `denature`, and
  `rna_lib_37C_2min` → `37C_2min` are still condition-name candidates only;
- `rna_library_trial1` and `rna_library_trial2` remain candidate identifiers
  for the public trial runs;
- `source_material_id=1` and `source_material_id=2` are candidate metadata,
  not proof that integer `1`/`2` means processed `pdb_library_1`/`pdb_library_2`;
- no XML element, alias, library name, or sample field binds either trial run
  to one of `pdb_library_1`, `pdb_library_2`, or `pdb_library_3`;
- the XML therefore cannot distinguish the unresolved trial mapping, and it
  cannot replace author-defined crosswalk evidence or the contract-required
  manual review.

No label was admitted from this audit. In particular, the audit does not turn
the read-depth or raw-FASTQ prefix correlations into identities.

## Failure and recovery evidence

The first script invocation attempted network retrieval but received five
`URLError` failures before any XML file was written:

| Item | Path | SHA256 |
|---|---|---|
| Failed-attempt JSON | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/sra_full_xml_metadata_20260802T163000Z.json` | `432921294f06945a912f67bc25ba370946b2a79789304f625d98c08f9e596ff0` |
| Failed-attempt log | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/sra_full_xml_metadata_20260802T163000Z.log` | `cee7454bb6916aea80f1a1a2663ffaa3cdce28af36918baf2ff87647782d3a24` |

The retrieval was then rerun through the same official endpoint, with the raw
XML responses saved before the offline parse. The failed attempt is retained
as an execution/transport incident; it is not counted as a scientific result.

## Gate effect

- full XML metadata audit: **PASS / complete for the selected five runs**;
- accession-preserving raw-to-processed crosswalk: **still unresolved**;
- trial1/trial2 processed-library assignment: **ambiguous / candidate only**;
- manual matching acceptance: **pending**;
- primary labels admitted: **false**;
- Phase 0 gate: **not passed**;
- Phase 0.5 specification freeze: **not authorized**;
- GPU training: **not started**.

## Next admissible action

Continue only with provenance-preserving recovery of an author-defined
crosswalk or the contract-required manual review. If no such evidence can be
recovered, retain the unresolved mapping as an explicit blocker and proceed
only along the contract's falsification/alternative path. Do not infer labels
from `source_material_id`, condition names, read depth, or sequence-prefix
correlations.
