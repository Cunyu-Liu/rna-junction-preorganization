# RNA Junction Preorganization v1.2 — Contract Landing

Landing date: 2026-08-04 (Asia/Shanghai)
Landing status: `CLEAN_V1_2_BOUND_NEW_RUN_IN_PROGRESS`

## Authority

The clean v1.2 DOCX is the scientific and engineering authority. The redline is
decision context only; it cannot override the clean document. Historical v1.1
contracts, the 2026-08-03 v1.2 worktree, its manifest, reports, and replay outputs
are preserved as historical context and are `STALE_NOT_AUTHORITATIVE` for this run.

| Artifact | Source / remote landing | SHA-256 | Status |
|---|---|---|---|
| clean v1.2 contract | `contract/1.2.docx` | `3ad0c9997cdea8e510f80424c4b011062f0f95a8bf8879a4659a847adcab22a0` | `HASH_MATCH` |
| decision redline | `contract/1.1_to_1.2_redline.docx` | `c77d647de7f644eafe10292aa2474d230de4a9483952ee390b45c75983bca7a5` | `HASH_MATCH` |
| decision/claim log | `contract/v1.2_decision_and_claim_log.md` | `372e37a159195f6d5b21d57aca32cf1d679b4408d6ce9b43b079974a87adcf92` | `HASH_MATCH` |

The clean contract was fully extracted before implementation: 373 paragraphs,
81 tables, and 454 body-level blocks. Local render QA sampled the rendered
pages; no clipping or overlap was observed. CJK fallback glyphs in the bundled
render environment are an environment/font note, not a byte or text-authority
change.

## This run

```text
run_id = v1_2_tecto_qmap_codex_20260804T074900Z
parent_run_id = v1_2_tecto_qmap_20260803
host = bms-18937653-012
worktree = /home/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z
branch = codex/v1_2_tecto_qmap_codex_20260804T074900Z
base_commit = d4768817c4d4bc5fe469762fd0d4fa921a4e7c11
run_root = /mnt/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z
manifest = manifests/canonical_manifest_v1_2_tecto_qmap_codex_20260804T074900Z.json
```

The worktree was created from the historical v1.2 code commit but does not use
the historical worktree or its outputs as current evidence. Builders and
finalizers are rebound through `scripts/runtime_config.py` to the run-specific
worktree, run root, manifest, and clean-contract hash. The actual clean DOCX
hash is checked before a formal gate can pass.

## Frozen initial machine state

```text
CURRENT_OPERATIONAL_STATE = BLOCKED_AT_TECTO_DATA_ADMISSION
CURRENT_SCIENTIFIC_DISPOSITION = CONDITIONAL_CANDIDATE
CURRENT_DMS_CROSSWALK = ASSUMED_PERMANENTLY_UNAVAILABLE_V1_2
CURRENT_DMS_PRIMARY_LABELS = NOT_ADMITTED_FINAL_V1_2
CURRENT_DMS_REPLAY = ENGINEERING_EVIDENCE_ONLY
CURRENT_DMS_JOINT_TRANSPORT = CLOSED_NOT_AUTHORIZED
QMAPSEQ_ROLE = MANDATORY_COMPLETION_GATE_FOR_STRONG_MANUSCRIPT
SCIENTIFIC_UNLOCK = NO_UNLOCK
T0 = BLOCKED
S0/T1/M0/T2/T3 = NOT_STARTED
Q0/Q1/Q2/Q3/Q4/Q5 = NOT_STARTED
qmap_terminal_disposition = NOT_ADJUDICATED
claim_class = NOT_ADJUDICATED
```

No current-DMS primary labels, fitting, split/threshold/feature selection,
effect-size claim, joint model, test-time input, or scientific claim is allowed
in v1.2. Tecto and qMaPseq remain independent DAGs; qMaPseq does not unlock
tecto, and Q4/Q5 cannot retroactively change frozen tecto choices.

## Scientific boundary

- Primary estimand: one target-specific thermodynamic functional `Φ*` for one
  frozen `K*` and condition `c*`.
- Partial identification is first-class: report identified intervals/sets,
  coverage, width, and dominant uncertainty source whenever point identification
  fails.
- Tecto DAG: `T0 → S0 → T1 → M0 → T2 → T3 → optional sequence-only → claim adjudication`.
- qMaPseq DAG: `Q0 → Q1 → Q2 → Q3 → Q4 → Q5 → claim adjudication`.
- Primary split: motif-family outer holdout with component closure; random row,
  nucleotide, replicate, condition-crossing, and giant-component-breaking splits
  are prohibited.
- The qMaPseq attrition contract preserves `98 = 84 fitted + 11 right-censored
  + 2 closing-pair structural-QC + 1 alternate/unknown`; censored rows are not
  deleted.

## Source snapshot policy

Source artifacts are copied into the new run root before rebuilding. Their
checksums and original locations are recorded in a run-local source snapshot
manifest. Historical derived outputs may be used only as a route or checksum
cross-check; they cannot satisfy a current gate without regeneration and
current-run provenance.

## Finalizer policy

Only the finalizer may write `PASS`. Every formal gate must bind the clean
contract hash, current code commit, current run ID, schema-valid canonical
manifest, required artifacts, checksums, and gate-specific tests. Failed or
incomplete criteria remain `RUNNING`/`BLOCKED` with an explicit
`PARTIAL_ENGINEERING_EVIDENCE` or failure decision; no stale report is promoted.
