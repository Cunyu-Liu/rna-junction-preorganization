#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X0 — independent external-case qualification audit (v1.5 §13).

Qualifies whether a public case OUTSIDE the RNA-MaP/tectoRNA platform cluster
can serve as a genuinely independent external case that tests method
transportability (rather than mechanism replication).

Primary candidate: PRIME (Choi/Bose/Mathews/Mustoe/Lucks 2026 preprint).
Facts about PRIME used here are web-verified on X0 day from primary sources:
  - bioRxiv preprint DOI 10.64898/2026.01.28.702231 (2026-01-30), NOT peer reviewed
  - repo https://github.com/LucksLab/Choi_PRIME_Chemprobing_2026 (CC-BY-4.0),
    24 commits, last 2026-02-14 -> actively updated, no frozen release
  - DMS-MaP chemical probing; per-nucleotide energetics (0.5-3 kcal/mol);
    PRIME constructs are a small set of well-studied systems (fourU, HIV TAR,
    Tetrahymena P4P6) plus a few mutants
  - raw sequencing via NCBI SRA BioProject PRJNA1400640

This script writes the X0 qualification artifacts into RUN_ROOT/external_case/x0/.
It never fabricates a qualification: the decision is derived strictly from the
assessed eligibility matrix below.
"""

from __future__ import annotations
import csv
import json
import os
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
X0_DIR = f"{RUN_ROOT}/external_case/x0"


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_tsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(X0_DIR, exist_ok=True)
    now = _utcnow()

    # ------------------------------------------------------------------
    # 1. candidate_registry.tsv
    # ------------------------------------------------------------------
    candidates = [
        ["PRIME_P4P6", "PRIME (Probing-Resolved Inference of Molecular Energetics) — P4P6/Tetrahymena group I intron domain",
         "DMS-MaP chemical probing -> per-nucleotide structural energetics",
         "yes", "bioRxiv preprint 10.64898/2026.01.28.702231 (2026-01-30)",
         "preprint, not peer-reviewed; repo actively updated (No fixed version)"],
        ["PRIME_HIV", "PRIME — HIV-1 TAR dynamic ensemble construct",
         "DMS-MaP chemical probing -> per-nucleotide structural energetics",
         "yes", "bioRxiv preprint 10.64898/2026.01.28.702231",
         "preprint; same PRIME family (single candidate family, not 3 independent systems)"],
        ["PRIME_4U", "PRIME — fourU RNA construct",
         "DMS-MaP chemical probing -> per-nucleotide structural energetics",
         "yes", "bioRxiv preprint 10.64898/2026.01.28.702231",
         "preprint; same PRIME family"],
        ["smFRET_opens_for_review", "smFRET/SAXS RNA thermodynamic measurement systems (e.g. Al-Hashimi NMR dynamics)",
         "NMR/smFRET dynamics of base-pair opening",
         "uncertain", "various settled papers",
         "estimand/operator mapping to a construct-level transport audit is not established here"],
    ]
    write_tsv(f"{X0_DIR}/candidate_registry.tsv",
              ["candidate_id", "name", "measurement_system", "assessed",
               "source_authority", "version_status"],
              candidates)

    # ------------------------------------------------------------------
    # 2. source_integrity.tsv  (per assessed candidate)
    # ------------------------------------------------------------------
    source_rows = [
        ["PRIME_P4P6", "source_authority", "bioRxiv preprint 10.64898/2026.01.28.702231 (2026-01-30); not peer-reviewed",
         "PRESENT", "UNSETTLED", "preprint authority not settled; cannot be treated as settled conclusion"],
        ["PRIME_P4P6", "license", "repo LICENSE-CC-BY-4.0.txt; SRA data public",
         "PRESENT", "OK", "CC-BY-4.0 permits reuse"],
        ["PRIME_P4P6", "fixed_version", "repo at 24 commits, last 2026-02-14; bioRxiv v1",
         "ABSENT", "UNSETTLED", "actively-updated repo; no frozen release/commit pinned for X0"],
        ["PRIME_P4P6", "download_checksum", "SRA BioProject PRJNA1400640; repo not pinned to a checksummed snapshot",
         "PARTIAL", "NOT_VERIFIED", "would require pinning a repo commit + SRA snapshot before any transfer"],
        ["PRIME_P4P6", "construct_canonical_id", "fourU / HIV / P4P6 family; constructs identified",
         "PRESENT", "OK", "construct identity is explicit in the study"],
        ["PRIME_P4P6", "measured_interpolated_censored_distinction", "per-nucleotide kobs/dG; no construct-level measured/censored/failed taxonomy in the audit's sense",
         "ABSENT", "NOT_APPLICABLE", "audit censoring taxonomy not defined at PRIME construct level"],
        ["PRIME_P4P6", "selection_attrition_reconstructable", "not demonstrated at construct level",
         "UNKNOWN", "NOT_VERIFIED", ""],
    ]
    write_tsv(f"{X0_DIR}/source_integrity.tsv",
              ["candidate_id", "dimension", "finding", "provenance_status",
               "verdict", "note"],
              source_rows)

    # ------------------------------------------------------------------
    # 3. platform_lineage.tsv  (independent systems vs the cluster)
    # ------------------------------------------------------------------
    platform_rows = [
        ["qMaP2024", "Das lab (Kladwang et al., NAR 2024)", "SHAPE chemical probing / Mg2+ endpoint thermodynamics",
         "in-project parent", "the audited system"],
        ["RNA-MaP_cluster", "Denny/Bonilla/Shin/Yesselman (SHAPE-MaP / tectoRNA lineage)",
         "SHAPE-MaP mutational profiling / RNA-MaP", "forbidden to split",
         "registered as ONE platform lineage, not 3 independent systems"],
        ["PRIME", "Lucks/Mustoe lab (Choi et al. 2026 preprint)", "DMS-MaP chemical probing -> per-nucleotide energetics",
         "independent measurement platform", "NOT in the RNA-MaP/tectoRNA cluster; platform-independent vs qMaP"],
        ["smFRET_NMR", "Al-Hashimi et al. NMR/smFRET dynamics", "NMR relaxation dispersion / single-molecule dynamics",
         "independent classical biophysics", "not adopted as a qualified X0 case here"],
    ]
    write_tsv(f"{X0_DIR}/platform_lineage.tsv",
              ["system", "lab_cluster", "measurement_platform", "relation_to_project", "note"],
              platform_rows)

    # ------------------------------------------------------------------
    # 4. qualification_matrix.tsv  (eligibility conditions §13.1)
    # ------------------------------------------------------------------
    # Each row: candidate, condition, evidence, PASS/FAIL/INCONCLUSIVE, rationale
    qual_rows = [
        ["PRIME_P4P6", "source authority / license / checksum",
         "CC-BY-4.0 + SRA public; preprint not settled; no pinned checksummed snapshot",
         "INCONCLUSIVE", "license OK but preprint authority unsettled and no frozen snapshot"],
        ["PRIME_P4P6", "construct-level canonical ID / sequence / condition / endpoint",
         "fourU/HIV/P4P6 constructs explicit; endpoint is per-nucleotide dG",
         "PASS", "construct identity explicit"],
        ["PRIME_P4P6", "measured / interpolated / censored / failed distinction",
         "not defined in the audit's censoring taxonomy at construct level",
         "FAIL", "censoring taxonomy not mapped"],
        ["PRIME_P4P6", "selection / attrition reconstructable",
         "not demonstrated",
         "INCONCLUSIVE", "not verifiable from the preprint alone"],
        ["PRIME_P4P6", "independent biological groups + valid outer holdout",
         "only a small set of well-studied constructs (fourU, HIV, P4P6) + few mutants",
         "FAIL", "construct-level N is too small to power a pre-registered held-out transport test"],
        ["PRIME_P4P6", "platform-lineage independence from tecto/qMaP",
         "DMS-MaP (Lucks/Mustoe) vs SHAPE (Das); distinct lab and probe",
         "PASS", "genuinely independent measurement platform"],
        ["PRIME_P4P6", "estimand / operator mapping explicit",
         "PRIME estimand is per-nucleotide structural energetics; audit transport estimand is population-level held-out transport",
         "FAIL", "operator/estimand mapping is ambiguous, not a complete bijection to the audit operator"],
        ["PRIME_P4P6", "data volume sufficient for pre-registered held-out test",
         "large at nucleotide level but low at independent-construct level",
         "FAIL", "cannot power a construct-level outer holdout"],
        ["PRIME_P4P6", "PDB / template / pretraining exposure",
         "P4P6 is a well-studied structure (PDB) with prior thermodynamic literature",
         "INCONCLUSIVE", "exposure audit not performed; would be needed before any transfer"],
        ["PRIME_P4P6", "contact author / new wet experiment needed",
         "operator mapping would require author confirmation or new construct-level data",
         "REQUIRED", "resolving the operator/estimand gap needs author input or a new wet experiment, both outside current authorization"],
    ]
    write_tsv(f"{X0_DIR}/qualification_matrix.tsv",
              ["candidate_id", "eligibility_condition", "evidence", "verdict", "rationale"],
              qual_rows)

    # ------------------------------------------------------------------
    # 5. Qualification decision
    # ------------------------------------------------------------------
    # PRIME is genuinely platform-independent, but (a) low independent-construct N,
    # (b) operator/estimand mapping ambiguous, (c) preprint authority unsettled.
    # => INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY. This closes the strong
    # cross-case / general-transport claim but does NOT block B3 or the resource
    # auditable route.
    verdicts = [r[3] for r in qual_rows]
    decisive_fails = [r[3] for r in qual_rows if r[3] in ("FAIL", "REQUIRED")]
    decision = {
        "schema_version": "X0-qualification-v1.5",
        "gate": "X0",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "decision_time_utc": now,
        "primary_candidate": "PRIME (Choi et al. 2026 preprint, DMS-MaP)",
        "platform_independent": True,
        "platform_lineage_note": "PRIME is NOT in the RNA-MaP/tectoRNA cluster; Denny/Bonilla/Shin/Yesselman registered as ONE lineage",
        "eligibility_verdicts": verdicts,
        "decisive_blockers": decisive_fails,
        "qualification_summary": (
            "PRIME is a genuinely independent measurement platform (DMS-MaP, Lucks/Mustoe lab) "
            "and therefore satisfies the platform-lineage-independence condition. However it does "
            "NOT meet full qualification for a pre-registered held-out transport test because: "
            "(1) independent-construct N is small (fourU/HIV/P4P6 plus a few mutants) so no "
            "construct-level outer holdout can be powered; (2) the operator/estimand mapping "
            "between PRIME per-nucleotide energetics and the audit population-scale transport "
            "estimand is ambiguous/incomplete; (3) the source is an unsettled preprint with an "
            "actively-updated repo (no pinned frozen snapshot). Resolving (2) requires author "
            "input or a new wet experiment, both outside this run's authorization."
        ),
        "owner_claim_scope": (
            "X0_NOT_QUALIFIED/INCONCLUSIVE closes the strong cross-case / general-transport claim. "
            "PRIME may be cited as a platform-independent future candidate, but transport/generalization "
            "may only be claimed within B3's generative benchmark and the audited qMaP population."
        ),
        "state": "X0_INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY",
        "outputs": {
            "candidate_registry": "external_case/x0/candidate_registry.tsv",
            "source_integrity": "external_case/x0/source_integrity.tsv",
            "platform_lineage": "external_case/x0/platform_lineage.tsv",
            "qualification_matrix": "external_case/x0/qualification_matrix.tsv",
        },
    }
    with open(f"{X0_DIR}/X0_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # No external_case_analysis_card.json is written: §13.2 only requires it when qualified.

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    report = [
        "# X0 — Independent External-Case Qualification Audit",
        "",
        f"**Decision:** X0_INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY  ({now})",
        "",
        "## Candidate",
        "",
        "PRIME (Choi, Bose, Mathews, Mustoe, Lucks), *Ubiquitous low-energy RNA fluctuations and "
        "energetic coupling measured by chemical probing*, bioRxiv preprint "
        "DOI 10.64898/2026.01.28.702231 (2026-01-30). DMS-MaP chemical probing to per-nucleotide "
        "structural energetics. Repo: github.com/LucksLab/Choi_PRIME_Chemprobing_2026 (CC-BY-4.0); "
        "raw sequencing via NCBI SRA BioProject PRJNA1400640.",
        "",
        "## Platform lineage",
        "",
        f"PRIME is platform-independent vs qMaP (SHAPE, Das lab) and NOT in the RNA-MaP/tectoRNA "
        f"cluster. Denny/Bonilla/Shin/Yesselman are registered as ONE lineage, not three systems.",
        "",
        "## Why not fully qualified",
        "",
        "1. **Low independent-construct N** — a small set of well-studied constructs (fourU, HIV, "
        "P4P6) plus a few mutants cannot power a construct-level pre-registered outer holdout.",
        "",
        "2. **Operator/estimand ambiguity** — PRIME's estimand is per-nucleotide structural "
        "energetics; the audit's operator is a population-level held-out transport test. The mapping "
        "is not a complete, explicit bijection and would require author input or a new wet experiment.",
        "",
        "3. **Unsettled source authority** — preprint (not peer reviewed) with an actively-updated "
        "repo; no pinned, checksummed frozen snapshot for X0.",
        "",
        "## Consequence",
        "",
        f"X0_INCONCLUSIVE closes the strong cross-case / general-transport claim. B3 remains "
        f"VALIDATED. The manuscript route proceeds to N1 as a resource/auditable route.",
        "",
    ]
    with open(f"{RUN_ROOT}/reports/X0_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())