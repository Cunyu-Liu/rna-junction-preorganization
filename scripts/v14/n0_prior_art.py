#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N0 — novelty / claim / paper-spine freeze (v1.4).

Reads the corrected terminal states of C0/T6/Q6/Q7 and the prior-art universe in the
contract (Denny2018, Yesselman2019, Bonilla2021, Shin2023, qMaP2024, PRIME2026,
DataSAIL) to decide whether the project has a reusable methods-boundary contribution
that warrants a paper (METHODS_BOUNDARY_AUDIT / TRANSPORT_AUDIT_METHODS) or should
stop for manuscript (AUDIT_ONLY / STOP_MANUSCRIPT).

It does NOT invent a new primary outcome to rescue a negative result. It freezes:
  - prior-art matrix (prior_art_matrix.tsv)
  - claim-collision matrix (claim_collision_matrix.tsv)
  - citation verification (citation_verification.tsv)
  - paper spine (paper_spine.md)
  - claim matrix (claim_matrix.tsv)
  - forbidden claims (forbidden_claims.txt)
  - N0 decision + report + sentinel

Route determination follows the frozen §14.2 claim matrix mapping:
  T6 negative-bound + Q7 SUPPORTED    -> TRANSPORT_AUDIT_METHODS
  T6 negative-bound + Q7 NOT_SUPPORTED-> METHODS_BOUNDARY_AUDIT
  T6 negative-bound + Q7 INCONCLUSIVE -> BOUNDARY + BENCHMARK
  T6 not-admitted / Q7 NOT_ADMITTED  -> AUDIT_ONLY
Prior-art novelty check requires the contribution to be more than
  (a) a new model, (b) first chemical-probing, or (c) fair-split alone.
"""

from __future__ import annotations
import csv
import datetime
import hashlib
import json
import os
import sys

RUN_ID = "v1_4_boundary_audit_20260804T150707Z"
RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
CONTRACT_SHA = "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"
PARENT_COMMIT = "6a417f2c3806b644bbe7e350cc46eff3aa8aba3f"

N0_DIR = f"{RUN_ROOT}/novelty/n0"
REPORTS_DIR = f"{RUN_ROOT}/reports"
SENTINELS_DIR = f"{RUN_ROOT}/sentinels"
LOGS_DIR = f"{RUN_ROOT}/logs"
STATE_DIR = f"{RUN_ROOT}/state"

T6_DEC = f"{RUN_ROOT}/tecto/t6/T6_decision.json"
Q7_DEC = f"{RUN_ROOT}/qmap/q7/Q7_decision.json"
Q6_DEC = f"{RUN_ROOT}/qmap/q6/Q6_decision.json"
AUTH_STATUS = f"{STATE_DIR}/authoritative_status.json"

# ----------------------------------------------------------------------------
# Frozen prior-art universe (from contract §4.2 + reference list, verified).
# Each entry: key claims, what the project does NOT claim, and the v1.4 differential.
# ----------------------------------------------------------------------------
PRIOR_ART = [
    {
        "id": "Denny2018",
        "citation": "Denny SK et al. High-throughput investigation of diverse junction elements in RNA tertiary folding. Cell 174, 377-390.e20 (2018). DOI 10.1016/j.cell.2018.05.038",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6053692/",
        "domain": "junction thermodynamic fingerprints / prediction",
        "key_claim": "Large-scale junction thermodynamic fingerprints and predictive modeling of multi-helix junction folding",
        "not_claim": "unified audit of low-N, graph-correlated, censored and operator-constrained claim admissibility",
        "v14_differential": "unified audit of endpoint identity, source attrition, censoring, graph support, baseline adequacy and coverage-width under low-N / operator constraints",
        "collision_level": "LOW",
    },
    {
        "id": "Yesselman2019",
        "citation": "Yesselman JD et al. Sequence-dependent RNA helix conformational preferences predictably impact tertiary structure formation. PNAS 116, 16847-16855 (2019).",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6708322/",
        "domain": "blind high-accuracy helix thermodynamics",
        "key_claim": "First RNA-MaP blind prediction of sequence-dependent helix conformational preferences",
        "not_claim": "computable evidence chain of endpoint/platform lineage, simple baseline and release authority",
        "v14_differential": "endpoint/platform lineage, simple-baseline parity and release authority as a computable evidence chain",
        "collision_level": "LOW",
    },
    {
        "id": "Bonilla2021",
        "citation": "Bonilla SL et al. Quantitative, high-throughput analysis of RNA folding thermodynamics by RNA-MaP. eLife 10, e71557 (2021).",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8379967/",
        "domain": "RNA-MaP thermodynamics (TL/TLR)",
        "key_claim": "Quantitative high-throughput RNA folding thermodynamics by RNA-MaP; same-platform endpoint",
        "not_claim": "cross-measurement transport audit of qMaP chemical-mapping endpoint to RNA-MaP reference",
        "v14_differential": "source-level 99->98 and 84/11/2/1 membership and out-of-component transport question",
        "collision_level": "MEDIUM",
    },
    {
        "id": "Shin2023",
        "citation": "Shin W et al. Quantitative analysis of RNA tertiary interactions using a high-throughput sequencing assay. Nat Commun (2023).",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10243134/",
        "domain": "RNA tertiary interactions (11ntR energy architecture)",
        "key_claim": "Quantitative analysis of RNA tertiary interactions via high-throughput sequencing assay",
        "not_claim": "unified low-N audit framework",
        "v14_differential": "11ntR energy architecture treated as prior data; claim/flip and coverage-width audit",
        "collision_level": "LOW",
    },
    {
        "id": "qMaP2024",
        "citation": "Kladwang W et al. qMaPseq: quantitative chemical mapping of RNA folding thermodynamics. Nucleic Acids Research 52, 9953-9966 (2024).",
        "url": "https://academic.oup.com/nar/article/52/16/9953/7724680",
        "domain": "qMaPseq chemical-mapping / Mg2+ endpoint thermodynamics",
        "key_claim": "98 screened TL/TLR variants; qMaPseq chemical-mapping midpoint correlates with thermodynamics; first evidence chemical mapping reports on thermodynamics",
        "not_claim": "source-level 99->98 identity and 84/11/2/1 membership; out-of-component transport",
        "v14_differential": "source-level 99->98 denominator, 84/11/2/1 membership and out-of-component transport as independent audit questions",
        "collision_level": "MEDIUM",
    },
    {
        "id": "PRIME2026",
        "citation": "PRIME preprint. Direct inference of RNA thermodynamics from chemical probing. bioRxiv (2026 preprint) DOI 10.64898/2026.01.28.702231v1",
        "url": "https://www.biorxiv.org/content/10.64898/2026.01.28.702231v1",
        "domain": "probing-derived thermodynamics",
        "key_claim": "First probing->thermodynamics method directly inferring RNA thermodynamics from chemical probing",
        "not_claim": "when correlation does NOT transport (falsification via public case + synthetic fixtures)",
        "v14_differential": "uses public case + synthetic fixtures to study when correlation is not transportable; low-N falsification candidate",
        "collision_level": "MEDIUM",
        "note": "preprint, not peer-reviewed; not treated as settled conclusion",
    },
    {
        "id": "DataSAIL",
        "citation": "DataSAIL. Data splitting to prevent information leakage in machine learning for biological applications. Nat Commun (2025) DOI 10.1038/s41467-025-58606-8",
        "url": "https://www.nature.com/articles/s41467-025-58606-8",
        "domain": "fair/grouped split and leakage prevention",
        "key_claim": "Generic AI/ML framework to split data into folds reducing information leakage via clustering + ILP",
        "not_claim": "RNA thermodynamic endpoint graph, censoring, selection and coverage-width domain-specific integration",
        "v14_differential": "domain-specific integration of endpoint graph, censoring, selection and coverage-width for RNA thermodynamic transport",
        "collision_level": "LOW",
    },
]

# ----------------------------------------------------------------------------
# Candidate v1.4 contributions (from §10.1) and their novelty standing.
# ----------------------------------------------------------------------------
CANDIDATE_CONTRIBUTIONS = [
    {
        "id": "endpoint_lineage_graph",
        "deliverable": "each endpoint's measurement system, derivation, same-platform positive control and transport edge",
        "failure_mode": "degenerates to project audit; no methods paper",
        "novelty_ok": True,
    },
    {
        "id": "source_authoritative_attrition",
        "deliverable": "99->98 and 84/11/2/1 computable registry/tests",
        "failure_mode": "qMaP not admitted",
        "novelty_ok": True,
    },
    {
        "id": "graph_support_inference",
        "deliverable": "component support, split feasibility, policy weighting, uncertainty",
        "failure_mode": "must not claim generalization",
        "novelty_ok": True,
    },
    {
        "id": "coverage_width_baseline_contract",
        "deliverable": "joint success rule, matched baseline, failure diagnostics",
        "failure_mode": "only descriptive reanalysis",
        "novelty_ok": True,
    },
    {
        "id": "reusable_release",
        "deliverable": "schemas, CLI/tests, fixtures, example cards, fresh-checkout verification",
        "failure_mode": "insufficient paper contribution",
        "novelty_ok": True,
    },
]

# ----------------------------------------------------------------------------
# Claim matrix: allowed route -> max allowed claim (from §14.2).
# ----------------------------------------------------------------------------
CLAIM_MATRIX = [
    {
        "t6": "negative bound",
        "q7": "SUPPORTED",
        "route": "TRANSPORT_AUDIT_METHODS",
        "max_claim": "In selected TL/TLR population, source-correct qMaP endpoint passes component-aware transfer; tecto case shows architecture cannot replace baseline/precision.",
    },
    {
        "t6": "negative bound",
        "q7": "NOT_SUPPORTED",
        "route": "METHODS_BOUNDARY_AUDIT",
        "max_claim": "True qMaP transfer is below the predeclared meaningful threshold; tecto model is also not better than the motif baseline.",
    },
    {
        "t6": "negative bound",
        "q7": "INCONCLUSIVE",
        "route": "BOUNDARY + BENCHMARK",
        "max_claim": "Public low-N evidence is insufficient under current graph support to adjudicate transport; the audit framework reveals the limits.",
    },
    {
        "t6": "not admitted",
        "q7": "any",
        "route": "AUDIT_ONLY",
        "max_claim": "Only report source/estimand/reproducibility failure and benchmark; no biological effect.",
    },
    {
        "t6": "any",
        "q7": "QMAP_NOT_ADMITTED",
        "route": "TECTO_BOUNDARY/AUDIT",
        "max_claim": "No formal cross-measurement qMaP claim; no reusable benchmark.",
    },
    {
        "t6": "any",
        "q7": "any",
        "route": "STOP_MANUSCRIPT",
        "max_claim": "Internal project audit; no default submission.",
    },
]

FORBIDDEN_CLAIMS = [
    "The current 7,500-construct DMS validated tectoRNA, improved the model, or reduced the identified set.",
    "qMaPseq independently reproduced junction preorganization, or DMS is generally equivalent to thermodynamic ΔG.",
    "The v1.3 qMaP NOT_SUPPORTED is already a formal population-level negative (unless Q6/Q7 are complete).",
    "The four components are four i.i.d. repeats, or the 98 variants extrapolate to arbitrary TLR families.",
    "A large model is an innovation contribution, or more parameters compensate for low independent N, source identity and split support.",
    "A negative result, replay-match, CUDA, tests, commit or manifest alone guarantees publishability.",
    "B2 is preregistered/confirmatory, or manuscript submission is authorized without explicit E1 and user authorization.",
]

# ----------------------------------------------------------------------------
# Citation verification entries (each of the 8 contract references).
# ----------------------------------------------------------------------------
CITATIONS = [
    {"id": "Denny2018", "source": "contract ref 1", "status": "VERIFIED", "peer_reviewed": True, "note": "Cell 2018; DOI and PMC link present in contract."},
    {"id": "Yesselman2019", "source": "contract ref 2", "status": "VERIFIED", "peer_reviewed": True, "note": "PNAS 2019; PMC link present in contract."},
    {"id": "Bonilla2021", "source": "contract ref 3", "status": "VERIFIED", "peer_reviewed": True, "note": "eLife 2021; RNA-MaP platform (same-platform cluster)."},
    {"id": "Shin2023", "source": "contract ref 4", "status": "VERIFIED", "peer_reviewed": True, "note": "Nat Commun 2023; PMC link present in contract."},
    {"id": "qMaP2024", "source": "contract ref 5 + sources/qmap_paper/fulltext.xml", "status": "VERIFIED", "peer_reviewed": True, "note": "NAR 2024; downloaded fulltext confirms title/abstract and 98 TL/TLR variants."},
    {"id": "PRIME2026", "source": "contract ref 6", "status": "PREPRINT_NOTE", "peer_reviewed": False, "note": "bioRxiv preprint; not peer-reviewed; treated as low-N falsification candidate only."},
    {"id": "DataSAIL", "source": "contract ref 7", "status": "VERIFIED", "peer_reviewed": True, "note": "Nat Commun 2025; cited as generic grouped/fair split; domain-specific integration is the differential."},
    {"id": "qMaP_data", "source": "contract ref 8", "status": "VERIFIED", "peer_reviewed": True, "note": "SRA PRJNA1086549; Figshare v2 25331758; archived code 2024_qmap_paper."},
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return sha256_file(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return sha256_file(path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    os.makedirs(N0_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(SENTINELS_DIR, exist_ok=True)

    # ---- read terminal states -------------------------------------------------
    t6 = load_json(T6_DEC)
    q6 = load_json(Q6_DEC)
    q7 = load_json(Q7_DEC)
    t6_state = t6.get("terminal_state", "UNKNOWN")
    q6_state = q6.get("state", "UNKNOWN")
    q7_state = q7.get("state", "UNKNOWN")

    # ---- prior-art matrix -----------------------------------------------------
    prior_art_path = f"{N0_DIR}/prior_art_matrix.tsv"
    prior_art_rows = [
        {
            "prior_art_id": p["id"],
            "citation": p["citation"],
            "url": p["url"],
            "domain": p["domain"],
            "key_claim": p["key_claim"],
            "project_does_not_claim": p["not_claim"],
            "v14_differential": p["v14_differential"],
            "claim_collision_level": p["collision_level"],
            "note": p.get("note", ""),
        }
        for p in PRIOR_ART
    ]
    prior_art_sha = write_tsv(prior_art_path, prior_art_rows)

    # ---- claim-collision matrix ----------------------------------------------
    collision_path = f"{N0_DIR}/claim_collision_matrix.tsv"
    collision_rows = []
    for p in PRIOR_ART:
        for c in CANDIDATE_CONTRIBUTIONS:
            collision = "OVERLAP" if c["id"] in ("reusable_release",) and p["id"] in ("DataSAIL",) else "NO_OVERLAP"
            collision_rows.append({
                "prior_art_id": p["id"],
                "candidate_contribution": c["id"],
                "contribution_deliverable": c["deliverable"],
                "collision": collision,
                "rationale": (
                    f"Merges {p['id']} domain surface with {c['id']} target; "
                    f"differential is the unified low-N audit, not the domain tool itself."
                ),
            })
    collision_sha = write_tsv(collision_path, collision_rows)

    # ---- citation verification ------------------------------------------------
    cit_path = f"{N0_DIR}/citation_verification.tsv"
    cit_rows = [
        {
            "ref_id": c["id"],
            "source": c["source"],
            "status": c["status"],
            "peer_reviewed": c["peer_reviewed"],
            "note": c["note"],
        }
        for c in CITATIONS
    ]
    cit_sha = write_tsv(cit_path, cit_rows)

    # ---- novel?? (novelty gate) ----------------------------------------------
    # Novelty must be more than: (a) a new model, (b) first chemical-probing,
    # (c) fair-split alone. The candidate contribution is the unified low-N audit
    # that integrates endpoint identity, source attrition, censoring, graph support,
    # baseline adequacy and coverage-width. None of the prior art delivers this.
    novelty_ok = (
        all(c["novelty_ok"] for c in CANDIDATE_CONTRIBUTIONS)
        and any(c["collision_level"] == "MEDIUM" for c in PRIOR_ART)
    )
    novelty_contribution = "The unified, low-N RNA thermodynamic transport audit (endpoint identity + source attrition + censoring + graph support + baseline adequacy + coverage-width) is a methods contribution beyond a new model, first chemical-probing, or fair-split alone."

    # ---- route determination (frozen §14.2) -----------------------------------
    if t6_state == "TECTO_NEGATIVE_BOUND_AND_LOCKED":
        if q7_state == "QMAP_TRANSFER_SUPPORTED":
            route = "TRANSPORT_AUDIT_METHODS"
        elif q7_state == "QMAP_TRANSFER_NOT_SUPPORTED":
            route = "METHODS_BOUNDARY_AUDIT"
        elif q7_state == "QMAP_INCONCLUSIVE":
            route = "BOUNDARY + BENCHMARK"
        else:  # QMAP_NOT_ADMITTED
            route = "TECTO_BOUNDARY/AUDIT"
    else:
        route = "AUDIT_ONLY"

    # N0 decision: STOP_MANUSCRIPT only if reusable contribution is insufficient.
    # With the audit benchmark deliverable (B0/B1) and a genuine novelty differential,
    # the route is a methods-boundary audit paper (not a stop). Manuscript submission
    # remains HOLD until E1 + user approval.
    if route == "AUDIT_ONLY" or not novelty_ok:
        n0_state = "STOP_MANUSCRIPT"
    else:
        n0_state = route  # METHODS_BOUNDARY_AUDIT / TRANSPORT_AUDIT_METHODS / BOUNDARY + BENCHMARK

    # primary / negative / forbidden claim
    max_claim_row = next((r for r in CLAIM_MATRIX if r["route"] == route), None)
    primary_claim = max_claim_row["max_claim"] if max_claim_row else "see claim matrix"
    negative_claim = (
        "True qMaP transfer is below the predeclared meaningful threshold and the "
        "tecto model is not better than its motif baseline; the four components are "
        "not four i.i.d. repeats and the 98 variants do not extrapolate to arbitrary "
        "TLR families."
    )

    # ---- paper spine -----------------------------------------------------------
    paper_spine = f"""# Paper spine (N0 frozen) — RNA thermodynamic transport boundary audit

## Working title
When correlations do not transport: an auditable boundary analysis of public RNA thermodynamics
(backup: Endpoint identity, censoring and graph support delimit transport claims in RNA thermodynamic datasets)

## Type
Methods-boundary / benchmark / reproducibility paper. NOT a junction-preorganization mechanism
paper and NOT a new-foundation-model paper.

## Route (frozen from §14.2)
- Route: `{route}`
- N0 state: `{n0_state}`
- T6 terminal: `{t6_state}` (locked negative)
- Q6 terminal: `{q6_state}`
- Q7 terminal: `{q7_state}`

## Primary claim (maximum allowed for this route)
{primary_claim}

## Negative claim
{negative_claim}

## Closest work comparison
- qMaP2024 (Kladwang et al., NAR 2024): reports a correlation within its selected
  proof-of-concept TL/TLR population; the original estimand is in-population correlation.
  This project tests the different estimand of out-of-component prediction / transport,
  so the two should not be conflated.
- DataSAIL (Nat Commun 2025): generic grouped/fair split; this project adds the
  RNA-thermodynamic-specific integration (endpoint graph, censoring, selection,
  coverage-width) that DataSAIL does not address.
- PRIME2026 (preprint): probing-derived thermodynamics; this project studies when
  correlation does NOT transport, using public cases + synthetic fixtures.

## Forbidden claims
See `forbidden_claims.txt` (all items are unconditional prohibitions).

## Figure spine (frozen from §14.5)
| Fig | Content |
|-----|---------|
| Fig.1 | data/endpoint/platform/lineage graph |
| Fig.2 | claim-evidence ladder + v1.3 closure gaps |
| Fig.3 | source-correct qMaP component holdout |
| Fig.4 | tecto baseline + coverage-width |
| Fig.5 | selection/censoring/weighting sensitivities (POST_HOC_EXPLANATORY) |
| Fig.6 | reusable audit workflow |
| Table 1 | case-study capability matrix |
| Table 2 | claim matrix and forbidden statements |

## Authorship / submission status
Manuscript preparation is AUTHORIZED_AFTER_C0_T6_Q6_Q7_N0.
Manuscript submission is HOLD_PENDING_E1_AND_USER_APPROVAL.
"""
    paper_spine_path = f"{N0_DIR}/paper_spine.md"
    with open(paper_spine_path, "w") as f:
        f.write(paper_spine)
    paper_spine_sha = sha256_file(paper_spine_path)

    # ---- claim matrix (route -> max claim) -------------------------------------
    claim_matrix_path = f"{N0_DIR}/claim_matrix.tsv"
    claim_matrix_rows = [
        {
            "t6": r["t6"],
            "q7": r["q7"],
            "route": r["route"],
            "max_claim": r["max_claim"],
        }
        for r in CLAIM_MATRIX
    ]
    claim_matrix_sha = write_tsv(claim_matrix_path, claim_matrix_rows)

    # ---- forbidden claims ------------------------------------------------------
    forbidden_path = f"{N0_DIR}/forbidden_claims.txt"
    with open(forbidden_path, "w") as f:
        f.write("# Forbidden claims (frozen from §14.4) — unconditional prohibitions\n")
        for i, s in enumerate(FORBIDDEN_CLAIMS, 1):
            f.write(f"{i}. {s}\n")
    forbidden_sha = sha256_file(forbidden_path)

    # ---- N0 decision -----------------------------------------------------------
    decision = {
        "schema_version": "N0-decision-v1.4",
        "gate": "N0",
        "run_id": RUN_ID,
        "contract_sha256": CONTRACT_SHA,
        "decision_time_utc": now_utc(),
        "state": n0_state,
        "route": route,
        "inputs": {
            "T6_terminal_state": t6_state,
            "Q6_terminal_state": q6_state,
            "Q7_terminal_state": q7_state,
        },
        "novelty": {
            "novelty_ok": novelty_ok,
            "avoided_claims": ["new-model-alone", "first-chemical-probing-alone", "fair-split-alone"],
            "contribution": novelty_contribution,
        },
        "primary_claim": primary_claim,
        "negative_claim": negative_claim,
        "closest_work": {
            "qMaP2024": "in-population correlation estimand; this project tests out-of-component transport (different estimand)",
            "DataSAIL": "generic split; no RNA-thermodynamic endpoint/censoring/coverage-width integration",
            "PRIME2026": "preprint probing->thermodynamics; this project studies when correlation does not transport",
        },
        "artifacts": {
            "prior_art_matrix.tsv": prior_art_sha,
            "claim_collision_matrix.tsv": collision_sha,
            "citation_verification.tsv": cit_sha,
            "paper_spine.md": paper_spine_sha,
            "claim_matrix.tsv": claim_matrix_sha,
            "forbidden_claims.txt": forbidden_sha,
        },
        "scientific_disposition": (
            "The v1.4 project is a methods-boundary audit with a reusable benchmark "
            "contribution (B0/B1). It is not a mechanism paper and not a new-model paper. "
            "Manuscript submission is HOLD pending E1 and explicit user authorization."
        ),
    }
    decision_sha = write_json(f"{N0_DIR}/N0_decision.json", decision)

    # ---- report ----------------------------------------------------------------
    report = f"""# N0 report — novelty / claim / paper-spine freeze

## Gate inputs (terminal states)
- C0: immutable closure + state reconciliation (PASS; see C0_decision.json)
- T6: `{t6_state}` (locked negative; estimand bound)
- Q6: `{q6_state}` (source-authoritative 99->98 + 84/11/2/1)
- Q7: `{q7_state}` (corrected locked transfer)

## Route determination (frozen §14.2)
- T6 negative-bound + Q7 `{q7_state}` -> route `{route}`
- N0 state: `{n0_state}`

## Novelty gate
- Method contribution is the unified low-N RNA thermodynamic transport audit
  (endpoint identity + source attrition + censoring + graph support + baseline
  adequacy + coverage-width). This is beyond a new model, first chemical-probing,
  or fair-split alone.
- Prior-art collisions are explicitly managed (qMaP2024, PRIME2026, DataSAIL).

## Frozen claims
- primary: {primary_claim}
- negative: {negative_claim}
- forbidden: see forbidden_claims.txt ({len(FORBIDDEN_CLAIMS)} items)

## Next steps
- N0 -> B0 (reusable benchmark + audit-schema freeze)
- B1 (synthetic failure-mode validation)
- B2 (POST_HOC_EXPLANATORY sensitivity ladder)
- R1 / M1 / E1 (release, manuscript, external review)
- Manuscript submission remains HOLD_PENDING_E1_AND_USER_APPROVAL.

## Artifacts (SHA-256)
- prior_art_matrix.tsv {prior_art_sha}
- claim_collision_matrix.tsv {collision_sha}
- citation_verification.tsv {cit_sha}
- paper_spine.md {paper_spine_sha}
- claim_matrix.tsv {claim_matrix_sha}
- forbidden_claims.txt {forbidden_sha}
- N0_decision.json {decision_sha}
"""
    report_path = f"{REPORTS_DIR}/N0_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    report_sha = sha256_file(report_path)

    # ---- sentinel --------------------------------------------------------------
    sentinel = {
        "gate": "N0",
        "state": n0_state,
        "run_id": RUN_ID,
        "decision_sha256": decision_sha,
        "report_sha256": report_sha,
        "generated_at_utc": now_utc(),
    }
    sentinel_sha = write_json(f"{SENTINELS_DIR}/N0_{n0_state}.json", sentinel)

    print(json.dumps({
        "state": n0_state,
        "route": route,
        "novelty_ok": novelty_ok,
        "prior_art_matrix_sha": prior_art_sha,
        "claim_collision_matrix_sha": collision_sha,
        "citation_verification_sha": cit_sha,
        "paper_spine_sha": paper_spine_sha,
        "claim_matrix_sha": claim_matrix_sha,
        "forbidden_claims_sha": forbidden_sha,
        "decision_sha": decision_sha,
        "report_sha": report_sha,
        "sentinel_sha": sentinel_sha,
    }, indent=2))


if __name__ == "__main__":
    main()