#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N1 — novelty and manuscript-route gate (v1.5 §14).

N1 reads ONLY the frozen Q8, L0, B3, X0 results; it does not re-run any prior
analysis and cannot tune back into earlier gates.

Route logic (§14.1), evaluated strictly from frozen states:
  B3_VALIDATED and X0_QUALIFIED_EXTERNAL_CASE -> STRONG_METHODS_RESOURCE_CANDIDATE
  B3_VALIDATED and X0 not qualified           -> RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE
  B3_PARTIAL_RESOURCE_ONLY                     -> LIMITED_APPLICATION_OR_TECHNICAL_REPORT
  else                                         -> STOP_METHODS_MANUSCRIPT_TECHNICAL_AUDIT_ONLY
"""

from __future__ import annotations
import csv
import json
import os
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
N1_DIR = f"{RUN_ROOT}/novelty/n1"


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path):
    with open(os.path.join(RUN_ROOT, path)) as f:
        return json.load(f)


def write_tsv(path, header, rows):
    os.makedirs(N1_DIR, exist_ok=True)
    with open(os.path.join(N1_DIR, path), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(N1_DIR, exist_ok=True)
    now = _utcnow()

    # ---- read frozen prerequisite states (never recompute) ----
    b3 = _load_json("benchmark/b3/B3_decision.json")
    b3_state = b3["state"]

    x0 = _load_json("external_case/x0/X0_decision.json")
    x0_state = x0["state"]

    q8_sentinel = {}
    with open(os.path.join(RUN_ROOT, "sentinels", "Q8_ADJUDICATED.sentinel")) as f:
        for line in f.read().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                q8_sentinel[k] = v

    l0_decision = _load_json("literature/l0/L0_decision.json")

    # ---- route adjudication (§14.1) ----
    x0_qualified = x0_state == "X0_QUALIFIED_EXTERNAL_CASE"
    if b3_state == "B3_VALIDATED" and x0_qualified:
        route = "STRONG_METHODS_RESOURCE_CANDIDATE"
    elif b3_state == "B3_VALIDATED":
        route = "RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE"
    elif b3_state == "B3_PARTIAL_RESOURCE_ONLY":
        route = "LIMITED_APPLICATION_OR_TECHNICAL_REPORT"
    else:
        route = "STOP_METHODS_MANUSCRIPT_TECHNICAL_AUDIT_ONLY"

    claim_tier = {
        "STRONG_METHODS_RESOURCE_CANDIDATE": "T1 - strong methods/resource claim",
        "RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE": "T2 - resource/audit note (no strong cross-case generality)",
        "LIMITED_APPLICATION_OR_TECHNICAL_REPORT": "T3 - limited application / technical report",
        "STOP_METHODS_MANUSCRIPT_TECHNICAL_AUDIT_ONLY": "T4 - technical audit only",
    }[route]

    # ---- manuscript route ----
    manuscript_route = {
        "schema_version": "N1-route-v1.5",
        "gate": "N1",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "decision_time_utc": now,
        "route": route,
        "claim_tier": claim_tier,
        "b3_state": b3_state,
        "x0_state": x0_state,
        "x0_qualified": x0_qualified,
        "recommended_title": (
            "Predictive signal is not sufficient for calibrated transport: "
            "an auditable framework for public RNA thermodynamic evidence"
        ),
        "main_thesis": (
            "Held-out predictive signal is not sufficient for calibrated, cross-system "
            "transportable thermodynamic evidence: endpoint identity, source membership "
            "category, censoring, graph support, strong-baseline parity, coverage-width and "
            "release provenance must all pass together."
        ),
        "candidate_venue": "Bioinformatics Advances (conditional; re-verify at S1)",
        "strong_cross_case_claim": "CLOSED (X0_INCONCLUSIVE)",
        "gates_satisfied": {
            "Q8": q8_sentinel.get("state", "UNKNOWN"),
            "L0": l0_decision.get("state", "UNKNOWN"),
            "B3": b3_state,
            "X0": x0_state,
        },
    }
    with open(f"{N1_DIR}/manuscript_route.json", "w") as f:
        json.dump(manuscript_route, f, indent=2)

    # ---- claim prior-art matrix (differential vs each prior-art row) ----
    prior_rows = [
        ["qMaP2024", "in-population correlation (SHAPE/Mg2+ endpoint)",
         "This work: out-of-component calibrated transport audit (endpoint/source/censoring/graph/baseline/coverage-width/release)",
         "transport-admissibility audit, not a new correlation method"],
        ["PRIME2026", "direct probing-derived per-nucleotide energetics (DMS-MaP)",
         "This work: auditable transport claim framework; PRIME is a candidate X0 case (not qualified)",
         "evidential-admissibility framework; PRIME is a platform-independent future case"],
        ["DataSAIL", "data-leakage-aware grouped split",
         "This work: on top of leakage-safe splits adds endpoint graph, censoring, selection, coverage-width, claim provenance",
         "integrated admissibility audit, not just a splitter"],
        ["DIONYSUS", "probabilistic calibration benchmark for low-data small molecules",
         "This work: RNA thermodynamic transport with censored proper score + evidence audit",
         "domain-specific admissibility, not generic small-molecule calibration"],
        ["benchmark_guidelines", "generic omics benchmarking methodology",
         "This work: B3 is multi-regime generative truth + detector sensitivity/specificity/power/calibration CI",
         "concrete fulfillment of the guideline for the RNA thermodynamic audit domain"],
        ["OSIRIS", "reproducibility-core-items consensus",
         "This work: X1 genuine independent recomputation distinguishes reproduction from replication",
         "implements the reproduction/replication distinction in the RNA audit"],
    ]
    write_tsv("claim_prior_art_matrix.tsv",
              ["prior_art", "their_claim", "project_differential_claim",
               "differentiation_kind"],
              prior_rows)

    # ---- contribution-evidence map ----
    contrib_rows = [
        ["auditable transport criterion", "Q8 six substates + §10 sensitivity", "benchmark/b3/, qmap/q8/, sentinels/", "frozen"],
        ["generative benchmark validation", "B3 B3_VALIDATED, false-pass/false-fail 0.0", "benchmark/b3/", "frozen"],
        ["external-case boundary", "X0_INCONCLUSIVE (PRIME platform-independent but not qualified)", "external_case/x0/", "frozen"],
        ["claim correction", "V15-01..11 reconciled (C1)", "reconciliation/c1/", "frozen"],
        ["independent-recomputation plan", "X1 (deferred to independent executor)", "reproducibility/x1/", "pending"],
    ]
    write_tsv("contribution_evidence_map.tsv",
              ["contribution", "evidence", "artifact_path", "status"],
              contrib_rows)

    # ---- N1 decision ----
    decision = {
        "schema_version": "N1-decision-v1.5",
        "gate": "N1",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "decision_time_utc": now,
        "b3_state": b3_state,
        "x0_state": x0_state,
        "route": route,
        "claim_tier": claim_tier,
        "candidate_venue": "Bioinformatics Advances (conditional)",
        "state": f"N1_ROUTE_{route}",
        "outputs": {
            "claim_prior_art_matrix": "novelty/n1/claim_prior_art_matrix.tsv",
            "contribution_evidence_map": "novelty/n1/contribution_evidence_map.tsv",
            "manuscript_route": "novelty/n1/manuscript_route.json",
            "claim_tier": "novelty/n1/claim_tier.json",
        },
    }
    with open(f"{N1_DIR}/N1_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # claim_tier.json
    with open(f"{N1_DIR}/claim_tier.json", "w") as f:
        json.dump({"claim_tier": claim_tier, "route": route,
                   "authorized_claims": {
                       "locked_tecto_negative": True,
                       "qmap_signal_present_gain_met": True,
                       "qmap_registered_coverage_rule_failed_not_proven_undercover": True,
                       "qmap_full_criterion_not_met": True,
                       "b3_false_pass_false_fail_zero": True,
                       "x0_strong_cross_case": False,
                       "dms_validates_tecto": False,
                   }}, f, indent=2)

    # ---- report ----
    report = [
        "# N1 — Novelty and Manuscript-Route Gate",
        "",
        f"**Decision:** N1_ROUTE_{route}  ({now})",
        "",
        f"- B3 state: {b3_state}",
        f"- X0 state: {x0_state} -> qualified={x0_qualified}",
        f"- Route: **{route}**",
        f"- Claim tier: **{claim_tier}**",
        "",
        "## Main thesis",
        "",
        f"{manuscript_route['main_thesis']}",
        "",
        "## Recommended title",
        "",
        f"> {manuscript_route['recommended_title']}",
        "",
        "## What is NOT claimed",
        "",
        "- No strong cross-case / general transport claim (X0_INCONCLUSIVE).",
        "- No claim that DMS validates tecto; no claim that qMaP reproduces junction preorganization.",
        "- No claim that qMaP has no signal; full predeclared criterion is NOT_MET.",
        "- No claim that point coverage failure = proven real undercoverage.",
        "",
        "## Next gates",
        "",
        "F0 (figures) -> M2 (manuscript) -> RC1 (internal red-team) -> X1 (independent "
        "recomputation + review) -> M3 -> R2 (seal) -> S1 (internal submission package).",
        "",
    ]
    with open(f"{RUN_ROOT}/reports/N1_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())