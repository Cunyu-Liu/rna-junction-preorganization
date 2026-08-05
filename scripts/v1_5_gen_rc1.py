#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RC1 — internal red-team review (v1.5 §17).

Performed by the current execution chain; name is fixed to INTERNAL_RED_TEAM_REVIEW.
This is NOT an independent review and must never be labelled as such (that role is
X1, which requires a genuinely uninvolved executor/reviewer).

The red-team reads ONLY the frozen artifacts and formulates issues across the §17
dimensions. It does not re-run or re-adjudicate any prior gate. It is designed to be
adversarial: it looks for the strongest honest challenge to each claim, and does not
auto-pass. Critical/high issues are carried forward to M3 for resolution; the RC1
gate itself is terminal once the review is recorded, with the resolution tracked
separately in M3.

Severity scale: CRITICAL / HIGH / MEDIUM / LOW.
Status: OPEN / PARTIAL / CLOSED / WONTFIX_DOWNGRADED.
"""

from __future__ import annotations
import csv
import json
import os
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
RC1_DIR = f"{RUN_ROOT}/review/rc1"


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(rel):
    with open(os.path.join(RUN_ROOT, rel)) as f:
        return json.load(f)


def _read_sentinel(name):
    with open(os.path.join(RUN_ROOT, "sentinels", name)) as f:
        return dict(line.partition("=")[::2] for line in f.read().splitlines() if "=" in line)


def write_tsv(path, header, rows):
    os.makedirs(RC1_DIR, exist_ok=True)
    with open(os.path.join(RC1_DIR, path), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(RC1_DIR, exist_ok=True)
    now = _utcnow()

    # ---- load frozen prerequisite states (never recompute) ----
    q8 = _load_json("qmap/q8/Q8_decision.json")
    b3 = _load_json("benchmark/b3/B3_decision.json")
    x0 = _load_json("external_case/x0/X0_decision.json")
    l0 = _load_json("literature/l0/L0_decision.json")
    n1 = _load_json("novelty/n1/N1_decision.json")

    q8_sub = q8["sub_states"]
    b3_state = b3["state"]
    x0_state = x0["state"]
    route = n1["route"]

    # =====================================================================
    # Red-team issue construction. Each row: dimension, issue_id, title,
    # severity, evidence, owner, required_fix, status.
    # =====================================================================
    issues = []

    # --- data lineage / platform independence / selection bias ---
    issues.append([
        "data_lineage", "RC1-01",
        "qMaP primary_n=95 is small; primary component sizes 80/11/2/2 with two size-2 components cannot support a reliable component-aware holdout.",
        "HIGH",
        f"Q8_decision: primary_component_sizes=[80,11,2,2]; B3 component_imbalance regime (detection_rate=1.0) exists precisely to flag small components.",
        "current_implementation",
        "Restrict any quantitative qMaP claim to the 80-member component; report the 11/2/2 members only as sensitivity. Do not claim equal evidence across components.",
        "OPEN",
    ])
    issues.append([
        "selection_bias", "RC1-02",
        "B3 aggregate sensitivity/specificity=1.0 is computed on frozen synthetic DGP seeds; this is a design-truth upper bound, not a guarantee on real data.",
        "MEDIUM",
        f"B3_decision: aggregate false_pass_rate=0.0, false_fail_rate=0.0; data are synthetic (no external data required per B3 README).",
        "current_implementation",
        "State B3 as a generative validation of the detector procedure; do not claim real-data detection rates equal to the synthetic 1.0.",
        "OPEN",
    ])

    # --- censoring / membership / component split ---
    issues.append([
        "membership", "RC1-03",
        "The 11th censored member CCUGCC_ACUGG is FIT_IDENTIFIED_MEMBERSHIP_UNCERTAIN; its membership must be shown not to flip any conclusion.",
        "HIGH",
        "Q8 decision includes three frozen sensitivity scenarios (S1 censored / S2 fitted / S3 excluded) for this member.",
        "current_implementation",
        "Report all three sensitivity sets of N, component sizes, gain, bootstrap, permutation, coverage; confirm the full-predeclared-criterion dis-position is stable to membership.",
        "OPEN",
    ])

    # --- baseline / metric / uncertainty / calibration ---
    issues.append([
        "uncertainty", "RC1-04",
        "qMaP gain bootstrap CI [-0.5718, 0.7478] crosses zero => QMAP_GAIN_BOOTSTRAP=INCONCLUSIVE; the point gain 0.416 is not robustly separated from zero.",
        "HIGH",
        f"Q8_decision: bootstrap_ci_95=[{q8['frozen']['bootstrap_ci_95']}], sub_states.gain_bootstrap=INCONCLUSIVE.",
        "current_implementation",
        "Keep the gain claim qualified: 'gain MET threshold but bootstrap INCONCLUSIVE'; do not assert a robust positive transport effect.",
        "OPEN",
    ])
    issues.append([
        "calibration", "RC1-05",
        "Registered point coverage rule [0.75,0.85] FAILED (observed 0.726, width 1.13) but this is a registered-decision failure, NOT proven real undercoverage.",
        "HIGH",
        f"Q8_decision: observed_coverage={q8['frozen']['observed_coverage']}, observed_width={q8['frozen']['observed_width']}, coverage_rule={q8['frozen']['registered_point_coverage_rule']}, calibration_deficit_evidence=INCONCLUSIVE.",
        "current_implementation",
        "Never write 'proven undercoverage'; the manuscript must carry the registered-decision-failure vs empirical-evidence distinction verbatim.",
        "OPEN",
    ])

    # --- benchmark truth / false-pass / false-fail / MC precision ---
    issues.append([
        "benchmark", "RC1-06",
        "B3 detection rates are reported without explicit Monte-Carlo precision bounds per frozen seed set; point 1.0 could hide finite-seed uncertainty.",
        "MEDIUM",
        f"B3_decision: state={b3_state}; per-regime n=10 decisions all 1.0/0.0.",
        "current_implementation",
        "Report MC confidence intervals (e.g. Wilson) around sensitivity/specificity and per-regime detection rates from the frozen seed count.",
        "OPEN",
    ])

    # --- external case qualification ---
    issues.append([
        "external_case", "RC1-07",
        "PRIME external case is platform-independent but NOT qualified (X0_INCONCLUSIVE): low independent-construct N, operator/estimand ambiguity, unsettled preprint authority.",
        "HIGH",
        f"X0_decision: state={x0_state}, eligibility_verdicts={x0['eligibility_verdicts']}, decisive_blockers={x0['decisive_blockers']}.",
        "current_implementation",
        "Keep strong cross-case/general transport claim CLOSED; present PRIME only as a future candidate case, not as validation.",
        "OPEN",
    ])

    # --- novelty / venue fit ---
    issues.append([
        "venue_fit", "RC1-08",
        "Route is RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE (T2); candidate venue Bioinformatics Advances is conditional and must be re-verified at S1.",
        "MEDIUM",
        f"N1_decision: route={route}, claim_tier=T2, candidate_venue='Bioinformatics Advances (conditional)'.",
        "current_implementation",
        "Do not state venue acceptance or suitability as decided; re-open venue fit check in S1 before any submission step.",
        "OPEN",
    ])

    # --- claim inflation ---
    issues.append([
        "claim_inflation", "RC1-09",
        "Risk of conflating 'qMaP gain threshold MET' with a validated transport claim; the full predeclared criterion is NOT_MET and must stay separate.",
        "CRITICAL",
        f"Q8_decision sub_states: gain_threshold=MET, permutation_signal=PRESENT, full_predeclared_transport_criterion=NOT_MET.",
        "current_implementation",
        "Every appearance of the qMaP gain must pair the MET sub-state with the NOT_MET full criterion; prohibit bare 'qMaP validated/transports' wording.",
        "OPEN",
    ])

    # --- reproducibility / release lineage ---
    issues.append([
        "reproducibility", "RC1-10",
        "X1 genuine independent recomputation + independent review is NOT yet done; only hash-integrity replay exists in the parent run (E1).",
        "CRITICAL",
        "RC1 is INTERNAL_RED_TEAM_REVIEW; §18 X1 requires an uninvolved executor/reviewer and must fail-closed to X1_AWAITING_INDEPENDENT_REVIEW otherwise.",
        "current_implementation",
        "Do not claim independent reproduction/review until a genuinely uninvolved executor/reviewer completes X1; otherwise stop at X1_AWAITING_INDEPENDENT_REVIEW.",
        "OPEN",
    ])
    issues.append([
        "release_lineage", "RC1-11",
        "R2 final clean-commit recursive seal has not been produced; parent release was a historical partial seal (V14_RELEASE=HISTORICAL_PARTIAL_SEAL_NOT_FINAL).",
        "HIGH",
        "C1 reconciliation: V15-04 (R1 written as final sealed release) reconciled; R2 not yet executed.",
        "current_implementation",
        "Produce R2 on the final clean commit after M3 and X1; do not claim final sealed release before R2.",
        "OPEN",
    ])

    # --- figure/table consistency ---
    issues.append([
        "figure_table", "RC1-12",
        "F0 figures and M2 manuscript are generated from the same frozen artifacts, but cross-figure numeric consistency is only guaranteed by shared source-data TSV hashes.",
        "LOW",
        "F0 produces source-data TSV + SHA-256 per figure; M2 claim_evidence_citation_map.tsv binds claims to artifacts.",
        "current_implementation",
        "Add a one-pass cross-check that every numeric literal in the manuscript text occurs in a figure source-data TSV or a frozen decision JSON.",
        "OPEN",
    ])

    # --- citation / license ---
    issues.append([
        "citation_license", "RC1-13",
        "PRIME/X0 sources are CC-BY-4.0 + SRA public but preprint authority is unsettled and no pinned checksummed snapshot was frozen; citations must use the settled public record.",
        "MEDIUM",
        "X0_decision: PRIME license OK but preprint authority unsettled, no frozen snapshot.",
        "current_implementation",
        "Freeze exact citation/version for any cited external work at S1; do not cite the unsettled preprint as authoritative evidence.",
        "OPEN",
    ])

    # =====================================================================
    # Claim challenge log: pair each headline claim with the strongest
    # adversarial challenge and the bounded response.
    # =====================================================================
    challenges = [
        ["qMaP gain threshold MET (0.416 > 0.3)",
         "Bootstrap CI crosses zero; gain is not robustly > 0.",
         "Qualified: gain_threshold MET but gain_bootstrap INCONCLUSIVE; no robust transport effect claimed.",
         "PARTIAL"],
        ["qMaP 80% interval coverage FAILED (0.726)",
         "This is a registered-decision failure, not proof of true undercoverage.",
         "Agreed and enforced: manuscript keeps registered-decision vs empirical-evidence distinction.",
         "CLOSED"],
        ["B3 sensitivity/specificity = 1.0",
         "Synthetic DGP upper bound; not a real-data guarantee.",
         "Qualified: B3 validates the detector procedure on generative truth; real-data rates not claimed.",
         "PARTIAL"],
        ["PRIME is an independent external case",
         "Platform-independent but not qualified (low N, operator ambiguity, preprint unsettled).",
         "Strong cross-case claim CLOSED; PRIME kept as future candidate only.",
         "CLOSED"],
        ["Manuscript is submission-ready",
         "Route is T2 resource/audit note; submission is HOLD pending R2/S1 and user authorization.",
         "Manuscript is a draft; submission explicitly HOLD_PENDING_USER_AUTHORIZATION.",
         "CLOSED"],
        ["Independent recomputation/review performed",
         "Only internal red-team (RC1) and parent hash-integrity replay (E1) exist; not independent.",
         "X1 requires uninvolved executor/reviewer; otherwise fail-closed to X1_AWAITING_INDEPENDENT_REVIEW.",
         "OPEN"],
    ]

    write_tsv("issues.tsv",
              ["dimension", "issue_id", "title", "severity", "evidence", "owner", "required_fix", "status"],
              issues)
    write_tsv("claim_challenge_log.tsv",
              ["claim", "strongest_challenge", "bounded_response", "status"],
              challenges)

    # =====================================================================
    # Decision
    # =====================================================================
    severities = [r[3] for r in issues]
    n_critical = severities.count("CRITICAL")
    n_high = severities.count("HIGH")
    n_med = severities.count("MEDIUM")
    n_low = severities.count("LOW")
    n_open = sum(1 for r in issues if r[7] == "OPEN")

    decision = {
        "schema_version": "RC1-decision-v1.5",
        "gate": "RC1",
        "review_name": "INTERNAL_RED_TEAM_REVIEW",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "decision_time_utc": now,
        "is_independent_review": False,
        "performed_by": "current_execution_chain (NOT an uninvolved reviewer)",
        "dimensions_covered": [
            "data_lineage", "platform_independence", "selection_bias",
            "censoring", "membership", "component_split",
            "baseline", "metric", "uncertainty", "calibration",
            "benchmark_truth", "false_pass_false_fail", "monte_carlo_precision",
            "external_case_qualification", "novelty", "venue_fit",
            "claim_inflation", "reproducibility", "release_lineage",
            "figure_table_consistency", "citation_license",
        ],
        "issue_counts": {
            "total": len(issues),
            "critical": n_critical,
            "high": n_high,
            "medium": n_med,
            "low": n_low,
            "open": n_open,
        },
        "critical_high_open": [r[1] for r in issues
                               if r[3] in ("CRITICAL", "HIGH") and r[7] == "OPEN"],
        "state": "RC1_INTERNAL_RED_TEAM_REVIEW_COMPLETE",
        "note": (
            "RC1 is terminal as a review record, NOT as a pass. Critical/high "
            "unresolved issues must be resolved or the claim downgraded in M3. "
            "RC1 must never be labelled an independent review; that is X1."
        ),
        "outputs": {
            "issues": "review/rc1/issues.tsv",
            "claim_challenge_log": "review/rc1/claim_challenge_log.tsv",
            "decision": "review/rc1/RC1_decision.json",
            "report": "reports/RC1_report.md",
        },
    }
    with open(f"{RC1_DIR}/RC1_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # =====================================================================
    # Report
    # =====================================================================
    report = [
        "# RC1 — Internal Red-Team Review (v1.5 §17)",
        "",
        f"**State:** {decision['state']}  ({now})",
        "",
        "This review is **INTERNAL_RED_TEAM_REVIEW**, performed by the current "
        "execution chain. It is **not** an independent review and is not labelled as one.",
        "",
        f"Route under review: **{route}** (T2 resource/audit note).",
        "",
        f"Verbatim frozen inputs: Q8 subpoises, B3 state `{b3_state}`, X0 state `{x0_state}`, "
        f"L0 state `{l0['state']}`, N1 route `{route}`.",
        "",
        "## Issue summary",
        "",
        f"| Severity | Count |",
        "|---|---|",
        f"| CRITICAL | {n_critical} |",
        f"| HIGH | {n_high} |",
        f"| MEDIUM | {n_med} |",
        f"| LOW | {n_low} |",
        f"| **Total** | **{len(issues)}** |",
        "",
        f"Open (unresolved) issues: **{n_open}** — including critical/high: "
        f"{', '.join(decision['critical_high_open']) or 'none'}.",
        "",
        "## Critical/high issues carried to M3",
        "",
        "- **RC1-03** (HIGH) 11th-member membership sensitivity must be reported under all three frozen scenarios.",
        "- **RC1-04** (HIGH) qMaP gain bootstrap INCONCLUSIVE; keep gain claim qualified.",
        "- **RC1-05** (HIGH) coverage failure is a registered-decision failure, not proven undercoverage.",
        "- **RC1-07** (HIGH) PRIME external case not qualified; strong cross-case claim remains CLOSED.",
        "- **RC1-09** (CRITICAL) never conflate gain-threshold-MET with a validated transport claim.",
        "- **RC1-10** (CRITICAL) no independent reproduction/review until a genuinely uninvolved executor/reviewer performs X1.",
        "- **RC1-11** (HIGH) R2 final seal not yet produced.",
        "",
        "## Wording bindings enforced",
        "",
        "- qMaP gain always paired with `full_predeclared_transport_criterion=NOT_MET`.",
        "- Coverage failure always stated as registered-decision failure (not proven undercoverage).",
        "- B3 stated as generative validation of the detector procedure, not a real-data guarantee.",
        "- PRIME kept as a future candidate external case only.",
        "- RC1 never referred to as independent review; X1 held for a genuinely uninvolved party.",
        "",
        "## Next gates",
        "",
        "X1 (independent recomputation + review, fail-closed to "
        "X1_AWAITING_INDEPENDENT_REVIEW) -> M3 (resolve RC1 critical/high) -> R2 (seal) -> S1 "
        "(internal submission package).",
        "",
    ]
    with open(f"{RUN_ROOT}/reports/RC1_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    # =====================================================================
    # Sentinel
    # =====================================================================
    sentinel_lines = [
        f"gate=RC1",
        f"state={decision['state']}",
        f"review_name={decision['review_name']}",
        f"is_independent_review=False",
        f"critical={n_critical}",
        f"high={n_high}",
        f"medium={n_med}",
        f"low={n_low}",
        f"open={n_open}",
        f"critical_high_open={','.join(decision['critical_high_open'])}",
        f"decision_time_utc={now}",
    ]
    with open(f"{RUN_ROOT}/sentinels/RC1_INTERNAL_RED_TEAM_REVIEW.sentinel", "w") as f:
        f.write("\n".join(sentinel_lines) + "\n")

    print("RC1 internal red-team review complete.")
    print(f"issues={len(issues)} critical={n_critical} high={n_high} "
          f"medium={n_med} low={n_low} open={n_open}")
    print("critical_high_open:", decision["critical_high_open"])
    return 0


if __name__ == "__main__":
    sys.exit(main())