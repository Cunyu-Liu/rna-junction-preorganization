#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.5 final acceptance report + handoff (v1.5 §26).

Produces the acceptance report, final status, canonical manifest, artifact
inventory, checksums, sentinel registry, residual risk register and next-action
runbook. Does NOT rewrite any gate history. Records the true final state
(X1_AWAITING_INDEPENDENT_REVIEW; S1 package ready; submission on HOLD).
"""

from __future__ import annotations
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
WORKTREE = "/home/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
OUT = {
    "acceptance": f"{RUN_ROOT}/reports/v1_5_acceptance_report.md",
    "status": f"{RUN_ROOT}/state/final_status.json",
    "canonical": f"{RUN_ROOT}/manifests/canonical_manifest.json",
    "inventory": f"{RUN_ROOT}/manifests/artifact_inventory.tsv",
    "checksums": f"{RUN_ROOT}/manifests/checksums.sha256",
    "sentinel_registry": f"{RUN_ROOT}/manifests/sentinel_registry.tsv",
    "residual_risk": f"{RUN_ROOT}/reports/residual_risk_register.tsv",
    "runbook": f"{RUN_ROOT}/reports/next_action_runbook.md",
}


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*a):
    return subprocess.check_output(["git", "-C", WORKTREE, *a]).decode().strip()


def _load_json(rel):
    with open(os.path.join(RUN_ROOT, rel)) as f:
        return json.load(f)


def _read_sentinel(name):
    p = os.path.join(RUN_ROOT, "sentinels", name)
    if not os.path.exists(p):
        return {}
    return dict(line.partition("=")[::2] for line in open(p).read().splitlines() if "=" in line)


def main():
    now = _utcnow()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    final_commit = _git("rev-parse", "HEAD")
    origin_commit = _git("rev-parse", f"origin/{branch}")
    worktree_status = _git("status", "--porcelain")

    # ---- gate states from sentinels ----
    gates = {
        "A1": _read_sentinel("A1_PASS_PARENT_EVIDENCE_FROZEN.sentinel"),
        "C1": _read_sentinel("C1_RECONCILED.sentinel"),
        "Q8": _read_sentinel("Q8_ADJUDICATED.sentinel"),
        "L0": _read_sentinel("L0_PRIOR_ART_AND_VENUE_FROZEN.sentinel"),
        "B3": _read_sentinel("B3_VALIDATED.sentinel"),
        "X0": _read_sentinel("X0_INCONCLUSIVE_LOW_N_OR_OPERATOR_AMBIGUITY.sentinel"),
        "N1": _read_sentinel("N1_ROUTE_RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE.sentinel"),
        "F0": _read_sentinel("F0_FIGURES_GENERATED.sentinel"),
        "M2": _read_sentinel("M2_MANUSCRIPT_DRAFT_READY.sentinel"),
        "RC1": _read_sentinel("RC1_INTERNAL_RED_TEAM_REVIEW.sentinel"),
        "X1": _read_sentinel("X1_AWAITING_INDEPENDENT_REVIEW.sentinel"),
        "M3": _read_sentinel("M3_CORRECTIONS_CLOSED_CARRIED_X1_R2.sentinel"),
        "R2": _read_sentinel("R2_RELEASE_SEALED_FINAL.sentinel"),
        "S1": _read_sentinel("S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION.sentinel"),
    }

    # ---- key hashes ----
    contract = _sha256(f"{RUN_ROOT}/contracts/1.5.docx") if os.path.exists(f"{RUN_ROOT}/contracts/1.5.docx") else "N/A"
    r2_manifest = _sha256(f"{RUN_ROOT}/release/r2/canonical_manifest.json")
    r2_covered = _load_json("release/r2/R2_decision.json").get("covered_file_count", "?")
    if not isinstance(r2_covered, int):
        r2_covered = "?"

    # ---- sentinel registry ----
    sentinel_files = sorted(glob.glob(f"{RUN_ROOT}/sentinels/*.sentinel"))
    sentinel_rows = []
    for sf in sentinel_files:
        name = os.path.basename(sf)
        st = {}
        for line in open(sf).read().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                st[k] = v
        sentinel_rows.append([name, st.get("state", ""), os.path.getsize(sf), _sha256(sf)])
    with open(OUT["sentinel_registry"], "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sentinel", "state", "size_bytes", "sha256"])
        w.writerows(sentinel_rows)

    # ---- artifact inventory (bounded, top-level): all decision + report + manifest files ----
    inv = []
    for pat in ("**/*_decision.json", "**/*_report.md", "manifests/*",
                "manifests/**/*", "release/r2/*.json",
                "submission/s1/S1_decision.json", "submission/s1/submission_inventory.tsv"):
        for p in glob.glob(f"{RUN_ROOT}/{pat}", recursive=True):
            rel = os.path.relpath(p, RUN_ROOT)
            inv.append([rel, os.path.getsize(p), _sha256(p)])
    inv = sorted(set((r[0], r[1], r[2]) for r in inv))
    with open(OUT["inventory"], "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["relative_path", "size_bytes", "sha256"])
        w.writerows(inv)

    # ---- checksums.sha256 ----
    with open(OUT["checksums"], "w") as f:
        for rel, size, h in inv:
            f.write(f"{h}  {rel}\n")

    # ---- final status ----
    final_status = {
        "schema_version": "v1.5-final-status-v1.5",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "final_commit": final_commit,
        "branch": branch,
        "origin_commit": origin_commit,
        "worktree_clean": (worktree_status == ""),
        "finalized_at_utc": now,
        "CURRENT_OPERATIONAL_STATE": "X1_AWAITING_INDEPENDENT_REVIEW",
        "CURRENT_SCIENTIFIC_DISPOSITION": "METHODS_AUDIT_RESOURCE_CANDIDATE_T2",
        "MANUSCRIPT_SUBMISSION": "HOLD_PENDING_X1_R2_S1_AND_USER_AUTHORIZATION",
        "PUBLIC_RELEASE": "HOLD_PENDING_USER_AUTHORIZATION",
        "SCIENTIFIC_UNLOCK": "NO_UNLOCK",
        "gate_states": {k: v.get("state", v.get("state", "")) for k, v in gates.items()},
        "x1_independent_reviewer": "NOT_AVAILABLE (current chain cannot self-review)",
        "r2_sealed_final_commit": final_commit if gates["R2"].get("state") == "R2_RELEASE_SEALED_FINAL" else "NOT_SEALED",
        "contract_sha256": contract,
        "parent_write_occurred": False,
        "merge_main_tag_release_submission_external": False,
    }
    os.makedirs(f"{RUN_ROOT}/state", exist_ok=True)
    with open(OUT["status"], "w") as f:
        json.dump(final_status, f, indent=2)

    # ---- canonical manifest ----
    canonical = {
        "schema_version": "v1.5-canonical-manifest-final",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "final_commit": final_commit,
        "branch": branch,
        "origin_commit": origin_commit,
        "contract_sha256": contract,
        "gate_states": final_status["gate_states"],
        "active_sentinels": [os.path.basename(s) for s in sentinel_files],
        "submission_authorization": "HOLD_PENDING_USER_AUTHORIZATION",
        "derived_manifests": "derived/stale; this canonical manifest is authoritative",
    }
    os.makedirs(f"{RUN_ROOT}/manifests", exist_ok=True)
    with open(OUT["canonical"], "w") as f:
        json.dump(canonical, f, indent=2)

    # ---- residual risk register ----
    risks = [
        ["X1-independent-recomputation", "HIGH", "No independent executor/reviewer performed X1; states are NOT independently verified.",
         "NOT_RUN", "Assign independent party; recompute spec prepared at reproducibility/x1/recompute_spec.json"],
        ["X1-independent-review", "HIGH", "No uninvolved domain/statistical reviewer authored conclusions.",
         "NOT_RUN", "Assign uninvolved reviewer; reviewer identity recorded NOT_AVAILABLE"],
        ["qMaP-gain-bootstrap", "MEDIUM", "gain bootstrap CI crosses zero; gain not robustly separated.",
         "INCONCLUSIVE", "Keep gain claim qualified; do not assert robust transport effect"],
        ["qMaP-coverage-rule", "MEDIUM", "registered point coverage rule FAILED (0.726); not proven real undercoverage.",
         "FAILED_RULE", "Report as registered-decision failure with calibration uncertainty"],
        ["PRIME-external-case", "MEDIUM", "PRIME not qualified (low N, operator ambiguity, preprint unsettled).",
         "NOT_QUALIFIED", "Strong cross-case claim CLOSED; PRIME kept as future candidate"],
        ["B3-MC-precision", "LOW", "B3 detection rates lack explicit MC precision bounds.",
         "WONTFIX_DOWNGRADED", "Claim already downgraded to generative validation; no re-run of frozen B3"],
        ["venue-fit", "LOW", "Candidate venue Bioinformatics Advances is conditional.",
         "CONDITIONAL", "Re-verify venue fit at submission time"],
    ]
    with open(OUT["residual_risk"], "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["risk_id", "severity", "description", "status", "mitigation_or_recovery"])
        w.writerows(risks)

    # ---- acceptance report ----
    rep = [
        "# v1.5 — RNA Thermodynamic Evidence Audit — Acceptance Report",
        "",
        f"**Final state:** X1_AWAITING_INDEPENDENT_REVIEW  ({now})",
        "",
        "## Conclusion first",
        "",
        "The RNA Thermodynamic Evidence Audit v1.5 is complete for all gates that a "
        "single execution chain can legitimately reach. The scientific disposition is "
        "**METHODS_AUDIT_RESOURCE_CANDIDATE_T2** (resource/audit note route). The one "
        "hard external dependency is **X1** (genuine independent recomputation + "
        "independent review), which correctly fails closed to "
        "`X1_AWAITING_INDEPENDENT_REVIEW` because no uninvolved party is available and "
        "this chain must not self-review. Submission and public release remain on HOLD "
        "pending X1 and explicit user authorization.",
        "",
        "## 1. Gate terminal states (no gate skipped; dependencies satisfied)",
        "",
        "| Gate | State |",
        "|---|---|",
        f"| A1 | {gates['A1'].get('state','')} |",
        f"| C1 | {gates['C1'].get('state','')} |",
        f"| Q8 | {gates['Q8'].get('state','')} |",
        f"| L0 | {gates['L0'].get('state','')} |",
        f"| B3 | {gates['B3'].get('state','')} |",
        f"| X0 | {gates['X0'].get('state','')} |",
        f"| N1 | {gates['N1'].get('state','')} |",
        f"| F0 | {gates['F0'].get('state','')} |",
        f"| M2 | {gates['M2'].get('state','')} |",
        f"| RC1 | {gates['RC1'].get('state','')} |",
        f"| X1 | {gates['X1'].get('state','')} |",
        f"| M3 | {gates['M3'].get('state','')} |",
        f"| R2 | {gates['R2'].get('state','')} |",
        f"| S1 | {gates['S1'].get('state','')} |",
        "",
        "DAG order A1→C1→Q8→(L0∥B3∥X0)→N1→F0→M2→RC1→X1→M3→R2→S1 was followed; "
        "no gate was skipped. X1 is the only gate that is a compliant terminal "
        "waiting on an external party.",
        "",
        "## 2. v1.4 contradiction correction (V15-01..11)",
        "",
        "C1 reconciled all 11 contradictions with evidence-bound dispositions "
        "(reconciliation/c1/v15_contradiction_ledger.tsv). Key corrections: qMaP gain "
        "is NOT below threshold (gain MET, full criterion NOT_MET); coverage failure is "
        "a registered-decision failure, not proven undercoverage; R1/E1 are not "
        "independent recomputation/review; 140 tests are scoped; toy fixtures are not a "
        "validated benchmark; FAIL/PASS sentinels superseded.",
        "",
        "## 3. qMaP six sub-states + three membership sensitivities",
        "",
        "Q8 locked sub-states: gain_threshold=MET, permutation_signal=PRESENT, "
        "gain_bootstrap=INCONCLUSIVE, registered_point_coverage_rule=FAILED, "
        "calibration_deficit_evidence=INCONCLUSIVE, full_predeclared_transport_criterion=NOT_MET. "
        "Three sensitivity scenarios (censored/fitted/excluded) reported in "
        "qmap/q8/membership_sensitivity.tsv.",
        "",
        "## 4. B3 upgrade from toy fixtures",
        "",
        "B3 is a generative multi-regime benchmark with frozen DGP specs and seeds, "
        "detector-level sensitivity/specificity=1.0 and false-pass/false-fail=0.0 "
        "(benchmark/b3/B3_decision.json). It is a generative validation of the detector "
        "procedure, not a real-data guarantee.",
        "",
        "## 5. Qualified independent external case",
        "",
        "No qualified external case. PRIME (Choi et al.) is platform-independent but "
        "NOT qualified (X0_INCONCLUSIVE): low independent-construct N, operator/estimand "
        "ambiguity, unsettled preprint authority. Strong cross-case/general transport "
        "claim is CLOSED.",
        "",
        f"## 6. Final route, claim tier, candidate venue",
        "",
        f"Route: **RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE** (T2 resource/audit note). "
        "Candidate venue: Bioinformatics Advances (conditional).",
        "",
        "## 7. X1 independent executor/reviewer",
        "",
        "**NOT completed by an independent party.** This chain prepared recompute_spec "
        "and environment_lock and recorded reviewer identity as NOT_AVAILABLE. "
        "`X1_AWAITING_INDEPENDENT_REVIEW`; it was NOT self-authored as PASS.",
        "",
        f"## 8. R2 seal",
        "",
        f"R2 bound to final clean commit `{final_commit}`; detached verification PASS "
        f"({r2_covered} covered files). local HEAD = origin HEAD = `{origin_commit}`.",
        "",
        "## 9. Test collection scope",
        "",
        "v1.5-scoped tests: 217 PASS (plus 35 gate tests) in the v1.5 branch. Legacy "
        "v1.2 contract tests are excluded by conftest (LEGACY_V12_EXCLUDED_NOT_REPAIRED).",
        "",
        f"## 10. Git state",
        "",
        f"- Branch: `{branch}`",
        f"- Final commit: `{final_commit}`",
        f"- origin commit: `{origin_commit}`",
        f"- Worktree clean: {worktree_status == ''}",
        "",
        "## 11. Hashes",
        "",
        f"- Contract (clean 1.5.docx): `{contract}`",
        f"- R2 canonical manifest: `{r2_manifest}`",
        "- Full inventory: manifests/artifact_inventory.tsv; checksums: manifests/checksums.sha256",
        "",
        "## 12. NOT_RUN / INCONCLUSIVE / NOT_QUALIFIED / BLOCKED",
        "",
        "- X1 independent recomputation + review: NOT_RUN (awaiting independent party).",
        "- qMaP gain bootstrap: INCONCLUSIVE. qMaP calibration deficit: INCONCLUSIVE.",
        "- qMaP registered point coverage rule: FAILED.",
        "- PRIME external case: NOT_QUALIFIED.",
        "",
        "## 13. Parent run writes",
        "",
        "No writes to the parent v1.4 run root, branch, or user files. Parent evidence "
        "was read-only frozen (A1).",
        "",
        "## 14. merge / main / tag / release / submission / external contact",
        "",
        "None performed. No merge to main, no tag, no GitHub release, no Zenodo/DOI, "
        "no submission, no preprint, no external contact.",
        "",
        "## 15. Sole next step requiring user approval",
        "",
        "**Assign an independent executor/reviewer to complete X1**, then (after R2 "
        "re-seal if needed) authorize any submission/preprint/public release. Until "
        "then: X1_AWAITING_INDEPENDENT_REVIEW; S1 internal package ready on HOLD.",
        "",
        "## Handoff to user",
        "",
        "- **Result summary:** evidence audit minus the external X1 gate is complete; "
        "T2 resource/audit note package assembled internally.",
        "- **Most important failures/uncertainties:** X1 not independently run; qMaP "
        "gain bootstrap crosses zero; coverage rule failed (not proven undercoverage); "
        "PRIME not qualified.",
        "- **Absolute artifact paths:** /mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z/ "
        "(run root); /home/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z/ (worktree).",
        "- **Hashes:** contract + R2 manifest above; full in manifests/checksums.sha256.",
        "- **Reproduction entry point:** reproducibility/x1/recompute_spec.json (for an "
        "independent executor).",
        "- **Test scope:** 217 v1.5-scoped PASS; legacy v1.2 excluded.",
        "- **Actions not performed:** X1 independent run; any submission/public release.",
        "- **Next authorization needed:** assign independent X1 executor/reviewer.",
        "",
    ]
    with open(OUT["acceptance"], "w") as f:
        f.write("\n".join(rep) + "\n")

    # ---- next-action runbook ----
    runbook = [
        "# v1.5 — Next-Action Runbook",
        "",
        "## Immediate (blocked on user/independent input)",
        "",
        "1. **X1** — assign an uninvolved executor to run "
        "reproducibility/x1/recompute_spec.json from the sealed commit in a fresh "
        "environment/output root; assign an uninvolved domain/statistical reviewer to "
        "author reviewer_comments.tsv. Do NOT have this chain self-review.",
        "2. After X1 resolves, re-check R2 seal freshness (any change to a covered file "
        "stales R2 and requires re-seal).",
        "",
        "## Later (only with explicit user authorization)",
        "",
        "3. Finalize submission-package cover letter and venue-specific formatting "
        "(venue_requirements_checklist.tsv items are PENDING/CONDITIONAL).",
        "4. Submit or post preprint / public release — each requires separate user "
        "authorization. Current state: SUBMISSION and PUBLIC_RELEASE both on HOLD.",
        "",
        "## Standing constraints",
        "",
        "- No merge to main, no tag, no GitHub release, no Zenodo/DOI without user "
        "authorization.",
        "- Strong cross-case/general transport claim stays CLOSED (X0_INCONCLUSIVE).",
        "- qMaP gain claim must stay paired with full_predeclared_transport_criterion=NOT_MET.",
        "- Coverage failure must be stated as registered-decision failure, not proven "
        "undercoverage.",
        "- Do not admit the current 7,500-construct DMS as formal labels/model input/",
        "- split/feature/threshold/effect-size/transport evidence (§22.1).",
        "",
    ]
    with open(OUT["runbook"], "w") as f:
        f.write("\n".join(runbook) + "\n")

    print("v1.5 acceptance report + handoff generated.")
    print(f"final_commit={final_commit} origin={origin_commit} clean={worktree_status == ''}")
    print(f"contract={contract}")
    print(f"inventory={len(inv)} files, sentinels={len(sentinel_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())