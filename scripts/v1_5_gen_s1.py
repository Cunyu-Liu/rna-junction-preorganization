#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 — internal submission package (v1.5 §21).

Assembles the internal submission package ONLY because the route allows it
(T2 resource/audit note), M3 is closed, and R2 is sealed. This is an INTERNAL
package only — it does NOT submit, post a preprint, publicly release, or contact
any journal. Final state stays:
    MANUSCRIPT_SUBMISSION = HOLD_PENDING_USER_AUTHORIZATION
    PUBLIC_RELEASE       = HOLD_PENDING_USER_AUTHORIZATION

X1 remains AWAITING_INDEPENDENT_REVIEW; the package is assembled but submission
is held pending X1/R2/S1 and user authorization.
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import sys

from datetime import datetime, timezone

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
S1_DIR = f"{RUN_ROOT}/submission/s1"
M2_DIR = f"{RUN_ROOT}/manuscript/m2"
F0_DIR = f"{RUN_ROOT}/figures/f0"


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    os.makedirs(S1_DIR, exist_ok=True)
    now = _utcnow()

    # ---- preconditions: route allows, M3 closed, R2 sealed ----
    n1 = json.load(open(f"{RUN_ROOT}/novelty/n1/N1_decision.json"))
    route = n1["route"]
    assert route in ("RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE",
                     "LIMITED_APPLICATION_OR_TECHNICAL_REPORT"), \
        f"route {route} does not allow an internal submission package"
    assert os.path.exists(f"{RUN_ROOT}/sentinels/M3_CORRECTIONS_CLOSED_CARRIED_X1_R2.sentinel")
    assert os.path.exists(f"{RUN_ROOT}/sentinels/R2_RELEASE_SEALED_FINAL.sentinel")

    # ---- assemble manuscript / supplement / cover letter ----
    _copy(f"{M2_DIR}/manuscript.docx", f"{S1_DIR}/manuscript_final.docx")
    _copy(f"{M2_DIR}/manuscript.pdf", f"{S1_DIR}/manuscript_final.pdf")
    _copy(f"{M2_DIR}/supplement.docx", f"{S1_DIR}/supplement_final.docx")
    _copy(f"{M2_DIR}/supplement.pdf", f"{S1_DIR}/supplement_final.pdf")
    _copy(f"{M2_DIR}/cover_letter_draft.docx", f"{S1_DIR}/cover_letter_draft.docx")
    _copy(f"{M2_DIR}/claim_evidence_citation_map.tsv", f"{S1_DIR}/claim_evidence_citation_map.tsv")

    # ---- figures + source data ----
    for img in sorted(os.listdir(f"{F0_DIR}/main")):
        _copy(f"{F0_DIR}/main/{img}", f"{S1_DIR}/figures/{img}")
    for sd in sorted(os.listdir(f"{F0_DIR}/source_data")):
        _copy(f"{F0_DIR}/source_data/{sd}", f"{S1_DIR}/source_data/{sd}")

    # ---- reporting checklists ----
    _copy(f"{M2_DIR}/reporting_checklist.tsv", f"{S1_DIR}/reporting_checklists/reporting_checklist.tsv")
    _copy(f"{RUN_ROOT}/corrections/m3/final_claim_evidence_map.tsv",
          f"{S1_DIR}/reporting_checklists/final_claim_evidence_map.tsv")

    # ---- data availability ----
    data_avail = """# Data Availability (internal draft)

This is an **internal draft**. Do not publicly release until the user authorizes.

- **Primary data**: the project uses public RNA thermodynamic / chem-probing data.
  The qMaP and tectoRNA analyses are built on frozen public datasets whose source,
  license and checksum are recorded in the run provenance registry
  (`provenance/` and `sources/`). DO NOT treat the current 7,500-construct DMS as a
  formal label, model input, split, feature, threshold, effect size or joint
  transport evidence (§22.1).
- **External case**: PRIME (Choi et al.) is CC-BY-4.0 + SRA public but remains an
  unsettled preprint and is NOT a qualified external case (X0_INCONCLUSIVE).
- **Synthetic data**: the B3 benchmark is fully synthetic, generated from frozen
  DGP specs and frozen seeds; no external data required.
- No new wet-lab data were produced in this project.
"""
    with open(f"{S1_DIR}/data_availability.md", "w") as f:
        f.write(data_avail)

    # ---- code availability ----
    code_avail = """# Code Availability (internal draft)

- Branch: `codex/v1_5_manuscript_readiness_20260805T052052Z`
- Final commit: `41899d8cd432a8b5bc55c766c843382b1df84476`
- Remote: `git@github.com:Cunyu-Liu/rna-junction-preorganization.git`

This is an **internal draft**. Do not create a public release, Zenodo/DOI, or tag
until the user authorizes. R2 is sealed; editing the branch after R2 auto-stales R2.
"""
    with open(f"{S1_DIR}/code_availability.md", "w") as f:
        f.write(code_avail)

    # ---- venue requirements checklist ----
    venue_rows = [
        ["route", "RNA_THERMODYNAMIC_AUDIT_RESOURCE_NOTE (T2)", "matches route", "OK"],
        ["target_venue", "Bioinformatics Advances (conditional)", "re-verify at S1", "CONDITIONAL"],
        ["manuscript_word_count_policy", "not yet checked against venue", "verify at submission", "PENDING"],
        ["figure_limits", "8 main figures", "check venue limits", "PENDING"],
        ["supplement_format", "docx/pdf prepared", "check venue rules", "PENDING"],
        ["reporting_checklist", "available", "confirm venue requires", "PENDING"],
        ["data_availability", "draft prepared", "confirm venue policy", "PENDING"],
        ["code_availability", "draft prepared", "confirm venue policy", "PENDING"],
        ["cover_letter", "draft prepared", "confirm venue requirements", "PENDING"],
        ["independent_review", "X1 AWAITING_INDEPENDENT_REVIEW", "must be resolved before submission", "BLOCKER"],
    ]
    with open(f"{S1_DIR}/venue_requirements_checklist.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item", "status_now", "action_at_submission", "flag"])
        w.writerows(venue_rows)

    # ---- submission inventory ----
    inv_rows = []
    for root, _, files in os.walk(S1_DIR):
        for name in sorted(files):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, S1_DIR)
            inv_rows.append([rel, os.path.getsize(p), _sha256(p)])
    with open(f"{S1_DIR}/submission_inventory.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["relative_path", "size_bytes", "sha256"])
        w.writerows(inv_rows)

    # ---- decision ----
    decision = {
        "schema_version": "S1-decision-v1.5",
        "gate": "S1",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "decision_time_utc": now,
        "route": route,
        "m3_closed": True,
        "r2_sealed": True,
        "x1_state": "X1_AWAITING_INDEPENDENT_REVIEW",
        "package_file_count": len(inv_rows),
        "state": "S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION",
        "manuscript_submission": "HOLD_PENDING_USER_AUTHORIZATION",
        "public_release": "HOLD_PENDING_USER_AUTHORIZATION",
        "note": (
            "Internal submission package assembled. Submission and public release "
            "are HOLD pending X1/R2/S1 and explicit user authorization. X1 remains "
            "X1_AWAITING_INDEPENDENT_REVIEW and is a submission blocker."
        ),
        "outputs": {
            "manuscript": "submission/s1/manuscript_final.docx/pdf",
            "supplement": "submission/s1/supplement_final.docx/pdf",
            "figures": "submission/s1/figures/",
            "source_data": "submission/s1/source_data/",
            "reporting_checklists": "submission/s1/reporting_checklists/",
            "data_availability": "submission/s1/data_availability.md",
            "code_availability": "submission/s1/code_availability.md",
            "cover_letter": "submission/s1/cover_letter_draft.docx",
            "venue_requirements": "submission/s1/venue_requirements_checklist.tsv",
            "submission_inventory": "submission/s1/submission_inventory.tsv",
            "decision": "submission/s1/S1_decision.json",
            "report": "reports/S1_report.md",
        },
    }
    with open(f"{S1_DIR}/S1_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # ---- report ----
    report = [
        "# S1 — Internal Submission Package (v1.5 §21)",
        "",
        f"**State:** S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION  ({now})",
        "",
        f"- Route: {route} (T2 resource/audit note).",
        "- M3 closed: True; R2 sealed: True.",
        "- X1: X1_AWAITING_INDEPENDENT_REVIEW (submission blocker).",
        "",
        "## Package contents",
        "",
        "- manuscript_final.docx / .pdf",
        "- supplement_final.docx / .pdf",
        "- figures/ (8 main figures)",
        "- source_data/",
        "- reporting_checklists/ (reporting checklist + final claim-evidence map)",
        "- data_availability.md",
        "- code_availability.md",
        "- cover_letter_draft.docx",
        "- venue_requirements_checklist.tsv",
        "- submission_inventory.tsv (SHA-256 per file)",
        "",
        "## Hold",
        "",
        "**MANUSCRIPT_SUBMISSION = HOLD_PENDING_USER_AUTHORIZATION**",
        "**PUBLIC_RELEASE = HOLD_PENDING_USER_AUTHORIZATION**",
        "",
        "Submission, preprint posting, public release, or contacting any journal "
        "requires separate user authorization. X1 must be resolved before submission.",
        "",
        "## Next",
        "",
        "Final acceptance report + handoff (scientific unit).",
        "",
    ]
    with open(f"{RUN_ROOT}/reports/S1_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    # ---- sentinel ----
    with open(f"{RUN_ROOT}/sentinels/S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION.sentinel", "w") as f:
        f.write(
            "gate=S1\n"
            f"state={decision['state']}\n"
            f"route={route}\n"
            f"m3_closed=True\n"
            f"r2_sealed=True\n"
            f"x1_state=X1_AWAITING_INDEPENDENT_REVIEW\n"
            f"package_file_count={len(inv_rows)}\n"
            f"manuscript_submission=HOLD_PENDING_USER_AUTHORIZATION\n"
            f"public_release=HOLD_PENDING_USER_AUTHORIZATION\n"
            f"decision_time_utc={now}\n"
        )

    print("S1 internal submission package assembled.")
    print(f"route={route} files={len(inv_rows)}")
    print("state=S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION")
    return 0


if __name__ == "__main__":
    sys.exit(main())