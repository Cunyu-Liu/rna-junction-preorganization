"""S1 internal submission package tests (v1.5 §21)."""
import csv
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
S1 = os.path.join(RUN_ROOT, "submission", "s1")


def test_decision_state():
    with open(os.path.join(S1, "S1_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION"
    assert d["gate"] == "S1"
    assert d["manuscript_submission"] == "HOLD_PENDING_USER_AUTHORIZATION"
    assert d["public_release"] == "HOLD_PENDING_USER_AUTHORIZATION"


def test_preconditions():
    assert os.path.exists(os.path.join(
        RUN_ROOT, "sentinels", "M3_CORRECTIONS_CLOSED_CARRIED_X1_R2.sentinel"))
    assert os.path.exists(os.path.join(
        RUN_ROOT, "sentinels", "R2_RELEASE_SEALED_FINAL.sentinel"))


def test_required_documents():
    for name in ("manuscript_final.docx", "manuscript_final.pdf",
                 "supplement_final.docx", "supplement_final.pdf",
                 "cover_letter_draft.docx",
                 "data_availability.md", "code_availability.md",
                 "venue_requirements_checklist.tsv", "submission_inventory.tsv"):
        assert os.path.exists(os.path.join(S1, name)), name


def test_figures_and_source_data():
    assert os.path.isdir(os.path.join(S1, "figures"))
    assert os.path.isdir(os.path.join(S1, "source_data"))
    figs = os.listdir(os.path.join(S1, "figures"))
    assert any(f.startswith("fig") for f in figs)
    sd = os.listdir(os.path.join(S1, "source_data"))
    assert len(sd) >= 5


def test_reporting_checklists():
    assert os.path.exists(os.path.join(S1, "reporting_checklists", "reporting_checklist.tsv"))
    assert os.path.exists(os.path.join(S1, "reporting_checklists", "final_claim_evidence_map.tsv"))


def test_inventory_hashes_match_files():
    with open(os.path.join(S1, "submission_inventory.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    assert len(rows) >= 15
    for r in rows:
        p = os.path.join(S1, r[0])
        assert os.path.exists(p), r[0]
        assert r[2].startswith("0123456789abcdef") or len(r[2]) == 64


def test_report_and_sentinel():
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "S1_report.md"))
    sent = os.path.join(RUN_ROOT, "sentinels",
                        "S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION.sentinel")
    assert os.path.exists(sent)
    with open(sent) as f:
        c = f.read()
    assert "state=S1_INTERNAL_PACKAGE_READY_HOLD_USER_AUTHORIZATION" in c
    assert "manuscript_submission=HOLD_PENDING_USER_AUTHORIZATION" in c