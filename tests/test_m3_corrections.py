"""M3 correction closure tests (v1.5 §19)."""
import csv
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
M3 = os.path.join(RUN_ROOT, "corrections", "m3")


def test_decision_state():
    with open(os.path.join(M3, "M3_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "M3_CORRECTIONS_CLOSED_CARRIED_TO_X1_R2"
    assert d["gate"] == "M3"


def test_all_rc1_issues_disposed():
    with open(os.path.join(M3, "correction_ledger.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    assert len(rows) == 13, f"expected 13 RC1 issues, got {len(rows)}"
    for r in rows:
        assert r[3] in ("CLOSED", "WONTFIX_DOWNGRADED", "CARRIED_TO_X1", "CARRIED_TO_R2")


def test_carried_to_x1_present():
    with open(os.path.join(M3, "M3_decision.json")) as f:
        d = json.load(f)
    assert "RC1-10" in d["carried_to_x1"]
    assert d["x1_state_at_m3"] == "X1_AWAITING_INDEPENDENT_REVIEW"


def test_carried_to_r2_present():
    with open(os.path.join(M3, "M3_decision.json")) as f:
        d = json.load(f)
    assert "RC1-11" in d["carried_to_r2"]


def test_counts_consistent():
    with open(os.path.join(M3, "M3_decision.json")) as f:
        d = json.load(f)
    assert (d["rc1_issue_total"] ==
            d["rc1_issue_closed"] + d["rc1_issue_downgraded"] + d["rc1_issue_carried"])


def test_revalidation_and_claim_map():
    assert os.path.exists(os.path.join(M3, "affected_artifact_revalidation.tsv"))
    with open(os.path.join(M3, "final_claim_evidence_map.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    assert len(rows) >= 7
    assert "FROZEN" in ",".join(r[3] for r in rows)


def test_report_and_sentinel():
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "M3_report.md"))
    sent = os.path.join(RUN_ROOT, "sentinels", "M3_CORRECTIONS_CLOSED_CARRIED_X1_R2.sentinel")
    assert os.path.exists(sent)
    with open(sent) as f:
        c = f.read()
    assert "state=M3_CORRECTIONS_CLOSED_CARRIED_TO_X1_R2" in c
    assert "carried_to_x1=RC1-10" in c