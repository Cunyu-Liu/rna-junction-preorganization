"""RC1 internal red-team review tests (v1.5 §17)."""
import csv
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
RC1 = os.path.join(RUN_ROOT, "review", "rc1")


def test_decision_state():
    with open(os.path.join(RC1, "RC1_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "RC1_INTERNAL_RED_TEAM_REVIEW_COMPLETE"
    assert d["gate"] == "RC1"
    assert d["review_name"] == "INTERNAL_RED_TEAM_REVIEW"


def test_not_independent():
    with open(os.path.join(RC1, "RC1_decision.json")) as f:
        d = json.load(f)
    assert d["is_independent_review"] is False
    assert "current_execution_chain" in d["performed_by"].lower()


def test_issues_tsv_schema():
    with open(os.path.join(RC1, "issues.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header = rows[0]
    assert header == ["dimension", "issue_id", "title", "severity",
                      "evidence", "owner", "required_fix", "status"]
    assert len(rows) >= 10
    for r in rows[1:]:
        assert r[3] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert r[7] in ("OPEN", "PARTIAL", "CLOSED", "WONTFIX_DOWNGRADED")
        assert r[1].startswith("RC1-")


def test_critical_and_high_present():
    with open(os.path.join(RC1, "issues.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    sev = [r[3] for r in rows]
    assert "CRITICAL" in sev and "HIGH" in sev


def test_dimensions_covered():
    with open(os.path.join(RC1, "RC1_decision.json")) as f:
        d = json.load(f)
    dims = set(d["dimensions_covered"])
    for required in ("data_lineage", "censoring", "membership", "benchmark_truth",
                     "external_case_qualification", "claim_inflation",
                     "reproducibility", "release_lineage", "calibration"):
        assert required in dims, f"missing dimension: {required}"


def test_claim_challenge_log():
    with open(os.path.join(RC1, "claim_challenge_log.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["claim", "strongest_challenge", "bounded_response", "status"]
    assert len(rows) >= 4


def test_report_and_sentinel():
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "RC1_report.md"))
    sent = os.path.join(RUN_ROOT, "sentinels", "RC1_INTERNAL_RED_TEAM_REVIEW.sentinel")
    assert os.path.exists(sent)
    with open(sent) as f:
        content = f.read()
    assert "is_independent_review=False" in content
    assert "state=RC1_INTERNAL_RED_TEAM_REVIEW_COMPLETE" in content


def test_issue_counts_match():
    with open(os.path.join(RC1, "issues.tsv")) as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    with open(os.path.join(RC1, "RC1_decision.json")) as f:
        d = json.load(f)
    counts = d["issue_counts"]
    assert counts["total"] == len(rows)
    assert counts["total"] == counts["critical"] + counts["high"] + counts["medium"] + counts["low"]
    assert counts["critical"] + counts["high"] >= 1