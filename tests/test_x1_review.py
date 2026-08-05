"""X1 independent recomputation + review fail-closed tests (v1.5 §18)."""
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
X1 = os.path.join(RUN_ROOT, "reproducibility", "x1")


def test_decision_state():
    with open(os.path.join(X1, "X1_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "X1_AWAITING_INDEPENDENT_REVIEW"
    assert d["gate"] == "X1"


def test_not_performed_statuses():
    with open(os.path.join(X1, "X1_decision.json")) as f:
        d = json.load(f)
    assert d["independent_recomputation_status"] == "NOT_PERFORMED"
    assert d["independent_review_status"] == "NOT_PERFORMED"


def test_reviewer_not_available():
    with open(os.path.join(X1, "reviewer_identity_and_independence.json")) as f:
        r = json.load(f)
    assert r["independent_recomputation_executor"]["status"] == "NOT_AVAILABLE"
    assert r["independent_reviewer"]["status"] == "NOT_AVAILABLE"
    assert r["fabrication_guard"]


def test_recompute_spec_targets():
    with open(os.path.join(X1, "recompute_spec.json")) as f:
        s = json.load(f)
    ids = [t["id"] for t in s["recompute_targets"]]
    for req in ("T6_LOCKED_METRICS", "Q8_SIX_SUBSTATES", "B3_PRIMARY_METRICS",
                "X0_PRIMARY_RESULT", "MAIN_FIGURE_SOURCE_DATA"):
        assert req in ids
    assert "numeric_abs" in s["pre_registered_tolerances"]
    assert s["fail_closed_rule"]


def test_environment_lock():
    with open(os.path.join(X1, "environment_lock.json")) as f:
        e = json.load(f)
    assert "python" in e and "platform" in e


def test_report_and_sentinel():
    assert os.path.exists(os.path.join(RUN_ROOT, "reports", "X1_report.md"))
    sent = os.path.join(RUN_ROOT, "sentinels", "X1_AWAITING_INDEPENDENT_REVIEW.sentinel")
    assert os.path.exists(sent)
    with open(sent) as f:
        c = f.read()
    assert "state=X1_AWAITING_INDEPENDENT_REVIEW" in c