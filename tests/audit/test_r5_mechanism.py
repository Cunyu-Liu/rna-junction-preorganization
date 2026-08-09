"""Unit tests for the R5 mechanism / claim-matrix module."""
import pytest

from audit.r5_mechanism import (CLAIMS, build_claim_matrix, _pooled_macro,
                                paper_outline, limitations)


def test_claim_matrix_populated():
    m = build_claim_matrix("/tmp/fake_root", {}, {})
    assert m["n_claims"] == len(CLAIMS) > 0
    for r in m["rows"]:
        assert r["claim"] and r["evidence_label"] and r["decision"]
        assert r["evidence_label"] in ("FACT_CONFIRMED", "INFERENCE",
                                       "UNKNOWN_NOT_ASSERTED",
                                       "REQUIRES_NEW_EVIDENCE")


def test_no_sota_or_submission_claim():
    m = build_claim_matrix("/tmp/fake_root", {}, {})
    for r in m["rows"]:
        assert "SOTA" not in r["decision"].upper().split("/")[0] or \
            r["decision"] == "SOTA_NOT_ADJUDICATED"
    sub = [r for r in m["rows"] if "投稿" in r["claim"]]
    assert sub and sub[0]["decision"] == "NO_SUBMISSION_AUTHORIZATION"


def test_pooled_macro_measured():
    rows = [{"y": 1.0, "cens": False, "mu": 1.0, "sigma": 0.5, "jid": "j0"},
            {"y": 1.5, "cens": False, "mu": 1.5, "sigma": 0.5, "jid": "j0"},
            {"y": 2.0, "cens": False, "mu": 2.0, "sigma": 0.5, "jid": "j1"}]
    assert _pooled_macro(rows) is not None


def test_paper_outline_and_limitations_nonempty():
    assert "Benchmark" in paper_outline()
    assert "UNKNOWN_NOT_ASSERTED" in limitations()
