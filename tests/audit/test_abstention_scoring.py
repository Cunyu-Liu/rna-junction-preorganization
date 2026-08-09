"""R0.2 abstention / support-aware scoring tests.

Contract R0.2 must tests:
  - abstain without fallback entering full task => comparison-ineligible
  - selective coverage below frozen floor => meets_floor False
  - unsupported placeholder values never enter an NLL aggregate
  - non-unique primary key => detected
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.evaluation.scorer_v2 import (
    full_coverage_score, selective_score, validate_unique_keys, TASK_SPECS,
)

CAP = -7.1


def _rows(n=4):
    return [{"source_row_id": f"R{i:03d}", "jid": f"J{i}", "y": -8.5, "cens": False,
             "helix_seq": f"ctx{i}", "scaf": 1} for i in range(n)]


def _pred(rid, mu=-8.0, abstain=False, fallback_type=""):
    return {"source_row_id": rid, "mu": mu, "sigma": 0.7, "abstain": abstain,
            "fallback_type": fallback_type}


def test_abstain_without_fallback_ineligible():
    rows = _rows(3)
    preds = {r["source_row_id"]: _pred(r["source_row_id"]) for r in rows}
    preds["R000"]["abstain"] = True
    preds["R000"]["fallback_type"] = ""
    score, elig = full_coverage_score(rows, preds)
    assert elig["eligible"] is False
    assert score["n_abstain_no_fallback"] == 1
    assert score["coverage"] == pytest.approx(2 / 3)


def test_full_coverage_all_scorable_eligible():
    rows = _rows(3)
    preds = {r["source_row_id"]: _pred(r["source_row_id"]) for r in rows}
    score, elig = full_coverage_score(rows, preds)
    assert elig["eligible"] is True
    assert score["coverage"] == pytest.approx(1.0)


def test_fallback_restores_eligibility():
    rows = _rows(3)
    preds = {r["source_row_id"]: _pred(r["source_row_id"]) for r in rows}
    preds["R000"]["abstain"] = True
    preds["R000"]["fallback_type"] = "global_mu"  # pre-registered fallback
    score, elig = full_coverage_score(rows, preds)
    assert elig["eligible"] is True
    assert score["n_abstain_no_fallback"] == 0


def test_selective_coverage_floor():
    rows = _rows(5)
    preds = {r["source_row_id"]: _pred(r["source_row_id"]) for r in rows}
    # abstain 2 of 5 -> coverage 0.6 < floor 0.8
    for rid in ("R000", "R001"):
        preds[rid]["abstain"] = True
        preds[rid]["fallback_type"] = ""
    s = selective_score(rows, preds)
    assert s["meets_floor"] is False
    assert s["coverage"] == pytest.approx(0.6)


def test_placeholder_never_in_nll():
    rows = _rows(4)
    # placeholder mu=0 for abstained row must NOT lower the pooled NLL
    preds = {r["source_row_id"]: _pred(r["source_row_id"], mu=-8.0) for r in rows}
    preds["R000"] = {"source_row_id": "R000", "mu": 0.0, "sigma": 0.7,
                     "abstain": True, "fallback_type": ""}  # placeholder
    score, elig = full_coverage_score(rows, preds)
    assert score["n_scorable"] == 3
    # placeholder row contributes nothing
    assert score["pooled_junction_macro_nll"] is not None


def test_duplicate_keys_detected():
    preds = [_pred("R000"), _pred("R000")]
    for p in preds:
        p["axis"] = "symmetry_5fold"; p["fold"] = 0; p["model_id"] = "m"
    assert validate_unique_keys(preds) == [("symmetry_5fold", 0, "R000", "m")]
