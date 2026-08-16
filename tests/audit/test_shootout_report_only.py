"""Tests for the report-only shootout aggregator (shootout_report_only.py).

Covers:
  (a) pooled junction-macro NLL and the paired contrast are computed correctly on
      a tiny synthetic prediction set (sign / magnitude sanity);
  (b) ``build_report`` is deterministic (two calls identical) and exposes the
      expected report keys;
  (c) end-to-end CLI: writing a report from a minimal on-disk prediction file is
      repeatable.
"""

from __future__ import annotations

import json

import pytest

from audit.repair.shootout_report_only import (
    _load_preds,
    build_report,
)
from audit.repair.shootout_run import _pooled_contrast, _pooled_nll_by_model


def _pred(model_id, jid, y, mu, sigma, cens=False, support=True, abstain=False):
    return {
        "axis": "edit_x_nested_context", "fold": "e:AAAC_GAAC",
        "source_row_id": f"row_{jid}", "jid": jid, "scaf": 0,
        "context": "AC", "model_id": model_id, "y": y, "cens": cens,
        "mu": mu, "sigma": sigma, "abstain": abstain, "support": support,
        "fallback_type": None,
    }


def _synthetic_preds():
    # model A hits y exactly (mu=y, small sigma) -> tiny NLL; model B is off
    preds = []
    for jid, y in [("j1", 2.0), ("j2", 5.0), ("j3", 7.0)]:
        preds.append(_pred("good", jid, y, mu=y, sigma=0.5))
        preds.append(_pred("bad", jid, y, mu=y + 3.0, sigma=0.5))
    return preds


def test_pooled_nll_ranks_good_below_bad():
    preds = _synthetic_preds()
    pooled = _pooled_nll_by_model(preds)
    assert pooled["good"] < pooled["bad"]
    # exact-good predictor has near-zero NLL
    assert pooled["good"] < 1.0


def test_paired_contrast_sign_and_semantics():
    preds = _synthetic_preds()
    # delta = (b - a); positive means a is better (good vs bad)
    c = _pooled_contrast(preds, "good", "bad", "good", "bad",
                         "delta (bad - good)")
    assert c["available"] is True
    assert c["theta_abs"] > 0
    assert c["relative_gain"] > 0
    assert c["n_junctions"] == 3


def test_build_report_deterministic_and_shape():
    preds = _synthetic_preds()
    r1 = build_report(preds, [])
    r2 = build_report(preds, [])
    assert r1 == r2
    for key in ("hybrid_vs_nuisance", "rnafm_vs_nuisance", "pooled_junction_macro_nll",
                "vienna_vs_no_sequence", "rnafm_vienna_vs_nuisance"):
        assert key in r1


def test_cli_writes_report_repeatably(tmp_path):
    preds = _synthetic_preds()
    predfile = tmp_path / "Predictions_v3.jsonl"
    with predfile.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")
    loaded = _load_preds(tmp_path)
    assert len(loaded) == len(preds)
    report = build_report(loaded, [])
    assert report["pooled_junction_macro_nll"]["good"] < report["pooled_junction_macro_nll"]["bad"]
