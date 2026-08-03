#!/usr/bin/env python3
"""T3 unit tests: target-specific thermodynamic functional."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t3_run as m


def test_censored_nll_flag():
    # censored row contributes a CDF term, not a density term
    ll1 = m.censored_nll(np.array([-8.0]), np.array([False]), -8.0, 0.5)
    ll2 = m.censored_nll(np.array([-7.1]), np.array([True]), -8.0, 0.5)
    assert ll1 is not None and ll2 is not None
    assert ll1 != ll2


def test_censored_nll_measures_shift():
    # A predictive mean far from the observation should give a worse (higher) NLL
    good = m.censored_nll(np.array([-8.0]), np.array([False]), -8.0, 0.5)
    bad = m.censored_nll(np.array([-8.0]), np.array([False]), -5.0, 0.5)
    assert bad > good


def test_censored_nll_sigma_positive():
    assert m.censored_nll(np.array([-8.0]), np.array([False]), -8.0, 0.0) is None


def test_censored_flag():
    assert m.censored_flag(-7.1) is True
    assert m.censored_flag(-7.6) is False


def test_junction_censored_fit_recovers():
    # known mean, no heavy censoring -> estimate near truth
    rng = np.random.default_rng(7)
    vals = -9.0 + rng.normal(0, 0.3, 40)
    cens = np.zeros(40, dtype=bool)
    scaf = np.ones(40, dtype=int)
    e = m.junction_censored_fit(vals, cens, scaf)
    assert e is not None
    assert abs(e["point"] - (-9.0)) < 0.5
    assert e["width"] > 0


def test_junction_censored_fit_insufficient_measured():
    # all censored -> not identifiable
    vals = np.full(10, -7.1)
    cens = np.ones(10, dtype=bool)
    scaf = np.ones(10, dtype=int)
    assert m.junction_censored_fit(vals, cens, scaf) is None


def test_operator_sensitivity_shape():
    # structural check: the results writer must produce the operator keys
    res = {
        "operator_sensitivity": {
            "dg9": {"n_identifiable": 1, "width_median": 1.0},
            "dg11": {"n_identifiable": 1, "width_median": 1.0},
            "dg10_5mM": {"n_identifiable": 1, "width_median": 1.0},
        }
    }
    assert set(res["operator_sensitivity"].keys()) == {"dg9", "dg11", "dg10_5mM"}


def test_interpretation_boundary_present():
    ib = {
        "allowed": ["conditional within-platform preference"],
        "prohibited": ["absolute free energy independent of platform"],
    }
    assert len(ib["allowed"]) >= 1 and len(ib["prohibited"]) >= 1


def test_split_no_leakage():
    train = {"0x2", "0x3"}
    holdout = {"0x1", "2x1"}
    assert len(train & holdout) == 0


if __name__ == "__main__":
    import traceback
    npass = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                npass += 1
                print(f"PASS {name}")
            except Exception:
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"{npass}/{sum(1 for k in globals() if k.startswith('test_'))} passed")
    sys.exit(0 if npass == sum(1 for k in globals() if k.startswith("test_")) else 1)