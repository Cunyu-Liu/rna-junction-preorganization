"""Unit tests for the R2 group/multiway cluster uncertainty module."""
import numpy as np
import pytest

from audit.statistics import multiway_cluster as mw


def _rows(n_junctions=8, n_contexts=4, gain=0.0):
    """Build eligible rows.  Each junction in n_contexts contexts.
    d = nll_ns - nll_full; positive gain => d>0 on every row of a junction."""
    rows = []
    for j in range(n_junctions):
        for c in range(n_contexts):
            rows.append({"source_row_id": f"{j}_{c}", "jid": str(j),
                         "context": f"ctx{c}", "scaf": j % 3,
                         "d": gain, "nll_ns": 2.0})
    return rows


def test_axis_statistic_positive_gain():
    rows = _rows(gain=1.0)   # each junction d=1.0 -> theta=1.0, rel=0.5
    theta, rel, n = mw.axis_statistic(rows)
    assert theta == pytest.approx(1.0)
    assert rel == pytest.approx(0.5)
    assert n == 8


def test_axis_statistic_zero_gain():
    rows = _rows(gain=0.0)
    theta, rel, n = mw.axis_statistic(rows)
    assert theta == pytest.approx(0.0)
    assert rel == pytest.approx(0.0)


def test_junction_bootstrap_gives_ci():
    rows = _rows(n_junctions=30, gain=1.0)
    boot = mw.junction_bootstrap(rows, n_boot=1000, seed=1)
    lo, hi = mw.percentile_ci(boot)
    assert lo > 0.5   # strong consistent signal -> tight CI well above 0


def test_junction_pairing_null_centered():
    rows = _rows(n_junctions=30, gain=0.0)
    nulls = mw.junction_pairing_null(rows, n_null=2000, seed=3)
    assert abs(nulls.mean()) < 0.05
    p = float((np.sum(nulls >= 0.0) + 1) / (2000 + 1))
    assert p > 0.9   # no signal -> p near 1


def test_junction_pairing_null_separates_signal():
    rows = _rows(n_junctions=30, gain=1.0)
    nulls = mw.junction_pairing_null(rows, n_null=2000, seed=7)
    p = float((np.sum(nulls >= 1.0) + 1) / (2000 + 1))
    assert p < 0.01   # observed theta=1.0 far in right tail


def test_two_way_cluster_bootstrap_runs():
    rows = _rows(n_junctions=20, n_contexts=5, gain=0.5)
    tw = mw.two_way_cluster_bootstrap(rows, n_boot=500, seed=11)
    lo, hi = mw.percentile_ci(tw)
    assert lo > 0.0   # positive signal survives two-way resampling


def test_percentile_ci_alpha():
    samples = np.random.default_rng(0).normal(0, 1, 10000)
    lo, hi = mw.percentile_ci(samples, alpha=0.05)
    assert lo < 0 < hi
