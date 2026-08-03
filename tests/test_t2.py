"""Unit tests for the T2 tecto-only inference gate.

Verify the censored-inversion helpers, the censoring rule, and the
random-effects generalization model on controlled inputs. These are NOT a
biological success claim.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np  # noqa: E402
import t2_run  # noqa: E402


def _make_junction(vals):
    """Build a single-junction row group from dg values + censoring + scaffolds."""
    rows = []
    for i, v in enumerate(vals):
        rows.append({"dg10": v, "cens": t2_run.censored_flag(v), "scaf": i % 9})
    return rows


def test_censored_flag():
    assert t2_run.censored_flag(-7.1) is True
    assert t2_run.censored_flag(-7.1000001) is True  # within tolerance
    assert t2_run.censored_flag(-7.6) is False
    assert t2_run.censored_flag(-15.0) is False


def test_junction_censored_fit_recovers_mean():
    # 9 scaffolds, ~20 members each, small noise, no censoring -> estimate near true
    rng = np.random.default_rng(0)
    true_mu = -9.0
    vals = []
    for sc in range(9):
        for _ in range(20):
            vals.append(true_mu + rng.normal(0, 0.2))
    rows = _make_junction(vals)
    est = t2_run.junction_censored_fit(
        np.array([r["dg10"] for r in rows]),
        np.array([r["cens"] for r in rows]),
        np.array([r["scaf"] for r in rows]))
    assert est is not None
    assert abs(est["point"] - true_mu) < 0.3
    assert est["lower"] < true_mu < est["upper"]


def test_junction_censored_fit_out_of_range():
    # fewer than 2 measured rows -> not identifiable
    rows = [{"dg10": -7.1, "cens": True, "scaf": 0},
            {"dg10": -7.1, "cens": True, "scaf": 1}]
    est = t2_run.junction_censored_fit(
        np.array([r["dg10"] for r in rows]),
        np.array([r["cens"] for r in rows]),
        np.array([r["scaf"] for r in rows]))
    assert est is None


def test_fit_random_effects_recovers_tau2():
    # est_i = mu + u_i + e_i, u_i ~ N(0, tau2), e_i ~ N(0, se_i^2)
    rng = np.random.default_rng(1)
    mu = -9.0
    tau2 = 0.25
    n = 200
    ests = mu + rng.normal(0, np.sqrt(tau2), n)
    ses = np.full(n, 0.1)
    mu_hat, tau2_hat, se_mu = t2_run.fit_random_effects(ests, ses)
    assert abs(mu_hat - mu) < 0.2
    assert 0.1 < tau2_hat < 0.5


def test_thresholds():
    assert t2_run.CAP == -7.1
    assert t2_run.MIN_EFFECT == 1.0
    assert t2_run.WIDTH_MAX == 1.0
    assert t2_run.SPLIT_SEED == 20260803