"""Unit tests for the Denny-train-only and RNAMake/physical-prior baselines
(contract §4.2 / §9.2 / §9.3).  Includes a finite-difference gradient check for
the Denny censored objective (guards against the historical gradient-sign bug
class) and interface/feature sanity checks for the physical prior head.
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.benchmark.denny_train_only import (
    DENNY_TRAIN_ONLY, _objective, hermite, TAU, SIGMA_Q, RIDGE)
from audit.benchmark.physical_prior import (
    build_physical_cache, fit_physical_head, predict_physical_head,
    _contig, physical_feature_names)


# ---------------------------------------------------------------------------
# Denny-train-only
# ---------------------------------------------------------------------------
def test_denny_gradient_finite_difference():
    """Analytic vs central finite-difference gradient (with bias) must agree."""
    nodes, lw = hermite(48)
    rng = np.random.default_rng(0)
    n_j = 20
    X = rng.standard_normal((n_j, 5))
    flat_j, flat_y, flat_c = [], [], []
    for j in range(n_j):
        for _ in range(3):
            flat_j.append(j)
            flat_y.append(rng.normal() * 0.7)
            flat_c.append(bool(rng.random() < 0.3))
    panel = {"jids": list(range(n_j)),
             "flat_j": np.array(flat_j, dtype=int),
             "flat_y": np.array(flat_y),
             "flat_c": np.array(flat_c)}
    theta = np.array([-5.0, 0.1, -0.2, 0.3, 0.05, -0.1])
    nll, g = _objective(theta, panel, X, nodes, lw, RIDGE)
    eps = 1e-6
    gn = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        gn[i] = (_objective(tp, panel, X, nodes, lw, RIDGE)[0]
                 - _objective(tm, panel, X, nodes, lw, RIDGE)[0]) / (2 * eps)
    max_rel = float(np.max(np.abs(g - gn) / (np.abs(gn) + 1e-9)))
    assert max_rel < 1e-4


def test_denny_model_predicts_in_data_range():
    """The shared bias must let the model represent the mean of y (not ~0)."""
    rng = np.random.default_rng(1)
    train = []
    for i in range(40):
        train.append({"jid": "j%d" % i, "junction_seq": "ACGU%02d" % i,
                      "scaf": int(i % 3) + 1, "y": float(-9.0 + rng.standard_normal()),
                      "cens": bool(i % 4 == 0), "source_row_id": "r%d" % i})
    test = [{"jid": "j%d" % i, "junction_seq": "ACGU%02d" % i,
             "scaf": 1, "y": 0.0, "cens": False, "source_row_id": "t%d" % i}
            for i in range(40)]
    fit, pred = DENNY_TRAIN_ONLY["denny_train_only"]
    model = fit(train)
    assert model["success"]
    mu, sigma, cp, support, abstain = pred(model, test)
    assert mu.shape == (len(test),)
    # mean of predictions should be near the data mean (~-9), not near 0
    assert abs(float(np.mean(mu)) - (-9.0)) < 1.0
    assert np.all(np.isfinite(mu))
    assert set(support.tolist()) == {True}
    assert set(abstain.tolist()) == {False}
    assert np.all(sigma > 0)


# ---------------------------------------------------------------------------
# Physical prior (RNAMake-style)
# ---------------------------------------------------------------------------
def test_physical_contig():
    assert _contig("CUAG_CUAAG") == "CUAGCUAAG"
    assert _contig(None) == ""


def test_physical_cache_feature_dim():
    cache = build_physical_cache(["CUAG_CUAAG", "CUAG_CUUAG", "GCGC_GCAGC"])
    assert len(cache) == 3
    for k, v in cache.items():
        assert v.shape == (6,)
        assert np.all(np.isfinite(v))
    assert len(physical_feature_names()) == 6


def test_physical_head_fit_predict_shapes():
    cache = build_physical_cache(["CUAG_CUAAG", "CUAG_CUUAG", "GCGC_GCAGC",
                                  "GCGC_GCAGC", "ACGU_ACGU", "UGCA_UGCA"])
    rng = np.random.default_rng(2)
    train = [{"junction_seq": s, "y": float(-9.0 + rng.standard_normal()),
              "cens": bool(i % 3 == 0), "source_row_id": "r%d" % i}
             for i, s in enumerate(["CUAG_CUAAG", "CUAG_CUUAG", "GCGC_GCAGC"])]
    test = [{"junction_seq": s, "y": 0.0, "cens": False,
             "source_row_id": "t%d" % i}
            for i, s in enumerate(["ACGU_ACGU", "UGCA_UGCA"])]
    model = fit_physical_head(train, cache)
    assert model["gate"]["success"]
    mu, sigma, cp, support, abstain = predict_physical_head(model, test, cache)
    assert mu.shape == (len(test),) and sigma.shape == (len(test),)
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0)
    assert set(support.tolist()) == {True} and set(abstain.tolist()) == {False}
