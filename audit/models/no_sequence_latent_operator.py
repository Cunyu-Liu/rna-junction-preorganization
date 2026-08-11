"""Matched no-sequence latent-operator Tobit (R2 core contrast).

Purpose (contract §9.2 / §12.3): the ONLY difference vs corrected v1.31 must be
the sequence/physical representation.  Both models share:
    q_j  ~ N(location_j, sigma_q^2)
    Y_js | q_j ~ N(a_s + b_s q_j, tau^2)
    right-censored marginal via Gauss-Hermite, same ridge/slope_ridge, same
    bounds, same optimizer budget.

Difference:
  - corrected_v1_31 : location_j = X_j @ theta  (per-junction sequence features)
  - no_sequence      : location_j = theta_0      (single shared intercept)

So the no-sequence model cannot use junction identity or sequence composition to
inform the latent location; the only scaffold-level calibration remains the
intercept a_s and slope b_s.  If the full model's apparent gain survives the
matched ablation (i.e. full vs no-sequence > 0) on identical folds/rows, that
gain is attributable to the sequence map, not to scaffold calibration or to the
latent/scaffold machinery that both models share.

Predictive sigma mirrors corrected_v1_31: sqrt(tau^2 + b_s^2 * sigma_q^2).
unseen scaffold -> abstain (no placeholder scoring).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.numerics.v131_corrected_objective import (
    hermite, pack, unpack, bounds, corrected_objective_and_grad,
    CAP, TAU, SIGMA_Q)

RIDGE = 5.0
SLOPE_RIDGE = 5.0
GH = 48
MAXITER = 500
MAXLS = 40


def _panel(train_rows):
    """Build train panel with an intercept-only sequence map.

    The location matrix has a single column of ones, so the latent location is a
    shared scalar theta_0 across all junctions (no sequence input).
    """
    scaffolds = sorted({int(r["scaf"]) for r in train_rows})
    si = {s: i for i, s in enumerate(scaffolds)}
    ji = {}
    flat_j, flat_s, flat_y, flat_c = [], [], [], []
    for r in train_rows:
        j = str(r["jid"])
        if j not in ji:
            ji[j] = len(ji)
        flat_j.append(ji[j])
        flat_s.append(si[int(r["scaf"])])
        flat_y.append(float(r["y"]))
        flat_c.append(bool(r["cens"]))
    panel = {"jids": sorted(ji, key=ji.get), "scaffolds": scaffolds,
             "flat_j": np.asarray(flat_j, dtype=int),
             "flat_s": np.asarray(flat_s, dtype=int),
             "flat_y": np.asarray(flat_y, dtype=float),
             "flat_c": np.asarray(flat_c, dtype=bool)}
    n_j = len(panel["jids"])
    X = np.ones((n_j, 1))  # single shared intercept, no per-junction features
    ref = scaffolds.index(2) if 2 in scaffolds else 0
    return panel, X, ji, scaffolds, ref


def make_no_sequence_adapter():
    """Return (fit, predict) for the matched no-sequence latent operator."""
    def fit(train_rows):
        panel, X_tr, _, scaffolds, ref = _panel(train_rows)
        nf = X_tr.shape[1]
        ns = len(panel["scaffolds"])
        nodes, lw = hermite(GH)
        p0 = pack(np.zeros(nf), np.zeros(ns), np.zeros(ns), ref)
        res = minimize(lambda p: corrected_objective_and_grad(
            p, panel, X_tr, nodes, lw, RIDGE, SLOPE_RIDGE, ref)[0],
            p0, jac=lambda p: corrected_objective_and_grad(
                p, panel, X_tr, nodes, lw, RIDGE, SLOPE_RIDGE, ref)[1],
            method="L-BFGS-B", bounds=bounds(nf, ns, ref),
            options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-7, "maxls": MAXLS})
        theta, a, b = unpack(res.x, nf, ns, ref)
        return {"kind": "no_sequence_latent_operator", "theta": theta,
                "a": a, "b": b, "ref": ref, "scaffolds": scaffolds,
                "n_junctions": len(panel["jids"]),
                "success": bool(res.success), "nit": int(res.nit),
                "optimizer_message": str(res.message),
                "final_grad_norm": float(np.linalg.norm(res.jac)),
                "grad": np.asarray(res.jac, dtype=float),
                "beta": np.asarray(res.x, dtype=float),
                "bounds": bounds(nf, ns, ref)}

    def predict(model, test_rows):
        si = {s: i for i, s in enumerate(model["scaffolds"])}
        theta, a, b = model["theta"], model["a"], model["b"]
        n = len(test_rows)
        mu = np.zeros(n)
        sigma = np.full(n, TAU)
        cp = np.zeros(n)
        support = np.ones(n, dtype=bool)
        abstain = np.zeros(n, dtype=bool)
        loc = float(theta[0])  # shared latent location for every junction
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) not in si:
                abstain[i] = True
                support[i] = False
                mu[i] = 0.0
                sigma[i] = TAU
            else:
                s = si[int(r["scaf"])]
                mu[i] = a[s] + b[s] * loc
                sigma[i] = float(np.sqrt(TAU * TAU + (b[s] * SIGMA_Q) ** 2))
            cp[i] = float(np.exp(np.clip(log_ndtr((mu[i] - CAP) / sigma[i]), -50.0, 0.0)))
        return mu, sigma, cp, support, abstain

    return fit, predict


NO_SEQUENCE_LATENT_OPERATOR = {
    "no_sequence_latent_operator": make_no_sequence_adapter(),
}
