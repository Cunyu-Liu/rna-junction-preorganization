"""ViennaRNA-representation latent-operator adapter (head-diagnosis follow-up).

Identical to corrected_v1_31 EXCEPT the sequence map: the 63-D position/
composition features are replaced by ViennaRNA thermodynamic/secondary-structure
features.  The latent-operator head, GH integration, scaffold intercept/slope,
ridge, bounds, optimizer budget and strict projected-gradient gate are all
unchanged.

Model:
    q_j ~ N(f_theta(seq), sigma_q^2),  f = x_j @ theta,  x_j = ViennaRNA features
    Y_js | q_j ~ N(a_s + b_s q_j, tau^2)
    right-censored marginal via Gauss-Hermite (same objective).

The only difference from corrected_v1_31 is the feature builder, so any gain
vs the no-sequence model is attributable to the ViennaRNA folding representation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_features import build_raw_by_jid, fit_scaler, transform
from audit.numerics.v131_corrected_objective import (
    hermite, pack, unpack, bounds, corrected_objective_and_grad,
    CAP, TAU, SIGMA_Q)

RIDGE = 5.0
SLOPE_RIDGE = 5.0
GH = 48
MAXITER = 500
MAXLS = 40


def _panel_and_x(train_rows, test_rows):
    tr_jids = sorted({str(r["jid"]) for r in train_rows})
    te_jids = sorted({str(r["jid"]) for r in test_rows})
    by_jid = build_raw_by_jid(train_rows + test_rows)
    mean, sd = fit_scaler(tr_jids, by_jid)
    X_tr = transform(tr_jids, by_jid, mean, sd)
    X_te = transform(te_jids, by_jid, mean, sd)
    scaffolds = sorted({int(r["scaf"]) for r in train_rows})
    si = {s: i for i, s in enumerate(scaffolds)}
    ji_tr = {j: i for i, j in enumerate(tr_jids)}
    ji_te = {j: i for i, j in enumerate(te_jids)}
    flat_j, flat_s, flat_y, flat_c = [], [], [], []
    for r in train_rows:
        flat_j.append(ji_tr[str(r["jid"])])
        flat_s.append(si[int(r["scaf"])])
        flat_y.append(float(r["y"]))
        flat_c.append(bool(r["cens"]))
    panel = {"jids": tr_jids, "scaffolds": scaffolds,
             "flat_j": np.asarray(flat_j, dtype=int),
             "flat_s": np.asarray(flat_s, dtype=int),
             "flat_y": np.asarray(flat_y, dtype=float),
             "flat_c": np.asarray(flat_c, dtype=bool)}
    ref = scaffolds.index(2) if 2 in scaffolds else 0
    return panel, X_tr, X_te, ji_te, scaffolds, ref, tr_jids, te_jids


def make_vienna_latent_adapter():
    """Return (fit, predict) for the ViennaRNA-representation latent operator."""
    def fit(train_rows):
        panel, X_tr, _, _, _, ref, tr_jids, te_jids = _panel_and_x(train_rows, train_rows)
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
        by_jid = build_raw_by_jid(train_rows)
        return {"kind": "vienna_latent_operator", "theta": theta, "a": a, "b": b,
                "ref": ref, "scaffolds": panel["scaffolds"],
                "by_jid": by_jid, "tr_jids": tr_jids,
                "success": bool(res.success), "nit": int(res.nit),
                "optimizer_message": str(res.message),
                "final_grad_norm": float(np.linalg.norm(res.jac)),
                "grad": np.asarray(res.jac, dtype=float),
                "beta": np.asarray(res.x, dtype=float),
                "bounds": bounds(nf, ns, ref)}

    def predict(model, test_rows):
        mean, sd = fit_scaler(model["tr_jids"], model["by_jid"])
        te_jids = sorted({str(r["jid"]) for r in test_rows})
        by_jid = build_raw_by_jid(test_rows)
        X_te = transform(te_jids, by_jid, mean, sd)
        je = {j: i for i, j in enumerate(te_jids)}
        si = {s: i for i, s in enumerate(model["scaffolds"])}
        theta, a, b = model["theta"], model["a"], model["b"]
        n = len(test_rows)
        mu = np.zeros(n)
        sigma = np.full(n, TAU)
        cp = np.zeros(n)
        support = np.ones(n, dtype=bool)
        abstain = np.zeros(n, dtype=bool)
        q = X_te @ theta
        for i, r in enumerate(test_rows):
            j = je[str(r["jid"])]
            if int(r["scaf"]) not in si:
                abstain[i] = True
                support[i] = False
                mu[i] = 0.0
                sigma[i] = TAU
            else:
                s = si[int(r["scaf"])]
                mu[i] = a[s] + b[s] * q[j]
                sigma[i] = float(np.sqrt(TAU * TAU + (b[s] * SIGMA_Q) ** 2))
            cp[i] = float(np.exp(np.clip(log_ndtr((mu[i] - CAP) / sigma[i]), -50.0, 0.0)))
        return mu, sigma, cp, support, abstain

    return fit, predict


VIENNA_LATENT_OPERATOR = {
    "vienna_latent_operator": make_vienna_latent_adapter(),
}