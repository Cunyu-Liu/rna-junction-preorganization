"""Denny-train-only thermodynamic-fingerprint baseline (contract §4.2 / §9.2).

Denny et al. (Cell 2018) attribute junction energetics to a per-junction
thermodynamic fingerprint measured across scaffolds.  Their NATIVE fingerprint
uses the target junctions measured information (oracle).  This baseline is the
strict TRAIN-ONLY reconstruction: it estimates the fingerprint from the junction
sequence alone, using only outer-train rows, and applies it to junctions in the
test fold WITHOUT any per-scaffold operator calibration.

Model (right-censored, Gauss-Hermite marginalization):
    q_j  ~ N(X_j @ theta, sigma_q^2)          # sequence -> fingerprint latent
    Y_js | q_j ~ N(q_j, tau^2)                # NO scaffold operator (a_s, b_s)
placeholder: the single shared fingerprint is the intrinsic preorganization.

This is the closest faithful train-only analog of the Denny native fingerprint
and is contrastive to corrected_v1_31 (which adds per-scaffold a_s/b_s) and to
no_sequence_latent_operator (which keeps operators but drops the sequence map).
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.features import build_raw_by_jid, fit_scaler, transform

CAP = -7.1
TAU = 0.7
SIGMA_Q = 1.0
RIDGE = 5.0
GH = 48
MAXITER = 500
MAXLS = 40
THETA_BOUND = 4.0


def hermite(nodes):
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    log_w = np.log(np.maximum(w, 1e-300)) - 0.5 * math.log(math.pi)
    return x, log_w


def _objective(theta, panel, X, nodes, log_w, ridge):
    """Return (nll, grad). theta[0] is the shared bias; theta[1:] are per-feature.

    The latent is q_j = bias + X_j @ theta[1:]; without the bias the model cannot
    represent the mean of Y (which is far from 0), so the bias is required for a
    fair sequence-only fingerprint reconstruction.
    """
    n_groups = len(panel["jids"])
    bias = theta[0]
    f = np.asarray(X @ theta[1:], dtype=float) + bias
    q = f[:, None] + math.sqrt(2.0 * SIGMA_Q) * nodes[None, :]
    y = panel["flat_y"][:, None]
    c = panel["flat_c"][:, None]
    z = (y - q[panel["flat_j"]]) / TAU
    mu_q = q[panel["flat_j"]]
    ll = np.where(c,
                  log_ndtr((mu_q - CAP) / TAU),
                  -0.5 * math.log(2.0 * math.pi) - math.log(TAU) - 0.5 * z * z)
    grouped = np.zeros((n_groups, len(nodes)), dtype=float)
    for k in range(len(nodes)):
        np.add.at(grouped[:, k], panel["flat_j"], ll[:, k])
    log_marg = logsumexp(grouped + log_w[None, :], axis=1)
    D = -float(np.sum(log_marg))
    posterior = np.exp(grouped + log_w[None, :] - log_marg[:, None])

    # score of a single observation w.r.t. the latent q
    score_mu = np.where(
        c,
        np.exp(-0.5 * ((mu_q - CAP) / TAU) ** 2 - 0.5 * math.log(2.0 * math.pi)
               - log_ndtr((mu_q - CAP) / TAU)) / TAU,
        (y - q[panel["flat_j"]]) / (TAU * TAU),
    )
    # d log_marg_j / d f_j  via chain rule over GH nodes
    score_jk = np.zeros_like(grouped)
    for k in range(len(nodes)):
        np.add.at(score_jk[:, k], panel["flat_j"], score_mu[:, k])
    grad_f = np.sum(posterior * score_jk, axis=1)  # dD/d f_j (unscaled)
    grad_bias_data = -float(np.sum(grad_f))
    grad_feat_data = -(X.T @ grad_f)
    grad_theta_data = np.concatenate([[grad_bias_data], grad_feat_data])
    # regularize per-feature coefficients only (bias/intercept is unregularized,
    # consistent with v1.31 scaffold intercepts a_s)
    grad_theta_reg = np.concatenate([[0.0], float(ridge) * theta[1:]])
    R = 0.5 * float(ridge) * float(np.dot(theta[1:], theta[1:]))
    total = D + R
    grad = (grad_theta_data + grad_theta_reg) / n_groups
    return total / n_groups, grad


def make_denny_train_only_adapter():
    def fit(train_rows):
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = build_raw_by_jid(train_rows)
        mean, sd = fit_scaler(tr_jids, by_jid)
        X = transform(tr_jids, by_jid, mean, sd)
        ji = {j: i for i, j in enumerate(tr_jids)}
        flat_j, flat_y, flat_c = [], [], []
        for r in train_rows:
            flat_j.append(ji[str(r["jid"])])
            flat_y.append(float(r["y"]))
            flat_c.append(bool(r["cens"]))
        panel = {"jids": tr_jids, "flat_j": np.asarray(flat_j, dtype=int),
                 "flat_y": np.asarray(flat_y, dtype=float),
                 "flat_c": np.asarray(flat_c, dtype=bool)}
        nf = X.shape[1]
        nodes, log_w = hermite(GH)
        # bias init = unweighted mean of y (measured rows); features start at 0
        meas_y = [float(r["y"]) for r in train_rows if not r["cens"]]
        bias0 = float(np.mean(meas_y)) if meas_y else -9.0
        p0 = np.concatenate([[bias0], np.zeros(nf)])
        bias_hi = max(20.0, abs(bias0) * 2 + 20.0)
        res = minimize(lambda p: _objective(p, panel, X, nodes, log_w, RIDGE)[0],
                       p0, jac=lambda p: _objective(p, panel, X, nodes, log_w, RIDGE)[1],
                       method="L-BFGS-B",
                       bounds=[(-bias_hi, bias_hi)] + [(-THETA_BOUND, THETA_BOUND)] * nf,
                       options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-7, "maxls": MAXLS})
        return {"kind": "denny_train_only", "theta": np.asarray(res.x, dtype=float),
                "by_jid": by_jid, "tr_jids": tr_jids,
                "success": bool(res.success), "nit": int(res.nit),
                "final_grad_norm": float(np.linalg.norm(res.jac))}

    def predict(model, test_rows):
        mean, sd = fit_scaler(model["tr_jids"], model["by_jid"])
        te_jids = sorted({str(r["jid"]) for r in test_rows})
        by_jid = build_raw_by_jid(test_rows)
        X_te = transform(te_jids, by_jid, mean, sd)
        je = {j: i for i, j in enumerate(te_jids)}
        theta = model["theta"]
        n = len(test_rows)
        mu = np.zeros(n)
        sigma = np.full(n, TAU)
        cp = np.zeros(n)
        support = np.ones(n, dtype=bool)
        abstain = np.zeros(n, dtype=bool)
        q = theta[0] + X_te @ theta[1:]
        for i, r in enumerate(test_rows):
            j = je[str(r["jid"])]
            mu[i] = q[j]
            sigma[i] = float(np.sqrt(TAU * TAU + SIGMA_Q * SIGMA_Q))
            cp[i] = float(np.exp(np.clip(log_ndtr((mu[i] - CAP) / sigma[i]), -50.0, 0.0)))
        return mu, sigma, cp, support, abstain

    return fit, predict


DENNY_TRAIN_ONLY = {"denny_train_only": make_denny_train_only_adapter()}
