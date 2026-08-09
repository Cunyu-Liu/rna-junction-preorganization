"""P0.5 minimum-eligibility censored baselines.

All baselines are nuisance-only: they use NO junction sequence information.
They are fit only on the outer-training rows of each fold and evaluated on the
frozen test rows under the shared right-censored NLL metric (tau=0.7 fixed to
match the modeling convention, so comparisons are on equal footing).

Models (contract P0.5 minimum set, first three):
  global_censor_intercept : single location mu, no structure
  train_only_scaffold     : per-scaffold location mu_s
  scaffold_context_hierarchy : mu = b0 + scaffold_effect + context_effect
                               with L2 (partial-pooling) shrinkage

Each returns per-row (mu, sigma, support, abstain) for the test rows.
The likelihood is right-censored Gaussian (measured density + censored
survival), identical to audit.evaluation.metrics.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

CAP = -7.1
TAU = 0.7
RIDGE = 1.0   # partial-pooling shrinkage for hierarchy

# R0.2: single unified censored objective (correct gradient, optimizer gate).
from audit.core.censored_objective import CensoredObjective, fit_lbfgs


def _fit(Xtr, ytr, ctr, ridge=RIDGE):
    """Censored MLE with L2 ridge (shrinkage toward 0) using the unified
    R0.2 objective.  Returns (beta, gate_record); gate_record carries the
    optimizer/convergence diagnostics required by the fold gate."""
    nf = Xtr.shape[1]
    obj = CensoredObjective(Xtr, ytr, ctr)
    rec = fit_lbfgs(obj, np.zeros(nf), ridge=ridge, maxiter=2000, gtol=1e-8)
    return rec["beta"], rec


def _predict(beta, Xte, abstain_cols=None):
    mu = Xte @ beta
    sigma = np.full(len(mu), TAU)
    a = (mu - CAP) / TAU
    censor_prob = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
    if abstain_cols is not None:
        # cols [i] with no training evidence -> abstain
        support = np.ones(len(mu), dtype=bool)
        for col in abstain_cols:
            if col >= 0:
                support &= (np.abs(Xte[:, col]) > 1e-8)
        abstain = ~support
    else:
        support = np.ones(len(mu), dtype=bool)
        abstain = np.zeros(len(mu), dtype=bool)
    return mu, sigma, censor_prob, support, abstain


def _rows_to_arrays(rows):
    y = np.asarray([r["y"] for r in rows], dtype=float)
    cens = np.asarray([r["cens"] for r in rows], dtype=bool)
    return y, cens


def fit_global(train):
    y, cens = _rows_to_arrays(train)
    X = np.ones((len(y), 1))
    beta, gate = _fit(X, y, cens, ridge=0.0)
    return {"kind": "global", "beta": beta, "intercept": True, "gate": gate}


def predict_global(model, test):
    y, cens = _rows_to_arrays(test)
    X = np.ones((len(y), 1))
    mu, sigma, cp, support, abstain = _predict(model["beta"], X)
    return mu, sigma, cp, support, abstain


def fit_scaffold(train):
    scafs = sorted({r["scaf"] for r in train})
    idx = {s: i for i, s in enumerate(scafs)}
    X = np.zeros((len(train), len(scafs)))
    for i, r in enumerate(train):
        X[i, idx[r["scaf"]]] = 1.0
    y, cens = _rows_to_arrays(train)
    beta, gate = _fit(X, y, cens, ridge=0.0)
    return {"kind": "scaffold", "beta": beta, "scafs": scafs, "idx": idx, "gate": gate}


def predict_scaffold(model, test):
    scafs, idx = model["scafs"], model["idx"]
    X = np.zeros((len(test), len(scafs)))
    for i, r in enumerate(test):
        if r["scaf"] in idx:
            X[i, idx[r["scaf"]]] = 1.0
    mu, sigma, cp, support, abstain = _predict(model["beta"], X)
    # abstain scaffolds unseen in train
    abstain = np.asarray([r["scaf"] not in idx for r in test], dtype=bool)
    support = ~abstain
    return mu, sigma, cp, support, abstain


def fit_hierarchy(train, ridge=RIDGE):
    """mu = b0 + scaffold_effect + context_effect with L2 shrinkage."""
    scafs = sorted({int(r["scaf"]) for r in train})
    ctxs = sorted({str(r["helix_seq"]) for r in train})
    si = {s: i + 1 for i, s in enumerate(scafs)}   # +1 leaves col0 as intercept
    ci = {c: i + 1 + len(scafs) for i, c in enumerate(ctxs)}
    nf = 1 + len(scafs) + len(ctxs)
    X = np.zeros((len(train), nf))
    X[:, 0] = 1.0
    for i, r in enumerate(train):
        X[i, si[int(r["scaf"])]] = 1.0
        X[i, ci[str(r["helix_seq"])]] = 1.0
    y, cens = _rows_to_arrays(train)
    beta, gate = _fit(X, y, cens, ridge=ridge)
    return {"kind": "hierarchy", "beta": beta, "scafs": scafs, "ctxs": ctxs,
            "si": si, "ci": ci, "gate": gate}


def predict_hierarchy(model, test):
    scafs, ctxs = model["scafs"], model["ctxs"]
    nf = model["beta"].shape[0]
    X = np.zeros((len(test), nf))
    X[:, 0] = 1.0
    for i, r in enumerate(test):
        if int(r["scaf"]) in model["si"]:
            X[i, model["si"][int(r["scaf"])]] = 1.0
        if str(r["helix_seq"]) in model["ci"]:
            X[i, model["ci"][str(r["helix_seq"])]] = 1.0
    mu, sigma, cp, support, abstain = _predict(model["beta"], X)
    abstain = np.asarray(
        [(int(r["scaf"]) not in model["si"] or str(r["helix_seq"]) not in model["ci"])
         for r in test], dtype=bool)
    support = ~abstain
    return mu, sigma, cp, support, abstain


BASELINES = {
    "global_censor_intercept": (fit_global, predict_global),
    "train_only_scaffold": (fit_scaffold, predict_scaffold),
    "scaffold_context_hierarchy": (fit_hierarchy, predict_hierarchy),
}
