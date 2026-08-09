"""R0.2 unified right-censored Gaussian objective with correct gradients.

This is the SINGLE objective used by every parametric baseline so the
censored-row gradient sign error replicated across the old baselines can never
recur.  It also provides an optimizer-convergence gate so a "finished" fit
cannot be mistaken for a convergent fit (contract R0.2 / P0 checklist items 2-3).

Objective (per-row, right-censored Gaussian, tau fixed):
  measured rows : NLL_m = 0.5*log(2*pi) + log(tau) + 0.5*((y-mu)/tau)^2
  censored rows : NLL_c = -log P(Y>=CAP) = -log Phi((mu-CAP)/tau)

Gradients:
  dNLL/dmu (measured) = -(y-mu)/tau^2
  dNLL/dmu (censored) = -phi((mu-CAP)/tau) / (tau * Phi((mu-CAP)/tau))
    [raising mu raises P(Y>=CAP) and LOWERS the censored NLL]

Both directions are validated by a central finite-difference test
(tests/audit/test_censored_objective_fd.py) with tolerance <=1e-4 on a
well-conditioned fixture and <=1e-3 near a real init.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

CAP = -7.1
TAU = 0.7
EPS = 1e-8


def survival_nll(a):
    """-log Phi(a), clipped at 50 to stay finite."""
    return -np.clip(log_ndtr(np.asarray(a, dtype=float)), -50.0, 50.0)


def censored_score(a, tau=TAU):
    """d/da[-log Phi(a)] = -phi(a)/Phi(a); sign such that raising mu lowers NLL.

    Returns the derivative of NLL w.r.t. mu, i.e. -(phi/Phi)/tau.
    """
    a = np.asarray(a, dtype=float)
    # phi(a)/Phi(a) computed stably via log-space to avoid overflow
    phi = np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi))
    Phi = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
    ratio = np.where(Phi > 1e-300, phi / np.maximum(Phi, 1e-300), 0.0)
    return -ratio / tau  # dNLL/dmu


class CensoredObjective:
    """Linear-predictor right-censored objective.

    mu = X @ beta.  Returns (nll, grad, diag) where diag carries per-fold
    optimizer/convergence diagnostics for the fold gate.
    """

    def __init__(self, X, y, cens, tau=TAU, cap=CAP):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.cens = np.asarray(cens, dtype=bool)
        self.tau = tau
        self.cap = cap

    def value_grad(self, beta):
        X = self.X
        mu = X @ beta
        tau = self.tau
        y = self.y
        cens = self.cens
        grad = np.zeros_like(beta, dtype=float)
        nll = 0.0
        m = ~cens
        if m.any():
            z = (y[m] - mu[m]) / tau
            nll += 0.5 * float(np.sum(z * z))
            grad += -(X[m].T @ z) / tau
        c = cens
        if c.any():
            a = (mu[c] - self.cap) / tau
            nll += float(np.sum(survival_nll(a)))
            score = censored_score(a, tau)
            grad += X[c].T @ score
        # constant per-row term (same across models; kept for comparability)
        n_const = float(len(y))
        nll += n_const * (0.5 * np.log(2.0 * np.pi) + np.log(tau))
        return nll, grad

    def nll(self, beta):
        return self.value_grad(beta)[0]


def add_ridge(beta, ridge, mask_intercept=True):
    """L2 penalty on coefficients (optionally excluding column 0 intercept).

    Returns (reg_nll, reg_grad).
    """
    beta = np.asarray(beta, dtype=float)
    if mask_intercept and len(beta) > 0:
        pen_idx = np.arange(1, len(beta))
    else:
        pen_idx = np.arange(len(beta))
    if len(pen_idx) == 0:
        return 0.0, np.zeros_like(beta)
    reg = 0.5 * ridge * float(beta[pen_idx] @ beta[pen_idx])
    g = np.zeros_like(beta)
    g[pen_idx] = ridge * beta[pen_idx]
    return reg, g


def fit_lbfgs(objective, beta0, ridge=0.0, mask_intercept=True,
              maxiter=2000, gtol=1e-8, bounds=None):
    """Fit beta under CensoredObjective + optional ridge; returns a gate record.

    The returned dict always contains the optimizer/convergence diagnostics so
    a caller can decide whether the fold has numerical eligibility.
    """
    from scipy.optimize import minimize
    beta0 = np.asarray(beta0, dtype=float)

    def f(beta):
        nll, g = objective.value_grad(beta)
        if ridge > 0.0:
            reg, greg = add_ridge(beta, ridge, mask_intercept=mask_intercept)
            nll += reg
            g = g + greg
        return nll, g

    res = minimize(f, beta0, jac=True, method="L-BFGS-B",
                   bounds=bounds,
                   options={"maxiter": int(maxiter), "gtol": float(gtol)})
    beta = np.asarray(res.x, dtype=float)
    nll, grad = f(beta)
    grad_norm = float(np.linalg.norm(grad))
    n_nan_inf = int(np.sum(~np.isfinite(beta))) + int(np.sum(~np.isfinite(grad)))
    # bound hits: parameters at their bound within tolerance
    bound_hits = 0
    if bounds is not None:
        for i, (lo, hi) in enumerate(bounds):
            if lo is not None and abs(beta[i] - lo) < 1e-6:
                bound_hits += 1
            if hi is not None and abs(beta[i] - hi) < 1e-6:
                bound_hits += 1
    return {
        "beta": beta,
        "nll": float(nll),
        "success": bool(res.success),
        "optimizer_message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
        "final_grad_norm": grad_norm,
        "n_nan_inf_params": n_nan_inf,
        "n_bound_hits": bound_hits,
        "converged": bool(res.success and np.isfinite(nll)
                        and np.all(np.isfinite(beta)) and np.all(np.isfinite(grad))
                        and grad_norm <= 1e-2),
    }


def objective_gate_passes(record):
    """Gate a fitted parametric fold. False => model x fold is comparison-ineligible."""
    return bool(record.get("converged")) and not record.get("n_nan_inf_params")
