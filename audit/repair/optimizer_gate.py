"""Strict optimizer qualification gate (P0.2).

The strict audit found that ``res.success`` from SciPy was treated as
convergence even when the final raw gradient norm was large
(e.g. motif ~36, k-mer ~7, position-additive ~11, frozen RNA-FM head ~55).
L-BFGS-B reports success on a scaled projected-gradient criterion, so a raw
large norm is not a proof of a bad fit for high-dim one-hot problems -- but the
audit demands an explicit, reproducible gate instead of an implicit
``success`` flag.

This module provides two explicit gates:

* ``unbounded_fit_gate``  : final ||grad||_inf (or norm) below a tolerance.
* ``bounded_fit_gate``    : projected-gradient + bound-hit accounting, i.e. a
  parameter is "converged" if it is interior with small gradient, or at a bound
  with the gradient pointing in the feasible direction (projected gradient ~ 0).

Every gate returns a record with ``eligible`` (bool) and the diagnostics
required by ConvergenceLedger_v3.  A fold that fails the gate is
comparison-ineligible and must NOT enter the primary leaderboard.
"""
from __future__ import annotations

import numpy as np

# Defaults: tuned for the audit's linear/Tobit parametric models.
DEFAULT_GRAD_TOL = 1e-3
DEFAULT_PROJ_GRAD_TOL = 1e-3
BOUND_EPS = 1e-6


def _projected_gradient(grad: np.ndarray, bounds, beta: np.ndarray) -> np.ndarray:
    """Projected gradient for box-constrained optimization.

    For a minimization with bounds [lo, hi]:
      - if beta[i] <= lo + eps and grad[i] > 0  -> param wants to go below lo,
        set projected grad to 0 (at lower active bound).
      - if beta[i] >= hi - eps and grad[i] < 0  -> param wants to go above hi,
        set projected grad to 0 (at upper active bound).
      - otherwise projected grad = grad[i].
    """
    g = np.asarray(grad, dtype=float).copy()
    b = np.asarray(beta, dtype=float)
    if bounds is None:
        return g
    for i, (lo, hi) in enumerate(bounds):
        if i >= len(g):
            break
        if lo is not None and b[i] <= float(lo) + BOUND_EPS and g[i] > 0:
            g[i] = 0.0
        elif hi is not None and b[i] >= float(hi) - BOUND_EPS and g[i] < 0:
            g[i] = 0.0
    return g


def _bound_hits(beta: np.ndarray, bounds) -> int:
    if bounds is None:
        return 0
    hits = 0
    for i, (lo, hi) in enumerate(bounds):
        if i >= len(beta):
            break
        if lo is not None and abs(beta[i] - float(lo)) < BOUND_EPS:
            hits += 1
        if hi is not None and abs(beta[i] - float(hi)) < BOUND_EPS:
            hits += 1
    return hits


def _finite(diag: dict) -> bool:
    return bool(np.all(np.isfinite(diag.get("beta", []))))


def unbounded_fit_gate(beta, grad, *, success=True, grad_tol=DEFAULT_GRAD_TOL,
                       grad_norm="inf") -> dict:
    beta = np.asarray(beta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    if grad_norm == "inf":
        norm = float(np.max(np.abs(grad))) if grad.size else 0.0
    else:
        norm = float(np.linalg.norm(grad))
    eligible = bool(success and np.isfinite(norm) and norm <= grad_tol
                    and np.all(np.isfinite(beta)))
    return {
        "eligible": eligible,
        "final_grad_norm": norm,
        "grad_norm_kind": grad_norm,
        "grad_tol": grad_tol,
        "success": bool(success),
        "n_nan_inf_params": int(np.sum(~np.isfinite(beta))) + int(np.sum(~np.isfinite(grad))),
    }


def bounded_fit_gate(beta, grad, bounds, *, success=True,
                     proj_grad_tol=DEFAULT_PROJ_GRAD_TOL) -> dict:
    """Gate a bounded parametric fold via the projected gradient.

    eligible iff: optimizer success AND projected-gradient norm <= tol AND all
    params finite.  Bound hits are reported as diagnostics (a converged
    solution may legitimately sit on a bound).
    """
    beta = np.asarray(beta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    bounds = [(b[0], b[1]) if isinstance(b, (tuple, list)) else b for b in (bounds or [])]
    pg = _projected_gradient(grad, bounds, beta)
    pg_norm = float(np.linalg.norm(pg))
    n_hits = _bound_hits(beta, bounds)
    eligible = bool(success and np.isfinite(pg_norm) and pg_norm <= proj_grad_tol
                    and np.all(np.isfinite(beta)))
    return {
        "eligible": eligible,
        "projected_grad_norm": pg_norm,
        "proj_grad_tol": proj_grad_tol,
        "final_grad_norm_raw": float(np.linalg.norm(grad)),
        "n_bound_hits": n_hits,
        "success": bool(success),
        "n_nan_inf_params": int(np.sum(~np.isfinite(beta))) + int(np.sum(~np.isfinite(grad))),
    }


def gate_from_fit(model: dict, bounds=None, *, grad_tol=DEFAULT_GRAD_TOL,
                  proj_grad_tol=DEFAULT_PROJ_GRAD_TOL) -> dict:
    """Convenience: build a gate record from a fitted model dict.

    The model dict may carry ``beta/params`` and ``grad``/``final_grad_norm``.
    If a raw ``final_grad_norm`` is provided but no full gradient vector, we
    fall back to comparing that scalar against ``grad_tol`` (unbounded) or
    against ``proj_grad_tol`` when bounds are present but we cannot project.
    """
    beta = np.asarray(model.get("beta") if model.get("beta") is not None
                      else model.get("params") if model.get("params") is not None
                      else [], dtype=float)
    grad = np.asarray(model.get("grad"), dtype=float) if model.get("grad") is not None else None
    success = bool(model.get("success", True))
    if grad is not None and grad.size == beta.size:
        if bounds is not None:
            return bounded_fit_gate(beta, grad, bounds, success=success,
                                    proj_grad_tol=proj_grad_tol)
        return unbounded_fit_gate(beta, grad, success=success, grad_tol=grad_tol)
    # scalar fallback
    rn = model.get("final_grad_norm")
    if rn is None:
        return {"eligible": False, "reason": "no_gradient_diagnostic"}
    tol = proj_grad_tol if bounds is not None else grad_tol
    eligible = bool(success and float(rn) <= tol and np.all(np.isfinite(beta)))
    return {
        "eligible": eligible,
        "final_grad_norm": float(rn),
        "grad_tol": tol,
        "success": bool(success),
        "n_bound_hits": _bound_hits(beta, bounds) if bounds is not None else 0,
        "n_nan_inf_params": int(np.sum(~np.isfinite(beta))),
    }