"""Diagnostic: characterize P0.3 G3/G4/G5 failures.

Questions:
1. Is GH 'non-convergence' caused by re-optimizing at each node count (confound),
   or genuine quadrature error? -> evaluate objective at FIXED params across nodes.
2. Is synthetic recovery failure due to optimizer maxiter exhaustion, heavy ridge,
   or genuine identifiability failure? -> vary maxiter and ridge, track grad_norm.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v131_corrected_objective as v
import synthetic_operator_fixture as synth

CAP = -7.1
RIDGE = 100.0
SLOPE_RIDGE = 5.0


def build(n_junc, censor_frac, seed, with_signal=True):
    nf = 63
    theta_star = (np.random.default_rng(123).normal(0, 1, nf) * 0.3 if with_signal else np.zeros(nf))
    a_star = np.array([-8.0, -8.5, -9.0, -7.5, -8.2, -8.8, -7.9, -8.6, -8.4])
    b_star = np.array([1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 1.15, 0.85, 1.0])
    seqs, X = synth.make_sequences(n_junc, 9, seed)
    panel, X, q_star = synth.make_panel(seqs, X, theta_star, a_star, b_star, 9,
                                        censor_frac, seed + 1)
    ref = panel["scaffolds"].index(2)
    return panel, X, q_star, theta_star, ref


def gh_fixed_params(panel, X, ref):
    """Quadrature-only convergence: same params, vary node count."""
    nf = X.shape[1]
    ns = len(panel["scaffolds"])
    theta = np.random.default_rng(9).normal(0, 1, nf) * 0.2
    a = np.array([-8.0, -8.5, -9.0, -7.5, -8.2, -8.8, -7.9, -8.6, -8.4])
    logb = np.full(ns, 0.0)
    p = v.pack(theta, a, logb, ref)
    out = {}
    for n in (12, 24, 48, 64):
        nodes, lw = v.hermite(n)
        obj, grad = v.corrected_objective_and_grad(p, panel, X, nodes, lw, RIDGE, SLOPE_RIDGE, ref)
        out[n] = float(obj)
    return out


def recovery(n_junc, censor_frac, maxiter, ridge, seed=42):
    panel, X, q_star, theta_star, ref = build(n_junc, censor_frac, seed, with_signal=True)
    nf = X.shape[1]
    ns = len(panel["scaffolds"])
    nodes, lw = v.hermite(24)
    p0 = v.pack(np.zeros(nf), np.zeros(ns), np.zeros(ns), ref)
    trace = []
    def cb(xk):
        trace.append(float(np.linalg.norm(
            v.corrected_objective_and_grad(xk, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[1])))
    res = minimize(lambda p: v.corrected_objective_and_grad(
        p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[0],
        p0, jac=lambda p: v.corrected_objective_and_grad(
            p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[1],
        method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-9, "maxls": 40},
        callback=cb)
    theta, a, b = v.unpack(res.x, nf, ns, ref)
    th, ts = np.asarray(theta), np.asarray(theta_star)
    corr = float(np.corrcoef(th, ts)[0, 1]) if np.std(th) > 0 else 0.0
    return {
        "n_junc": n_junc, "censor_frac": censor_frac, "maxiter": maxiter, "ridge": ridge,
        "success": bool(res.success), "nit": int(res.nit),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "theta_corr": corr, "theta_scale": float(np.sum(th*ts)/max(np.sum(ts*ts),1e-12)),
        "final_trace_grad": trace[-1] if trace else None,
    }


def main():
    out = {}
    # GH quadrature-only
    panel, X, q_star, ts, ref = build(120, 0.2, 42, with_signal=True)
    gh = gh_fixed_params(panel, X, ref)
    out["gh_fixed_params"] = gh
    out["gh_fixed_params_diff_24_48"] = abs(gh[24]-gh[48])
    out["gh_fixed_params_diff_12_48"] = abs(gh[12]-gh[48])

    # recovery: sweep maxiter and ridge
    rec = []
    for cf in (0.0, 0.2, 0.5):
        for mi in (400, 1500):
            for ridge in (100.0, 5.0, 1.0):
                rec.append(recovery(120, cf, mi, ridge))
    out["recovery_sweep"] = rec
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
