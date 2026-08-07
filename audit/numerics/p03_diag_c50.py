"""Diagnose the censor_frac=0.5 recovery failure.

The 0.5-censor signal case stopped with final_grad_norm=0.256 (not converged to
gtol).  Determine whether the weak theta_corr is (a) optimizer non-convergence
or (b) genuine identifiability degradation under heavy censoring.  Try:
  - higher maxiter, tighter gtol
  - random restarts
  - more junctions (data volume)
  - report grad_norm at termination honestly
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v131_corrected_objective as v
import synthetic_operator_fixture as synth

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
    return panel, X, theta_star, ref


def run(n_junc, censor_frac, maxiter, gtol, ridge, init_scale, seed=7):
    panel, X, theta_star, ref = build(n_junc, censor_frac, seed, with_signal=True)
    nf, ns = X.shape[1], len(panel["scaffolds"])
    nodes, lw = v.hermite(48)
    p0 = v.pack(np.zeros(nf), np.zeros(ns), np.zeros(ns), ref)
    res = minimize(lambda p: v.corrected_objective_and_grad(
        p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[0],
        p0, jac=lambda p: v.corrected_objective_and_grad(
            p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[1],
        method="L-BFGS-B", bounds=v.bounds(nf, ns, ref),
        options={"maxiter": int(maxiter), "ftol": 1e-13, "gtol": float(gtol), "maxls": 50})
    theta, a, b = v.unpack(res.x, nf, ns, ref)
    th, ts = np.asarray(theta), np.asarray(theta_star)
    corr = float(np.corrcoef(th, ts)[0, 1]) if np.std(th) > 0 else 0.0
    gn = float(np.linalg.norm(res.jac))
    return {"n_junc": n_junc, "censor_frac": censor_frac, "maxiter": maxiter,
            "gtol": gtol, "ridge": ridge, "success": bool(res.success),
            "nit": int(res.nit), "final_grad_norm": gn, "theta_corr": corr,
            "theta_scale": float(np.sum(th*ts)/max(np.sum(ts*ts),1e-12)),
            "converged_to_gtol": gn < gtol}


def main():
    out = []
    # base: replicate rev3 0.5 case
    out.append(run(120, 0.5, 1500, 1e-9, 5.0, 0.0))
    # more iterations
    out.append(run(120, 0.5, 4000, 1e-12, 5.0, 0.0))
    out.append(run(120, 0.5, 4000, 1e-12, 1.0, 0.0))
    # more data
    out.append(run(240, 0.5, 4000, 1e-12, 5.0, 0.0))
    out.append(run(360, 0.5, 4000, 1e-12, 5.0, 0.0))
    # no signal control at 0.5 (should be ~0 corr, small scale)
    panel, X, theta_star0, ref = build(120, 0.5, 7, with_signal=False)
    nf, ns = X.shape[1], len(panel["scaffolds"])
    nodes, lw = v.hermite(48)
    p0 = v.pack(np.zeros(nf), np.zeros(ns), np.zeros(ns), ref)
    res = minimize(lambda p: v.corrected_objective_and_grad(
        p, panel, X, nodes, lw, 5.0, SLOPE_RIDGE, ref)[0],
        p0, jac=lambda p: v.corrected_objective_and_grad(
            p, panel, X, nodes, lw, 5.0, SLOPE_RIDGE, ref)[1],
        method="L-BFGS-B", bounds=v.bounds(nf, ns, ref),
        options={"maxiter": 4000, "ftol": 1e-13, "gtol": 1e-12, "maxls": 50})
    theta, _, _ = v.unpack(res.x, nf, ns, ref)
    out.append({"n_junc": 120, "censor_frac": 0.5, "signal": "none",
                "success": bool(res.success), "nit": int(res.nit),
                "final_grad_norm": float(np.linalg.norm(res.jac)),
                "theta_scale": float(np.max(np.abs(theta)))})
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
