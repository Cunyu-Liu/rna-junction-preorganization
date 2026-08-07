"""Multi-seed 50%-censor recovery distribution (honest variance, not a gate).

The rev4 run showed the 50%-censor single-draw recovery is borderline
(theta_corr~0.47-0.48) and the optimizer stops at a line-search failure
(grad_norm~0.3), while a different seed (diag) recovered 0.70 at convergence.
This script estimates the true recovery distribution across multiple seeds so
the G4 gate can be evaluated on a proper statistic instead of one noisy draw.
It also reports how often the optimizer converges vs. stops in line search.
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
N_JUNC = 240
N_SCAF = 9
NF = 63


def build(seed_junc, censor_frac):
    theta_star = np.random.default_rng(123).normal(0, 1, NF) * 0.3
    a_star = np.array([-8.0, -8.5, -9.0, -7.5, -8.2, -8.8, -7.9, -8.6, -8.4])
    b_star = np.array([1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 1.15, 0.85, 1.0])
    seqs, X = synth.make_sequences(N_JUNC, N_SCAF, seed_junc)
    panel, X, _ = synth.make_panel(seqs, X, theta_star, a_star, b_star, N_SCAF,
                                   censor_frac, seed_junc + 1)
    return panel, X, theta_star


def run(seed_junc, censor_frac, ridge, maxiter, maxls, gtol):
    panel, X, theta_star = build(seed_junc, censor_frac)
    nf, ns = X.shape[1], len(panel["scaffolds"])
    nodes, lw = v.hermite(48)
    ref = panel["scaffolds"].index(2)
    p0 = v.pack(np.zeros(nf), np.zeros(ns), np.zeros(ns), ref)
    res = minimize(lambda p: v.corrected_objective_and_grad(
        p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[0],
        p0, jac=lambda p: v.corrected_objective_and_grad(
            p, panel, X, nodes, lw, ridge, SLOPE_RIDGE, ref)[1],
        method="L-BFGS-B", bounds=v.bounds(nf, ns, ref),
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": float(gtol),
                 "maxls": int(maxls)})
    theta, a, b = v.unpack(res.x, nf, ns, ref)
    th, ts = np.asarray(theta), np.asarray(theta_star)
    corr = float(np.corrcoef(th, ts)[0, 1]) if np.std(th) > 0 and np.std(ts) > 0 else 0.0
    gn = float(np.linalg.norm(res.jac))
    return {"seed_junc": seed_junc, "censor_frac": censor_frac, "ridge": ridge,
            "maxiter": maxiter, "maxls": maxls, "gtol": gtol,
            "success": bool(res.success), "message": str(res.message),
            "nit": int(res.nit), "final_grad_norm": gn,
            "theta_corr": corr,
            "theta_scale": float(np.sum(th * ts) / max(np.sum(ts * ts), 1e-12)),
            "converged": gn < 1e-2}


def main():
    out = {"seeds": [], "summary": {}}
    # 50%-censor, single ridge=5, several junction seeds -> recovery distribution
    rows = []
    for sj in range(40, 50):
        rows.append(run(sj, 0.5, 5.0, 4000, 50, 1e-9))
    corrs = [r["theta_corr"] for r in rows]
    conv = [r["converged"] for r in rows]
    out["seeds"] = rows
    out["summary"] = {
        "n_seeds": len(rows),
        "theta_corr_min": min(corrs), "theta_corr_max": max(corrs),
        "theta_corr_median": float(np.median(corrs)),
        "theta_corr_mean": float(np.mean(corrs)),
        "n_converged": sum(conv),
        "n_line_search_fail": sum(1 for r in rows if not r["converged"]),
        "n_theta_corr_gt_05": sum(1 for c in corrs if c > 0.5),
        "note": "50% censoring is a low-information stress test (real data ~16%). "
                "Single-draw recovery is seed-sensitive; reporting the full "
                "distribution so G4 is evaluated honestly.",
    }
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
