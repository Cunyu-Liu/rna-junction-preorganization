"""r76: nonlinear monotone (isotonic / polynomial) mu calibration - last untested mu lever.

r46/r47/r51 tested only LINEAR affine mu correction (global/per-scaf EB slope).
r56b added per-context constant EB.  The r62-corrected residual-vs-mu plot shows
mild NONLINEARITY: high-mu bin (mu ~ -7.9..-6.7) mean residual -0.166, low-mu
bins ~ +0.03, overall slope -0.042.  A nonlinear monotone transform of the
ensemble mu (isotonic regression / low-order polynomial), fit LOO on OTHER
folds and applied to the held-out fold, is genuinely untested.

Design (honest, LOO, train-legal):
  - mu = r62-corrected mu (frozen).
  - For each held-out fold: on OTHER folds' measured rows, fit a monotone
    calibration f(mu) of the form:
      (A) isotonic regression on (mu, y) then map mu -> fitted value
      (B) cubic polynomial  y ~ poly(mu, 3)  (constrained monotone-ish)
    applied to measured rows only; censored rows keep r62 mu.
  - Sigma re-scanned on the corrected mu (decoupled, per-scaf x stratum) as in r62.
  - Compare pooled NLL vs r62 (0.7243) + edit-cluster CI.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _pooled, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)
from audit.repair.r62_decoupled_frozen import _calibrate_r62

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def _edit_ci(ens_dict, base_dict):
    jid_edit = {}
    for rid, p in ens_dict.items():
        jid_edit.setdefault(p["jid"], str(p["fold"]).split(":", 1)[1])
    jid_d = defaultdict(list)
    for rid, p in ens_dict.items():
        if rid not in base_dict:
            continue
        nll_e = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        nll_b = float(row_nll([base_dict[rid]["y"]], [base_dict[rid]["cens"]],
                              [base_dict[rid]["mu"]], [base_dict[rid]["sigma"]])[0])
        jid_d[p["jid"]].append(nll_b - nll_e)
    by_edit = defaultdict(list)
    for j, vals in jid_d.items():
        by_edit[jid_edit.get(j, "?")].append(float(np.mean(vals)))
    rng = np.random.default_rng(17)
    boots = []
    for _ in range(2000):
        ch = rng.choice(list(by_edit), size=len(by_edit), replace=True)
        boots.append(float(np.mean([v for e in ch for v in by_edit[e]])))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for g in keep for x in g]))
    return {"ci": [round(lo, 4), round(hi, 4)], "ci_lower_gt_0": bool(lo > 0),
            "leave_one_largest": round(leave1, 4), "largest": largest,
            "n_edit": len(by_edit)}


def main():
    elig33 = _elig([R33_LEDGER])
    elig34 = _elig([R34_LEDGER])
    elig35 = _elig([R35_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows24 = _load(R24)
    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[T7] = _by_rid(rows24, T7, elig24)
    members[T7_S99] = _by_rid(rows24, T7_S99, elig24)
    members[T7_S2026] = _by_rid(rows24, T7_S2026, elig24)
    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ref = members[ALL_MEMBERS[0]]
    ens = {}
    for rid in common:
        p0 = ref[rid]
        gmu = float(np.mean([members[m][rid]["mu"] for m in GBDT]))
        mmu = float(np.mean([members[m][rid]["mu"] for m in MLP]))
        ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": int(p0["scaf"]),
                    "context": str(p0.get("context", "?")), "y": p0["y"],
                    "cens": p0["cens"], "mu": 0.5 * gmu + 0.5 * mmu}
    folds = sorted(set(ens[r]["fold"] for r in ens))

    cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)
    print("r62 baseline =", round(_pooled(cal62), 4))

    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        by_fold[p["fold"]][rid] = {**p}

    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import minimize_scalar
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma

    def _fit_poly(f, deg, ridge=0.0):
        mu = np.asarray([p["mu"] for p in f], dtype=float)
        y = np.asarray([p["y"] for p in f], dtype=float)
        X = np.vander(mu, deg + 1)
        if ridge > 0:
            A = X.T @ X + ridge * np.eye(deg + 1)
            b = X.T @ y
            coef = np.linalg.solve(A, b)
        else:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        def apply(mu_arr):
            return np.polynomial.polynomial.polyval(mu_arr, coef[::-1])
        return apply

    def _fit_iso(f):
        from sklearn.isotonic import IsotonicRegression
        mu = np.asarray([p["mu"] for p in f], dtype=float)
        y = np.asarray([p["y"] for p in f], dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip").fit(mu, y)
        return lambda mu_arr: iso.predict(mu_arr)

    def _recal(f, apply_fn):
        """Apply nonlinear mu transform + decoupled sigma re-scan (r62 Stage 2)."""
        # corrected mu for all rows
        out = {}
        for rid, p in f.items():
            if p["cens"]:
                out[rid] = {**p}
            else:
                out[rid] = {**p, "mu": float(apply_fn(np.asarray([p["mu"]]))[0])}
        return out

    def _sigma_rescan(cal_rows, folds):
        """Decoupled per-scaf x stratum sigma on the corrected mu (r62 Stage 2)."""
        by_fold = defaultdict(dict)
        for rid, p in cal_rows.items():
            by_fold[p["fold"]][rid] = p
        cal = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            s_global, _ = _scan_sigma(other, grid=GRID)
            sm_global, _ = _scan_sigma(other, cens_mask=False, grid=GRID)
            sc_global, _ = _scan_sigma(other, cens_mask=True, grid=GRID)
            by_scaf = defaultdict(dict)
            for rid, p in other.items():
                by_scaf[int(p["scaf"])][rid] = p
            strat_sigma = {}
            for sc, rows_sc in by_scaf.items():
                n = len(rows_sc)
                n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
                entry = {}
                if n - n_c >= 15:
                    entry["sigma_m"] = _scan_sigma(rows_sc, cens_mask=False, grid=GRID)[0]
                else:
                    entry["sigma_m"] = sm_global if sm_global is not None else s_global
                if n_c >= 15:
                    entry["sigma_c"] = _scan_sigma(rows_sc, cens_mask=True, grid=GRID)[0]
                else:
                    entry["sigma_c"] = sc_global if sc_global is not None else s_global
                strat_sigma[sc] = entry
            for rid, p in by_fold[f].items():
                sc = int(p["scaf"])
                e = strat_sigma.get(sc, {})
                sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
                cal[rid] = {**p, "sigma": float(sig)}
        return cal

    results = {}
    for name, deg, ridge in (("poly2", 2, 1.0), ("poly3", 3, 5.0),
                             ("poly3r10", 3, 10.0), ("iso", None, 0.0)):
        cal_all = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            fit_rows = [p for p in other.values() if not p["cens"]]
            if len(fit_rows) < 50:
                cal_all.update(by_fold[f])
                continue
            if name == "iso":
                apply = _fit_iso(fit_rows)
            else:
                apply = _fit_poly(fit_rows, deg, ridge=ridge)
            # apply to the held-out fold (corrected mu)
            corr = _recal(by_fold[f], apply)
            cal_all.update(corr)
        # decoupled sigma re-scan
        cal_final = _sigma_rescan(cal_all, folds)
        nll = _pooled(cal_final)
        results[name] = round(nll, 4)
        print(f"[{name}] nonlinear mu + sigma rescan: {nll:.4f} "
              f"(delta vs r62 {nll-0.7243:+.4f})")
        if name == "iso":
            ci = _edit_ci(cal_final, cal62)
            print(f"  edit CI vs r62 = {ci['ci']} lower_gt_0={ci['ci_lower_gt_0']} "
                  f"leave1={ci['leave_one_largest']}")

    out = {"r62": round(_pooled(cal62), 4), "results": results,
           "note": "nonlinear monotone mu calibration (isotonic / poly2-3) + decoupled sigma rescan"}
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r76_nonlinear_mu_calibration.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
