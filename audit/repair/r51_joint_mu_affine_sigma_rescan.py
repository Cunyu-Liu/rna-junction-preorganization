"""r51: JOINT measured-only affine mu correction + sigma RE-SCAN on corrected mu.

MOTIVATION (the gap r46/r47 left open):
  - r45 (frozen, 0.7907) re-scans sigma per scaffold x stratum but NEVER
    corrects mu.  The residual diagnostic shows the largest remaining error is
    MEASURED-layer systematic bias: scaf9 measured bias -0.996, scaf1 -0.445,
    overall measured-resid mean -0.106, measured-resid RMSE 0.6116.
  - r46/r47 tried measured-only mu correction BUT they scanned sigma_m on the
    UNCORRECTED mu (stale sigma) and used the OLD grid floor 0.4.  Because the
    pooled NLL at the corrected mu is by construction <= NLL at any fixed stale
    sigma, re-scanning sigma on the CORRECTED mu can only improve (or tie) the
    stale-sigma result.  This joint lever has never been tested.

r51 = r45 stratum sigma + measured-only affine mu correction (a + b*mu), with
sigma_m RE-SCANNED on the corrected-mu measured rows, all under the corrected
grid floor 0.05 (MetricSpec floor), LOO on the OTHER folds' OOF rows (no test
label leakage).  Censored rows keep mu EXACTLY and keep sigma_c (r45), so the
censored-side survival likelihood is never damaged by the mu correction.

Variants (mu-side):
  global_affine   : single (a,b) fit on ALL measured rows of other folds
  per_scaf_affine : per-operator (a_s, b_s)
  per_scaf_ridge  : per-operator slope ridge-shrunk toward 1.0
  per_scaf_eb     : per-operator slope empirical-Bayes shrunk toward global

Estimands (pooled-OOF junction-macro right-censored Gaussian NLL):
  - frozen 0.7
  - per-scaf sigma (r38)
  - per-scaf x stratum sigma (r45, corrected grid floor 0.05)
  - r51 joint (this run; corrected grid floor 0.05)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.shootout_run import _eligible_keys

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
R33 = f"{R}/r33_xgboost_full/Predictions_v3.jsonl"
R34 = f"{R}/r34_gbdt_seeds_full/Predictions_v3.jsonl"
R35 = f"{R}/r35_gbdt_hp_full/Predictions_v3.jsonl"
R24 = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"

R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R34_LEDGER = f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet"
R35_LEDGER = f"{R}/r35_gbdt_hp_full/ConvergenceLedger_v3.parquet"
R24_LEDGERS = [
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
]

XGB = "xgboost_censored_hybrid"
XGB_S99 = "xgboost_censored_hybrid_s99"
XGB_S2026 = "xgboost_censored_hybrid_s2026"
XGB_LR03 = "xgboost_censored_hybrid_hp_lr03"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUIS = "motif_topology_hierarchy"
ALL_MEMBERS = [XGB, XGB_LR03, XGB_S99, XGB_S2026, T7, T7_S99, T7_S2026]

GRID = np.arange(0.05, 1.6, 0.01)  # corrected r45 grid (MetricSpec floor 0.05)


def _load(path):
    return [json.loads(l) for l in open(path)]


def _elig(paths):
    import pandas as pd
    frames = [pd.read_parquet(p) for p in paths]
    conv = [dict(r) for r in pd.concat(frames, ignore_index=True).to_dict("records")]
    return _eligible_keys(conv)


def _by_rid(rows, model_id, eligible):
    out = {}
    for p in rows:
        if p["model_id"] == model_id and (model_id, p["fold"]) in eligible \
                and p["support"] and not p["abstain"]:
            out[p["source_row_id"]] = p
    return out


def _pooled(ens):
    jd = defaultdict(list)
    for rid, p in ens.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


def _pooled_strata(ens):
    jd_m = defaultdict(list)
    jd_c = defaultdict(list)
    for rid, p in ens.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        (jd_c if p["cens"] else jd_m)[p["jid"]].append(nll)
    out = {}
    for name, d in (("measured", jd_m), ("censored", jd_c)):
        out[name] = float(np.mean([np.mean(v) for v in d.values()])) if d else None
    return out


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
    for _ in range(1000):
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


def _scan_sigma(rows, cens_mask=None, grid=None):
    """Vectorized junction-macro NLL scan over sigma (corrected grid by default)."""
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    if cens_mask is None:
        sel = np.ones(len(y), dtype=bool)
    else:
        sel = cens == cens_mask
    if not sel.any():
        return None, np.inf
    y, cens, mu, jid = y[sel], cens[sel], mu[sel], jid[sel]
    uniq, jcode = np.unique(jid, return_inverse=True)
    jcounts = np.bincount(jcode, minlength=len(uniq))
    grid = grid if grid is not None else GRID
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
        jm = sums / jcounts
        nll = float(np.mean(jm[jcounts > 0]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _ols(x, y):
    """y ~ a + b*x, returns (a, b)."""
    A = np.vstack([np.ones(len(x)), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _ridge_ols(x, y, lam=1.0, b_prior=1.0):
    n = len(x)
    A = np.vstack([np.ones(n), x]).T
    A_aug = np.vstack([A, [[0.0, np.sqrt(lam)]]])
    y_aug = np.hstack([np.asarray(y), [b_prior * np.sqrt(lam)]])
    beta, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    return float(beta[0]), float(beta[1])


def _calibrate_r51(ens, folds, mode="global_affine", ridge_lam=0.0,
                   eb_kappa=20.0, min_rows=15, min_meas=10):
    """Joint LOO calibration: affine mu correction on measured rows + sigma_m
    RE-SCANNED on the corrected-mu measured rows (corrected grid floor 0.05).

    mode:
      "global_affine":   single (a,b) for all measured rows, shared across scaf
      "per_scaf_affine": per-scaffold (a_s, b_s) on measured rows only
      "per_scaf_ridge":  per-scaffold ridge-regressed toward b=1 (lam=ridge_lam)
      "per_scaf_eb":     per-scaffold EB shrink of b_s toward global slope

    Censored rows: mu unchanged, sigma_c from r45 (scan on censored rows of
    OTHER folds).  Measured rows: mu_cal = a + b*mu, sigma_m scanned on the
    corrected mu of OTHER folds' measured rows.
    """
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    grid = GRID
    cal = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        sm_global, _ = _scan_sigma(other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma(other, cens_mask=True, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p

        # global affine on ALL measured rows of OTHER folds (also EB prior)
        all_meas = [p for p in other.values() if not p["cens"]]
        if len(all_meas) >= min_meas:
            gx = np.asarray([p["mu"] for p in all_meas])
            gy = np.asarray([p["y"] for p in all_meas])
            a_g, b_g = _ols(gx, gy)
        else:
            a_g, b_g = 0.0, 1.0

        # per-scaf affine (a_s, b_s) fit on measured rows of OTHER folds
        affine = {}
        for sc, rows_sc in by_scaf.items():
            meas = [p for p in rows_sc.values() if not p["cens"]]
            if len(meas) < min_meas:
                continue
            mx = np.asarray([p["mu"] for p in meas])
            my = np.asarray([p["y"] for p in meas])
            if mode == "global_affine":
                affine[sc] = (a_g, b_g)
            elif mode == "per_scaf_affine":
                affine[sc] = _ols(mx, my)
            elif mode == "per_scaf_ridge":
                affine[sc] = _ridge_ols(mx, my, lam=ridge_lam)
            elif mode == "per_scaf_eb":
                a_raw, b_raw = _ols(mx, my)
                n_m = len(meas)
                lam_s = n_m / (n_m + eb_kappa)
                b = lam_s * b_raw + (1.0 - lam_s) * b_g
                a = float(np.mean(my)) - b * float(np.mean(mx))
                affine[sc] = (a, b)

        # Build the corrected-mu rows of OTHER folds for sigma_m re-scan
        corr_meas_by_scaf = defaultdict(dict)
        corr_meas_all = {}
        for rid, p in other.items():
            if p["cens"]:
                continue
            sc = int(p["scaf"])
            a, b = affine.get(sc, (0.0, 1.0))
            mu_c = a + b * p["mu"]
            row_c = {**p, "mu": float(mu_c)}
            corr_meas_by_scaf[sc][rid] = row_c
            corr_meas_all[rid] = row_c

        # sigma_m re-scan on corrected mu: per-scaf with fallback to global
        strat_sigma = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {}
            corr_rows = corr_meas_by_scaf.get(sc, {})
            if len(corr_rows) >= min_rows:
                sm, _ = _scan_sigma(corr_rows, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                sm_glob_corr, _ = _scan_sigma(corr_meas_all, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm_glob_corr if sm_glob_corr is not None else s_global
            if n_c >= min_rows:
                sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry

        # Apply to the held-out fold
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            if p["cens"]:
                sig = e.get("sigma_c", s_global)
                mu_new = p["mu"]
            else:
                sig = e.get("sigma_m", s_global)
                a, b = affine.get(sc, (0.0, 1.0))
                mu_new = a + b * p["mu"]
            cal[rid] = {**p, "mu": float(mu_new), "sigma": sig}
        fit_log[f] = {
            "stratum_sigma": {
                str(k): {"sigma_m": round(v["sigma_m"], 3),
                         "sigma_c": round(v["sigma_c"], 3)}
                for k, v in sorted(strat_sigma.items())},
            "affine": {str(k): {"a": round(v[0], 4), "b": round(v[1], 4)}
                       for k, v in sorted(affine.items())},
            "global_affine": {"a": round(a_g, 4), "b": round(b_g, 4)},
            "global_fallback": round(float(s_global), 3),
        }
    return cal, fit_log


def _calibrate_r45(ens, folds):
    """Corrected r45 reference (grid floor 0.05), no mu change."""
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    grid = GRID
    cal = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        sm_global, _ = _scan_sigma(other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma(other, cens_mask=True, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        strat_sigma = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {}
            if n - n_c >= min(15, n):
                sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if n_c >= 15:
                sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal[rid] = {**p, "sigma": sig}
    return cal


def main():
    print("Loading predictions...", file=sys.stderr)
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
    nuis = _by_rid(rows33, NUIS, elig33)

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ens = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS]))}

    folds = sorted(set(ens[r]["fold"] for r in ens))
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                for r in nuis}), 4),
        "equal_weight_7mem_nll_frozen07": round(
            _pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
        "n_folds": len(folds),
        "n_rows": len(ens),
        "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01,
                       "note": "MetricSpec floor (corrected r45 grid)"},
    }

    # r45 reference (corrected grid floor 0.05)
    cal_r45 = _calibrate_r45(ens, folds)
    out["per_scaf_stratum_sigma_loo_nll"] = round(_pooled(cal_r45), 4)
    out["strata_nll_r45"] = _pooled_strata(cal_r45)

    # r51 joint variants
    for mode in ("global_affine", "per_scaf_affine", "per_scaf_ridge",
                 "per_scaf_eb"):
        key = f"r51_{mode}"
        if mode == "per_scaf_ridge":
            for lam in (0.5, 1.0, 2.0, 5.0):
                sk = f"{key}_lam{lam:g}"
                cal, fl = _calibrate_r51(ens, folds, mode=mode, ridge_lam=lam)
                out[sk] = round(_pooled(cal), 4)
                out[f"{sk}_strata"] = _pooled_strata(cal)
                out[f"{sk}_fit_log"] = fl
        elif mode == "per_scaf_eb":
            for kappa in (5.0, 10.0, 20.0, 50.0):
                sk = f"{key}_kappa{kappa:g}"
                cal, fl = _calibrate_r51(ens, folds, mode=mode, eb_kappa=kappa)
                out[sk] = round(_pooled(cal), 4)
                out[f"{sk}_strata"] = _pooled_strata(cal)
                out[f"{sk}_fit_log"] = fl
        else:
            cal, fl = _calibrate_r51(ens, folds, mode=mode)
            out[key] = round(_pooled(cal), 4)
            out[f"{key}_strata"] = _pooled_strata(cal)
            out[f"{key}_fit_log"] = fl

    r45_nll = out["per_scaf_stratum_sigma_loo_nll"]
    best_key = min([k for k in out if k.startswith("r51_")
                    and isinstance(out[k], (int, float))],
                   key=lambda k: out[k])
    out["best_r51_nll"] = out[best_key]
    out["best_r51_key"] = best_key
    out["best_r51_vs_r45_delta"] = round(out[best_key] - r45_nll, 4)

    # Edit-cluster CI for the best r51 variant vs nuisance (r45-calibrated)
    if "ridge" in best_key:
        best_mode = "per_scaf_ridge"
        best_lam = float(best_key.split("lam")[1])
        cal_best, _ = _calibrate_r51(ens, folds, mode=best_mode, ridge_lam=best_lam)
    elif "eb" in best_key:
        best_mode = "per_scaf_eb"
        best_kappa = float(best_key.split("kappa")[1])
        cal_best, _ = _calibrate_r51(ens, folds, mode=best_mode, eb_kappa=best_kappa)
    else:
        best_mode = best_key.replace("r51_", "")
        cal_best, _ = _calibrate_r51(ens, folds, mode=best_mode)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_best_r51_vs_nuisance"] = _edit_ci(cal_best, cal_nuis)

    out["note"] = (
        "r51 lever: JOINT measured-only affine mu correction + sigma_m RE-SCAN "
        "on the corrected mu, under the corrected grid floor 0.05, LOO.  This "
        "closes the gap left by r46/r47, which corrected mu but kept the sigma "
        "scanned on the UNCORRECTED mu (stale sigma) and used the old grid "
        "floor 0.4.  Censored rows keep mu exactly and keep sigma_c (r45), so "
        "the censored-side survival likelihood is never damaged.  All applied "
        "LOO on the OTHER folds' OOF rows (no test-label leakage)."
    )
    Path(f"{R}/r51_joint_mu_affine_sigma_rescan.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
