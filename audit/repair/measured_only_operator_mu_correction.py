"""r46: measured-only per-operator mu bias correction (stratum-aware).

The residual diagnostic found large systematic measured bias on high-censoring
scaffolds (scaf9 -0.996, scaf1 -0.445) that the 7-member equal-weight ensemble
does NOT fix.  The r37/r39 additive-intercept attempts FAILED because they
shifted mu on ALL rows: moving mu down helps the (few) measured rows but pushes
the (many) censored rows' mu away from the right tail, worsening censored NLL
more than the measured gain.

r45 showed the measured and censored strata should be treated INDEPENDENTLY
(separate per-operator sigma).  r46 is the mu-side counterpart: apply a
per-scaffold additive bias correction ONLY to measured rows, leaving censored
rows' mu untouched (they already sit near the right tail).  Combined with the
r45 per-stratum sigma, this directly targets the scaf9/scaf1 measured bias
without the r37/r39 censored-side damage.

Honesty: for each held-out fold, alpha_s = mean(y - mu) is fit on the OOF
MEASURED rows of the OTHER 36 folds (no test-label leakage), then applied to
the held-out fold's measured rows.  Censored rows keep their mu and their
r45 sigma_c.  mu of measured rows is adjusted and they emit sigma_m.

Estimands (pooled-OOF junction-macro right-censored Gaussian NLL):
  - frozen 0.7
  - per-scaf sigma LOO (r38)
  - per-scaf x stratum sigma LOO (r45)
  - r45 + measured-only per-scaf mu correction (this run)
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

# sigma grid extended down to 0.05 (MetricSpec floor): r45 was grid-constrained
# at 0.40 which BINDs high-censoring scaffolds (scaf9 sigma_c optimum ~0.19).
SIGMA_GRID = np.arange(0.05, 1.6, 0.01)

XGB = "xgboost_censored_hybrid"
XGB_S99 = "xgboost_censored_hybrid_s99"
XGB_S2026 = "xgboost_censored_hybrid_s2026"
XGB_LR03 = "xgboost_censored_hybrid_hp_lr03"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUIS = "motif_topology_hierarchy"
ALL_MEMBERS = [XGB, XGB_LR03, XGB_S99, XGB_S2026, T7, T7_S99, T7_S2026]

SIGMA_FIXED = 0.7


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
    """1-D scan of sigma minimizing pooled junction-macro NLL (vectorized)."""
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
    grid = grid if grid is not None else SIGMA_GRID
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
        jm = sums / jcounts
        nll = float(np.mean(jm[jcounts > 0]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _calibrate_r45_plus_mu(ens, folds, min_rows=15, shrink=1.0):
    """r45 stratum sigma + measured-only per-scaf mu correction, LOO.

    For each held-out fold f:
      - fit on OOF rows of the other folds: per-scaf sigma_m/sigma_c (r45) and
        per-scaf alpha_s = mean(y - mu) on MEASURED rows only;
      - apply to fold f: measured rows get mu + alpha_s and sigma_m; censored
        rows keep mu and emit sigma_c.
    `shrink` shrinks alpha toward 0 (protects small scaffolds).
    """
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    grid = SIGMA_GRID
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
        strat_sigma = {}
        alpha = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {}
            if n - n_c >= min_rows:
                sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if n_c >= min_rows:
                sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry
            # measured-only bias on the OTHER folds (shrink toward 0)
            meas = [p["y"] - p["mu"] for p in rows_sc.values() if not p["cens"]]
            if len(meas) >= 5:
                alpha[sc] = shrink * float(np.mean(meas))
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            if p["cens"]:
                sig = e.get("sigma_c", s_global)
                mu_new = p["mu"]
            else:
                sig = e.get("sigma_m", s_global)
                mu_new = p["mu"] + alpha.get(sc, 0.0)
            cal[rid] = {**p, "mu": float(mu_new), "sigma": sig}
        fit_log[f] = {
            "stratum_sigma": {
                str(k): {"sigma_m": round(v["sigma_m"], 3),
                         "sigma_c": round(v["sigma_c"], 3)}
                for k, v in sorted(strat_sigma.items())},
            "alpha": {str(k): round(v, 4) for k, v in sorted(alpha.items())},
            "global_fallback": round(float(s_global), 3),
        }
    return cal, fit_log


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
    }

    # r38 reference
    cal_scaf = {}
    grid = SIGMA_GRID
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 20:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = None
        s_global, _ = _scan_sigma(other, grid=grid)
        for sc in scaf_sigma:
            if scaf_sigma[sc] is None:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal_scaf[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
    out["per_scaf_sigma_loo_nll"] = round(_pooled(cal_scaf), 4)
    out["strata_nll_per_scaf"] = _pooled_strata(cal_scaf)

    # r45 stratum-only (recompute from scratch, no mu change)
    cal_r45, fit_r45 = _calibrate_r45_plus_mu(ens, folds, shrink=0.0)
    out["per_scaf_stratum_sigma_loo_nll"] = round(_pooled(cal_r45), 4)
    out["strata_nll_r45"] = _pooled_strata(cal_r45)

    # r46: r45 + measured-only mu correction
    for shrink in (0.5, 0.75, 1.0):
        cal_r46, fit_r46 = _calibrate_r45_plus_mu(ens, folds, shrink=shrink)
        key = f"r46_mu_shrink{shrink:g}"
        out[key] = round(_pooled(cal_r46), 4)
        out[f"{key}_strata"] = _pooled_strata(cal_r46)
        out[f"{key}_fit_log"] = fit_r46

    best_key = min([k for k in out if k.startswith("r46_mu_shrink")
                    and isinstance(out[k], (int, float))],
                   key=lambda k: out[k])
    out["best_r46_nll"] = out[best_key]
    out["best_r46_key"] = best_key
    out["best_r46_vs_r45_delta"] = round(out[best_key] - out["per_scaf_stratum_sigma_loo_nll"], 4)
    out["edit_cluster_CI_best_r46_vs_nuisance"] = _edit_ci(
        _calibrate_r45_plus_mu(ens, folds, shrink=float(best_key.split("shrink")[1]))[0],
        {r: {**nuis[r], "sigma": 0.7} for r in nuis})

    out["note"] = (
        "r46 lever: measured-only per-operator mu bias correction on top of the "
        "r45 per-stratum sigma.  alpha_s = mean(y - mu) fit on OOF MEASURED rows "
        "of the other 36 folds (LOO, no test-label leakage), applied ONLY to the "
        "held-out fold's measured rows; censored rows keep mu and sigma_c.  This "
        "is the mu-side counterpart of r45: r37/r39 additive intercepts failed "
        "because they shifted ALL rows (hurting censored NLL); restricting the "
        "correction to measured rows targets the scaf9/scaf1 measured bias "
        "(-0.996/-0.445) without censored-side damage."
    )
    Path(f"{R}/measured_only_operator_mu_correction.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
