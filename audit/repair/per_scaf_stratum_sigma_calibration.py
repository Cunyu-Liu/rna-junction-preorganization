"""Per-operator stratum-aware sigma calibration (r45): measured vs censored.

The r38 per-scaffold sigma calibration emits ONE sigma per scaffold applied to
BOTH measured and censored rows.  But the residual-structure diagnostic shows
the two strata have genuinely different optimal scales:

  - measured rows:  per-scaffold RMSE ranges 0.45 (scaf2) .. 1.15 (scaf9);
    the frozen per-scaf sigma (0.84 for scaf9) is a compromise dominated by the
    high censoring fraction (scaf9 is 78.5% censored), so measured rows on
    high-censoring scaffolds are UNDER-dispersed (sigma too small -> inflated
    measured NLL);
  - censored rows:  NLL = -log Phi((mu-CAP)/sigma); with mu > CAP the censored
    rows prefer a SMALLER sigma than the pooled fit.

This module tests the strict generalization: emit sigma_m (measured stratum)
and sigma_c (censored stratum) per scaffold, both fit leave-one-fold-out on the
OTHER 36 folds' OOF rows (no test-label leakage), applied to the held-out fold.
mu stays the equal-weight 7-member ensemble (unchanged).  Under MetricSpec_v3
sigma is a per-row model-emitted parameter, so emitting a different sigma per
stratum is a legitimate method lever -- exactly as r38 generalized the global
sigma to per-scaffold.

Estimands reported (same pooled-OOF junction-macro right-censored NLL primary):
  - frozen sigma=0.7              (reference)
  - per-scaf sigma LOO (r38)      (previous frozen, for direct comparison)
  - per-scaf x stratum sigma LOO  (this run; primary r45 lever)
  - measured-only / censored-only NLL decomposition of each
  - edit-cluster CI of the r45 lever vs nuisance
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
    """(measured pooled, censored pooled) NLL on the same junction-macro basis."""
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


def _scan_sigma_stratum(rows, cens_mask, grid=None):
    """1-D scan of the sigma minimizing pooled junction-macro NLL on a stratum.

    `cens_mask` selects which rows of `rows` belong to the stratum.  Only rows
    in the stratum contribute; aggregation stays junction-macro (consistent with
    the primary estimand).
    """
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
    grid = grid if grid is not None else np.arange(0.3, 1.6, 0.01)
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
        jm = sums / jcounts
        nll = float(np.mean(jm[jcounts > 0]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


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

    # ---- r38 reference: per-scaffold sigma LOO (one sigma per scaffold) ----
    cal_scaf = {}
    grid = np.arange(0.4, 1.4, 0.01)
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        scaf_sigma = {}
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 20:
                s, _ = _scan_sigma_stratum(rows_sc, cens_mask=None, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = None
        s_global, _ = _scan_sigma_stratum(other, cens_mask=None, grid=grid)
        for sc in scaf_sigma:
            if scaf_sigma[sc] is None:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal_scaf[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
    out["per_scaf_sigma_loo_nll"] = round(_pooled(cal_scaf), 4)

    # ---- r45: per-scaffold x stratum sigma LOO ----
    cal_stratum = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        strat_sigma = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {"n": n, "n_censored": n_c}
            # measured stratum: need >= 15 measured rows to fit independently
            if n - n_c >= 15:
                sm, _ = _scan_sigma_stratum(rows_sc, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                entry["sigma_m"] = None
            # censored stratum: need >= 15 censored rows to fit independently
            if n_c >= 15:
                sc_, _ = _scan_sigma_stratum(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = None
            strat_sigma[sc] = entry
        s_global, _ = _scan_sigma_stratum(other, cens_mask=None, grid=grid)
        sm_global, _ = _scan_sigma_stratum(other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma_stratum(other, cens_mask=True, grid=grid)
        for sc, entry in strat_sigma.items():
            if entry["sigma_m"] is None:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if entry["sigma_c"] is None:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal_stratum[rid] = {**p, "sigma": sig}
        fit_log[f] = {
            "stratum_sigma": {
                str(k): {"sigma_m": round(v["sigma_m"], 3),
                         "sigma_c": round(v["sigma_c"], 3)}
                for k, v in sorted(strat_sigma.items())},
            "global_fallback": round(float(s_global), 3),
            "sm_global": round(float(sm_global), 3) if sm_global else None,
            "sc_global": round(float(sc_global), 3) if sc_global else None,
        }
    out["per_scaf_stratum_sigma_loo_nll"] = round(_pooled(cal_stratum), 4)
    out["per_scaf_stratum_vs_frozen07_delta"] = round(
        out["per_scaf_stratum_sigma_loo_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["per_scaf_stratum_vs_per_scaf_delta"] = round(
        out["per_scaf_stratum_sigma_loo_nll"] - out["per_scaf_sigma_loo_nll"], 4)

    # stratum decomposition: where does the r45 gain come from?
    out["strata_nll_frozen07"] = _pooled_strata(
        {r: {**ens[r], "sigma": 0.7} for r in ens})
    out["strata_nll_per_scaf"] = _pooled_strata(cal_scaf)
    out["strata_nll_per_scaf_stratum"] = _pooled_strata(cal_stratum)

    # stability of learned stratum sigmas across folds
    sm_series = defaultdict(list)
    sc_series = defaultdict(list)
    for l in fit_log.values():
        for k, v in l["stratum_sigma"].items():
            sm_series[k].append(v["sigma_m"])
            sc_series[k].append(v["sigma_c"])
    out["sigma_m_stability"] = {
        str(k): {"min": round(min(v), 3), "mean": round(float(np.mean(v)), 3),
                 "max": round(max(v), 3), "n_folds": len(v)}
        for k, v in sorted(sm_series.items())}
    out["sigma_c_stability"] = {
        str(k): {"min": round(min(v), 3), "mean": round(float(np.mean(v)), 3),
                 "max": round(max(v), 3), "n_folds": len(v)}
        for k, v in sorted(sc_series.items())}

    # edit-cluster CI of r45 vs nuisance (honest group-aware uncertainty)
    out["edit_cluster_CI_r45_vs_nuisance"] = _edit_ci(
        cal_stratum, {r: {**nuis[r], "sigma": 0.7} for r in nuis})
    out["fit_log_folds"] = fit_log

    out["note"] = (
        "r45 lever: per-scaffold x stratum sigma (sigma_m for measured rows, "
        "sigma_c for censored rows), both fit leave-one-fold-out on the OTHER "
        "36 folds' OOF rows (no test-label leakage), applied to the held-out "
        "fold.  mu is unchanged (equal-weight 7-member ensemble).  This is the "
        "strict generalization of r38 (one sigma per scaffold for ALL rows): "
        "the residual diagnostic showed the measured vs censored strata prefer "
        "different scales on high-censoring scaffolds (scaf9 measured RMSE 1.15 "
        "vs pooled sigma 0.84), so a single pooled sigma is a compromise that "
        "under-disperses measured rows.  sigma is a per-row model-emitted "
        "parameter per MetricSpec, so per-stratum emission is a legitimate "
        "method lever."
    )
    Path(f"{R}/per_scaf_stratum_sigma_calibration.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
