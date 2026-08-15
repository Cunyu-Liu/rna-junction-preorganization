"""Fair horizontal table for the r45 per-scaffold x stratum sigma calibration.

The r45 lever emits per-scaffold sigma separately for the measured (sigma_m)
and censored (sigma_c) strata, both fit leave-one-fold-out on the OTHER folds'
OOF rows and applied to the held-out fold.  For a fair horizontal contrast the
SAME procedure must be applied to EVERY comparator (nuisance, each single
member, the 3x t7 ensemble, and the 7-member ensemble).  Otherwise calibrating
only the winning ensemble would bias the comparison.

Estimands (MetricSpec_v3 pooled-junction-macro right-censored Gaussian NLL):
  - frozen sigma=0.7             : r24/r34/r35 freeze (all models emit 0.7)
  - per-scaffold sigma LOO (r38) : previous frozen, for direct comparison
  - per-scaffold x stratum LOO   : r45 lever applied to every comparator
Relative gain and edit-cluster CI are computed against the nuisance baseline
at the SAME sigma treatment, so the horizontal table is apples-to-apples.
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
GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


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
    """1-D scan of the sigma minimizing pooled junction-macro NLL (vectorized).

    cens_mask=None scans ALL rows; True/False restrict to that stratum.
    Junction-macro aggregation uses np.bincount (vectorized, no Python loop
    over grid points), so a full 37-fold LOO calibration runs in seconds.
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
    # map junction ids to dense ints for bincount
    uniq, jcode = np.unique(jid, return_inverse=True)
    jcounts = np.bincount(jcode, minlength=len(uniq))
    grid = grid if grid is not None else np.arange(0.4, 1.4, 0.01)
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
        jm = sums / jcounts
        nll = float(np.mean(jm[jcounts > 0]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _per_scaf_calibrate(preds, folds, min_rows=20):
    """LOO per-scaffold sigma (r38). Returns (calibrated, sigma_map)."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
    cal = {}
    sigma_map = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= min_rows:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
        sigma_map[f] = scaf_sigma
    return cal, sigma_map


def _per_scaf_stratum_calibrate(preds, folds, min_rows=15):
    """LOO per-scaffold x stratum sigma (r45). Returns (calibrated, sigma_map)."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
    cal = {}
    sigma_map = {}
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
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal[rid] = {**p, "sigma": sig}
        sigma_map[f] = strat_sigma
    return cal, sigma_map


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
    ens7 = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens7[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                     "y": ref["y"], "cens": ref["cens"],
                     "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS]))}
    ens3 = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens3[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                     "y": ref["y"], "cens": ref["cens"],
                     "mu": float(np.mean([members[m][rid]["mu"] for m in MLP]))}

    models = {
        "nuisance": nuis,
        "t7_s99": members[T7_S99],
        "xgb_lr03": members[XGB_LR03],
        "3x_t7_ensemble": ens3,
        "7mem_mixed_ensemble": ens7,
    }

    frozen = {}
    for name, preds in models.items():
        frozen[name] = round(_pooled({r: {**p, "sigma": 0.7} for r, p in preds.items()}), 4)

    calib_scaf = {}
    for name, preds in models.items():
        folds = sorted(set(preds[r]["fold"] for r in preds))
        cal, _ = _per_scaf_calibrate(preds, folds)
        calib_scaf[name] = round(_pooled(cal), 4)

    calib_strat = {}
    for name, preds in models.items():
        folds = sorted(set(preds[r]["fold"] for r in preds))
        cal, _ = _per_scaf_stratum_calibrate(preds, folds)
        calib_strat[name] = round(_pooled(cal), 4)

    out = {
        "frozen_sigma_07_nll": frozen,
        "per_scaf_sigma_loo_nll": calib_scaf,
        "per_scaf_stratum_sigma_loo_nll": calib_strat,
    }
    out["frozen_sigma_07_rel_gain_pct"] = {
        k: round(100.0 * (frozen["nuisance"] - v) / frozen["nuisance"], 2)
        for k, v in frozen.items() if k != "nuisance"}
    out["per_scaf_rel_gain_pct"] = {
        k: round(100.0 * (calib_scaf["nuisance"] - v) / calib_scaf["nuisance"], 2)
        for k, v in calib_scaf.items() if k != "nuisance"}
    out["per_scaf_stratum_rel_gain_pct"] = {
        k: round(100.0 * (calib_strat["nuisance"] - v) / calib_strat["nuisance"], 2)
        for k, v in calib_strat.items() if k != "nuisance"}

    cal_ens7, _ = _per_scaf_stratum_calibrate(
        ens7, sorted(set(ens7[r]["fold"] for r in ens7)))
    cal_nuis, _ = _per_scaf_stratum_calibrate(
        nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_7mem_stratum_vs_nuisance_stratum"] = _edit_ci(cal_ens7, cal_nuis)
    out["edit_cluster_CI_7mem_stratum_vs_nuisance_frozen07"] = _edit_ci(
        cal_ens7, {r: {**nuis[r], "sigma": 0.7} for r in nuis})

    out["n_rows"] = len(common)
    out["note"] = (
        "Fair horizontal table: the r45 per-scaffold x stratum sigma "
        "calibration (sigma_m for measured, sigma_c for censored, both LOO on "
        "other folds' OOF rows) is applied to EVERY comparator.  Relative gains "
        "and edit-cluster CIs are computed at the SAME sigma treatment.  r38 "
        "(per-scaffold, single sigma) and frozen-0.7 shown for comparison."
    )
    Path(f"{R}/per_scaf_stratum_sigma_horizontal_table.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
