"""Per-operator (per-scaffold) sigma calibration: the r38 method lever.

The residual-structure diagnostic found strong operator heterogeneity:
  - per-scaffold measured RMSE ranges 0.45 (scaf2) .. 1.15 (scaf9);
  - scaffold 9 is 78.5% censored, scaffold 1 is 59.2% censored -- the high
    censoring operators carry the largest predictive uncertainty;
  - the global sigma-only LOO calibration already improved the 7-member
    ensemble from 0.8527 -> 0.8460 by emitting sigma~0.62 instead of 0.7.

A per-operator sigma lets each scaffold emit its OWN calibrated scale, which
directly targets the operator heterogeneity that a single global sigma cannot.
Like the r37 sigma-only calibration this is honest: for every held-out fold the
per-scaffold sigmas are fit ONLY on the OOF rows of the OTHER 36 folds, then
applied to the held-out fold.  mu stays the equal-weight ensemble.

Estimands reported:
  - frozen sigma=0.7            (r24/r34/r35 freeze, for reference)
  - global sigma-only LOO        (r37 positive, sigma~0.62)
  - per-scaffold sigma LOO       (this run; primary r38 lever)
  - per-scaffold sigma + global  (two-scale; diagnostic)
All on the SAME pooled-OOF junction-macro right-censored Gaussian NLL
(MetricSpec_v3 primary) and the same 37 joint-blocked folds.
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


def _scan_sigma(rows, grid=None):
    """1-D scan of the sigma minimizing pooled NLL over a set of rows."""
    if not rows:
        return None, np.inf
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    grid = grid if grid is not None else np.arange(0.4, 1.2, 0.01)
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        by = defaultdict(list)
        for j, loss in zip(jid, losses):
            by[str(j)].append(float(loss))
        nll = float(np.mean([np.mean(v) for v in by.values()]))
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

    # ---- reference: global sigma-only LOO (r37 positive) ----
    cal_global = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=np.arange(0.4, 1.2, 0.01))
        for rid, p in by_fold[f].items():
            cal_global[rid] = {**p, "sigma": s_global}
    out["global_sigma_only_loo_nll"] = round(_pooled(cal_global), 4)

    # ---- r38: per-scaffold sigma LOO ----
    cal_scaf = {}
    fit_log = {}
    grid = np.arange(0.4, 1.4, 0.01)
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        # per-scaffold sigma on the OTHER folds
        scaf_sigma = {}
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        for sc, rows_sc in by_scaf.items():
            # require a minimum number of rows to fit a scaffold's sigma
            if len(rows_sc) >= 20:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = None
        # fallback for scaffolds with too few rows: global sigma on other folds
        s_global, _ = _scan_sigma(other, grid=grid)
        for sc in scaf_sigma:
            if scaf_sigma[sc] is None:
                scaf_sigma[sc] = s_global
        # apply to held-out fold
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal_scaf[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
        fit_log[f] = {
            "scaf_sigma": {str(k): round(v, 3) for k, v in sorted(scaf_sigma.items())},
            "global_fallback": round(float(s_global), 3),
        }
    out["per_scaf_sigma_loo_nll"] = round(_pooled(cal_scaf), 4)
    out["per_scaf_sigma_loo_vs_frozen07_delta"] = round(
        out["per_scaf_sigma_loo_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["per_scaf_sigma_loo_vs_global_delta"] = round(
        out["per_scaf_sigma_loo_nll"] - out["global_sigma_only_loo_nll"], 4)
    out["edit_cluster_CI_per_scaf_vs_nuisance"] = _edit_ci(
        cal_scaf, {r: {**nuis[r], "sigma": 0.7} for r in nuis})
    # stability of per-scaffold sigma across folds
    scaf_series = defaultdict(list)
    for l in fit_log.values():
        for k, v in l["scaf_sigma"].items():
            scaf_series[k].append(v)
    out["per_scaf_sigma_stability"] = {
        str(k): {"min": round(min(v), 3), "mean": round(float(np.mean(v)), 3),
                 "max": round(max(v), 3), "n_folds": len(v)}
        for k, v in sorted(scaf_series.items())
    }
    out["fit_log_folds"] = fit_log

    out["note"] = (
        "r38 lever: per-scaffold (per-operator) sigma, fit leave-one-fold-out "
        "on the OTHER 36 folds' OOF rows (no test-label leakage), applied to "
        "the held-out fold.  mu is unchanged (equal-weight 7-member ensemble). "
        "This targets the operator heterogeneity found in the residual "
        "diagnostic (scaf9 RMSE 1.15 high-censoring vs scaf2 0.45).  sigma is "
        "a model-emitted parameter per MetricSpec, so recalibration is a "
        "legitimate method improvement; frozen-0.7 estimand reported too."
    )
    Path(f"{R}/per_scaf_sigma_calibration.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
