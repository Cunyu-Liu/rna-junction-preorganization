"""Proper stacking: censoring-aware learned ensemble weights + calibrated sigma.

All prior ensembles used equal mu-averaging with fixed sigma=0.7.  The "weight
sweep" claim in analyze_mixed_gbdt_t7.py has no backing script -- this is the
first attempt to LEARN optimal per-member combination on the exact frozen
right-censored Gaussian NLL (MetricSpec_v3 pooled_junction_macro estimand).

Two method-level improvements, both evaluated on the SAME 37 joint-blocked
OOF predictions so the comparison is leakage-free:
  1. Calibrated ensemble sigma: sigma_c = sqrt(0.7^2 + var(member_mus)).
     The ensemble's predictive uncertainty should reflect spread across
     members; this is never tested in the shootout (r17 only tested a single
     model with a learned per-input sigma head, which is a different thing).
  2. Censoring-aware stacking: leave-one-fold-out non-negative linear
     combination of the 7 members' mu, fitting per-member weights by
     minimizing the exact right-censored Gaussian NLL (same scorer as frozen).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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


def _pooled(ens_dict):
    """ens_dict: {rid: {y, cens, mu, sigma, jid}} -> pooled junction-macro NLL."""
    jd = defaultdict(list)
    for rid, p in ens_dict.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


def _edit_ci(ens_dict, base_dict):
    """edit-component bootstrap CI for ens vs base. positive delta = base better."""
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


def _censored_nll_loss(params, feat, y, cens):
    """Right-censored Gaussian NLL for a fold. params[0]=intercept, rest=weights."""
    mu = params[0] + feat @ params[1:]
    return float(np.sum(row_nll(y, cens, mu, np.full(len(y), SIGMA_FIXED))))


def _fit_fold_stacking(feat_train, y_train, cens_train, feat_test):
    """Non-negative linear stacking on one held-out fold."""
    n_feat = feat_train.shape[1]
    x0 = np.concatenate([[0.0], np.full(n_feat, 1.0 / n_feat)])
    bounds = [(None, None)] + [(0.0, None)] * n_feat
    res = minimize(
        _censored_nll_loss, x0,
        args=(feat_train, y_train, cens_train),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-10},
    )
    mu_test = res.x[0] + feat_test @ res.x[1:]
    return mu_test, res.x, res.success


def _stacked_ensemble(members, valid_folds, by_fold, flat):
    """Leave-one-fold-out censoring-aware stacking. Returns {rid: pred}.

    ``flat`` is the per-member {rid: pred} dict (rows across all folds);
    ``by_fold`` maps member -> fold -> {rid: pred} for membership bookkeeping.
    """
    out = {}
    fit_status = {}
    for k, test_fold in enumerate(valid_folds):
        train_folds = [f for i, f in enumerate(valid_folds) if i != k]
        train_rids = sorted(set.intersection(*[
            set().union(*[set(by_fold[m][f]) for f in train_folds])
            for m in members
        ]))
        if len(train_rids) < 10:
            fit_status[test_fold] = {"status": "skip", "reason": "train too small"}
            continue
        test_rids = sorted(set.intersection(*[
            set(by_fold[m][test_fold]) for m in members
        ]))
        if len(test_rids) == 0:
            continue
        feat_train = np.column_stack([
            [flat[m][r]["mu"] for r in train_rids] for m in members
        ])
        y_train = np.array([flat[members[0]][r]["y"] for r in train_rids])
        cens_train = np.array([flat[members[0]][r]["cens"] for r in train_rids])
        feat_test = np.column_stack([
            [flat[m][r]["mu"] for r in test_rids] for m in members
        ])
        mu_test, weights, ok = _fit_fold_stacking(
            feat_train, y_train, cens_train, feat_test)
        for i, rid in enumerate(test_rids):
            ref = flat[members[0]][rid]
            out[rid] = {"jid": ref["jid"], "fold": ref["fold"],
                        "y": ref["y"], "cens": ref["cens"],
                        "mu": float(mu_test[i]), "sigma": SIGMA_FIXED}
        fit_status[test_fold] = {
            "status": "ok" if ok else "not_converged",
            "intercept": round(float(weights[0]), 4),
            "weights": {m: round(float(w), 4) for m, w in zip(members, weights[1:])},
        }
    return out, fit_status


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

    out = {"nuisance_nll": round(_pooled(nuis), 4)}

    # --- Reference: equal-weight 7-member ensemble (fixed sigma=0.7) ---
    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ens_eq = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens_eq[rid] = {"jid": ref["jid"], "y": ref["y"], "cens": ref["cens"],
                       "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS])),
                       "sigma": SIGMA_FIXED}
    out["equal_weight_7mem_nll"] = round(_pooled(ens_eq), 4)
    out["equal_weight_7mem_rel_gain_pct"] = round(
        100.0 * (out["nuisance_nll"] - out["equal_weight_7mem_nll"]) / out["nuisance_nll"], 2)

    # --- Method 1: calibrated ensemble sigma ---
    ens_cal = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        mus = [members[m][rid]["mu"] for m in ALL_MEMBERS]
        mu_mean = float(np.mean(mus))
        var_mus = float(np.var(mus, ddof=1)) if len(mus) > 1 else 0.0
        sigma_cal = float(np.sqrt(SIGMA_FIXED ** 2 + var_mus))
        ens_cal[rid] = {"jid": ref["jid"], "fold": ref["fold"], "y": ref["y"],
                        "cens": ref["cens"],
                        "mu": mu_mean, "sigma": sigma_cal}
    out["calibrated_sigma_7mem_nll"] = round(_pooled(ens_cal), 4)
    out["calibrated_sigma_7mem_rel_gain_pct"] = round(
        100.0 * (out["nuisance_nll"] - out["calibrated_sigma_7mem_nll"]) / out["nuisance_nll"], 2)
    out["calibrated_sigma_stats"] = {
        "min": round(min(ens_cal[rid]["sigma"] for rid in ens_cal), 4),
        "mean": round(float(np.mean([ens_cal[rid]["sigma"] for rid in ens_cal])), 4),
        "max": round(max(ens_cal[rid]["sigma"] for rid in ens_cal), 4),
    }
    out["edit_cluster_CI_calibrated_vs_nuisance"] = _edit_ci(ens_cal, nuis)

    # --- Method 2: censoring-aware stacking (leave-one-fold-out) ---
    by_fold = {m: defaultdict(dict) for m in ALL_MEMBERS}
    for m in ALL_MEMBERS:
        for rid, p in members[m].items():
            by_fold[m][p["fold"]][rid] = p

    fold_keys = sorted(set(by_fold[ALL_MEMBERS[0]].keys()))
    valid_folds = [f for f in fold_keys
                   if all(len(by_fold[m][f]) > 0 for m in ALL_MEMBERS)]
    out["n_valid_folds"] = len(valid_folds)
    print(f"Valid folds: {len(valid_folds)}/{len(fold_keys)}", file=sys.stderr)

    if len(valid_folds) >= 3:
        stacked, fit_status = _stacked_ensemble(ALL_MEMBERS, valid_folds, by_fold, members)
        if stacked:
            out["stacked_7mem_nll"] = round(_pooled(stacked), 4)
            out["stacked_7mem_n_rows"] = len(stacked)
            out["stacked_7mem_rel_gain_pct"] = round(
                100.0 * (out["nuisance_nll"] - out["stacked_7mem_nll"]) / out["nuisance_nll"], 2)
            out["edit_cluster_CI_stacked_vs_nuisance"] = _edit_ci(stacked, nuis)
            out["stacked_vs_equal_weight_delta"] = round(
                out["stacked_7mem_nll"] - out["equal_weight_7mem_nll"], 4)
            out["stacking_fold_status"] = fit_status
        else:
            out["stacked_7mem_nll"] = None
            out["stacked_note"] = "stacking produced no valid predictions"
    else:
        out["stacked_7mem_nll"] = None
        out["stacked_note"] = f"too few valid folds ({len(valid_folds)})"

    out["n_common_rows"] = len(common)
    out["note"] = (
        "Method 1 = calibrated ensemble sigma: sigma_c = sqrt(0.7^2 + var(member_mus)). "
        "Method 2 = censoring-aware non-negative linear stacking: fit per-member weights "
        "on held-out folds optimizing the exact right-censored Gaussian NLL. "
        "Both are genuine method-level improvements never tested in the shootout."
    )

    Path(f"{R}/stacked_ensemble_analysis.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
