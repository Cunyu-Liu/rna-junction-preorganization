"""Operator-aware ensemble calibration (OEC): post-hoc, leakage-free.

Diagnostic finding (residual_structure_diagnostic.json):
  - the 7-member equal-weight ensemble leaves systematic per-scaffold bias
    (scaffold 9: -0.99, scaffold 1: -0.44) that the raw ensemble does NOT fix
    (nuisance has the same scaffold-9 bias), and
  - the ensemble's actual measured residual RMSE is 0.61, so the emitted
    sigma=0.7 is over-dispersed (optimal global sigma 0.62 -> NLL 0.8419 vs
    0.8527 at 0.7).

This script tests a genuine METHOD-LEVEL calibration layer on top of the
frozen OOF predictions, with NO test-label leakage:

  For each held-out fold k (edit component):
    - calibration is fit ONLY on the OOF rows of the OTHER 36 folds
      (predictions whose training sets exclude those rows -> honest);
    - per-scaffold additive intercept  alpha_s = mean(y - mu) on measured rows;
    - global (or per-scaffold) sigma s* minimizing pooled NLL on other folds;
    - apply to fold k:  mu_cal = mu + alpha_scaf(row), sigma = s*.

Reported as a separate estimand (calibrated) alongside the frozen-0.7 primary,
so the comparison is transparent.  The same calibration procedure is applied to
every comparator (nuisance, single members) for a fair horizontal contrast.
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

CAP = -7.1


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


def _best_sigma(rows):
    """1-D scan of the sigma minimizing pooled NLL on a set of rows.

    Vectorized across the sigma grid (no per-sigma Python loop over rows).
    """
    items = list(rows.values())
    n = len(items)
    if n == 0:
        return 0.7, np.inf
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    sigmas = np.arange(0.45, 0.90, 0.01)
    best_s, best_n = None, np.inf
    for s in sigmas:
        losses = row_nll(y, cens, mu, np.full(n, float(s)))
        by = defaultdict(list)
        for j, loss in zip(jid, losses):
            by[str(j)].append(float(loss))
        nll = float(np.mean([np.mean(v) for v in by.values()]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _per_scaf_alpha(rows, min_per_scaf=5, shrink=0.9):
    """Per-scaffold additive intercept fit on measured rows of a calibration set.

    `shrink` pulls each offset toward 0 to protect small-scaffold estimates.
    """
    resid = defaultdict(list)
    for rid, p in rows.items():
        if not p["cens"]:
            resid[int(p["scaf"])].append(p["y"] - p["mu"])
    alpha = {}
    for s, vals in resid.items():
        if len(vals) >= min_per_scaf:
            alpha[s] = shrink * float(np.mean(vals))
    return alpha


def _apply_calibration(rows, alpha, sigma):
    out = {}
    for rid, p in rows.items():
        s = int(p["scaf"])
        out[rid] = {"jid": p["jid"], "fold": p["fold"], "scaf": p["scaf"],
                    "y": p["y"], "cens": p["cens"],
                    "mu": p["mu"] + alpha.get(s, 0.0), "sigma": sigma}
    return out


def _oec(ens, folds, fit_sigma_global=True, use_alpha=True):
    """Leave-one-fold-out calibration. Returns calibrated {rid: pred}.

    - use_alpha=True:  per-scaffold additive intercept (alpha) applied to mu.
    - fit_sigma_global=True: per-fold sigma recalibrated on other folds.
    Either/both can be disabled to isolate the lever.
    """
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    cal = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        alpha = _per_scaf_alpha(other) if use_alpha else {}
        if fit_sigma_global:
            sigma, _ = _best_sigma(other)
        else:
            sigma = 0.7
        cal.update(_apply_calibration(by_fold[f], alpha, sigma))
        fit_log[f] = {"alpha": {str(k): round(v, 4) for k, v in sorted(alpha.items())},
                      "sigma": round(float(sigma), 3)}
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
    nuis_folds = sorted(set(nuis[r]["fold"] for r in nuis))

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                 for r in nuis}), 4),
        "equal_weight_7mem_nll_frozen07": round(
            _pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
    }

    # ---- calibrated ensemble: leave-one-fold-out ----
    cal_ens, ens_log = _oec(ens, folds, fit_sigma_global=True, use_alpha=True)
    out["oec_7mem_nll"] = round(_pooled(cal_ens), 4)
    out["oec_7mem_rel_gain_pct"] = round(
        100.0 * (out["nuisance_nll_frozen07"] - out["oec_7mem_nll"])
        / out["nuisance_nll_frozen07"], 2)
    out["oec_vs_equal_weight_frozen07_delta"] = round(
        out["oec_7mem_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["edit_cluster_CI_oec_vs_nuisance"] = _edit_ci(
        cal_ens, {r: {**nuis[r], "sigma": 0.7} for r in nuis})
    out["oec_sigma_summary"] = {
        "min": round(min(l["sigma"] for l in ens_log.values()), 3),
        "mean": round(float(np.mean([l["sigma"] for l in ens_log.values()])), 3),
        "max": round(max(l["sigma"] for l in ens_log.values()), 3),
    }
    # mean |alpha| per scaffold across folds
    scaf_alphas = defaultdict(list)
    for l in ens_log.values():
        for s, a in l["alpha"].items():
            scaf_alphas[s].append(a)
    out["oec_mean_alpha_by_scaf"] = {
        str(s): round(float(np.mean(v)), 4) for s, v in sorted(scaf_alphas.items())}
    out["oec_n_rows"] = len(cal_ens)

    # ---- isolated levers (leave-one-fold-out, honest) ----
    cal_sigma_only, sig_log = _oec(ens, folds, fit_sigma_global=True, use_alpha=False)
    out["sigma_only_loo_nll"] = round(_pooled(cal_sigma_only), 4)
    out["sigma_only_loo_vs_frozen07_delta"] = round(
        out["sigma_only_loo_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["alpha_only_loo_nll"] = round(
        _pooled(_oec(ens, folds, fit_sigma_global=False, use_alpha=True)[0]), 4)
    out["alpha_only_loo_vs_frozen07_delta"] = round(
        out["alpha_only_loo_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["sigma_only_loo_sigma_summary"] = {
        "min": round(min(l["sigma"] for l in sig_log.values()), 3),
        "mean": round(float(np.mean([l["sigma"] for l in sig_log.values()])), 3),
        "max": round(max(l["sigma"] for l in sig_log.values()), 3),
    }

    # ---- calibrated nuisance (fair horizontal contrast) ----
    cal_nuis, _ = _oec({r: {**nuis[r]} for r in nuis}, nuis_folds,
                       fit_sigma_global=True, use_alpha=True)
    out["nuisance_nll_calibrated"] = round(_pooled(cal_nuis), 4)

    # ---- calibrated best single member (t7_s99) ----
    best_single = members[T7_S99]
    folds_s = sorted(set(best_single[r]["fold"] for r in best_single))
    cal_single, _ = _oec({r: {**best_single[r]} for r in best_single}, folds_s,
                         fit_sigma_global=True, use_alpha=True)
    out["t7_s99_nll_frozen07"] = round(
        _pooled({r: {**best_single[r], "sigma": 0.7} for r in best_single}), 4)
    out["t7_s99_nll_calibrated"] = round(_pooled(cal_single), 4)

    # ---- member-wise calibrated single NLL (who benefits most from OEC) ----
    out["member_nll_frozen07"] = {
        m: round(_pooled({r: {**members[m][r], "sigma": 0.7} for r in members[m]}), 4)
        for m in ALL_MEMBERS}
    member_cal = {}
    member_sigma_only = {}
    for m in ALL_MEMBERS:
        rows_m = members[m]
        f_m = sorted(set(rows_m[r]["fold"] for r in rows_m))
        cal_m, _ = _oec({r: {**rows_m[r]} for r in rows_m}, f_m,
                        fit_sigma_global=True, use_alpha=True)
        member_cal[m] = round(_pooled(cal_m), 4)
        cal_ms, _ = _oec({r: {**rows_m[r]} for r in rows_m}, f_m,
                         fit_sigma_global=True, use_alpha=False)
        member_sigma_only[m] = round(_pooled(cal_ms), 4)
    out["member_nll_calibrated"] = member_cal
    out["member_nll_sigma_only"] = member_sigma_only
    # pure sigma-only LOO for nuisance (isolates the ensemble-variance mechanism)
    cal_nuis_sig, _ = _oec({r: {**nuis[r]} for r in nuis}, nuis_folds,
                           fit_sigma_global=True, use_alpha=False)
    out["nuisance_nll_sigma_only"] = round(_pooled(cal_nuis_sig), 4)
    out["nuisance_nll_sigma_only_delta_vs_frozen"] = round(
        out["nuisance_nll_sigma_only"] - out["nuisance_nll_frozen07"], 4)

    out["note"] = (
        "OEC = leave-one-fold-out per-scaffold intercept calibration + "
        "fold-level sigma recalibration.  Calibration is fit ONLY on the OOF "
        "rows of the other 36 folds (no test-label leakage), and applied to "
        "the held-out fold.  Same procedure applied to every comparator for a "
        "fair horizontal contrast.  sigma is a model-emitted parameter per "
        "MetricSpec, so recalibration is a legitimate method improvement; the "
        "frozen-0.7 estimand is still reported for transparency."
    )
    Path(f"{R}/operator_calibrated_ensemble.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
