"""Residual-structure diagnostic for the 7-member mixed ensemble.

Goal: identify where the remaining pooled-OOF NLL error lives so the next
method-level change targets a real bottleneck instead of guessing.

Quantities computed on the SAME held-out OOF predictions:
  1. residual distribution by censoring stratum (measured / censored)
  2. per-scaffold (operator) residual std -- is operator error heterogeneous?
  3. per-edit-component residual std -- the independent unit
  4. optimal GLOBAL sigma for the equal-weight ensemble (1-D scan on OOF) --
     is the frozen 0.7 actually the NLL-optimal scale for THIS ensemble?
     (honest diagnostic: the model emits sigma, so this is a model parameter)
  5. CRPS / calibration summary (marginal predicted CDF vs empirical)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr

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


def _crps_row(y, cens, mu, sigma):
    """Right-censored CRPS for a Gaussian predictive (positive loss).

    Standard result (Avramidis/Matheson-Winkler):
      measured: crps = sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ), z=(y-mu)/sigma
      censored: y>=CAP recorded; predicted survival above CAP:
        crps = sigma * ( (a)*Phi(a) + phi(a) - 1/sqrt(pi) ), a=(CAP-mu)/sigma
    """
    if cens:
        a = (CAP - mu) / sigma
        return float(sigma * (a * ndtr(a) + np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
                              - 1.0 / np.sqrt(np.pi)))
    z = (y - mu) / sigma
    phiz = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return float(sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phiz - 1.0 / np.sqrt(np.pi)))


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

    # equal-weight ensemble
    ens = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS]))}

    out = {"nuisance_nll": round(_pooled(nuis), 4),
           "equal_weight_7mem_nll": round(_pooled({r: {**ens[r], "sigma": 0.7}
                                                    for r in ens}), 4)}

    # 1. residual by stratum
    res_meas = []
    res_cens_mu = []
    cens_mu = []
    for rid, p in ens.items():
        if p["cens"]:
            cens_mu.append(p["mu"])
            res_cens_mu.append(p["mu"] - CAP)
        else:
            res_meas.append(p["y"] - p["mu"])
    out["stratum"] = {
        "n_measured": len(res_meas),
        "n_censored": len(res_cens_mu),
        "measured_resid_rmse": round(float(np.sqrt(np.mean(np.square(res_meas)))), 4),
        "measured_resid_mean": round(float(np.mean(res_meas)), 4),
        "censored_mu_above_cap_frac": round(
            float(np.mean(np.asarray(cens_mu) > CAP)), 4),
        "censored_mean_mu_minus_cap": round(float(np.mean(res_cens_mu)), 4),
    }

    # 2. per-scaffold residual std (measured rows)
    scaf_resid = defaultdict(list)
    for rid, p in ens.items():
        if not p["cens"]:
            scaf_resid[int(p["scaf"])].append(p["y"] - p["mu"])
    out["per_scaffold_measured_rmse"] = {
        str(k): round(float(np.sqrt(np.mean(np.square(v)))), 4)
        for k, v in sorted(scaf_resid.items())}
    out["per_scaffold_rmse_range"] = {
        "min": round(min(float(np.sqrt(np.mean(np.square(v)))) for v in scaf_resid.values()), 4),
        "max": round(max(float(np.sqrt(np.mean(np.square(v)))) for v in scaf_resid.values()), 4),
        "n_scaf": len(scaf_resid),
    }

    # 3. optimal global sigma for the equal-weight ensemble (1-D scan on OOF)
    best_sigma, best_nll = None, np.inf
    for s in np.arange(0.45, 0.9, 0.01):
        nll = _pooled({r: {**ens[r], "sigma": float(s)} for r in ens})
        if nll < best_nll:
            best_nll, best_sigma = nll, s
    out["optimal_global_sigma"] = {
        "sigma": round(float(best_sigma), 3),
        "nll_at_opt_sigma": round(float(best_nll), 4),
        "nll_at_frozen_0.7": round(_pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
        "delta_vs_frozen": round(float(best_nll) - _pooled({r: {**ens[r], "sigma": 0.7}
                                                            for r in ens}), 4),
    }
    # per-fold optimal sigma (does the optimal scale vary by edit component?)
    folds = sorted(set(ens[r]["fold"] for r in ens))
    per_fold = {}
    for f in folds:
        rid_f = [r for r in ens if ens[r]["fold"] == f]
        if len(rid_f) < 10:
            continue
        best_s, best_n = None, np.inf
        for s in np.arange(0.45, 0.9, 0.01):
            nll = _pooled({r: {**ens[r], "sigma": float(s)} for r in rid_f})
            if nll < best_n:
                best_n, best_s = nll, s
        per_fold[f] = round(float(best_s), 2)
    out["per_fold_optimal_sigma"] = per_fold
    svalues = list(per_fold.values())
    out["per_fold_optimal_sigma_summary"] = {
        "min": min(svalues), "max": max(svalues),
        "mean": round(float(np.mean(svalues)), 3),
        "n_folds": len(svalues),
    }

    # 4. CRPS at sigma=0.7 (secondary metric)
    crps = [_crps_row(p["y"], p["cens"], p["mu"], 0.7) for rid, p in ens.items()]
    out["pooled_junction_macro_crps"] = round(
        float(np.mean(_group_mean(crps, [ens[r]["jid"] for r in ens]))), 4)

    # 5. are the members' mu errors correlated with a systematic operator offset?
    #    residual mean per scaffold from nuisance vs ensemble
    nuis_resid = defaultdict(list)
    for rid, p in nuis.items():
        if rid in ens and not p["cens"]:
            nuis_resid[int(p["scaf"])].append(p["y"] - p["mu"])
    out["per_scaffold_measured_bias"] = {
        "ensemble": {str(k): round(float(np.mean(v)), 4)
                     for k, v in sorted(scaf_resid.items())},
        "nuisance": {str(k): round(float(np.mean(v)), 4)
                     for k, v in sorted(nuis_resid.items())},
    }

    out["n_rows"] = len(common)
    out["note"] = (
        "Diagnostic only: locates remaining error. sigma is a model-emitted "
        "parameter (MetricSpec row_likelihood uses the emitted sigma), so the "
        "optimal-global-sigma scan is an honest 1-D calibration check, not a "
        "metric change. CRPS reported as secondary.")
    Path(f"{R}/residual_structure_diagnostic.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def _group_mean(vals, jids):
    by = defaultdict(list)
    for v, j in zip(vals, jids):
        by[j].append(v)
    return [np.mean(v) for v in by.values()]


if __name__ == "__main__":
    main()
