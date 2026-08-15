"""Diagnostic: is there an affine (slope) structure in the measured residuals?

All prior mu-correction levers (r37/r39 additive alpha on ALL rows, r46
additive alpha on measured rows only) assumed mu_cal = mu + alpha (b=1).
This diagnostic checks whether the measured residual y - mu has a slope
against mu (regression-to-mean / overconfidence): if the best fit is
y ~ a + b*mu with b != 1, then a measured-only AFFINE calibration
(mu_cal = a + b*mu on measured rows, LOO) is a genuine, untested lever.

Read-only over the frozen OOF predictions; reports per-scaffold slope, global
slope, and the NLL impact of a GLOBAL measured-only slope (upper bound).
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
ALL_MEMBERS = [XGB, XGB_LR03, XGB_S99, XGB_S2026, T7, T7_S99, T7_S2026]


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

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ens = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS]))}

    out = {"n_rows": len(ens), "n_folds": len(set(p["fold"] for p in ens.values()))}

    meas = [p for p in ens.values() if not p["cens"]]
    ym = np.asarray([p["y"] for p in meas], dtype=float)
    mum = np.asarray([p["mu"] for p in meas], dtype=float)
    scaf = np.asarray([int(p["scaf"]) for p in meas], dtype=int)

    # global OLS slope on measured rows: y ~ a + b*mu
    A = np.vstack([np.ones(len(mum)), mum]).T
    beta, *_ = np.linalg.lstsq(A, ym, rcond=None)
    out["global_measured_slope_ols"] = {
        "a": round(float(beta[0]), 4), "b": round(float(beta[1]), 4),
        "n_measured": len(mum)}
    # correlation between mu and residual on measured rows (slope signal)
    resid = ym - mum
    rho = float(np.corrcoef(mum, resid)[0, 1])
    out["corr_mu_resid_measured"] = round(rho, 4)

    # per-scaffold OLS slopes
    scaf_slope = {}
    for sc in sorted(set(scaf)):
        m = scaf == sc
        if m.sum() < 20:
            continue
        A2 = np.vstack([np.ones(int(m.sum())), mum[m]]).T
        b2, *_ = np.linalg.lstsq(A2, ym[m], rcond=None)
        scaf_slope[str(sc)] = {"a": round(float(b2[0]), 4),
                               "b": round(float(b2[1]), 4),
                               "n": int(m.sum())}
    out["per_scaf_measured_slope_ols"] = scaf_slope

    # NLL impact of a GLOBAL measured-only affine calibration (upper bound,
    # not LOO yet): mu_cal = a + b*mu on measured rows, censored mu unchanged,
    # sigma fixed at 0.7 for this diagnostic.
    by_jid = defaultdict(list)
    for p in ens.values():
        by_jid[p["jid"]].append(p)
    def pooled_nll(mu_map):
        jd = defaultdict(list)
        for rid, p in ens.items():
            m = mu_map.get(rid, p["mu"])
            nll = float(row_nll([p["y"]], [p["cens"]], [m], [0.7])[0])
            jd[p["jid"]].append(nll)
        return float(np.mean([np.mean(v) for v in jd.values()]))

    base_nll = pooled_nll({})
    mu_glob = {rid: beta[0] + beta[1] * p["mu"] if not p["cens"] else p["mu"]
               for rid, p in ens.items()}
    out["pooled_nll_frozen07_baseline"] = round(base_nll, 4)
    out["pooled_nll_global_measured_affine_infit"] = round(pooled_nll(mu_glob), 4)
    out["infit_affine_delta"] = round(pooled_nll(mu_glob) - base_nll, 4)

    # leave-one-fold-out global affine (honest upper bound for a GLOBAL slope)
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    folds = sorted(by_fold)
    cal = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        om = [p for p in other.values() if not p["cens"]]
        if len(om) < 20:
            continue
        oy = np.asarray([p["y"] for p in om])
        omu = np.asarray([p["mu"] for p in om])
        A3 = np.vstack([np.ones(len(omu)), omu]).T
        b3, *_ = np.linalg.lstsq(A3, oy, rcond=None)
        for rid, p in by_fold[f].items():
            cal[rid] = {"mu": b3[0] + b3[1] * p["mu"] if not p["cens"] else p["mu"],
                        "cens": p["cens"], "jid": p["jid"]}
    out["pooled_nll_global_measured_affine_loo"] = round(pooled_nll(
        {rid: cal[rid]["mu"] for rid in cal}), 4)

    out["note"] = (
        "Diagnostic only: checks for a measured-only affine (slope) structure "
        "in the ensemble's OOF measured residuals.  All prior mu levers were "
        "ADDITIVE (b=1); if b != 1 a measured-only affine calibration is a "
        "genuine untested lever.  The infit row is an upper bound (fits on all "
        "measured rows); the LOO row is the honest global-slope estimate."
    )
    Path(f"{R}/measured_affine_slope_diagnostic.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
