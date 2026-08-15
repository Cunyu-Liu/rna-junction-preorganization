"""r39: censoring-aware per-operator intercept calibration (test lever).

The r37 OEC alpha used measured-row means (mean(y-mu) per scaffold), which
HURT censored rows: shifting mu down on high-censoring scaffolds (1: 59%,
9: 78%) pushes censored rows (Y>=-7.1) away from the right tail, worsening
their survival NLL more than it fixes measured rows.

r39 instead fits the per-scaffold intercept alpha_s by minimizing the FULL
right-censored Gaussian NLL on the OTHER folds' OOF rows (same scorer as the
frozen metric), optionally combined with the r38 per-scaffold sigma.  This is
the censoring-aware version of the operator calibration -- it can correct the
systematic per-scaffold bias (scaf9 measured bias -0.99) WITHOUT degrading
censored rows.

All calibration is leave-one-fold-out: fit on the other 36 folds' OOF rows,
apply to the held-out fold.  mu = equal-weight 7-member ensemble.

Reported:
  - frozen sigma=0.7
  - per-scaffold sigma LOO (r38 positive, 0.8166)
  - per-scaffold alpha (censoring-aware, NLL-fit) + frozen sigma
  - per-scaffold alpha + per-scaffold sigma (both censoring-aware)
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


def _scan_sigma(rows, grid=None):
    if not rows:
        return None, np.inf
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    grid = grid if grid is not None else np.arange(0.4, 1.4, 0.01)
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


def _scan_alpha(rows, sigma, grid=None):
    """1-D scan of an additive intercept minimizing FULL censored NLL."""
    if not rows:
        return None, np.inf
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    grid = grid if grid is not None else np.arange(-1.0, 1.01, 0.02)
    best_a, best_n = None, np.inf
    for a in grid:
        losses = row_nll(y, cens, mu + a, np.full(len(y), float(sigma)))
        by = defaultdict(list)
        for j, loss in zip(jid, losses):
            by[str(j)].append(float(loss))
        nll = float(np.mean([np.mean(v) for v in by.values()]))
        if nll < best_n:
            best_n, best_a = nll, a
    return float(best_a), float(best_n)


def _calibrate(preds, folds, mode, min_rows=20):
    """Leave-one-fold-out calibration. mode in {sigma, alpha, alpha_sigma}."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
    a_grid = np.arange(-1.0, 1.01, 0.02)
    cal = {}
    fit_log = {}
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
        scaf_alpha = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) < min_rows:
                scaf_sigma[sc] = s_global
                scaf_alpha[sc] = 0.0
                continue
            s_sc, _ = _scan_sigma(rows_sc, grid=grid)
            scaf_sigma[sc] = s_sc
            if mode in ("alpha", "alpha_sigma"):
                s_use = s_sc if mode == "alpha_sigma" else 0.7
                a_sc, _ = _scan_alpha(rows_sc, s_use, grid=a_grid)
                scaf_alpha[sc] = a_sc
            else:
                scaf_alpha[sc] = 0.0
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            sig = scaf_sigma.get(sc, s_global) if mode in ("sigma", "alpha_sigma") else 0.7
            alp = scaf_alpha.get(sc, 0.0)
            cal[rid] = {**p, "mu": p["mu"] + alp, "sigma": sig}
        fit_log[f] = {
            "sigma": {str(k): round(v, 3) for k, v in sorted(scaf_sigma.items())},
            "alpha": {str(k): round(v, 3) for k, v in sorted(scaf_alpha.items())},
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

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                for r in nuis}), 4),
        "equal_weight_7mem_nll_frozen07": round(
            _pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
        "n_folds": len(folds),
        "n_rows": len(ens),
    }

    # r38 reference: per-scaffold sigma only
    cal_sig, _ = _calibrate(ens, folds, "sigma")
    out["per_scaf_sigma_nll"] = round(_pooled(cal_sig), 4)

    # r39a: censoring-aware per-scaffold alpha, frozen sigma
    cal_alpha, log_alpha = _calibrate(ens, folds, "alpha")
    out["per_scaf_alpha_nll"] = round(_pooled(cal_alpha), 4)
    out["per_scaf_alpha_vs_frozen07_delta"] = round(
        out["per_scaf_alpha_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["per_scaf_alpha_vs_sigma_delta"] = round(
        out["per_scaf_alpha_nll"] - out["per_scaf_sigma_nll"], 4)

    # r39b: censoring-aware alpha + per-scaffold sigma
    cal_as, log_as = _calibrate(ens, folds, "alpha_sigma")
    out["per_scaf_alpha_sigma_nll"] = round(_pooled(cal_as), 4)
    out["per_scaf_alpha_sigma_vs_frozen07_delta"] = round(
        out["per_scaf_alpha_sigma_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["per_scaf_alpha_sigma_vs_sigma_delta"] = round(
        out["per_scaf_alpha_sigma_nll"] - out["per_scaf_sigma_nll"], 4)

    # mean learned alpha per scaffold (r39a)
    scaf_alpha_series = defaultdict(list)
    for l in log_alpha.values():
        for k, v in l["alpha"].items():
            scaf_alpha_series[k].append(v)
    out["per_scaf_alpha_mean"] = {
        str(k): round(float(np.mean(v)), 3) for k, v in sorted(scaf_alpha_series.items())}

    out["note"] = (
        "r39 lever: censoring-aware per-scaffold intercept (fit by minimizing "
        "the FULL right-censored NLL on other folds' OOF rows, not the "
        "measured-row mean that broke r37 OEC-alpha).  r39a = alpha only "
        "(frozen sigma); r39b = alpha + per-scaffold sigma.  All leave-one-"
        "fold-out, no test-label leakage.  If r39a is positive, the systematic "
        "operator bias (scaf9 -0.99) is recoverable without hurting censored "
        "rows; the r38 sigma win is the benchmark (0.8166)."
    )
    Path(f"{R}/censoring_aware_operator_calibration.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
