"""r43: per-context (helix-context) sigma calibration -- the r38 extension.

The r38 per-scaffold sigma calibration (0.8166, +25.20%) captures operator-level
heteroscedasticity.  The residual diagnostic found that WITHIN a scaffold the
nested helix-contexts are also highly heterogeneous (e.g. scaf1 context RMSE
0.53..1.62 across 25 contexts; scaf4 0.47..1.26).  r43 tests whether per-context
sigma -- a finer granularity than per-scaffold -- extracts more of this
heteroscedastic residual.

Honest calibration (same protocol as r38): for every held-out fold, the
per-context sigma table is fit ONLY on the OTHER folds' OOF rows, then applied
to the held-out fold.  mu stays the equal-weight 7-member ensemble.

Contexts are nested in scaffolds, so a context-keyed sigma generalizes the
per-scaffold table.  Contexts with too few rows fall back to their scaffold's
sigma, then to the global sigma.  All leave-one-fold-out, no test-label leakage.
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
    grid = grid if grid is not None else np.arange(0.4, 1.6, 0.01)
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


def _calibrate_ctx(ens, folds, min_rows=15):
    """Leave-one-fold-out per-context sigma with scaf/global fallbacks.

    For each held-out fold:
      - fit per-context sigma on the OTHER folds' OOF rows (contexts with
        >= min_rows rows), fallback to per-scaffold sigma (fit on other folds),
        then to global sigma;
      - apply the mapped sigma to the held-out fold.
    """
    by_fold = defaultdict(dict)
    for rid, p in ens.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.6, 0.01)
    cal = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        # per-scaffold sigma (fallback level 2)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 20:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = s_global
        # per-context sigma (finest level 1)
        by_ctx = defaultdict(dict)
        for rid, p in other.items():
            by_ctx[(int(p["scaf"]), str(p["context"]))][rid] = p
        ctx_sigma = {}
        for key, rows_ctx in by_ctx.items():
            if len(rows_ctx) >= min_rows:
                s, _ = _scan_sigma(rows_ctx, grid=grid)
                ctx_sigma[key] = s
        # apply
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            ctx_key = (sc, str(p["context"]))
            if ctx_key in ctx_sigma:
                sigma = ctx_sigma[ctx_key]
            elif sc in scaf_sigma:
                sigma = scaf_sigma[sc]
            else:
                sigma = s_global
            cal[rid] = {**p, "sigma": sigma}
        fit_log[f] = {
            "n_ctx_learned": len(ctx_sigma),
            "n_ctx_total": len(by_ctx),
            "scaf_sigma": {str(k): round(v, 3) for k, v in sorted(scaf_sigma.items())},
            "global": round(float(s_global), 3),
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
    ref = members[ALL_MEMBERS[0]]
    ens = {}
    for rid in common:
        p0 = ref[rid]
        ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": p0["scaf"],
                    "context": p0["context"], "y": p0["y"], "cens": p0["cens"],
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

    # r38 reference: per-scaffold sigma only
    # (inline: same as _calibrate with mode='sigma' in per_scaf_sigma_calibration.py)
    cal_scaf = {}
    sgrid = np.arange(0.4, 1.4, 0.01)
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=sgrid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            scaf_sigma[sc] = _scan_sigma(rows_sc, grid=sgrid)[0] if len(rows_sc) >= 20 else s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal_scaf[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
    out["per_scaf_sigma_nll"] = round(_pooled(cal_scaf), 4)

    # r43: per-context sigma (scaf/global fallback)
    cal_ctx, fit_log = _calibrate_ctx(ens, folds)
    out["per_ctx_sigma_nll"] = round(_pooled(cal_ctx), 4)
    out["per_ctx_sigma_vs_frozen07_delta"] = round(
        out["per_ctx_sigma_nll"] - out["equal_weight_7mem_nll_frozen07"], 4)
    out["per_ctx_sigma_vs_scaf_delta"] = round(
        out["per_ctx_sigma_nll"] - out["per_scaf_sigma_nll"], 4)
    out["per_ctx_coverage"] = {
        "avg_n_ctx_learned": round(float(np.mean([l["n_ctx_learned"]
                                                  for l in fit_log.values()])), 1),
        "avg_n_ctx_total": round(float(np.mean([l["n_ctx_total"]
                                                for l in fit_log.values()])), 1),
    }
    out["fit_log_folds"] = fit_log
    out["note"] = (
        "r43 lever: per-helix-context sigma calibration (finer than r38 "
        "per-scaffold).  Contexts nested in scaffolds; contexts with too few "
        "rows fall back to scaffold sigma then global.  All leave-one-fold-out "
        "on the other folds' OOF rows.  If the within-scaffold context "
        "heterogeneity (scaf1 ctx RMSE 0.53..1.62) is real, per-ctx sigma "
        "should beat per-scaf; if the gain is marginal/negative, per-scaf is "
        "the right granularity.")
    Path(f"{R}/per_ctx_sigma_calibration.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
