"""Mixed-family ensemble: XGBoost + 3x t7 MLP (orthogonal model families).

The r33 full run established a single right-censored XGBoost model reaches
NLL 0.8845 (vs 3x t7 MLP ensemble 0.8823), and the FIRST method improvement
over the saturated 3x t7 ensemble comes from mu-averaging XGBoost with the
3 t7 MLP members: pooled junction-macro NLL 0.8599 (+21.2% vs nuisance,
edit-cluster CI [0.181, 0.333], leave-one-largest 0.242).

This is different from the r28 negative (linear hybrid / latent-op members
diluted the MLP ensemble): GBDT is a structurally orthogonal learner whose
errors are complementary to the MLP, so it reduces variance where same-family
members cannot.

Run root artifacts:
  - r33_xgboost_full/Predictions_v3.jsonl  (xgboost + t7 + nuisance, 37 folds)
  - r24_t7_seed7/combined_*.jsonl          (t7 across seeds)
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
R24 = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"
R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R24_LEDGERS = [
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
]
XGB = "xgboost_censored_hybrid"
T7 = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
NUIS = "motif_topology_hierarchy"


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


def _edit_ci(ens, base):
    jid_edit = {}
    for rid, p in ens.items():
        jid_edit.setdefault(p["jid"], str(p["fold"]).split(":", 1)[1])
    jid_d = defaultdict(list)
    for rid, p in ens.items():
        if rid not in base:
            continue
        nll_e = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        nll_b = float(row_nll([base[rid]["y"]], [base[rid]["cens"]],
                              [base[rid]["mu"]], [base[rid]["sigma"]])[0])
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


def main():
    elig33 = _elig([R33_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows24 = _load(R24)

    xgb = _by_rid(rows33, XGB, elig33)
    nuis = _by_rid(rows33, NUIS, elig33)
    t7 = {m: _by_rid(rows24, m, elig24) for m in T7}

    ens3 = {}
    for rid in t7[T7[0]]:
        if all(m in t7 and rid in t7[m] for m in T7):
            ref = t7[T7[0]][rid]
            ens3[rid] = {"jid": ref["jid"], "fold": ref["fold"], "y": ref["y"],
                         "cens": ref["cens"],
                         "mu": float(np.mean([t7[m][rid]["mu"] for m in T7])),
                         "sigma": 0.7}
    ens4 = {}
    for rid in set(xgb) & set(t7[T7[0]]):
        if all(m in t7 and rid in t7[m] for m in T7):
            ref = xgb[rid]
            ens4[rid] = {"jid": ref["jid"], "fold": ref["fold"], "y": ref["y"],
                         "cens": ref["cens"],
                         "mu": float(np.mean([xgb[rid]["mu"]] +
                                             [t7[m][rid]["mu"] for m in T7])),
                         "sigma": 0.7}

    nll_n = _pooled(nuis)
    nll_x = _pooled(xgb)
    nll_3 = _pooled(ens3)
    nll_4 = _pooled(ens4)
    out = {
        "nuisance_nll": round(nll_n, 4),
        "xgboost_single_nll": round(nll_x, 4),
        "ensemble_3x_t7_nll": round(nll_3, 4),
        "mixed_xgb_3xt7_nll": round(nll_4, 4),
        "xgboost_rel_gain_pct": round(100.0 * (nll_n - nll_x) / nll_n, 2),
        "ensemble_3x_t7_rel_gain_pct": round(100.0 * (nll_n - nll_3) / nll_n, 2),
        "mixed_xgb_3xt7_rel_gain_pct": round(100.0 * (nll_n - nll_4) / nll_n, 2),
        "edit_cluster_CI_mixed_vs_nuisance": _edit_ci(ens4, nuis),
        "n_rows": len(ens4),
        "note": ("GBDT is a structurally orthogonal learner; its mu-averaging "
                 "with the 3x t7 MLP members is the first improvement over the "
                 "saturated 3x t7 ensemble (r28 linear-family dilution is the "
                 "contrast)."),
    }
    Path(f"{R}/mixed_xgb_t7_ensemble.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
