"""Mixed-family ensemble: multi-seed GBDT + 3x t7 MLP.

Consumes materialized preds from r24 (3x t7 MLP seeds), r33 (Gaussian XGBoost
seed23 + nuisance), r34_gbdt_seeds_full (seeds s99/s2026), and
r35_gbdt_hp_full (hp_lr03: lr=0.03, 3000 rounds, the r35 scan winner at 0.8807
vs default 0.8845) on the same 37 blocked joint folds, restricted to
optimizer+full-coverage eligible rows (fail-closed).

Headline estimand: pooled junction-macro right-censored Gaussian NLL (sigma=0.7).

Ensembles (mu-average, fixed sigma=0.7):
  - GBDT family: best-3 {lr03, s99, s2026}; 4x {s23, lr03, s99, s2026}
  - MLP family:  3x t7 MLP {t7, t7_s99, t7_s2026}
  - mixed:       GBDT-family ensemble + MLP-family ensemble (2 members)

Also reports the single-model NLL ladder and an edit-cluster bootstrap CI for
the headline mixed ensemble vs the nuisance baseline.

Note: Student-t (df=7) GBDT was smoke-rejected (r34); hyperparameter scan r35
confirmed lr03 (lower LR, more rounds) as the best single GBDT.
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

GBDT_ORIG = [XGB, XGB_S99, XGB_S2026]
GBDT_BEST3 = [XGB_LR03, XGB_S99, XGB_S2026]
GBDT_ALL4 = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
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


def _ens(members, keys):
    base = members[keys[0]]
    out = {}
    for rid in base:
        if all(rid in members[k] for k in keys):
            ref = base[rid]
            out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "y": ref["y"],
                        "cens": ref["cens"], "sigma": 0.7,
                        "mu": float(np.mean([members[k][rid]["mu"] for k in keys]))}
    return out


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
    for m in MLP:
        members[m] = _by_rid(rows24, m, elig24)
    nuis = _by_rid(rows33, NUIS, elig33)

    out = {"nuisance_nll": round(_pooled(nuis), 4)}
    out["single_nll"] = {m: round(_pooled(members[m]), 4) for m in members}
    out["n_rows_single"] = {m: len(members[m]) for m in members}

    ens_g3o = _ens(members, GBDT_ORIG)
    ens_g3b = _ens(members, GBDT_BEST3)
    ens_g4 = _ens(members, GBDT_ALL4)
    ens_m3 = _ens(members, MLP)
    ens_prev4 = _ens(members, [XGB] + MLP)  # r34 headline (xgb + 3x t7)
    ens_prev6 = _ens(members, GBDT_ORIG + MLP)  # r34 6-member
    ens_mixed_o = _ens({"g": ens_g3o, "m": ens_m3}, ["g", "m"])
    ens_mixed_b = _ens({"g": ens_g3b, "m": ens_m3}, ["g", "m"])
    ens_mixed_4 = _ens({"g": ens_g4, "m": ens_m3}, ["g", "m"])

    out["ensemble_nll"] = {
        "gbdt_3x_orig": round(_pooled(ens_g3o), 4),
        "gbdt_3x_best_lr03": round(_pooled(ens_g3b), 4),
        "gbdt_4x": round(_pooled(ens_g4), 4),
        "mlp_t7_3x": round(_pooled(ens_m3), 4),
        "xgb_plus_3xt7_4member": round(_pooled(ens_prev4), 4),
        "mixed_orig_3xt7": round(_pooled(ens_mixed_o), 4),
        "mixed_best3_lr03_3xt7": round(_pooled(ens_mixed_b), 4),
        "mixed_4xgbdt_3xt7": round(_pooled(ens_mixed_4), 4),
    }
    out["ensemble_rel_gain_pct"] = {
        k: round(100.0 * (out["nuisance_nll"] - v) / out["nuisance_nll"], 2)
        for k, v in out["ensemble_nll"].items()}
    out["edit_cluster_CI_mixed_vs_nuisance"] = {
        "mixed_orig_3xt7": _edit_ci(ens_mixed_o, nuis),
        "mixed_best3_lr03_3xt7": _edit_ci(ens_mixed_b, nuis),
        "mixed_4xgbdt_3xt7": _edit_ci(ens_mixed_4, nuis),
    }
    out["n_rows"] = len(ens_mixed_b)
    out["note"] = ("r35 scan winner hp_lr03 (0.8807 vs 0.8845) folded into the "
                   "GBDT family; equal family-weight mu-mean is optimal (both "
                   "families matched), so mixed = 6/7-member equal mu-mean.")

    Path(f"{R}/mixed_gbdt_t7_ensemble.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

