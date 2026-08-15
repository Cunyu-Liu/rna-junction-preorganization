"""Mixed-family ensemble: GBDT + t7 MLP + kernel members.

Consumes materialized preds from r24 (3x t7 MLP seeds), r33 (Gaussian XGBoost
seed23 + nuisance), r34_gbdt_seeds_full (seeds s99/s2026), r35_gbdt_hp_full
(hp_lr03, the r35 scan winner), and r36_kernel_full (kernel RBF member) on the
same 37 blocked joint folds, restricted to optimizer+full-coverage eligible rows
(fail-closed).

Headline estimand: pooled junction-macro right-censored Gaussian NLL (sigma=0.7).

Families (mu-average, fixed sigma=0.7):
  - GBDT family: {s23, lr03, s99, s2026}
  - MLP family:  3x t7 MLP {t7, t7_s99, t7_s2026}
  - kernel family: kernel_censored_hybrid (single member)

Mixed ensembles: family-equal weights (each family 1/3) and the 2-family
GBDT+MLP optimum for comparison; the kernel's standalone NLL and error
correlation vs the two other families quantify whether it adds real
variance reduction (the r36 hypothesis).
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
R36 = f"{R}/r36_kernel_full2/Predictions_v3.jsonl"
R24 = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"
R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R34_LEDGER = f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet"
R35_LEDGER = f"{R}/r35_gbdt_hp_full/ConvergenceLedger_v3.parquet"
R36_LEDGER = f"{R}/r36_kernel_full2/ConvergenceLedger_v3.parquet"
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
KERNEL = "kernel_censored_hybrid"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUIS = "motif_topology_hierarchy"

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


def _error_corr(a, b):
    shared = sorted(set(a) & set(b))
    ea = np.asarray([a[r]["y"] - a[r]["mu"] for r in shared], dtype=float)
    eb = np.asarray([b[r]["y"] - b[r]["mu"] for r in shared], dtype=float)
    return float(np.corrcoef(ea, eb)[0, 1]), len(shared)


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
    elig36 = _elig([R36_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows36 = _load(R36)
    rows24 = _load(R24)

    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[KERNEL] = _by_rid(rows36, KERNEL, elig36)
    for m in MLP:
        members[m] = _by_rid(rows24, m, elig24)
    nuis = _by_rid(rows33, NUIS, elig33)

    out = {"nuisance_nll": round(_pooled(nuis), 4)}
    out["single_nll"] = {m: round(_pooled(members[m]), 4) for m in members}
    out["n_rows_single"] = {m: len(members[m]) for m in members}

    # diversity: kernel vs GBDT family avg and MLP family avg
    ens_g4 = _ens(members, GBDT)
    ens_m3 = _ens(members, MLP)
    k = members[KERNEL]
    c_g, n_g = _error_corr(ens_g4, k)
    c_m, n_m = _error_corr(ens_m3, k)
    out["kernel_diversity"] = {
        "corr_kernel_vs_gbdt4x": round(c_g, 4),
        "corr_kernel_vs_mlp3x": round(c_m, 4),
        "n_shared": n_g,
    }

    ens_mixed2 = _ens({"g": ens_g4, "m": ens_m3}, ["g", "m"])          # 2-family
    ens_mixed3 = _ens({"g": ens_g4, "m": ens_m3, "k": k}, ["g", "m", "k"])  # family-equal 1/3
    out["ensemble_nll"] = {
        "gbdt_4x": round(_pooled(ens_g4), 4),
        "mlp_t7_3x": round(_pooled(ens_m3), 4),
        "kernel_1x": round(_pooled(k), 4),
        "mixed_gbdt4x_mlp3x_2fam": round(_pooled(ens_mixed2), 4),
        "mixed_3fam_family_equal": round(_pooled(ens_mixed3), 4),
    }
    out["ensemble_rel_gain_pct"] = {
        kk: round(100.0 * (out["nuisance_nll"] - v) / out["nuisance_nll"], 2)
        for kk, v in out["ensemble_nll"].items()}
    out["edit_cluster_CI_mixed_vs_nuisance"] = {
        "mixed_gbdt4x_mlp3x_2fam": _edit_ci(ens_mixed2, nuis),
        "mixed_3fam_family_equal": _edit_ci(ens_mixed3, nuis),
    }
    out["n_rows"] = len(ens_mixed3)
    out["note"] = ("Kernel RBF member (r36) is the structurally orthogonal 3rd "
                   "family; family-equal weights assign each family 1/3. The "
                   "r34/r35 weight sweep validated family-equal as optimal when "
                   "families are matched in quality.")

    Path(f"{R}/mixed_gbdt_t7_ensemble.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

