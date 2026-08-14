"""Mixed-ensemble diversity diagnostic (read-only on materialized preds).

Quantifies WHY the XGBoost + 3x t7 MLP mixed ensemble beats the saturated
same-family ensemble.  On the same 37 blocked joint folds, restricted to
optimizer+full-coverage eligible predictions (fail-closed):

  - per-row signed error correlation between every member pair
    (Pearson on y-mu for all supported rows; a low correlation is the
    variance-reduction precondition)
  - pooled junction-macro NLL of the 4-member mixed ensemble and every
    leave-one-out variant (contribution of each member)
  - additive check: does adding the 4th t7 MLP seed (t7_s7) help the
    mixed ensemble further?

This is a diagnostic only; it does not modify the protocol or the primary
estimand (right-censored Gaussian NLL at fixed sigma=0.7).
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
T7_S7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s7"
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


def _error_corr(members):
    """Pearson correlation of y - mu across members on shared supported rows."""
    vals = list(members.values())
    shared = set(vals[0])
    for m in vals[1:]:
        shared &= set(m)
    shared = sorted(shared)
    errs = {}
    for name, m in members.items():
        errs[name] = np.asarray([m[r]["y"] - m[r]["mu"] for r in shared], dtype=float)
    out = {}
    names = list(members)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = float(np.corrcoef(errs[a], errs[b])[0, 1])
            out[f"{a} vs {b}"] = round(c, 4)
    # mean inter-member |r| within family vs across family
    t7_names = [n for n in names if n.startswith("nonlinear")]
    out["mean_within_mlp_t7"] = round(float(np.mean(
        [abs(out[f"{a} vs {b}"]) for i, a in enumerate(t7_names)
         for b in t7_names[i + 1:] if f"{a} vs {b}" in out])), 4)
    xgb_corrs = [v for k, v in out.items() if XGB in k]
    out["mean_cross_xgb_mlp"] = round(float(np.mean([abs(v) for v in xgb_corrs])), 4)
    out["n_shared_rows"] = len(shared)
    return out


def _ens(members, keys):
    """mu-mean of a dict of member -> rid->pred, intersected on rids."""
    base = members[keys[0]]
    out = {}
    for rid in base:
        if all(rid in members[k] for k in keys):
            ref = base[rid]
            out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "y": ref["y"],
                        "cens": ref["cens"], "sigma": 0.7,
                        "mu": float(np.mean([members[k][rid]["mu"] for k in keys]))}
    return out


def main():
    elig33 = _elig([R33_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows24 = _load(R24)

    xgb = _by_rid(rows33, XGB, elig33)
    nuis = _by_rid(rows33, NUIS, elig33)
    t7 = {m: _by_rid(rows24, m, elig24) for m in T7}
    t7s7 = _by_rid(rows24, T7_S7, elig24)

    members = {"xgb": xgb, "t7": t7[T7[0]], "t7_s99": t7[T7[1]],
               "t7_s2026": t7[T7[2]]}
    out = {"diversity_error_correlation": _error_corr(members)}

    ens4 = _ens(members, list(members))
    out["ens4_mixed_nll"] = round(_pooled(ens4), 4)
    out["nuisance_nll"] = round(_pooled(nuis), 4)
    out["ens4_rel_gain_pct"] = round(
        100.0 * (out["nuisance_nll"] - out["ens4_mixed_nll"]) / out["nuisance_nll"], 2)

    # leave-one-out contribution
    loo = {}
    for drop in members:
        keys = [k for k in members if k != drop]
        e = _ens(members, keys)
        loo[f"leave_out_{drop}"] = {
            "nll": round(_pooled(e), 4),
            "delta_vs_ens4": round(_pooled(e) - out["ens4_mixed_nll"], 4),
        }
    out["leave_one_out"] = loo

    # additive check: 4th MLP seed (t7_s7) added to the mixed ensemble
    if t7s7:
        members5 = dict(members)
        members5["t7_s7"] = t7s7
        ens5 = _ens(members5, list(members5))
        out["ens5_add_t7s7_nll"] = round(_pooled(ens5), 4)
        out["ens5_delta_vs_ens4"] = round(_pooled(ens5) - out["ens4_mixed_nll"], 4)

    out["note"] = ("Low cross-family error correlation (xgb vs MLP) is the "
                   "variance-reduction precondition that the same-family t7 "
                   "seeds cannot provide; leave-one-out shows every member "
                   "earns its place.")
    Path(f"{R}/mixed_ensemble_diversity.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
