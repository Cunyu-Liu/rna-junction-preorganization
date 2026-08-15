"""Fair horizontal table for the r38 per-scaffold sigma calibration.

Per-scaffold sigma is fit leave-one-fold-out on the OTHER folds' OOF rows and
applied to the held-out fold.  For a fair horizontal contrast the SAME
procedure must be applied to EVERY comparator (nuisance, each single member,
the 3x t7 ensemble, and the 7-member ensemble).  Otherwise calibrating only the
winning ensemble would bias the comparison.

Estimands (MetricSpec_v3 pooled-junction-macro right-censored Gaussian NLL):
  - frozen sigma=0.7      : r24/r34/r35 freeze (all models emit 0.7)
  - per-scaffold sigma LOO : r38 calibration applied to every comparator
Relative gain and edit-cluster CI are computed against the nuisance baseline
at the SAME sigma treatment, so the horizontal table is apples-to-apples.
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


def _per_scaf_calibrate(preds, folds, min_rows=20):
    """Leave-one-fold-out per-scaffold sigma. Returns (calibrated, sigma_map)."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
    cal = {}
    sigma_map = {}
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
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= min_rows:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
        sigma_map[f] = scaf_sigma
    return cal, sigma_map


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

    # equal-weight 7-member ensemble
    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ens7 = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens7[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                     "y": ref["y"], "cens": ref["cens"],
                     "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS]))}

    # 3x t7 ensemble
    ens3 = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens3[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                     "y": ref["y"], "cens": ref["cens"],
                     "mu": float(np.mean([members[m][rid]["mu"] for m in MLP]))}

    models = {
        "nuisance": nuis,
        "t7_s99": members[T7_S99],
        "xgb_lr03": members[XGB_LR03],
        "3x_t7_ensemble": ens3,
        "7mem_mixed_ensemble": ens7,
    }

    # frozen sigma=0.7 table
    frozen = {}
    for name, preds in models.items():
        frozen[name] = round(_pooled({r: {**p, "sigma": 0.7} for r, p in preds.items()}), 4)

    # per-scaffold sigma LOO table (applied to EVERY comparator)
    calib = {}
    calib_sigma = {}
    for name, preds in models.items():
        folds = sorted(set(preds[r]["fold"] for r in preds))
        cal, sigma_map = _per_scaf_calibrate(preds, folds)
        calib[name] = round(_pooled(cal), 4)
        calib_sigma[name] = sigma_map

    out = {
        "frozen_sigma_07_nll": frozen,
        "per_scaf_sigma_loo_nll": calib,
    }
    # relative gain vs nuisance at the SAME sigma treatment
    out["frozen_sigma_07_rel_gain_pct"] = {
        k: round(100.0 * (frozen["nuisance"] - v) / frozen["nuisance"], 2)
        for k, v in frozen.items() if k != "nuisance"}
    out["per_scaf_rel_gain_pct"] = {
        k: round(100.0 * (calib["nuisance"] - v) / calib["nuisance"], 2)
        for k, v in calib.items() if k != "nuisance"}

    # edit-cluster CI: 7mem per-scaf vs nuisance per-scaf (both calibrated)
    cal_ens7, _ = _per_scaf_calibrate(ens7, sorted(set(ens7[r]["fold"] for r in ens7)))
    cal_nuis, _ = _per_scaf_calibrate(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_7mem_per_scaf_vs_nuisance_per_scaf"] = _edit_ci(cal_ens7, cal_nuis)
    # and vs nuisance at frozen 0.7 (transparency: mixing sigma treatments)
    out["edit_cluster_CI_7mem_per_scaf_vs_nuisance_frozen07"] = _edit_ci(
        cal_ens7, {r: {**nuis[r], "sigma": 0.7} for r in nuis})

    out["n_rows"] = len(common)
    out["note"] = (
        "Fair horizontal table: the r38 per-scaffold sigma calibration is "
        "applied to EVERY comparator (fit leave-one-fold-out on other folds' "
        "OOF rows, applied to the held-out fold).  Relative gains and "
        "edit-cluster CIs are computed at the SAME sigma treatment.  sigma is "
        "a model-emitted parameter per MetricSpec, so calibration is a "
        "legitimate method improvement.  Frozen-0.7 table shown for reference."
    )
    Path(f"{R}/per_scaf_sigma_horizontal_table.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
