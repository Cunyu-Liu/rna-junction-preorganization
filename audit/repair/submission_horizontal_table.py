"""Final submission horizontal comparison table (corrected frozen method).

Combines the complete model-family comparison into ONE authoritative table:

  1. ALL families at frozen sigma=0.7 (the pre-calibration primary estimand),
     consumed from the same 37 blocked joint folds, optimizer+full-coverage
     eligible predictions only (fail-closed).
  2. The corrected r45 per-scaffold x stratum sigma calibration (extended grid
     floor 0.05 = MetricSpec floor, LOO, no leakage) applied to EVERY family.
  3. The frozen submission method (7-member mixed ensemble + per-scaf x stratum
     sigma) highlighted with its edit-cluster group CI.

Relative gain is always vs motif_topology_hierarchy (nuisance) at the SAME
sigma treatment so the horizontal comparison is apples-to-apples.  All numbers
trace to row-level predictions in the run roots.

Output: submission_horizontal_table.json + a markdown table for the manuscript.
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
R29 = f"{R}/r29_p05_rerun/Predictions_v3.jsonl"
R31 = f"{R}/r31_nuisance_only_full/Predictions_v3.jsonl"
R14 = f"{R}/r14_extended_mlp_scan/Predictions_v3.jsonl"

R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R34_LEDGER = f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet"
R35_LEDGER = f"{R}/r35_gbdt_hp_full/ConvergenceLedger_v3.parquet"
R24_LEDGERS = [
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
]
R29_LEDGER = f"{R}/r29_p05_rerun/ConvergenceLedger_v3.parquet"
R31_LEDGER = f"{R}/r31_nuisance_only_full/ConvergenceLedger_v3.parquet"
R14_LEDGER = f"{R}/r14_extended_mlp_scan/ConvergenceLedger_v3.parquet"

XGB = "xgboost_censored_hybrid"
XGB_S99 = "xgboost_censored_hybrid_s99"
XGB_S2026 = "xgboost_censored_hybrid_s2026"
XGB_LR03 = "xgboost_censored_hybrid_hp_lr03"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUS = "nonlinear_mlp_nuisance_only_t7"
NUIS = "motif_topology_hierarchy"
C131 = "corrected_v1_31"
NOSEQ = "no_sequence_latent_operator"
SCAF = "train_only_scaffold"
REGDEEP = "nonlinear_mlp_extended_hybrid_reg_deep"
T5 = "nonlinear_mlp_extended_hybrid_reg_deep_t"
T10 = "nonlinear_mlp_extended_hybrid_reg_deep_t10"
T7S7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s7"

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


def _ens_mu(members, keys, rid_common):
    out = {}
    for rid in rid_common:
        ref = members[keys[0]][rid]
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([members[k][rid]["mu"] for k in keys]))}
    return out


def _scan_sigma(rows, cens_mask=None, grid=None):
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    if cens_mask is None:
        sel = np.ones(len(y), dtype=bool)
    else:
        sel = cens == cens_mask
    if not sel.any():
        return None, np.inf
    y, cens, mu, jid = y[sel], cens[sel], mu[sel], jid[sel]
    uniq, jcode = np.unique(jid, return_inverse=True)
    jcounts = np.bincount(jcode, minlength=len(uniq))
    grid = grid if grid is not None else np.arange(0.05, 1.6, 0.01)
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
        jm = sums / jcounts
        nll = float(np.mean(jm[jcounts > 0]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _calibrate_r45(preds, folds, min_rows=15):
    """LOO per-scaf x stratum sigma (corrected r45, extended grid)."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.05, 1.6, 0.01)
    cal = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        sm_global, _ = _scan_sigma(other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma(other, cens_mask=True, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        strat_sigma = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {}
            if n - n_c >= min_rows:
                sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if n_c >= min_rows:
                sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal[rid] = {**p, "sigma": sig}
    return cal


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


def main():
    print("Loading predictions...", file=sys.stderr)
    elig33 = _elig([R33_LEDGER])
    elig34 = _elig([R34_LEDGER])
    elig35 = _elig([R35_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    elig29 = _elig([R29_LEDGER])
    elig31 = _elig([R31_LEDGER])
    elig14 = _elig([R14_LEDGER])
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows24 = _load(R24)
    rows29 = _load(R29)
    rows31 = _load(R31)
    rows14 = _load(R14)

    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[T7] = _by_rid(rows24, T7, elig24)
    members[T7_S99] = _by_rid(rows24, T7_S99, elig24)
    members[T7_S2026] = _by_rid(rows24, T7_S2026, elig24)
    members[NUS] = _by_rid(rows31, NUS, elig31)
    singles = {
        C131: _by_rid(rows29, C131, elig29),
        NOSEQ: _by_rid(rows29, NOSEQ, elig29),
        SCAF: _by_rid(rows29, SCAF, elig29),
        NUIS: _by_rid(rows29, NUIS, elig29),
        REGDEEP: _by_rid(rows14, REGDEEP, elig14),
        T5: _by_rid(rows24, T5, elig24),
        T10: _by_rid(rows24, T10, elig24),
        T7S7: _by_rid(rows24, T7S7, elig24),
    }
    for m in ALL_MEMBERS:
        singles[m] = members[m]

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ens7 = _ens_mu(members, ALL_MEMBERS, common)
    ens3 = _ens_mu(members, MLP, common)

    models = dict(singles)
    models["ENSEMBLE_3x_t7"] = ens3
    models["ENSEMBLE_MIXED_7"] = ens7

    out = {"n_rows": len(common), "n_folds": 37,
           "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01,
                          "note": "MetricSpec floor = 0.05 (corrected r45)"}}

    frozen = {}
    calib = {}
    for name, preds in models.items():
        folds = sorted(set(preds[r]["fold"] for r in preds))
        frozen[name] = round(_pooled({r: {**p, "sigma": 0.7} for r, p in preds.items()}), 4)
        calib[name] = round(_pooled(_calibrate_r45(preds, folds)), 4)

    out["frozen_sigma_07_nll"] = frozen
    out["r45_calibrated_nll"] = calib
    out["frozen_rel_gain_pct_vs_nuisance"] = {
        k: round(100.0 * (frozen[NUIS] - v) / frozen[NUIS], 2)
        for k, v in frozen.items() if k != NUIS}
    out["r45_rel_gain_pct_vs_nuisance"] = {
        k: round(100.0 * (calib[NUIS] - v) / calib[NUIS], 2)
        for k, v in calib.items() if k != NUIS}

    # edit-cluster CI for the frozen method vs nuisance (r45-calibrated)
    folds7 = sorted(set(ens7[r]["fold"] for r in ens7))
    cal7 = _calibrate_r45(ens7, folds7)
    cal_nuis = _calibrate_r45(models[NUIS], sorted(set(models[NUIS][r]["fold"] for r in models[NUIS])))
    out["frozen_method_r45_nll"] = round(_pooled(cal7), 4)
    out["frozen_method_vs_nuisance_r45"] = {
        "rel_gain_pct": round(100.0 * (_pooled(cal_nuis) - _pooled(cal7)) / _pooled(cal_nuis), 2),
        "edit_cluster_CI": _edit_ci(cal7, cal_nuis),
    }
    out["frozen_method_vs_nuisance_frozen07"] = {
        "rel_gain_pct": round(100.0 * (frozen[NUIS] - frozen["ENSEMBLE_MIXED_7"]) / frozen[NUIS], 2),
        "edit_cluster_CI": _edit_ci({r: {**ens7[r], "sigma": 0.7} for r in ens7},
                                    {r: {**models[NUIS][r], "sigma": 0.7} for r in models[NUIS]}),
    }

    out["note"] = (
        "DEFINITIVE submission horizontal table.  frozen_sigma_07 = every family "
        "at the pre-calibration frozen sigma 0.7; r45_calibrated = corrected "
        "per-scaf x stratum sigma (extended grid floor 0.05, LOO, no leakage) "
        "applied to EVERY family.  Relative gain always vs motif_topology_"
        "hierarchy (nuisance) at the SAME treatment.  Frozen submission method "
        "= 7-member mixed ensemble + r45 calibration."
    )
    Path(f"{R}/submission_horizontal_table.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
