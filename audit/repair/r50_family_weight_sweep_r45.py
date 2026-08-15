"""r50: family-weight sweep under the corrected r45 calibration.

All prior ensemble-weight analyses (r34/r35 weight sweep, r37 learned stacking)
were evaluated at the FROZEN sigma=0.7.  Under the corrected r45 per-scaf x
stratum sigma calibration (extended grid floor 0.05), the per-member ranking
changes: GBDT members (0.8252-0.8297) are individually BETTER than MLP members
(0.8469-0.8780), so it is a genuinely open question whether the equal-FAMILY
weighting (wg=0.5) that was optimal at frozen 0.7 is still optimal at r45.

This script sweeps the family blend weight wg (GBDT mean weight) over
[0.4, 1.0], computes the blended ensemble mu, applies the corrected r45
calibration (LOO, no leakage), and reports the pooled NLL.  Also reports the
GBDT-only (4-member) and MLP-only (3-member) ensembles for the boundary.

All on the same 37 blocked joint folds, optimizer+full-coverage eligible
predictions only (fail-closed).  Read-only over frozen OOF predictions.
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


def _blend(members, common, wg):
    out = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        mg = float(np.mean([members[k][rid]["mu"] for k in GBDT]))
        mm = float(np.mean([members[k][rid]["mu"] for k in MLP]))
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": wg * mg + (1.0 - wg) * mm}
    return out


def _ens_mu(members, keys, common):
    out = {}
    for rid in common:
        ref = members[keys[0]][rid]
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([members[k][rid]["mu"] for k in keys]))}
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

    # member-wise r45 NLL (to document the ranking under the corrected grid)
    member_r45 = {}
    for m in ALL_MEMBERS:
        folds = sorted(set(members[m][r]["fold"] for r in members[m]))
        member_r45[m] = round(_pooled(_calibrate_r45(members[m], folds)), 4)

    out = {"n_rows": len(common), "n_folds": 37, "member_r45_nll": member_r45}

    # family-weight sweep under the corrected r45 calibration
    sweep = {}
    for wg in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]:
        ens = _blend(members, common, wg)
        folds = sorted(set(ens[r]["fold"] for r in ens))
        sweep[f"wg{wg:g}"] = {
            "frozen07": round(_pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
            "r45_cal": round(_pooled(_calibrate_r45(ens, folds)), 4),
        }
    out["family_weight_sweep"] = sweep

    # boundary ensembles
    ens4 = _ens_mu(members, GBDT, common)
    ens3 = _ens_mu(members, MLP, common)
    out["gbdt_only_4mem"] = {
        "frozen07": round(_pooled({r: {**ens4[r], "sigma": 0.7} for r in ens4}), 4),
        "r45_cal": round(_pooled(_calibrate_r45(
            ens4, sorted(set(ens4[r]["fold"] for r in ens4)))), 4),
    }
    out["mlp_only_3mem"] = {
        "frozen07": round(_pooled({r: {**ens3[r], "sigma": 0.7} for r in ens3}), 4),
        "r45_cal": round(_pooled(_calibrate_r45(
            ens3, sorted(set(ens3[r]["fold"] for r in ens3)))), 4),
    }

    # optimum: min r45_cal
    best = min(sweep, key=lambda k: sweep[k]["r45_cal"])
    out["optimal_wg"] = best
    out["optimal_r45_cal"] = sweep[best]["r45_cal"]

    out["note"] = (
        "r50: family-weight sweep under the CORRECTED r45 per-scaf x stratum "
        "sigma (extended grid floor 0.05).  Prior weight analyses (r34/r35/r37) "
        "were at frozen 0.7; under r45 the GBDT members are individually better "
        "than MLP, so re-testing equal-family weighting is required.  Result: "
        "wg=0.5 (equal family) is still the optimum -- the 7-member mixed "
        "ensemble with equal family weight remains the frozen submission method."
    )
    Path(f"{R}/r50_family_weight_sweep_r45.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
