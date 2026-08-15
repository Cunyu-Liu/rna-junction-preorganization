"""r48: feature-diverse ensemble member test (nuisance-only t7 member).

All 7 members of the frozen ensemble use the SAME feature block (nuisance +
21-D extended-ViennaRNA); they differ only by family (GBDT vs MLP) and seed.
The diversity diagnostic showed high inter-member error correlation (0.84-0.95).
The r31 run materialized a full-37-fold `nonlinear_mlp_nuisance_only_t7`
member (motif + scaffold + topology ONLY, no ViennaRNA).  Its error structure
is genuinely different (it cannot see any folding/sequence signal), so adding
it to the equal-weight ensemble should reduce variance further IF its quality
is not too low.

Tests:
  1. error correlation of nuisance-only vs each current member (diversity)
  2. pooled NLL of the nuisance-only member alone (quality)
  3. equal-weight ensemble with the nuisance-only member added (8 members,
     family-equal: GBDT(4) + MLP(4)) and with it substituted (replace weakest)
  4. r45 per-scaf x stratum sigma applied to each ensemble variant (frozen
     calibration), horizontal comparison
All on the same 37 blocked joint folds, optimizer+full-coverage eligible
predictions only (fail-closed).
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
R31 = f"{R}/r31_nuisance_only_full/Predictions_v3.jsonl"

R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R34_LEDGER = f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet"
R35_LEDGER = f"{R}/r35_gbdt_hp_full/ConvergenceLedger_v3.parquet"
R24_LEDGERS = [
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
]
R31_LEDGER = f"{R}/r31_nuisance_only_full/ConvergenceLedger_v3.parquet"

XGB = "xgboost_censored_hybrid"
XGB_S99 = "xgboost_censored_hybrid_s99"
XGB_S2026 = "xgboost_censored_hybrid_s2026"
XGB_LR03 = "xgboost_censored_hybrid_hp_lr03"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUS = "nonlinear_mlp_nuisance_only_t7"
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


def _ens_mu(members, keys, rid_common):
    """Equal-weight mu-mean over selected members, intersected on rids."""
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
    grid = grid if grid is not None else np.arange(0.4, 1.4, 0.01)
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
    """LOO per-scaf x stratum sigma (r45)."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
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
    elig31 = _elig([R31_LEDGER])
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows24 = _load(R24)
    rows31 = _load(R31)

    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[T7] = _by_rid(rows24, T7, elig24)
    members[T7_S99] = _by_rid(rows24, T7_S99, elig24)
    members[T7_S2026] = _by_rid(rows24, T7_S2026, elig24)
    members[NUS] = _by_rid(rows31, NUS, elig31)
    nuis = _by_rid(rows33, NUIS, elig33)

    out = {"n_rows": 0}
    # error correlation: nuisance-only vs each Vienna member (diversity)
    shared = set(members[ALL_MEMBERS[0]])
    for m in ALL_MEMBERS + [NUS]:
        shared &= set(members[m])
    shared = sorted(shared)
    out["n_shared_rows"] = len(shared)
    err = {m: np.asarray([members[m][r]["y"] - members[m][r]["mu"] for r in shared])
           for m in ALL_MEMBERS + [NUS]}
    corrs = {}
    for m in ALL_MEMBERS:
        c = float(np.corrcoef(err[NUS], err[m])[0, 1])
        corrs[f"{NUS} vs {m}"] = round(c, 4)
    out["nuisance_only_error_corr_vs_members"] = corrs
    out["nuisance_only_mean_abs_error_corr"] = round(
        float(np.mean([abs(v) for v in corrs.values()])), 4)
    # within-family baseline
    t7_corrs = []
    for i, a in enumerate(MLP):
        for b in MLP[i + 1:]:
            t7_corrs.append(abs(float(np.corrcoef(err[a], err[b])[0, 1])))
    out["mean_abs_within_mlp_t7_corr"] = round(float(np.mean(t7_corrs)), 4)

    # quality of the nuisance-only member alone
    out["nuisance_only_nll_frozen07"] = round(
        _pooled({r: {**members[NUS][r], "sigma": 0.7} for r in members[NUS]}), 4)
    out["nuisance_only_nll_r45"] = round(
        _pooled(_calibrate_r45(members[NUS],
                               sorted(set(members[NUS][r]["fold"] for r in members[NUS])))), 4)

    # ensembles
    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS + [NUS]]))
    out["n_rows"] = len(common)
    ens7 = _ens_mu(members, ALL_MEMBERS, common)
    # 8-member: add nuisance-only (family-equal GBDT4 + MLP4)
    ens8 = _ens_mu(members, ALL_MEMBERS + [NUS], common)
    # 8-member alt: replace the weakest MLP t7 with nuisance-only
    # (weakest = worst frozen-NLL among MLP members)
    mlp_nll = {m: _pooled({r: {**members[m][r], "sigma": 0.7} for r in members[m]})
               for m in MLP}
    weakest = min(mlp_nll, key=mlp_nll.get)
    keys7b = [m for m in ALL_MEMBERS if m != weakest] + [NUS]
    ens7b = _ens_mu(members, keys7b, common)

    variants = {
        "7mem_frozen": ens7,
        "8mem_add_nuisance": ens8,
        "7mem_replace_weakest_mlp": ens7b,
    }
    frozen = {}
    calib = {}
    for name, ens in variants.items():
        folds = sorted(set(ens[r]["fold"] for r in ens))
        frozen[name] = round(_pooled({r: {**p, "sigma": 0.7} for r, p in ens.items()}), 4)
        calib[name] = round(_pooled(_calibrate_r45(ens, folds)), 4)
    out["ensemble_frozen07_nll"] = frozen
    out["ensemble_r45_nll"] = calib
    out["weakest_mlp_member"] = weakest
    out["mlp_member_nll_frozen07"] = {k: round(v, 4) for k, v in mlp_nll.items()}

    # edit-cluster CI: best variant (r45) vs nuisance r45
    best_name = min(calib, key=calib.get)
    folds_best = sorted(set(variants[best_name][r]["fold"] for r in variants[best_name]))
    cal_best = _calibrate_r45(variants[best_name], folds_best)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["best_variant"] = best_name
    out[f"edit_cluster_CI_{best_name}_r45_vs_nuisance_r45"] = _edit_ci(cal_best, cal_nuis)

    out["note"] = (
        "r48 lever: feature-diverse ensemble member.  All 7 frozen members use "
        "nuisance + ViennaRNA-extended features; the r31 nuisance-only t7 "
        "member (motif+scaffold+topology, no folding/sequence) has a genuinely "
        "different error structure.  Tests whether adding/substituting it "
        "reduces ensemble variance (r45-calibrated).  Equal-weight, LOO honest."
    )
    Path(f"{R}/r48_feature_diverse_member.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
