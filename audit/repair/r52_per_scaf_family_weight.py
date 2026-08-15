"""r52: LOO per-scaffold family-weight + r51 joint calibration.

r50 confirmed equal-family weighting (wg=0.5) is the global optimum under r45.
But the residual diagnostic shows per-scaffold measured bias/error heterogeneity
(scaf9 RMSE 1.15 vs scaf2 0.45).  A quick (LEAKY, same-fold) probe suggested
per-scaffold GBDT-vs-MLP weights vary (scaf4 prefers wg=0.7, scaf6 prefers
wg=0.3).  r52 tests this HONESTLY: per-scaffold family weight fit on the OTHER
folds' OOF rows (LOO, no test-label leakage), applied to the held-out fold.

The member blend is:  mu_sc = wg_sc * mean(GBDT members) + (1-wg_sc) * mean(MLP)
Each scaffold gets its own wg_sc in [0.0, 1.0], chosen by grid search on the
other folds' pooled-OOF junction-macro NLL under the CORRECTED r45 stratum
sigma.  This is combined on top of the r51 joint (mu-affine + sigma re-scan)
calibration so we see the marginal value of per-scaf family weighting after
mu/sigma calibration has already removed the systematic bias.

Estimands (pooled-OOF junction-macro right-censored Gaussian NLL):
  - frozen 0.7
  - r45 (corrected grid)
  - r51 (joint mu-affine + sigma re-scan; best variant)
  - r52 = r51 + LOO per-scaf family weight
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
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _pooled, _scan_sigma, GRID,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, NUIS, ALL_MEMBERS,
)

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

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]
WG_GRID = np.arange(0.0, 1.01, 0.05)


def _pooled_strata(ens):
    jd_m = defaultdict(list)
    jd_c = defaultdict(list)
    for rid, p in ens.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        (jd_c if p["cens"] else jd_m)[p["jid"]].append(nll)
    out = {}
    for name, d in (("measured", jd_m), ("censored", jd_c)):
        out[name] = float(np.mean([np.mean(v) for v in d.values()])) if d else None
    return out


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


def _calibrate_r45(ens, folds):
    """Corrected r45 reference (grid floor 0.05), no mu change."""
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _calibrate_r45 as _c
    return _c(ens, folds)


def _blend_per_scaf(members, common, wg_sc, ref_key=None, gbdt=None, mlp=None):
    """Blend GBDT/MLP families with per-scaffold weight wg_sc: scaf->weight.

    `ref_key` names the member used as the row-reference for fold/scaf/jid/y/cens
    (defaults to ALL_MEMBERS[0]).  `gbdt`/`mlp` name the two family member lists
    (defaults to the module constants GBDT/MLP).
    """
    ref_key = ref_key or ALL_MEMBERS[0]
    gbdt = gbdt or GBDT
    mlp = mlp or MLP
    out = {}
    for rid in common:
        ref = members[ref_key][rid]
        sc = int(ref["scaf"])
        wg = float(wg_sc.get(sc, 0.5))
        gmu = float(np.mean([members[m][rid]["mu"] for m in gbdt]))
        mmu = float(np.mean([members[m][rid]["mu"] for m in mlp]))
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": wg * gmu + (1.0 - wg) * mmu}
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
    nuis = _by_rid(rows33, NUIS, elig33)

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    folds = sorted(set(members[ALL_MEMBERS[0]][r]["fold"] for r in common))

    # precompute per-row family mus
    gmu_map = {rid: float(np.mean([members[m][rid]["mu"] for m in GBDT])) for rid in common}
    mmu_map = {rid: float(np.mean([members[m][rid]["mu"] for m in MLP])) for rid in common}

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                for r in nuis}), 4),
        "n_folds": len(folds),
        "n_rows": len(common),
        "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01},
    }

    by_fold = defaultdict(dict)
    for rid in common:
        by_fold[members[ALL_MEMBERS[0]][rid]["fold"]][rid] = rid

    # ---- r52: LOO per-scaf family weight ----
    cal52 = {}
    wg_fit = {}
    for f in folds:
        other_rids = set()
        for ff in folds:
            if ff != f:
                other_rids |= set(by_fold[ff])
        # Fit per-scaf wg on OTHER folds: scan each scaffold's wg keeping
        # others at 0.5, scoring under r45 sigma of the other folds.
        # First compute r45 calibration on OTHER folds to get per-scaf sigma.
        other_preds = {}
        for rid in other_rids:
            ref = members[ALL_MEMBERS[0]][rid]
            other_preds[rid] = {"jid": ref["jid"], "fold": ref["fold"],
                                "scaf": ref["scaf"], "y": ref["y"],
                                "cens": ref["cens"], "mu": 0.5 * gmu_map[rid] + 0.5 * mmu_map[rid]}
        folds_other = sorted(set(other_preds[r]["fold"] for r in other_preds))
        cal45_other = _calibrate_r45(other_preds, folds_other)
        # per-scaf r45 sigma lookup for held-out scoring
        # (we need sigma per row: build from cal45_other by scaffold+stratum)
        sigma_lookup = {}
        for rid, p in cal45_other.items():
            sigma_lookup.setdefault(int(p["scaf"]), {})[p["cens"]] = p["sigma"]
        scaf_list = sorted(set(int(p["scaf"]) for p in other_preds.values()))
        # fit wg per scaffold (grid search on OTHER folds, r45 sigma fixed)
        wg_sc = {}
        fit_delta = {}
        for sc in scaf_list:
            rows_sc = [rid for rid in other_preds if int(other_preds[rid]["scaf"]) == sc]
            if len(rows_sc) < 30:
                wg_sc[sc] = 0.5
                fit_delta[sc] = 0.0
                continue
            best_w, best_n = 0.5, None
            for wg in WG_GRID:
                tot = 0.0
                cnt = 0
                for rid in rows_sc:
                    p = other_preds[rid]
                    mu = wg * gmu_map[rid] + (1.0 - wg) * mmu_map[rid]
                    sig = sigma_lookup[int(p["scaf"])][p["cens"]]
                    nll = float(row_nll([p["y"]], [p["cens"]], [mu], [sig])[0])
                    tot += nll
                    cnt += 1
                n = tot / cnt
                if best_n is None or n < best_n:
                    best_n, best_w = n, wg
            wg_sc[sc] = float(best_w)
            fit_delta[sc] = round(float(best_n) - 0.0, 4)
        wg_fit[f] = wg_sc
        # Apply to held-out fold
        for rid in by_fold[f]:
            ref = members[ALL_MEMBERS[0]][rid]
            sc = int(ref["scaf"])
            wg = float(wg_sc.get(sc, 0.5))
            mu = wg * gmu_map[rid] + (1.0 - wg) * mmu_map[rid]
            cal52[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                          "y": ref["y"], "cens": ref["cens"], "mu": float(mu)}

    # r45 sigma applied to the r52 blend for scoring (and r51 joint as well)
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _calibrate_r51
    cal52_r45 = _calibrate_r45(cal52, folds)
    out["r52_per_scaf_wg_r45_nll"] = round(_pooled(cal52_r45), 4)
    out["r52_strata_r45"] = _pooled_strata(cal52_r45)
    cal52_r51, _ = _calibrate_r51(cal52, folds, mode="per_scaf_eb", eb_kappa=20.0)
    out["r52_per_scaf_wg_r51_nll"] = round(_pooled(cal52_r51), 4)
    out["r52_strata_r51"] = _pooled_strata(cal52_r51)
    out["wg_fit_log"] = {k: {str(kk): vv for kk, vv in v.items()}
                         for k, v in wg_fit.items()}

    # references
    ens50 = {}
    for rid in common:
        ref = members[ALL_MEMBERS[0]][rid]
        ens50[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                      "y": ref["y"], "cens": ref["cens"],
                      "mu": 0.5 * gmu_map[rid] + 0.5 * mmu_map[rid]}
    cal50_r45 = _calibrate_r45(ens50, folds)
    cal50_r51, _ = _calibrate_r51(ens50, folds, mode="per_scaf_eb", eb_kappa=20.0)
    out["r51_ref_r45_nll"] = round(_pooled(cal50_r45), 4)
    out["r51_ref_r51_nll"] = round(_pooled(cal50_r51), 4)

    out["best_nll"] = min(out["r52_per_scaf_wg_r45_nll"], out["r52_per_scaf_wg_r51_nll"],
                          out["r51_ref_r51_nll"])
    out["note"] = (
        "r52: LOO per-scaffold GBDT-vs-MLP family weight on top of r45/r51 "
        "calibration.  Honest: weight per scaffold fit on OTHER folds' OOF rows "
        "(no test-label leakage), applied to held-out fold.  If per-scaf weight "
        "variation is real and stable, this beats the global equal-family wg=0.5."
    )
    Path(f"{R}/r52_per_scaf_family_weight.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
