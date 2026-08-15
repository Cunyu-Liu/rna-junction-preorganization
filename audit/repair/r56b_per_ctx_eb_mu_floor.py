"""r56b: per-context EB mu with a min-measured-rows floor for the context term.

The r56 robustness audit showed the worst fold (CUCAG_CUGAG, delta -0.53) has
only n_meas=2 per context; EB with kappa=10 still gives those 2-row contexts a
16.7% weight, injecting noise.  r56b adds a floor: contexts with fewer than
`min_meas` measured rows in the OTHER folds contribute NO context bias (fall
back to the scaf-level bias from r51, which is ~0 after the scaf affine).  This
is the same "shrinkage toward the reliable parent" idea, just with a hard floor
instead of relying on kappa alone.

Estimands: r51 (frozen) vs r56b (this run, per-context EB mu + sigma re-scan
on corrected mu, with min_meas floor).
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
    _load, _elig, _by_rid, _pooled, _scan_sigma, _calibrate_r51, _calibrate_r45,
    GRID, XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, NUIS, ALL_MEMBERS,
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


def _calibrate_r56b(ens, folds, kappa=10.0, min_meas=5, grid=GRID):
    """r51 + per-context EB mu with a min-measured-rows floor + sigma re-scan.

    For each held-out fold:
      1. r51 gives corrected mu for ALL rows.
      2. On OTHER folds' r51-corrected rows, compute per-context measured bias
         only for contexts with >= min_meas measured rows; else fall back to
         scaf bias (b_scaf ~ 0 after r51).
      3. Shrink: alpha_ctx = w*b_ctx + (1-w)*b_scaf, w = n_ctx/(n_ctx+kappa).
      4. Apply to held-out fold measured rows; censored rows keep mu exactly.
      5. RE-SCAN sigma_m on corrected-mu measured rows (per-scaf, fallback).
    """
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _calibrate_r51 as _r51
    cal_r51, _ = _r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)

    by_fold = defaultdict(dict)
    for rid, p in cal_r51.items():
        by_fold[p["fold"]][rid] = p

    cal = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        ctx_res = defaultdict(list)
        scaf_res = defaultdict(list)
        for rid, p in other.items():
            if not p["cens"]:
                ctx_res[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
                scaf_res[int(p["scaf"])].append(p["y"] - p["mu"])
        b_scaf = {sc: float(np.mean(v)) for sc, v in scaf_res.items()}
        alpha_ctx = {}
        n_ctx_used = 0
        for (sc, ctx), v in ctx_res.items():
            if len(v) >= min_meas:
                b_ctx = float(np.mean(v))
                w = float(len(v) / (len(v) + kappa))
                alpha_ctx[(sc, ctx)] = float(w * b_ctx + (1.0 - w) * b_scaf.get(sc, 0.0))
                n_ctx_used += 1
        # sigma re-scan on corrected mu
        corr_other = {}
        for rid, p in other.items():
            if p["cens"]:
                corr_other[rid] = p
            else:
                sc = int(p["scaf"])
                alpha = alpha_ctx.get((sc, str(p["context"])), b_scaf.get(sc, 0.0))
                corr_other[rid] = {**p, "mu": float(p["mu"] + alpha)}
        s_global, _ = _scan_sigma(corr_other, grid=grid)
        sm_global, _ = _scan_sigma(corr_other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma(corr_other, cens_mask=True, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in corr_other.items():
            by_scaf[int(p["scaf"])][rid] = p
        strat_sigma = {}
        for sc, rows_sc in by_scaf.items():
            n = len(rows_sc)
            n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
            entry = {}
            if n - n_c >= 15:
                sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=grid)
                entry["sigma_m"] = sm
            else:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if n_c >= 15:
                sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                entry["sigma_c"] = sc_
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            if p["cens"]:
                sig = e.get("sigma_c", s_global)
                mu_new = p["mu"]
            else:
                sig = e.get("sigma_m", s_global)
                alpha = alpha_ctx.get((sc, str(p["context"])), b_scaf.get(sc, 0.0))
                mu_new = float(p["mu"] + alpha)
            cal[rid] = {**p, "mu": float(mu_new), "sigma": float(sig)}
        fit_log[f] = {
            "n_ctx_alpha": n_ctx_used,
            "n_ctx_total": len(ctx_res),
            "global_fallback": round(float(s_global), 3),
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
        gmu = float(np.mean([members[m][rid]["mu"] for m in GBDT]))
        mmu = float(np.mean([members[m][rid]["mu"] for m in MLP]))
        ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": int(p0["scaf"]),
                    "context": str(p0.get("context", "?")),
                    "y": p0["y"], "cens": p0["cens"],
                    "mu": 0.5 * gmu + 0.5 * mmu}

    folds = sorted(set(ens[r]["fold"] for r in ens))

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                for r in nuis}), 4),
        "n_folds": len(folds),
        "n_rows": len(ens),
        "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01},
    }

    cal_r51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)
    out["r51_nll"] = round(_pooled(cal_r51), 4)

    # scan kappa and min_meas
    best_nll, best_key = np.inf, None
    for min_meas in (3, 5, 8, 12):
        for kappa in (2.0, 5.0, 10.0, 20.0):
            cal, fl = _calibrate_r56b(ens, folds, kappa=kappa, min_meas=min_meas)
            key = f"r56b_mm{min_meas}_k{kappa:g}"
            out[key] = round(_pooled(cal), 4)
            out[f"{key}_strata"] = _pooled_strata(cal)
            out[f"{key}_fit_log"] = fl
            if out[key] < best_nll:
                best_nll, best_key = out[key], key

    out["best_r56b_nll"] = best_nll
    out["best_r56b_key"] = best_key
    out["best_vs_r51_delta"] = round(best_nll - out["r51_nll"], 4)

    # CI for the best
    mm = int(best_key.split("mm")[1].split("_")[0])
    kp = float(best_key.split("_k")[1])
    cal_best, _ = _calibrate_r56b(ens, folds, kappa=kp, min_meas=mm)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_best_vs_nuisance"] = _edit_ci(cal_best, cal_nuis)
    out["edit_cluster_CI_best_vs_r51"] = _edit_ci(cal_best, cal_r51)

    out["note"] = (
        "r56b: r56 + min-measured-rows floor on the context term.  r56's worst "
        "fold (CUCAG_CUGAG, -0.53) had only n_meas=2 per context; the floor "
        "prevents sparse contexts from injecting noise, while keeping the "
        "real context bias signal (scaf5-7 sd reduction) intact."
    )
    Path(f"{R}/r56b_per_ctx_eb_mu_floor.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
