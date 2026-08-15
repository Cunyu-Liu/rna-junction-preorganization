"""r54: per-context EB shrinkage sigma on top of r51 joint calibration.

r43 tried per-context sigma with HARD bucketing (min_rows=15 cutoff, fallback to
scaf then global) and the old grid floor 0.4.  Result: 0.831 (> 0.8166 per-scaf).
The failure mode: contexts just below the cutoff fall back entirely to scaf sigma,
creating a step discontinuity; contexts barely above the cutoff get a noisy per-context
estimate that overfits the few rows.

r54 fixes both: (1) EB shrinkage sigma = w * sigma_ctx + (1-w) * sigma_scaf, where
w = n_ctx / (n_ctx + kappa), smoothly interpolating between per-context and per-scaf
sigma; (2) corrected grid floor 0.05 (MetricSpec floor).  This is the same shrinkage
idea that worked for mu (r51 per-scaf EB affine) now applied to sigma.

The hierarchy is: context > scaf > global.  For each (scaf, context) with enough rows
to estimate a context sigma, the EB estimate shrinks it toward the scaf sigma.  The
scaf sigma itself is the per-scaf stratum sigma from r45 (scanned on the "other" folds).

Combined with r51 mu calibration (per-scaf EB affine mu + sigma re-scan on corrected
mu) and the equal-family wg=0.5 blend.

Estimands (pooled-OOF junction-macro right-censored Gaussian NLL):
  - frozen 0.7
  - r45 (corrected grid)
  - r51 (joint mu-affine + sigma re-scan; best variant)
  - r54 = r51 + per-context EB shrinkage sigma (this run)
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
    _load, _elig, _by_rid, _pooled, _scan_sigma, _calibrate_r45, _calibrate_r51,
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


def _calibrate_r54(ens, folds, kappa=10.0, min_ctx=5, grid=GRID):
    """Per-context EB shrinkage sigma on top of r51 mu calibration.

    Design (clean): run r51 once on ALL folds -> cal_r51 gives every row its
    corrected mu AND its per-scaf stratum sigma (the scaf target).  Then for
    each held-out fold f, estimate per-context EB sigma from the OTHER folds'
    already-corrected r51 rows (context sigma shrunk toward the scaf stratum
    sigma), and swap sigma on the held-out fold's cal_r51 rows.  mu stays the
    r51-corrected mu.  All fitting on OTHER folds' rows only (LOO, no leakage).
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
        # scaf x stratum sigma target from OTHER folds (already corrected mu)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        s_global, _ = _scan_sigma(other, grid=grid)
        sm_global, _ = _scan_sigma(other, cens_mask=False, grid=grid)
        sc_global, _ = _scan_sigma(other, cens_mask=True, grid=grid)
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
        # per-context EB shrinkage sigma
        by_ctx = defaultdict(dict)
        for rid, p in other.items():
            by_ctx[(int(p["scaf"]), str(p["context"]))][rid] = p
        ctx_sigma = {}
        for (sc, ctx), rows_ctx in by_ctx.items():
            meas = {r: p for r, p in rows_ctx.items() if not p["cens"]}
            cens = {r: p for r, p in rows_ctx.items() if p["cens"]}
            entry = {}
            scaf_sm = strat_sigma.get(sc, {}).get("sigma_m", s_global)
            scaf_sc = strat_sigma.get(sc, {}).get("sigma_c", s_global)
            if len(meas) >= min_ctx:
                sm_ctx, _ = _scan_sigma(meas, cens_mask=False, grid=grid)
                w = float(len(meas) / (len(meas) + kappa))
                entry["sigma_m"] = float(w * sm_ctx + (1.0 - w) * scaf_sm)
            else:
                entry["sigma_m"] = scaf_sm
            if len(cens) >= min_ctx:
                sc_ctx, _ = _scan_sigma(cens, cens_mask=True, grid=grid)
                w = float(len(cens) / (len(cens) + kappa))
                entry["sigma_c"] = float(w * sc_ctx + (1.0 - w) * scaf_sc)
            else:
                entry["sigma_c"] = scaf_sc
            ctx_sigma[(sc, ctx)] = entry
        # apply: swap sigma on held-out fold, keep r51 corrected mu
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            ctx_key = (sc, str(p["context"]))
            if ctx_key in ctx_sigma:
                sig = ctx_sigma[ctx_key].get("sigma_c" if p["cens"] else "sigma_m",
                                            s_global)
            else:
                e = strat_sigma.get(sc, {})
                sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal[rid] = {**p, "sigma": float(sig)}
        fit_log[f] = {
            "n_ctx": len(by_ctx),
            "n_ctx_meas_shrunk": sum(1 for k, v in ctx_sigma.items()
                                     if v["sigma_m"] != strat_sigma.get(k[0], {}).get("sigma_m", s_global)),
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

    # references
    cal_r45 = _calibrate_r45(ens, folds)
    out["r45_nll"] = round(_pooled(cal_r45), 4)
    cal_r51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)
    out["r51_nll"] = round(_pooled(cal_r51), 4)

    # r54: scan kappa
    for kappa in (2.0, 5.0, 10.0, 20.0, 50.0):
        cal, fl = _calibrate_r54(ens, folds, kappa=kappa)
        key = f"r54_kappa{kappa:g}"
        out[key] = round(_pooled(cal), 4)
        out[f"{key}_strata"] = _pooled_strata(cal)
        out[f"{key}_fit_log"] = fl

    best_key = min([k for k in out if k.startswith("r54_")
                    and isinstance(out[k], (int, float))],
                   key=lambda k: out[k])
    out["best_r54_nll"] = out[best_key]
    out["best_r54_key"] = best_key
    out["best_vs_r51_delta"] = round(out[best_key] - out["r51_nll"], 4)

    # Edit-cluster CI for the best r54 variant vs nuisance (r45-calibrated)
    best_kappa = float(best_key.split("kappa")[1])
    cal_best, _ = _calibrate_r54(ens, folds, kappa=best_kappa)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_best_r54_vs_nuisance"] = _edit_ci(cal_best, cal_nuis)
    # best r54 vs r51 (paired)
    out["edit_cluster_CI_best_vs_r51"] = _edit_ci(cal_best, cal_r51)

    out["note"] = (
        "r54: per-context EB shrinkage sigma on top of r51 joint calibration "
        "(mu-affine + sigma re-scan).  Smoothes the hard-bucket cutoff that "
        "caused r43 to fail (0.831 > 0.8166).  sigma_eb = w * sigma_ctx + "
        "(1-w) * sigma_scaf, w = n_ctx / (n_ctx + kappa).  Corrected grid "
        "floor 0.05.  If the within-scaf context heterogeneity is real and "
        "stable, shrinkage should beat per-scaf stratum alone."
    )
    Path(f"{R}/r54_per_ctx_eb_sigma.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()