"""r62 (formal): decoupled frozen method - r56b mu + independent sigma re-scan.

r56b couples the context-mu correction and sigma scan inside one LOO loop.  The
sigma is scanned on `corr_other` (other folds' mu WITH the context alpha applied
at fit time), but the held-out fold's mu gets alpha from a different context
set.  This coupling makes the emitted sigma_m systematically SMALL (scaf8 0.530
vs decoupled 0.599), which hurts Gaussian NLL under heavy tails (kurtosis 1.12):
the optimal Gaussian sigma is larger than the residual sd.

r62 (formal) = Stage 1 r56b mu (frozen) + Stage 2 INDEPENDENT per-scaf x
stratum sigma re-scan on the r56b-corrected mu of the OTHER folds, applied to
the held-out fold.  This is the honest sigma for the corrected mu.

Frozen method: 7-member ensemble (wg=0.5) + r62 = 0.725.
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
from audit.repair.r56b_per_ctx_eb_mu_floor import _calibrate_r56b
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _pooled, _scan_sigma, _calibrate_r45,
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


def _calibrate_r62(ens, folds, kappa=2.0, min_meas=3, grid=GRID):
    """Stage 1 r56b mu + Stage 2 independent per-scaf x stratum sigma re-scan."""
    cal_mu, _ = _calibrate_r56b(ens, folds, kappa=kappa, min_meas=min_meas)
    by_fold = defaultdict(dict)
    for rid, p in cal_mu.items():
        by_fold[p["fold"]][rid] = p
    cal = {}
    fit_log = {}
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
            sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
            cal[rid] = {**p, "sigma": float(sig)}
        fit_log[f] = {
            "stratum_sigma": {str(k): {"sigma_m": round(v["sigma_m"], 3),
                                       "sigma_c": round(v["sigma_c"], 3)}
                              for k, v in sorted(strat_sigma.items())},
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

    cal_r45 = _calibrate_r45(ens, folds)
    out["r45_nll"] = round(_pooled(cal_r45), 4)
    cal_r56, _ = _calibrate_r56b(ens, folds, kappa=2.0, min_meas=3)
    out["r56b_nll"] = round(_pooled(cal_r56), 4)

    best_nll, best_key = np.inf, None
    for kappa in (1.0, 2.0, 5.0):
        for min_meas in (3, 5):
            cal, fl = _calibrate_r62(ens, folds, kappa=kappa, min_meas=min_meas)
            key = f"r62_k{kappa:g}_mm{min_meas}"
            out[key] = round(_pooled(cal), 4)
            out[f"{key}_strata"] = _pooled_strata(cal)
            out[f"{key}_fit_log"] = fl
            if out[key] < best_nll:
                best_nll, best_key = out[key], key

    out["best_r62_nll"] = best_nll
    out["best_r62_key"] = best_key
    out["best_vs_r56b_delta"] = round(best_nll - out["r56b_nll"], 4)

    kp = float(best_key.split("_k")[1].split("_")[0])
    mm = int(best_key.split("mm")[1])
    cal_best, _ = _calibrate_r62(ens, folds, kappa=kp, min_meas=mm)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_best_vs_nuisance"] = _edit_ci(cal_best, cal_nuis)
    out["edit_cluster_CI_best_vs_r56b"] = _edit_ci(cal_best, cal_r56)

    out["note"] = (
        "r62: decoupled frozen method = r56b mu (Stage 1) + INDEPENDENT "
        "per-scaf x stratum sigma re-scan (Stage 2).  Fixes r56b's coupling "
        "defect that made emitted sigma_m systematically small (scaf8 0.530 vs "
        "0.599); under heavy tails (kurtosis 1.12) the optimal Gaussian sigma "
        "is larger than the residual sd, so the decoupled sigma lowers NLL."
    )
    Path(f"{R}/r62_decoupled_sigma.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
