"""r53: family-weight sweep under the r51 JOINT (mu-affine + sigma re-scan).

r50 swept family weights under the corrected r45 sigma-only calibration and
found wg=0.5 (equal-family) optimal.  r51 adds the measured-only affine mu
correction with sigma re-scanned on the corrected mu.  Since r51 changes the
calibration surface, the optimal family weight may shift, so the sweep must be
re-run under r51 (same logic as r50, different calibrator).

Also reports the member-mean blend (frozen submission method) under r51 so the
submission method can be updated if equal-family wg=0.5 is better.

Estimands (pooled-OOF junction-macro right-censored Gaussian NLL):
  - wg sweep {0.3..1.0} under r45 (reference) and r51 (this run)
  - member-mean blend under r51 (old frozen method)
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
    _load, _elig, _by_rid, _pooled, _calibrate_r45, _calibrate_r51,
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


def _blend(members, common, wg, gbdt=None, mlp=None, ref_key=None):
    out = {}
    gbdt = gbdt or GBDT
    mlp = mlp or MLP
    ref_key = ref_key or ALL_MEMBERS[0]
    for rid in common:
        ref = members[ref_key][rid]
        gmu = float(np.mean([members[k][rid]["mu"] for k in gbdt]))
        mmu = float(np.mean([members[k][rid]["mu"] for k in mlp]))
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"],
                    "mu": wg * gmu + (1.0 - wg) * mmu}
    return out


def _ens_mu(members, keys, common, ref_key=None):
    ref_key = ref_key or keys[0]
    out = {}
    for rid in common:
        ref = members[ref_key][rid]
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
    nuis = _by_rid(rows33, NUIS, elig33)

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    folds = sorted(set(members[ALL_MEMBERS[0]][r]["fold"] for r in common))

    out = {
        "nuisance_nll_frozen07": round(_pooled({r: {**nuis[r], "sigma": 0.7}
                                                for r in nuis}), 4),
        "n_folds": len(folds),
        "n_rows": len(common),
        "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01,
                       "note": "MetricSpec floor (corrected r45 grid), r51 re-scan"},
    }

    sweep = {}
    for wg in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]:
        ens = _blend(members, common, wg)
        r45_cal = _calibrate_r45(ens, folds)
        r51_cal, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)
        sweep[f"wg{wg:g}"] = {
            "frozen07": round(_pooled({r: {**ens[r], "sigma": 0.7} for r in ens}), 4),
            "r45_cal": round(_pooled(r45_cal), 4),
            "r51_cal": round(_pooled(r51_cal), 4),
        }
        print(f"wg={wg:g}  frozen07={sweep[f'wg{wg:g}']['frozen07']}  "
              f"r45={sweep[f'wg{wg:g}']['r45_cal']}  r51={sweep[f'wg{wg:g}']['r51_cal']}",
              file=sys.stderr)
    out["family_weight_sweep"] = sweep
    best = min(sweep, key=lambda k: sweep[k]["r51_cal"])
    out["optimal_wg"] = best
    out["optimal_r51_cal"] = sweep[best]["r51_cal"]
    out["optimal_r45_cal"] = sweep[best]["r45_cal"]

    # member-mean blend (old frozen submission method) under r51
    ens_mm = _ens_mu(members, ALL_MEMBERS, common)
    r51_mm, _ = _calibrate_r51(ens_mm, folds, mode="per_scaf_eb", eb_kappa=20.0)
    out["member_mean_r51_nll"] = round(_pooled(r51_mm), 4)
    out["member_mean_r45_nll"] = round(_pooled(_calibrate_r45(ens_mm, folds)), 4)

    # family-only / single-family references under r51
    for name, keys in (("gbdt_only", GBDT), ("mlp_only", MLP)):
        ens_fam = _ens_mu(members, keys, common)
        cal_fam, _ = _calibrate_r51(ens_fam, folds, mode="per_scaf_eb", eb_kappa=20.0)
        out[f"{name}_r51_nll"] = round(_pooled(cal_fam), 4)

    # Edit-cluster CI of the best wg blend under r51 vs nuisance (r45-calibrated)
    ens_best = _blend(members, common, float(best.replace("wg", "")))
    cal_best, _ = _calibrate_r51(ens_best, folds, mode="per_scaf_eb", eb_kappa=20.0)
    cal_nuis = _calibrate_r45(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))
    out["edit_cluster_CI_best_wg_r51_vs_nuisance"] = _edit_ci(cal_best, cal_nuis)
    # vs the OLD frozen (member-mean + r45)
    out["best_vs_old_frozen_delta"] = round(
        out["optimal_r51_cal"] - out["member_mean_r45_nll"], 4)

    out["note"] = (
        "r53: family-weight sweep under the r51 JOINT calibration "
        "(measured-only affine mu + sigma re-scan on corrected mu, corrected "
        "grid floor 0.05).  r52 showed per-scaf weights HURT (overfit); this "
        "sweep checks the global equal-family optimum has not shifted under r51."
    )
    Path(f"{R}/r53_family_weight_sweep_r51.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
