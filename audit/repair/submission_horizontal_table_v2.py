"""Definitive submission horizontal comparison table v2 (r51 joint calibration).

v2 adds the r51 JOINT treatment (measured-only affine mu correction + sigma
RE-SCAN on the corrected mu, corrected grid floor 0.05, LOO) to EVERY family,
alongside the r45 sigma-only calibration and the frozen sigma=0.7 estimand.

The FROZEN submission method is updated from r45 (0.7907) to the r51 joint
calibration on the 7-member mixed ensemble at equal-family weight wg=0.5
(0.7815), which removes the measured-layer systematic bias (scaf9 -0.996 ->
~0) that r45 could not correct.  r52 (per-scaf family weight) and r53 (weight
sweep under r51) are recorded; the equal-family blend stays the principled
frozen choice (near-optimal under both r45 and r51).

Columns (per model family):
  - frozen_sigma_07_nll   (pre-calibration primary estimand, sigma=0.7)
  - r45_calibrated_nll    (per-scaf x stratum sigma, corrected grid)
  - r51_calibrated_nll    (r45 + measured-only affine mu + sigma re-scan)
  - relative gains vs nuisance at the SAME treatment
Output: submission_horizontal_table_v2.json + printed markdown table.
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
    _calibrate_r45, _calibrate_r51, _pooled,
)

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


def _ens_mu(members, keys, rid_common, wg=0.5):
    out = {}
    for rid in rid_common:
        ref = members[keys[0]][rid]
        if wg is None:
            mu = float(np.mean([members[k][rid]["mu"] for k in keys]))
        else:
            gmu = float(np.mean([members[k][rid]["mu"] for k in GBDT]))
            mmu = float(np.mean([members[k][rid]["mu"] for k in MLP]))
            mu = wg * gmu + (1.0 - wg) * mmu
        out[rid] = {"jid": ref["jid"], "fold": ref["fold"], "scaf": ref["scaf"],
                    "y": ref["y"], "cens": ref["cens"], "mu": mu}
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
    ens7_wg05 = _ens_mu(members, ALL_MEMBERS, common, wg=0.5)
    ens7_mm = _ens_mu(members, ALL_MEMBERS, common, wg=None)
    ens3 = _ens_mu(members, MLP, common, wg=None)

    models = dict(singles)
    models["ENSEMBLE_3x_t7"] = ens3
    models["ENSEMBLE_MIXED_7_wg05"] = ens7_wg05
    models["ENSEMBLE_MIXED_7_membermean"] = ens7_mm

    out = {"n_rows": len(common), "n_folds": 37,
           "sigma_grid": {"floor": 0.05, "ceiling": 1.6, "step": 0.01,
                          "note": "MetricSpec floor (corrected r45 grid)"}}

    frozen = {}
    cal_r45 = {}
    cal_r51 = {}
    for name, preds in models.items():
        folds = sorted(set(preds[r]["fold"] for r in preds))
        frozen[name] = round(_pooled({r: {**p, "sigma": 0.7} for r, p in preds.items()}), 4)
        cal_r45[name] = round(_pooled(_calibrate_r45(preds, folds)), 4)
        cal_r51_c, _ = _calibrate_r51(preds, folds, mode="per_scaf_eb", eb_kappa=20.0)
        cal_r51[name] = round(_pooled(cal_r51_c), 4)

    out["frozen_sigma_07_nll"] = frozen
    out["r45_calibrated_nll"] = cal_r45
    out["r51_calibrated_nll"] = cal_r51
    out["frozen_rel_gain_pct_vs_nuisance"] = {
        k: round(100.0 * (frozen[NUIS] - v) / frozen[NUIS], 2)
        for k, v in frozen.items() if k != NUIS}
    out["r45_rel_gain_pct_vs_nuisance"] = {
        k: round(100.0 * (cal_r45[NUIS] - v) / cal_r45[NUIS], 2)
        for k, v in cal_r45.items() if k != NUIS}
    out["r51_rel_gain_pct_vs_nuisance"] = {
        k: round(100.0 * (cal_r51[NUIS] - v) / cal_r51[NUIS], 2)
        for k, v in cal_r51.items() if k != NUIS}

    # Frozen submission method = r51 joint on 7-member equal-family ensemble
    folds7 = sorted(set(ens7_wg05[r]["fold"] for r in ens7_wg05))
    cal7_r51, _ = _calibrate_r51(ens7_wg05, folds7, mode="per_scaf_eb", eb_kappa=20.0)
    cal_nuis_r51, _ = _calibrate_r51(models[NUIS], sorted(set(models[NUIS][r]["fold"] for r in models[NUIS])),
                                     mode="per_scaf_eb", eb_kappa=20.0)
    cal_nuis_r45 = _calibrate_r45(models[NUIS], sorted(set(models[NUIS][r]["fold"] for r in models[NUIS])))
    out["frozen_method_r51_nll"] = round(_pooled(cal7_r51), 4)
    out["frozen_method_r51_strata"] = _pooled_strata(cal7_r51)
    out["frozen_method_r51_vs_nuisance_r51"] = {
        "rel_gain_pct": round(100.0 * (_pooled(cal_nuis_r51) - _pooled(cal7_r51)) / _pooled(cal_nuis_r51), 2),
        "edit_cluster_CI": _edit_ci(cal7_r51, cal_nuis_r51),
    }
    out["frozen_method_r51_vs_nuisance_r45"] = {
        "rel_gain_pct": round(100.0 * (_pooled(cal_nuis_r45) - _pooled(cal7_r51)) / _pooled(cal_nuis_r45), 2),
        "edit_cluster_CI": _edit_ci(cal7_r51, cal_nuis_r45),
    }
    # vs old frozen (member-mean + r45)
    cal7_r45 = _calibrate_r45(ens7_mm, folds7)
    out["old_frozen_r45_nll"] = round(_pooled(cal7_r45), 4)
    out["r51_vs_old_frozen_delta"] = round(out["frozen_method_r51_nll"] - out["old_frozen_r45_nll"], 4)

    out["note"] = (
        "DEFINITIVE submission horizontal table v2.  frozen_sigma_07 = every "
        "family at the pre-calibration frozen sigma 0.7; r45_calibrated = "
        "per-scaf x stratum sigma (corrected grid floor 0.05, LOO); r51_"
        "calibrated = r45 + measured-only affine mu correction with sigma "
        "RE-SCANNED on the corrected mu (corrected grid, LOO).  FROZEN "
        "submission method = 7-member mixed ensemble at equal-family wg=0.5 "
        "+ r51 joint calibration = 0.7815, which removes the measured-layer "
        "systematic bias that r45 could not correct.  r52 (per-scaf family "
        "weight) and r53 (family-weight sweep under r51) confirm the "
        "equal-family blend remains the principled frozen choice."
    )
    Path(f"{R}/submission_horizontal_table_v2.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")

    # markdown table for the manuscript
    lines = ["| model | sigma=0.7 | r45 | r51 |",
             "|---|---:|---:|---:|"]
    for name in sorted(frozen, key=lambda k: cal_r51[k]):
        lines.append(f"| {name} | {frozen[name]:.4f} | {cal_r45[name]:.4f} | {cal_r51[name]:.4f} |")
    md = "\n".join(lines)
    Path(f"{R}/submission_horizontal_table_v2.md").write_text(md + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("\n=== markdown ===")
    print(md)


if __name__ == "__main__":
    main()
