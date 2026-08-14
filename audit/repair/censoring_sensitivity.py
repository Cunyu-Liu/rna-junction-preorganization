"""Censoring-sensitivity: pooled NLL on measured-only vs full rows (P0.5).

The right-censoring handling is a core protocol choice.  This analysis recomputes
the pooled junction-macro NLL for the key model families EXCLUDING censored rows
(measured-only) vs including them with the survival term (full), to show whether
model rankings depend on the censoring treatment.  Uses the SAME eligible folds
and the SAME fixed sigma=0.7 scorer as the primary benchmark table.
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
MODEL_RUNS = {
    "corrected_v1_31": "r29_p05_rerun",
    "no_sequence_latent_operator": "r29_p05_rerun",
    "motif_topology_hierarchy": "r29_p05_rerun",
    "nonlinear_mlp_nuisance_only_t7": "r31_nuisance_only_full",
    "nonlinear_mlp_extended_hybrid_reg_deep": "r14_extended_mlp_scan",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7": "r20_robust_t_df_sweep",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99": "r21_seed99_replication",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026": "r23_seed2026_replication",
}
T7_MEMBERS = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
LEDGER_PATHS = [
    f"{R}/r29_p05_rerun/ConvergenceLedger_v3.parquet",
    f"{R}/r31_nuisance_only_full/ConvergenceLedger_v3.parquet",
    f"{R}/r14_extended_mlp_scan/ConvergenceLedger_v3.parquet",
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
]


def _eligible_set():
    import pandas as pd
    frames = [pd.read_parquet(p) for p in LEDGER_PATHS]
    conv = [dict(r) for r in pd.concat(frames, ignore_index=True).to_dict("records")]
    return _eligible_keys(conv)


def load_model(model_id, eligible):
    run_dir = MODEL_RUNS[model_id]
    path = f"{R}/{run_dir}/Predictions_v3.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return [p for p in rows if p["model_id"] == model_id
            and (p["model_id"], p["fold"]) in eligible
            and p["support"] and not p["abstain"]]


def pooled(rows, measured_only=False):
    jid = defaultdict(list)
    for p in rows:
        if measured_only and p["cens"]:
            continue
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jid[p["jid"]].append(nll)
    if not jid:
        return None
    return float(np.mean([np.mean(v) for v in jid.values()]))


def main():
    eligible = _eligible_set()
    out = {"full": {}, "measured_only": {}, "censored_frac": 0.1624}
    print(f"{'model':48s} {'full':>7s} {'meas-only':>9s} {'diff':>7s}")
    for m in MODEL_RUNS:
        rows = load_model(m, eligible)
        full = pooled(rows, measured_only=False)
        mo = pooled(rows, measured_only=True)
        out["full"][m] = round(full, 4) if full else None
        out["measured_only"][m] = round(mo, 4) if mo else None
        diff = (full - mo) if (full is not None and mo is not None) else float("nan")
        print(f"{m:48s} {full:7.4f} {mo:9.4f} {diff:+7.4f}")

    # 3x t7 ensemble
    ens = []
    by_rid = defaultdict(dict)
    for m in T7_MEMBERS:
        for p in load_model(m, eligible):
            by_rid[p["source_row_id"]][m] = p
    for rid, d in by_rid.items():
        if not all(m in d for m in T7_MEMBERS):
            continue
        ref = d[T7_MEMBERS[0]]
        ens.append({"jid": ref["jid"], "y": ref["y"], "cens": ref["cens"],
                    "mu": float(np.mean([d[m]["mu"] for m in T7_MEMBERS])),
                    "sigma": 0.7})
    full = pooled(ens, False)
    mo = pooled(ens, True)
    out["full"]["ENSEMBLE_3x_t7"] = round(full, 4)
    out["measured_only"]["ENSEMBLE_3x_t7"] = round(mo, 4)
    print(f"{'ENSEMBLE_3x_t7':48s} {full:7.4f} {mo:9.4f} {full-mo:+7.4f}")

    # ranking consistency
    full_sorted = sorted(out["full"], key=out["full"].get)
    mo_sorted = sorted(out["measured_only"], key=out["measured_only"].get)
    out["rank_full"] = full_sorted
    out["rank_measured_only"] = mo_sorted
    out["rank_order_identical"] = full_sorted == mo_sorted
    print(f"\nfull ranking:      {full_sorted}")
    print(f"measured-only rank: {mo_sorted}")
    print(f"rank order identical: {out['rank_order_identical']}")
    Path(f"{R}/benchmark_censoring_sensitivity.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {R}/benchmark_censoring_sensitivity.json")


if __name__ == "__main__":
    main()
