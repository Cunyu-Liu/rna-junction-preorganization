"""Benchmark-track horizontal comparison table (contract P0.3/P0.5).

Assembles the final cross-model comparison from ALREADY-MATERIALIZED predictions
across all run roots.  All models are evaluated on the SAME 37 blocked joint
folds (edit_x_nested_context) with the SAME right-censored Gaussian scorer at
the SAME fixed sigma=0.7, restricted to folds that pass BOTH the optimizer and
the full-coverage eligibility gates (fail-closed; see shootout_run).

Columns (per model family):
  - pooled_junction_macro NLL  (PRIMARY estimand)
  - nested_context_macro NLL
  - scaffold_bundle_macro NLL
  - censored_frac covered
  - relative gain vs motif_topology_hierarchy (primary estimand)
  - 3x t7 ensemble computed by mu-averaging the three t7 members

Families included (run root -> model ids):
  - corrected_v1_31 (63-D seq map)            r29
  - no_sequence_latent_operator               r29
  - train_only_scaffold                       r29
  - motif_topology_hierarchy (nuisance)       r29
  - nuisance-only t7 MLP (no ViennaRNA)       r31
  - nonlinear_mlp_extended_hybrid_reg_deep    r14
  - t5/t7/t10 single (seed 0)                 r20
  - t7 s99 / s2026 / s7                       r21/r23/r24
  - 3x t7 ensemble (mu-mean)                  r24
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.estimands_v2 import (
    pooled_junction_macro, nested_context_macro, scaffold_bundle_macro,
)
from audit.evaluation.metrics import row_nll
from audit.repair.shootout_run import _eligible_keys

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"

# model -> (run_dir, ledger_dir_for_eligibility)
MODEL_RUNS = {
    "corrected_v1_31": ("r29_p05_rerun", "r29_p05_rerun"),
    "no_sequence_latent_operator": ("r29_p05_rerun", "r29_p05_rerun"),
    "train_only_scaffold": ("r29_p05_rerun", "r29_p05_rerun"),
    "motif_topology_hierarchy": ("r29_p05_rerun", "r29_p05_rerun"),
    "nonlinear_mlp_nuisance_only_t7": ("r31_nuisance_only_full", "r31_nuisance_only_full"),
    "nonlinear_mlp_extended_hybrid_reg_deep": ("r14_extended_mlp_scan", "r14_extended_mlp_scan"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t": ("r20_robust_t_df_sweep", "r20_robust_t_df_sweep"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t7": ("r20_robust_t_df_sweep", "r20_robust_t_df_sweep"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t10": ("r20_robust_t_df_sweep", "r20_robust_t_df_sweep"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99": ("r21_seed99_replication", "r21_seed99_replication"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026": ("r23_seed2026_replication", "r23_seed2026_replication"),
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s7": ("r24_t7_seed7", "r24_t7_seed7"),
    "xgboost_censored_hybrid": ("r33_xgboost_full", "r33_xgboost_full"),
    "xgboost_censored_hybrid_s99": ("r34_gbdt_seeds_full", "r34_gbdt_seeds_full"),
    "xgboost_censored_hybrid_s2026": ("r34_gbdt_seeds_full", "r34_gbdt_seeds_full"),
}
T7_MEMBERS = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
GBDT_MEMBERS = [
    "xgboost_censored_hybrid",
    "xgboost_censored_hybrid_s99",
    "xgboost_censored_hybrid_s2026",
]
NUIS = "motif_topology_hierarchy"

LEDGER_PATHS = [
    f"{R}/r29_p05_rerun/ConvergenceLedger_v3.parquet",
    f"{R}/r31_nuisance_only_full/ConvergenceLedger_v3.parquet",
    f"{R}/r14_extended_mlp_scan/ConvergenceLedger_v3.parquet",
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
    f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet",
    f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet",
]


def _eligible_set():
    import pandas as pd
    frames = [pd.read_parquet(p) for p in LEDGER_PATHS]
    conv = [dict(r) for r in pd.concat(frames, ignore_index=True).to_dict("records")]
    return _eligible_keys(conv)


def load_model(model_id, eligible):
    run_dir, ledger_dir = MODEL_RUNS[model_id]
    path = f"{R}/{run_dir}/Predictions_v3.jsonl"
    rows = [json.loads(l) for l in open(path)]
    rows = [p for p in rows if p["model_id"] == model_id
            and (p["model_id"], p["fold"]) in eligible]
    return rows


def _ctx_key(p):
    """Prediction rows carry the helix context under 'context'."""
    return str(p.get("context") or p.get("helix_seq"))


def pooled_junction_macro(rows, mu, sigma):
    by = defaultdict(list)
    for i, p in enumerate(rows):
        by[p["jid"]].append(float(row_nll([p["y"]], [p["cens"]], [mu[i]], [sigma[i]])[0]))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def nested_context_macro(rows, mu, sigma):
    by_jid = defaultdict(list)
    for i, p in enumerate(rows):
        by_jid[p["jid"]].append(float(row_nll([p["y"]], [p["cens"]], [mu[i]], [sigma[i]])[0]))
    jid_nll = {j: float(np.mean(v)) for j, v in by_jid.items()}
    ctx = defaultdict(list)
    for p in rows:
        if p["jid"] in jid_nll:
            ctx[_ctx_key(p)].append(jid_nll[p["jid"]])
    if not ctx:
        return None
    per_ctx = {c: float(np.mean(v)) for c, v in ctx.items()}
    return float(np.mean(list(per_ctx.values()))), per_ctx


def scaffold_bundle_macro(rows, mu, sigma):
    by_jid = defaultdict(list)
    for i, p in enumerate(rows):
        by_jid[p["jid"]].append(float(row_nll([p["y"]], [p["cens"]], [mu[i]], [sigma[i]])[0]))
    jid_nll = {j: float(np.mean(v)) for j, v in by_jid.items()}
    scaf = defaultdict(list)
    for p in rows:
        if p["jid"] in jid_nll:
            scaf[int(p["scaf"])].append(jid_nll[p["jid"]])
    if not scaf:
        return None
    per_scaf = {s: float(np.mean(v)) for s, v in scaf.items()}
    return float(np.mean(list(per_scaf.values()))), per_scaf


def _mu_sigma(rows):
    mu = np.array([p["mu"] for p in rows])
    sigma = np.array([p["sigma"] for p in rows])
    y = np.array([p["y"] for p in rows])
    cens = np.array([p["cens"] for p in rows], dtype=bool)
    return mu, sigma, y, cens


def _nll(rows):
    return [float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            for p in rows if p["support"] and not p["abstain"]]


def edit_cluster_ci(a_rows, b_rows):
    """Edit-cluster bootstrap 95% CI of delta=(b - a) (positive => a better)."""
    a_by = {p["source_row_id"]: p for p in a_rows}
    b_by = {p["source_row_id"]: p for p in b_rows}
    jid_edit = {}
    for p in a_rows:
        jid_edit.setdefault(p["jid"], str(p["fold"]).split(":", 1)[1])
    jid_d = defaultdict(list)
    for rid in set(a_by) & set(b_by):
        pa, pb = a_by[rid], b_by[rid]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
    by_edit = defaultdict(list)
    for j, vals in jid_d.items():
        by_edit[jid_edit.get(j, "?")].append(float(np.mean(vals)))
    edit_names = list(by_edit)
    rng = np.random.default_rng(17)
    boots = []
    for _ in range(1000):
        ch = rng.choice(edit_names, size=len(edit_names), replace=True)
        vals = [v for e in ch for v in by_edit[e]]
        boots.append(float(np.mean(vals)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for g in keep for x in g])) if keep else None
    return {"ci": [round(lo, 4), round(hi, 4)], "ci_lower_gt_0": bool(lo > 0),
            "n_edit": len(by_edit), "largest": largest,
            "leave_one_largest": round(leave1, 4) if leave1 is not None else None}


def main():
    eligible = _eligible_set()
    out = {}
    # rows keyed by source_row_id for ensemble + nuisance baseline
    nuis_rows = load_model(NUIS, eligible)
    nuis_jid = defaultdict(list)
    for p in nuis_rows:
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        nuis_jid[p["jid"]].append(nll)
    nuis_nll = float(np.mean([np.mean(v) for v in nuis_jid.values()]))

    print(f"{'model':52s} {'n_rows':>6s} {'pooled':>7s} {'ctx':>7s} {'scaf':>7s} "
          f"{'cens%':>5s} {'rel%':>7s}")
    for model_id in MODEL_RUNS:
        rows = load_model(model_id, eligible)
        rows = [p for p in rows if p["support"] and not p["abstain"]]
        if not rows:
            print(f"{model_id:52s}   no eligible rows")
            continue
        mu, sigma, y, cens = _mu_sigma(rows)
        pooled = pooled_junction_macro(rows, mu, sigma)
        ctx = nested_context_macro(rows, mu, sigma)
        scaf = scaffold_bundle_macro(rows, mu, sigma)
        cens_frac = float(np.mean(cens))
        jid = defaultdict(list)
        for p in rows:
            nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            jid[p["jid"]].append(nll)
        my_nll = float(np.mean([np.mean(v) for v in jid.values()]))
        rel = 100.0 * (nuis_nll - my_nll) / nuis_nll
        out[model_id] = {
            "n_rows": len(rows), "pooled_junction_macro": round(pooled, 4),
            "nested_context_macro": round(ctx[0], 4) if ctx else None,
            "scaffold_bundle_macro": round(scaf[0], 4) if scaf else None,
            "censored_frac": round(cens_frac, 4),
            "rel_gain_pct_vs_nuisance": round(rel, 2),
        }
        print(f"{model_id:52s} {len(rows):6d} {pooled:7.4f} "
              f"{ctx[0] if ctx else float('nan'):7.4f} "
              f"{scaf[0] if scaf else float('nan'):7.4f} "
              f"{100*cens_frac:5.1f} {rel:+7.2f}")

    # 3x t7 ensemble
    ens_rows = []
    by_rid = defaultdict(dict)
    for m in T7_MEMBERS:
        for p in load_model(m, eligible):
            if p["support"] and not p["abstain"]:
                by_rid[p["source_row_id"]][m] = p
    for rid, d in by_rid.items():
        if not all(m in d for m in T7_MEMBERS):
            continue
        ref = d[T7_MEMBERS[0]]
        ens_rows.append({
            "source_row_id": rid, "jid": ref["jid"], "fold": ref["fold"],
            "scaf": int(ref["scaf"]), "context": str(ref.get("context") or ref.get("helix_seq", "")),
            "y": ref["y"], "cens": ref["cens"],
            "mu": float(np.mean([d[m]["mu"] for m in T7_MEMBERS])),
            "sigma": 0.7, "support": True, "abstain": False,
        })
    mu = np.array([p["mu"] for p in ens_rows])
    sigma = np.array([p["sigma"] for p in ens_rows])
    cens = np.array([p["cens"] for p in ens_rows], dtype=bool)
    pooled_e = pooled_junction_macro(ens_rows, mu, sigma)
    ctx_e = nested_context_macro(ens_rows, mu, sigma)
    scaf_e = scaffold_bundle_macro(ens_rows, mu, sigma)
    jid = defaultdict(list)
    for p in ens_rows:
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jid[p["jid"]].append(nll)
    my_nll = float(np.mean([np.mean(v) for v in jid.values()]))
    rel_e = 100.0 * (nuis_nll - my_nll) / nuis_nll
    out["ENSEMBLE_3x_t7"] = {
        "n_rows": len(ens_rows), "pooled_junction_macro": round(pooled_e, 4),
        "nested_context_macro": round(ctx_e[0], 4) if ctx_e else None,
        "scaffold_bundle_macro": round(scaf_e[0], 4) if scaf_e else None,
        "censored_frac": round(float(np.mean(cens)), 4),
        "rel_gain_pct_vs_nuisance": round(rel_e, 2),
    }
    print(f"{'ENSEMBLE_3x_t7':52s} {len(ens_rows):6d} {pooled_e:7.4f} "
          f"{ctx_e[0] if ctx_e else float('nan'):7.4f} "
          f"{scaf_e[0] if scaf_e else float('nan'):7.4f} "
          f"{100*float(np.mean(cens)):5.1f} {rel_e:+7.2f}")

    # 6-member mixed ensemble: GBDT 3x + t7 MLP 3x (equal mu-mean).
    # The weight sweep on the family ensembles showed w=0.5 is optimal (the two
    # families have matched quality), so equal weighting is the honest optimum.
    MIXED_MEMBERS = T7_MEMBERS + GBDT_MEMBERS
    mix_by_rid = defaultdict(dict)
    for m in MIXED_MEMBERS:
        for p in load_model(m, eligible):
            if p["support"] and not p["abstain"]:
                mix_by_rid[p["source_row_id"]][m] = p
    mix_rows = []
    for rid, d in mix_by_rid.items():
        if not all(m in d for m in MIXED_MEMBERS):
            continue
        ref = d[MIXED_MEMBERS[0]]
        mix_rows.append({
            "source_row_id": rid, "jid": ref["jid"], "fold": ref["fold"],
            "scaf": int(ref["scaf"]), "context": str(ref.get("context") or ref.get("helix_seq", "")),
            "y": ref["y"], "cens": ref["cens"],
            "mu": float(np.mean([d[m]["mu"] for m in MIXED_MEMBERS])),
            "sigma": 0.7, "support": True, "abstain": False,
        })
    mu = np.array([p["mu"] for p in mix_rows])
    sigma = np.array([p["sigma"] for p in mix_rows])
    cens = np.array([p["cens"] for p in mix_rows], dtype=bool)
    pooled_x = pooled_junction_macro(mix_rows, mu, sigma)
    ctx_x = nested_context_macro(mix_rows, mu, sigma)
    scaf_x = scaffold_bundle_macro(mix_rows, mu, sigma)
    jid = defaultdict(list)
    for p in mix_rows:
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jid[p["jid"]].append(nll)
    my_nll = float(np.mean([np.mean(v) for v in jid.values()]))
    rel_x = 100.0 * (nuis_nll - my_nll) / nuis_nll
    out["ENSEMBLE_MIXED_6"] = {
        "n_rows": len(mix_rows), "pooled_junction_macro": round(pooled_x, 4),
        "nested_context_macro": round(ctx_x[0], 4) if ctx_x else None,
        "scaffold_bundle_macro": round(scaf_x[0], 4) if scaf_x else None,
        "censored_frac": round(float(np.mean(cens)), 4),
        "rel_gain_pct_vs_nuisance": round(rel_x, 2),
    }
    print(f"{'ENSEMBLE_MIXED_6 (3x GBDT + 3x t7)':52s} {len(mix_rows):6d} {pooled_x:7.4f} "
          f"{ctx_x[0] if ctx_x else float('nan'):7.4f} "
          f"{scaf_x[0] if scaf_x else float('nan'):7.4f} "
          f"{100*float(np.mean(cens)):5.1f} {rel_x:+7.2f}")

    # edit-cluster group-aware CI for the decisive contrasts
    t7s99 = load_model("nonlinear_mlp_extended_hybrid_reg_deep_t7_s99", eligible)
    ens_vs_nuis = edit_cluster_ci(ens_rows, nuis_rows)
    ens_vs_s99 = edit_cluster_ci(ens_rows, t7s99)
    nuis_vs_s99 = edit_cluster_ci(nuis_rows, t7s99)
    mix_vs_nuis = edit_cluster_ci(mix_rows, nuis_rows)
    out["edit_cluster_CI"] = {
        "ensemble_vs_nuisance": ens_vs_nuis,
        "ensemble_vs_t7_s99": ens_vs_s99,
        "nuisance_vs_t7_s99": nuis_vs_s99,
        "mixed6_vs_nuisance": mix_vs_nuis,
        "note": ("delta = (b - a); positive means the first (a) model is better. "
                 "Bootstrap unit = edit component (37)."),
    }
    print("\n=== edit-cluster group CI (positive => first better) ===")
    for name, d in (("ensemble_vs_nuisance", ens_vs_nuis),
                    ("ensemble_vs_t7_s99", ens_vs_s99),
                    ("nuisance_vs_t7_s99", nuis_vs_s99),
                    ("mixed6_vs_nuisance", mix_vs_nuis)):
        print(f"  {name:24s} CI={d['ci']} lower_gt_0={d['ci_lower_gt_0']} "
              f"n_edit={d['n_edit']} leave1={d['leave_one_largest']}")

    Path(f"{R}/benchmark_horizontal_table.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {R}/benchmark_horizontal_table.json")


if __name__ == "__main__":
    main()
