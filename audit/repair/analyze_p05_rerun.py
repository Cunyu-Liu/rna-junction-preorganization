"""P0.5 minimal-affected rerun analysis: true-joint contrast under fixed evaluator.

Consumes r29_p05_rerun (corrected_v1_31, no_sequence, motif_topology, train_only
scaffold on the 37 blocked joint folds, all folds optimizer+full-coverage
eligible, rows_hash recorded).  Computes the decisive matched contrasts:
  - corrected_v1_31 vs no_sequence_latent_operator (sequence map increment)
  - motif_topology_hierarchy vs no_sequence (best simple baseline)
and the 3x t7 MLP ensemble gain vs nuisance on the SAME folds (from r24 preds,
recomputed under the fixed eligible-only aggregation).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.shootout_run import _eligible_keys, _filter_eligible_preds

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
R29 = f"{R}/r29_p05_rerun/Predictions_v3.jsonl"
R24 = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"
R29_LEDGER = f"{R}/r29_p05_rerun/ConvergenceLedger_v3.parquet"
R24_LEDGER = f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet"

MLP_T7 = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
NUIS = "motif_topology_hierarchy"


def load(path):
    return [json.loads(l) for l in open(path)]


def pooled_nll(rows):
    jd = defaultdict(list)
    for p in rows:
        if p["support"] and not p["abstain"]:
            nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()])), len(jd)


def contrast(a_preds, b_preds):
    """delta = (b - a) per junction, macro-averaged; positive means a better."""
    by = defaultdict(dict)
    for p in a_preds + b_preds:
        if p["support"] and not p["abstain"]:
            by[p["source_row_id"]][p["model_id"]] = p
    jid_d = defaultdict(list)
    jid_base = defaultdict(list)
    n = 0
    for rid, d in by.items():
        if len(d) < 2:
            continue
        ids = list(d)
        pa, pb = d[ids[0]], d[ids[1]]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
        jid_base[pa["jid"]].append(dlb)
        n += 1
    theta = float(np.mean([np.mean(v) for v in jid_d.values()]))
    base = float(np.mean([np.mean(v) for v in jid_base.values()]))
    return {"n_rows": n, "n_junctions": len(jid_d), "theta_abs": round(theta, 4),
            "relative_gain_pct": round(100.0 * theta / base, 2) if base else None}


def main():
    r29 = load(R29)
    r29_conv = [dict(r) for r in __import__("pandas").read_parquet(R29_LEDGER).to_dict("records")]
    elig = _eligible_keys(r29_conv)
    r29e = [p for p in r29 if (p["model_id"], p["fold"]) in elig]
    print(f"r29 rows total={len(r29)} eligible-filtered={len(r29e)}")
    by_id = defaultdict(list)
    for p in r29e:
        by_id[p["model_id"]].append(p)
    print("\n=== r29 P0.5 true-joint pooled NLL (eligible-only) ===")
    for m in ("corrected_v1_31", "no_sequence_latent_operator",
              "motif_topology_hierarchy", "train_only_scaffold"):
        nll, nj = pooled_nll(by_id[m])
        print(f"  {m:32s} NLL={nll:.4f}  n_junc={nj}")

    print("\n=== decisive matched contrasts (positive = A better) ===")
    c1 = contrast(by_id["corrected_v1_31"], by_id["no_sequence_latent_operator"])
    print(f"  corrected_v1_31 vs no_sequence : theta={c1['theta_abs']} rel={c1['relative_gain_pct']}% rows={c1['n_rows']}")
    c2 = contrast(by_id["motif_topology_hierarchy"], by_id["no_sequence_latent_operator"])
    print(f"  motif_topology vs no_sequence   : theta={c2['theta_abs']} rel={c2['relative_gain_pct']}% rows={c2['n_rows']}")
    c3 = contrast(by_id["corrected_v1_31"], by_id["motif_topology_hierarchy"])
    print(f"  corrected_v1_31 vs motif_topology: theta={c3['theta_abs']} rel={c3['relative_gain_pct']}% rows={c3['n_rows']}")

    # 3x t7 MLP ensemble vs nuisance on same 37 joint folds.
    # Members' ledgers live in their own run dirs (r20/r21/r23/r24); merge all
    # four ConvergenceLedgers to build the eligible (model, fold) key set.
    import pandas as pd
    frames = []
    for d in ("r20_robust_t_df_sweep", "r21_seed99_replication",
              "r23_seed2026_replication", "r24_t7_seed7"):
        frames.append(pd.read_parquet(f"{R}/{d}/ConvergenceLedger_v3.parquet"))
    conv_all = pd.concat(frames, ignore_index=True)
    r24_conv = [dict(r) for r in conv_all.to_dict("records")]
    elig24 = _eligible_keys(r24_conv)
    r24 = load(R24)
    r24e = [p for p in r24 if (p["model_id"], p["fold"]) in elig24]
    by24 = defaultdict(list)
    for p in r24e:
        by24[p["model_id"]].append(p)
    ens_rows = []
    by_rid = defaultdict(dict)
    for m in MLP_T7:
        for p in by24[m]:
            by_rid[p["source_row_id"]][m] = p
    for rid, d in by_rid.items():
        if not all(m in d for m in MLP_T7):
            continue
        ref = d[MLP_T7[0]]
        ens_rows.append({"model_id": "ENS_3x_t7", "source_row_id": rid,
                         "jid": ref["jid"], "y": ref["y"], "cens": ref["cens"],
                         "mu": float(np.mean([d[m]["mu"] for m in MLP_T7])),
                         "sigma": 0.7, "support": True, "abstain": False})
    nll_e, nj_e = pooled_nll(ens_rows)
    nll_n, _ = pooled_nll(by24[NUIS])
    print("\n=== 3x t7 ensemble (fixed evaluator, eligible-only) ===")
    print(f"  ensemble NLL={nll_e:.4f}  nuisance NLL={nll_n:.4f}  "
          f"rel_gain={100.0*(nll_n-nll_e)/nll_n:+.2f}%  n_junc={nj_e}")

    # r31: nuisance-only t7 ablation (no ViennaRNA, same reg_deep t7 arch).
    r31 = load(f"{R}/r31_nuisance_only_full/Predictions_v3.jsonl")
    nus_t7 = [p for p in r31 if p["model_id"] == "nonlinear_mlp_nuisance_only_t7"]
    nll_nt7, _ = pooled_nll(nus_t7)
    print("\n=== matched ablation: nuisance-only t7 (no ViennaRNA) ===")
    print(f"  nuisance-only t7 NLL={nll_nt7:.4f}  "
          f"nonlinear-head gain vs nuisance={100.0*(nll_n-nll_nt7)/nll_n:+.2f}%  "
          f"ViennaRNA increment under nonlinear head={100.0*(nll_nt7-nll_e)/nll_nt7:+.2f}%")

    out = {
        "r29_pooled": {m: round(pooled_nll(by_id[m])[0], 4) for m in by_id},
        "contrasts": {"v131_vs_noseq": c1, "motif_vs_noseq": c2, "v131_vs_motif": c3},
        "ensemble_3x_t7": {"nll": round(nll_e, 4), "nuisance_nll": round(nll_n, 4),
                           "rel_gain_pct": round(100.0 * (nll_n - nll_e) / nll_n, 2)},
        "nuisance_only_t7_ablation": {"nll": round(nll_nt7, 4),
            "nonlinear_head_gain_pct": round(100.0 * (nll_n - nll_nt7) / nll_n, 2),
            "viennarna_increment_pct": round(100.0 * (nll_nt7 - nll_e) / nll_nt7, 2)},
    }
    Path(f"{R}/r29_p05_rerun/P05Analysis.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {R}/r29_p05_rerun/P05Analysis.json")


if __name__ == "__main__":
    main()
