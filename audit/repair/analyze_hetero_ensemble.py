"""Heterogeneous model-family mu-ensemble across existing run roots.

The project's iron rule so far: only clean variance reduction works.  The 3x t7
MLP ensemble (r24) averages same-architecture members; this analyzer tests the
NEXT variance-reduction axis -- averaging mu across DIFFERENT model families
(flat MLP, linear hybrid, linear latent-operator), which should be less
correlated than same-arch seeds.

All three run roots (r06, r08, r24) share the identical 37 blocked folds and
11,893 rows, so combining their held-out mu is a valid model combination (no
test leakage).  Ensemble rows are admitted only if supported+non-abstain in
EVERY member (coverage-matched), and evaluated with the shared fixed
sigma=0.7 exactly like the flat-MLP mu-ensemble.  Report pooled junction-macro
NLL, relative gain vs nuisance, and the edit-cluster bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
FILES = {
    "r06": f"{R}/r06_shootout/Predictions_v3.jsonl",
    "r08": f"{R}/r08_vienna_extended_hybrid/Predictions_v3.jsonl",
    "r24": f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl",
}
BASE = "motif_topology_hierarchy"
ENSEMBLE_ID = "HETERO_ENSEMBLE_MU"

# candidate members by family
MLP_T7 = [
    "nonlinear_mlp_extended_hybrid_reg_deep_t7",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99",
    "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026",
]
MLP_T7_S7 = MLP_T7 + ["nonlinear_mlp_extended_hybrid_reg_deep_t7_s7"]
LINEAR_HYBRID = ["vienna_linear_hybrid", "vienna_extended_linear_hybrid"]
LATENT_LIN = ["vienna_latent_operator", "no_sequence_latent_operator"]


def load_all():
    by_model = defaultdict(list)      # model_id -> list of rows
    for f in FILES.values():
        for line in open(f):
            p = json.loads(line)
            if p["support"] and not p["abstain"]:
                by_model[p["model_id"]].append(p)
    return by_model


def ensemble_rows(by_model, members):
    """Coverage-matched mu-ensemble across members (sigma fixed 0.7)."""
    by_rid = defaultdict(dict)
    for m in members:
        for p in by_model.get(m, []):
            by_rid[p["source_row_id"]][m] = p
    ens = []
    for rid, d in by_rid.items():
        if not all(m in d for m in members):
            continue
        ms = [d[m]["mu"] for m in members]
        ref = d[members[0]]
        ens.append({
            "source_row_id": rid, "jid": ref["jid"], "fold": ref["fold"],
            "model_id": ENSEMBLE_ID, "y": ref["y"], "cens": ref["cens"],
            "mu": float(np.mean(ms)), "sigma": 0.7,
            "abstain": False, "support": True,
        })
    return ens


def pooled_nll(rows):
    jd = defaultdict(list)
    for p in rows:
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()])), len(jd)


def contrast(ens_rows, base_rows, base_id=BASE):
    """delta = base_nll - ens_nll (positive means ensemble better), per-junction."""
    base_by = {p["source_row_id"]: p for p in base_rows if p["model_id"] == base_id}
    ens_by = {p["source_row_id"]: p for p in ens_rows}
    common = set(ens_by) & set(base_by)
    jid_d = defaultdict(list)
    jid_base = defaultdict(list)
    for rid in common:
        pa, pb = ens_by[rid], base_by[rid]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
        jid_base[pa["jid"]].append(dlb)
    theta = float(np.mean([np.mean(v) for v in jid_d.values()]))
    base = float(np.mean([np.mean(v) for v in jid_base.values()]))
    rel = theta / base if base else None
    return theta, rel, len(jid_d)


def edit_cluster_ci(ens_rows, base_rows, base_id=BASE):
    base_by = {p["source_row_id"]: p for p in base_rows if p["model_id"] == base_id}
    ens_by = {p["source_row_id"]: p for p in ens_rows}
    jid_edit = {p["jid"]: str(p["fold"]).split(":", 1)[1] for p in ens_rows}
    jid_d = defaultdict(list)
    for rid in set(ens_by) & set(base_by):
        pa, pb = ens_by[rid], base_by[rid]
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
        chosen = rng.choice(edit_names, size=len(edit_names), replace=True)
        vals = [v for e in chosen for v in by_edit[e]]
        boots.append(float(np.mean(vals)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for grp in keep for x in grp])) if keep else None
    return {"lo": lo, "hi": hi, "ci_lower_gt_0": bool(lo > 0),
            "n_edit": len(by_edit), "largest": largest,
            "leave_one_largest": leave1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/r28_hetero_ensemble.json")
    args = ap.parse_args()

    by_model = load_all()
    base_rows = by_model.get(BASE, [])
    base_nll, base_nj = pooled_nll([p for p in base_rows])
    combos = {
        "3x_t7_only": MLP_T7,
        "4x_t7_only": MLP_T7_S7,
        "3x_t7_plus_linhyb": MLP_T7 + LINEAR_HYBRID,
        "3x_t7_plus_ext_linhyb": MLP_T7 + ["vienna_extended_linear_hybrid"],
        "3x_t7_plus_latent": MLP_T7 + LATENT_LIN,
        "3x_t7_plus_ext_plus_latent": MLP_T7 + ["vienna_extended_linear_hybrid"] + LATENT_LIN,
    }
    out = {"base_nll": base_nll, "base_n_junctions": base_nj, "combos": {}}
    for name, members in combos.items():
        ens = ensemble_rows(by_model, members)
        nll, nj = pooled_nll(ens)
        theta, rel, ncommon = contrast(ens, by_model[BASE])
        ci = edit_cluster_ci(ens, by_model[BASE])
        out["combos"][name] = {
            "members": members, "n_ensemble_rows": len(ens), "n_junctions": nj,
            "pooled_nll": round(nll, 4),
            "relative_gain_pct": round(100.0 * rel, 2) if rel else None,
            "theta": round(theta, 4),
            "n_common_junctions": ncommon,
            "edit_cluster_95ci": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "ci_lower_gt_0": ci["ci_lower_gt_0"],
            "leave_one_largest": round(ci["leave_one_largest"], 4) if ci["leave_one_largest"] else None,
        }
        print(f"{name:32s} NLL={nll:.4f}  rel={100.0*rel if rel else 0:+.2f}%  "
              f"CI=[{ci['lo']:.3f},{ci['hi']:.3f}]  rows={len(ens)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
