"""Analyze r14 extended-MLP scan: pooled + edit-cluster CI for key contrasts.

Self-contained: only reads Predictions_v3.jsonl. The `fold` field encodes the
edit component ("e:AAAC_GAAC"), so jid->edit_component is derived from folds.
Reuses the exact contrast / edit-cluster bootstrap logic from shootout_run.py
so results are directly comparable to the shooter's report.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit.evaluation.metrics import row_nll


def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def edit_from_fold(fold):
    # fold like "e:AAAC_GAAC" -> "AAAC_GAAC"
    return str(fold).split(":", 1)[1]


def pooled_nll_by_model(all_preds):
    by = defaultdict(list)
    for p in all_preds:
        if p["support"] and not p["abstain"]:
            nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
            by[p["model_id"]].append((p["jid"], nll))
    out = {}
    for m, lst in by.items():
        jd = defaultdict(list)
        for j, n in lst:
            jd[j].append(n)
        out[m] = float(np.mean([np.mean(v) for v in jd.values()]))
    return out


def paired_rows(all_preds, a_id, b_id):
    by = defaultdict(dict)
    for p in all_preds:
        if p["model_id"] in (a_id, b_id):
            if p["support"] and not p["abstain"]:
                by[p["source_row_id"]][p["model_id"]] = p
    return {rid: d for rid, d in by.items() if a_id in d and b_id in d}


def pooled_contrast(all_preds, a_id, b_id, a_name, b_name):
    pairs = paired_rows(all_preds, a_id, b_id)
    if not pairs:
        return {"available": False, "reason": "no matched supported pairs"}
    jid_d = defaultdict(list)
    jid_base = defaultdict(list)
    for rid, d in pairs.items():
        pa, pb = d[a_id], d[b_id]
        dla = float(row_nll([pa["y"]], [pa["cens"]], [pa["mu"]], [pa["sigma"]])[0])
        dlb = float(row_nll([pb["y"]], [pb["cens"]], [pb["mu"]], [pb["sigma"]])[0])
        jid_d[pa["jid"]].append(dlb - dla)
        jid_base[pa["jid"]].append(dlb)
    theta = float(np.mean([np.mean(v) for v in jid_d.values()]))
    base = float(np.mean([np.mean(v) for v in jid_base.values()]))
    rel = theta / base if base else None
    return {
        "available": True,
        "contrast": f"{a_name} vs {b_name}",
        "n_rows": len(pairs), "n_junctions": len(jid_d),
        "theta_abs": theta,
        "relative_gain_pct": (rel * 100.0) if rel is not None else None,
        "hits_10pct_gate": bool(rel >= 0.10) if rel is not None else False,
        "n_positive_junctions": int(np.sum([np.mean(v) > 0 for v in jid_d.values()])),
    }


def edit_cluster_ci(all_preds, a_id, b_id):
    pairs = paired_rows(all_preds, a_id, b_id)
    if not pairs:
        return {"available": False}
    jid_edit = {p["jid"]: edit_from_fold(p["fold"]) for p in all_preds}
    jid_d = defaultdict(list)
    for rid, d in pairs.items():
        pa, pb = d[a_id], d[b_id]
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
    return {
        "available": True,
        "n_junctions": len(jid_d), "n_edit_components": len(by_edit),
        "mean_delta": float(np.mean([float(np.mean(v)) for v in jid_d.values()])),
        "edit_cluster_boot_95ci": [lo, hi],
        "ci_lower_gt_0": bool(lo > 0),
        "largest_edit_component": largest,
        "largest_edit_size": sizes[largest],
        "leave_one_largest_mean_delta": leave1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="+", default=[
        "reg_deep:motif_topology_hierarchy",
        "reg_light:motif_topology_hierarchy",
        "reg:motif_topology_hierarchy",
        "reg_deep:nonlinear_mlp_hybrid",
        "reg_deep:reg",
    ])
    args = ap.parse_args()

    all_preds = load_preds(args.preds)
    # The r14 run names its models with the full factory id; strip nothing.
    pooled = pooled_nll_by_model(all_preds)

    out = {"n_predictions": len(all_preds), "pooled_junction_macro_nll": pooled,
           "contrasts": {}}
    for pair in args.pairs:
        a_id, b_id = pair.split(":")
        name_a = a_id
        name_b = b_id
        pc = pooled_contrast(all_preds, a_id, b_id, name_a, name_b)
        ec = edit_cluster_ci(all_preds, a_id, b_id)
        out["contrasts"][f"{a_id}_vs_{b_id}"] = {
            "pooled": pc, "edit_cluster": ec,
            "note": f"positive delta = {name_a} (model A) is BETTER than {name_b}",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
