"""Analyze a row-level MU-ENSEMBLE of regularized nonlinear MLP variants.

The single best r16 model is nonlinear_mlp_extended_hybrid_reg_deep (pooled
junction-macro NLL 0.9742, +10.76% over the nuisance baseline, edit-cluster CI
[0.0017, 0.174] -- wide).  This analyzer tests the cheapest methodological
improvement: average the predictive location mu (all members share the fixed
sigma=0.7 of the right-censored Gaussian head) across a set of decorrelated
3-layer regularized variants on the SAME 21-D extended-ViennaRNA features.

Every member was evaluated on the SAME held-out blocked folds, so averaging
their held-out mu is a valid model combination (no test leakage).  We report:
  - pooled junction-macro NLL of the ensemble
  - relative gain vs the nuisance baseline (motif_topology_hierarchy)
  - edit-cluster bootstrap 95% CI and leave-one-largest robustness
All contrast logic mirrors analyze_r14_scan.py / shootout_run.py for comparability.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.evaluation.metrics import row_nll


DEFAULT_MEMBERS = [
    "nonlinear_mlp_extended_hybrid_reg_light",
    "nonlinear_mlp_extended_hybrid_reg",
    "nonlinear_mlp_extended_hybrid_reg_wider",
    "nonlinear_mlp_extended_hybrid_reg_deep",
]
BASE = "motif_topology_hierarchy"
REF_SINGLE = "nonlinear_mlp_extended_hybrid_reg_deep"
ENSEMBLE_ID = "ENSEMBLE_MLP_MU"


def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def edit_from_fold(fold):
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


def build_ensemble_rows(all_preds, members):
    """Return list of synthetic ENSEMBLE predictions (mu = mean of member mus).

    A row is admitted only if it is supported+non-abstain in EVERY member, so
    the comparison is coverage-matched across the whole ensemble.
    """
    by_member = {}
    for m in members:
        d = {}
        for p in all_preds:
            if p["model_id"] == m and p["support"] and not p["abstain"]:
                d[p["source_row_id"]] = p
        by_member[m] = d
    common = set.intersection(*[set(d) for d in by_member.values()])
    ens = []
    for rid in common:
        ms = [by_member[m][rid]["mu"] for m in members]
        ref = by_member[members[0]][rid]
        ens.append({
            "axis": ref["axis"], "fold": ref["fold"], "source_row_id": rid,
            "jid": ref["jid"], "scaf": ref["scaf"], "context": ref["context"],
            "model_id": ENSEMBLE_ID, "y": ref["y"], "cens": ref["cens"],
            "mu": float(np.mean(ms)), "sigma": 0.7,
            "abstain": False, "support": True, "fallback_type": None,
        })
    return ens


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
        "available": True, "contrast": f"{a_name} vs {b_name}",
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
        "available": True, "n_junctions": len(jid_d), "n_edit_components": len(by_edit),
        "mean_delta": float(np.mean([float(np.mean(v)) for v in jid_d.values()])),
        "edit_cluster_boot_95ci": [lo, hi],
        "ci_lower_gt_0": bool(lo > 0),
        "largest_edit_component": largest, "largest_edit_size": sizes[largest],
        "leave_one_largest_mean_delta": leave1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--ref-single", default=REF_SINGLE)
    args = ap.parse_args()

    all_preds = load_preds(args.preds)
    ens_rows = build_ensemble_rows(all_preds, args.members)
    combined = list(all_preds) + ens_rows

    pooled = pooled_nll_by_model(combined)

    out = {
        "n_predictions": len(all_preds),
        "n_ensemble_members": len(args.members),
        "members": args.members,
        "n_ensemble_rows": len(ens_rows),
        "pooled_junction_macro_nll": pooled,
        "single_best_member_nll": pooled.get("nonlinear_mlp_extended_hybrid_reg_deep"),
        "contrasts": {},
    }
    name_a = f"{ENSEMBLE_ID} ({len(args.members)}x mu-ensemble)"
    pc = pooled_contrast(combined, ENSEMBLE_ID, args.base, name_a, args.base)
    ec = edit_cluster_ci(combined, ENSEMBLE_ID, args.base)
    out["contrasts"][f"{ENSEMBLE_ID}_vs_{args.base}"] = {
        "pooled": pc, "edit_cluster": ec,
        "note": f"positive delta = {name_a} (model A) is BETTER than {args.base}",
    }
    # Also verify the ensemble improves on the single best member.
    if args.ref_single in pooled:
        pc2 = pooled_contrast(combined, ENSEMBLE_ID, args.ref_single,
                              name_a, args.ref_single)
        ec2 = edit_cluster_ci(combined, ENSEMBLE_ID, args.ref_single)
        out["contrasts"][f"{ENSEMBLE_ID}_vs_{args.ref_single}"] = {
            "pooled": pc2, "edit_cluster": ec2,
            "note": f"positive delta = {name_a} (model A) is BETTER than {args.ref_single}",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
