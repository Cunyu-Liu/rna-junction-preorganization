"""P4 gap closure: 3-seed final comparison + 1000-permutation sequence-pairing null.

Closes two contract deviations identified in the P4 audit:
  1. contract Phase 4 task "三seed最终比较" - run the split-unit bootstrap CI with
     3 seeds (0,1,2) and confirm the verdict is seed-consistent.
  2. contract Phase 4 task "1,000次null" - run 1000 sequence-pairing null
     permutations on the candidate-vs-strongest-baseline supported-NLL gain and
     report genuine vs null.

The candidate and baseline supported-NLL are deterministic (frozen P3 gates,
sealed P1 baselines, low-capacity edit-KNN). Only the bootstrap CI depends on the
seed; the null scrambles the junction<->sequence assignment to destroy any
sequence-identity signal while keeping support/coverage structure fixed.

No promotion is re-adjudicated here: this only supplies the two missing
diagnostic/robustness pieces. Evidence remains DEVELOPMENT_ONLY.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.models.support_aware_mixture import (
    support_features, fit_local, predict_gated, build_distance_cache, TAU,
    _seq_index, _dist_to_train_seq)
from audit.evaluation.metrics import junction_macro_nll

CANDIDATE = "support_aware_mixture"
AXES = ["symmetry_5fold", "edit_5fold", "context_lomo", "scaffold_lomo"]
STRONG = {
    "symmetry_5fold": "corrected_v1_31",
    "edit_5fold": "corrected_v1_31",
    "context_lomo": "train_only_scaffold",
    "scaffold_lomo": "edit_knn",
}


def load_rows(ledger_path):
    rows = {}
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("layer") != "admitted" or o.get("excluded"):
            continue
        rows[str(o["source_row_id"])] = o
    return rows


def load_splits(manifest_path):
    by_fold = defaultdict(set)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return by_fold


def load_frozen_gates(csv_path):
    out = {}
    for line in Path(csv_path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        if len(p) < 3 or p[0] == "axis":
            continue
        out[(p[0], int(p[1]))] = (None if p[2] == "None" else int(float(p[2])))
    return out


def load_p1(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[(o["axis"], int(o["fold"]), str(o["source_row_id"]), o["model_id"])] = (
            float(o["mu"]), float(o["sigma"]))
    return out


def _supported_nll(rows, mu_by_sid, sigma_by_sid):
    mus = [mu_by_sid[str(r["source_row_id"])] for r in rows]
    sigs = [sigma_by_sid[str(r["source_row_id"])] for r in rows]
    return junction_macro_nll(rows, mus, sigs)


def main(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    build_distance_cache(list(rows.values()))
    axes = cfg["axes"]
    gates = load_frozen_gates(Path(cfg["p3_gates"]))
    p1 = load_p1(Path(cfg["p1_preds"]))
    seeds = cfg["seeds"]
    n_null = cfg["n_null"]
    null_seed = cfg["null_seed"]

    universe = sorted({str(r["junction_seq"]) for r in rows.values()})
    univ_idx = {s: i for i, s in enumerate(universe)}
    n_univ = len(universe)

    # ---- deterministic per-fold candidate vs strongest baseline (coverage-matched) ----
    fold_gains = []      # axis,fold,strong,cand_sup_nll,strong_sup_nll,gain,rel_gain
    fold_data = []       # per fold: test_rows, sup_rows, sup_seq_uidx, base_mu_arr, strong_sup_nll
    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            continue
        by_fold = load_splits(mp)
        strong = STRONG[axis]
        for fold in sorted(by_fold.keys()):
            test_ids = by_fold[fold]
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            gate = gates.get((axis, fold), 1000)
            feats = support_features(train_rows, test_rows, 3)
            local = fit_local(train_rows)
            mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=gate)
            sup_idx = [i for i, s in enumerate(support) if s]
            sup_rows = [test_rows[i] for i in sup_idx]
            cand_sup_nll = junction_macro_nll(
                sup_rows, [mu[i] for i in sup_idx], [sigma[i] for i in sup_idx]) if sup_rows else None

            sup_sids = {str(r["source_row_id"]) for r in sup_rows}
            mu_by = {sid: p1[(axis, fold, sid, strong)][0]
                     for sid in sup_sids if (axis, fold, sid, strong) in p1}
            sig_by = {sid: p1[(axis, fold, sid, strong)][1]
                      for sid in sup_sids if (axis, fold, sid, strong) in p1}
            strong_sup_nll = _supported_nll(
                [r for r in sup_rows if str(r["source_row_id"]) in mu_by], mu_by, sig_by) if mu_by else None

            if cand_sup_nll is not None and strong_sup_nll is not None:
                gain = strong_sup_nll - cand_sup_nll
                rel = gain / strong_sup_nll if strong_sup_nll != 0 else 0.0
            else:
                gain, rel = None, None
            fold_gains.append({"axis": axis, "fold": fold, "strong": strong,
                               "cand_sup_nll": cand_sup_nll, "strong_sup_nll": strong_sup_nll,
                               "gain": gain, "rel_gain": rel})

            # per-fold base_mu over the full universe (for the null) as numpy array
            seq_idx_local = local["seq_idx"]
            node_val = local["node_val"]
            seqs = local["seqs"]
            train_seqs_idx = np.array([_seq_index(s) for s in seqs], dtype=np.int64)
            base_mu_arr = np.empty(n_univ)
            for s in universe:
                if s in seq_idx_local:
                    base_mu_arr[univ_idx[s]] = float(node_val[seq_idx_local[s]])
                else:
                    d = _dist_to_train_seq(train_seqs_idx, s)
                    k = min(local["k"], len(d))
                    idx = np.argsort(d)[:k]
                    base_mu_arr[univ_idx[s]] = float(np.mean(node_val[idx]))
            sup_seq_uidx = np.array([univ_idx[str(r["junction_seq"])] for r in sup_rows], dtype=np.int64)
            fold_data.append({"axis": axis, "fold": fold, "sup_rows": sup_rows,
                              "sup_seq_uidx": sup_seq_uidx, "base_mu_arr": base_mu_arr,
                              "strong_sup_nll": strong_sup_nll})

    # ---- 3-seed bootstrap CI over folds ----
    N_BOOT = cfg["n_boot"]
    per_seed = {}
    for seed in seeds:
        rows_for_seed = []
        for axis in axes:
            gs = [g for g in fold_gains if g["axis"] == axis and g["gain"] is not None]
            vals = np.asarray([g["gain"] for g in gs], dtype=float)
            rels = np.asarray([g["rel_gain"] for g in gs if g["rel_gain"] is not None], dtype=float)
            if len(vals) == 0:
                continue
            rng = np.random.default_rng(seed)
            means = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(vals), size=len(vals))
                means[b] = vals[idx].mean()
            lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
            rows_for_seed.append({
                "seed": seed, "axis": axis, "n_folds": len(vals),
                "strong_baseline": STRONG[axis],
                "observed_mean_gain": float(vals.mean()),
                "ci_low": float(lo), "ci_high": float(hi),
                "p_positive": float(np.mean(means > 0)),
                "n_folds_positive": int(np.sum(vals > 0)),
                "all_folds_positive": bool(np.all(vals > 0)),
                "mean_rel_gain": float(rels.mean()) if len(rels) else None,
                "promotion_eligible": bool(lo > 0 and np.all(vals > 0) and
                                           (float(rels.mean()) if len(rels) else 0.0) >= 0.10),
            })
        per_seed[str(seed)] = rows_for_seed
    pd.DataFrame([r for s in seeds for r in per_seed[str(s)]]
                 ).to_csv(out / "BootstrapIntervals_3seed.csv", index=False)

    # seed-consistency verdict
    seeds_comp = {}
    for axis in axes:
        eligible = {}
        cis = {}
        for s in seeds:
            r = next((r for r in per_seed[str(s)] if r["axis"] == axis), None)
            if r is None:
                continue
            eligible[s] = r["promotion_eligible"]
            cis[s] = [r["ci_low"], r["ci_high"]]
        seeds_comp[axis] = {
            "per_seed_promotion_eligible": eligible,
            "per_seed_ci_low_hi": cis,
            "all_seeds_consistent_not_promoted": bool(eligible and not any(eligible.values())),
        }

    # ---- 1000-permutation sequence-pairing null on supported-NLL gain ----
    rng_null = np.random.default_rng(null_seed)
    # per axis: genuine mean gain (over folds) and null distribution of mean gain
    genuine_by_axis = {}
    for axis in axes:
        gs = [g for g in fold_gains if g["axis"] == axis and g["gain"] is not None]
        genuine_by_axis[axis] = float(np.mean([g["gain"] for g in gs])) if gs else None

    null_dist = {a: [] for a in axes}
    for p in range(n_null):
        pi = rng_null.permutation(n_univ)
        for fd in fold_data:
            axis = fd["axis"]
            sup = fd["sup_rows"]
            if not sup:
                continue
            null_mu = fd["base_mu_arr"][pi[fd["sup_seq_uidx"]]]
            null_cand_nll = junction_macro_nll(sup, list(null_mu), [TAU] * len(sup))
            if fd["strong_sup_nll"] is not None:
                null_dist[axis].append(fd["strong_sup_nll"] - null_cand_nll)
    # aggregate per-axis: mean of fold-null-gains per permutation -> but simpler: per-axis
    # null distribution over folds (all permutation-fold pairs), compare genuine mean.
    null_rows = []
    for axis in axes:
        arr = np.asarray(null_dist[axis], dtype=float)
        if len(arr) == 0:
            continue
        lo, hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        genuine = genuine_by_axis[axis]
        null_rows.append({
            "axis": axis, "n_null_permutations": n_null,
            "n_null_fold_samples": int(len(arr)),
            "genuine_mean_gain": genuine,
            "null_2_5": float(lo), "null_97_5": float(hi),
            "genuine_gt_null_97_5": bool(genuine is not None and genuine > hi),
        })
    pd.DataFrame(null_rows).to_csv(out / "NullAdjudication_full.csv", index=False)

    # ---- summary ----
    summary = {
        "phase": "P4", "gap_closure": True, "state": "PASS",
        "candidate": CANDIDATE, "axes": axes,
        "seeds": seeds, "n_boot_per_seed": N_BOOT, "n_null": n_null, "null_seed": null_seed,
        "overall_promotion": "NOT_PROMOTED",
        "seed_consistency": seeds_comp,
        "sota_status": "SOTA_NOT_ADJUDICATED",
        "deliverables": ["BootstrapIntervals_3seed.csv", "NullAdjudication_full.csv",
                         "SeedsConsistency.json", "STATUS.json"],
    }
    (out / "SeedsConsistency.json").write_text(json.dumps(summary["seed_consistency"], indent=2) + "\n")
    (out / "STATUS.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
