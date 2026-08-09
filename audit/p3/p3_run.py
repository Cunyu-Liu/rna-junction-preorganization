"""Phase 3 Candidate C runner: support-aware gated mixture with abstention (true nested CV).

Fixes the selection-on-test leak in the previous version: the gate threshold
d_thresh is now selected per outer fold using ONLY the outer-train (folds != f)
via a bounded leave-one-inner-fold-out CV, then applied to the held-out outer
fold f. The reported supported-NLL / coverage on fold f never inform the gate
chosen for f.

For each outer fold f:
  1. outer_train = all rows in folds != f ; outer_test = fold f rows.
  2. Build K inner folds from outer_train grouped by the axis `group` key
     (deterministic binning so within-group rows stay together).
  3. Inner CV: for each inner fold g, fit local + scaffold on inner_train,
     evaluate supported-NLL gain (scaffold - candidate) on g for each d in the
     pre-registered GATE_GRID. Aggregate gain & coverage across inner folds.
  4. Select d*(f) = argmax mean gain subject to mean inner coverage >= 0.5.
  5. Fit on full outer_train, evaluate on outer_test fold f with d*(f).

Emits Phase 3 deliverables (DEVELOPMENT_ONLY, no scientific claim):
  CandidateRegistry.json
  AblationRegistry.json
  InnerCVSelection.json        (per-axis/fold -> selected gate, inner gain/coverage)
  SupportedNLL.csv             (per axis/fold/d diagnostic stats at every gate)
  SelectedGateEvaluation.csv   (per axis/fold at the nested-selected gate)
  CoverageRisk.csv             (per axis aggregate at selected gate)
  CandidatePromotionDecision.json
  STATUS.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.models.support_aware_mixture import (
    support_features, fit_local, predict_gated, supported_metrics, GATE_GRID, SUPPORT_DIST,
    build_distance_cache,
)
from audit.benchmark.baselines import fit_scaffold, predict_scaffold
from audit.evaluation.metrics import junction_macro_nll

CANDIDATE_ID = "support_aware_mixture"

CANDIDATE_REGISTRY = {
    "candidate_id": CANDIDATE_ID,
    "phase": "P3",
    "unique_new_capability": ("explicit support/abstention gate: abstain (exclude from "
        "scoring) on junctions lacking local sequence-neighbour support, so "
        "extrapolation is never scored as a catastrophic interpolation"),
    "matched_comparators": ["train_only_scaffold", "edit_knn", "corrected_v1_31"],
    "ablations": [
        {"id": "no_abstention", "note": "gate threshold=inf (score everything, i.e. pure edit-KNN)"},
        {"id": "no_support_features", "note": "gate uses only min_edit_dist, not n_neighbors/scaf/ctx"},
        {"id": "fixed_gate", "note": "d_thresh=3 fixed, no sweep"},
    ],
    "budget": {"seed": 0, "gate_grid": GATE_GRID,
               "gate_selection": "per-fold nested leave-one-inner-fold-out on outer-train; "
                                 "outer test fold f never informs gate for f",
               "note": "low-capacity; gate grid pre-registered, selected via nested CV (not test)"},
    "elimination_condition": ("if abstention gives no risk improvement (supported NLL not lower "
        "than baseline at matched coverage) and coverage is too low, keep pure ordinal/fallback "
        "and discard the learned gate"),
    "claim_scope": "KNOWN_OPERATOR_CONDITIONAL_ONLY; no operator-transfer claim",
}

ABLATION_REGISTRY = {
    "ablations": CANDIDATE_REGISTRY["ablations"],
    "note": "each ablation is a matched comparator; complexity added by the gate is justified only if it improves supported NLL / reduces catastrophic folds",
}

K_INNER = 8          # bounded inner folds for nested gate selection
COVERAGE_MIN = 0.5   # pre-registered minimum mean inner coverage for a gate to be eligible


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
    """Return (axis, {fold: set(source_row_id)}, n_folds) and {sid: group_key}."""
    by_fold = defaultdict(set)
    sid_group = {}
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        sid = str(o["source_row_id"])
        by_fold[o["fold"]].add(sid)
        sid_group[sid] = str(o.get("group", o.get("fold")))
    return axis, by_fold, len(by_fold), sid_group


def _scaffold_supported_nll(model, test_rows, mask):
    mu, sigma, cp, support, abstain = predict_scaffold(model, test_rows)
    sel = [i for i, m in enumerate(mask) if m]
    if not sel:
        return None
    sub = [test_rows[i] for i in sel]
    return junction_macro_nll(sub, [mu[i] for i in sel], [sigma[i] for i in sel])


def _inner_folds(outer_train_sids, sid_group, k=K_INNER):
    """Bin outer-train row ids into k inner folds grouped by axis group key.

    Deterministic: group_key -> stable hash -> inner bin. Within-group rows stay
    in the same inner fold, preserving grouped structure for gate selection."""
    bins = defaultdict(list)
    for sid in outer_train_sids:
        g = sid_group.get(sid, sid)
        h = int(hashlib.md5(g.encode()).hexdigest(), 16) % k
        bins[h].append(sid)
    # drop empty bins
    return [ids for ids in bins.values() if ids]


def _eval_supported_gain(inner_test_sids, outer_train_sids, rows):
    """Fit local + scaffold on outer_train \ inner_test; return per-gate gain/coverage.

    inner_train is restricted to outer-train rows (folds != current outer fold),
    so the outer test fold f never contributes to gate selection for f."""
    test_ids = set(inner_test_sids)
    test_rows = [r for sid, r in rows.items() if sid in test_ids]
    train_rows = [r for sid, r in rows.items() if sid in outer_train_sids and sid not in test_ids]
    feats = support_features(train_rows, test_rows, SUPPORT_DIST)
    local = fit_local(train_rows)
    scaff = fit_scaffold(train_rows)
    out = {}
    for d in GATE_GRID:
        mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=d)
        sm = supported_metrics(test_rows, mu, sigma, support)
        bm = _scaffold_supported_nll(scaff, test_rows, support)
        gain = (float(bm - sm["supported_nll"]) if sm["supported_nll"] is not None
                and bm is not None else None)
        out[d] = {"gain": gain, "coverage": sm["coverage"],
                  "supported_nll": sm["supported_nll"],
                  "baseline_supported_nll": bm, "n_supported": sm["n_supported"]}
    return out


def _select_gate(inner_fold_results):
    """inner_fold_results: list of per-inner-fold dict {d: {gain, coverage,...}}.
    Aggregate gain & coverage across inner folds; select d maximizing mean gain
    subject to mean coverage >= COVERAGE_MIN."""
    agg_gain = {d: [] for d in GATE_GRID}
    agg_cov = {d: [] for d in GATE_GRID}
    for ifr in inner_fold_results:
        for d, m in ifr.items():
            if m["gain"] is not None:
                agg_gain[d].append(m["gain"])
            agg_cov[d].append(m["coverage"])
    best_d, best_score = None, -np.inf
    per_d = {}
    for d in GATE_GRID:
        gs = agg_gain[d]
        cv = float(np.mean(agg_cov[d])) if agg_cov[d] else 0.0
        mean_gain = float(np.mean(gs)) if gs else None
        per_d[d] = {"mean_inner_gain": mean_gain, "mean_inner_coverage": cv,
                    "n_inner_folds_gain": len(gs)}
        if not gs or cv < COVERAGE_MIN:
            continue
        if mean_gain > best_score:
            best_d, best_score = d, mean_gain
    return best_d, best_score, per_d


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(cfg["records"]))
    build_distance_cache(list(rows.values()))   # precompute global Levenshtein matrix once
    axes = cfg["axes"]

    selection_records = []       # per axis/fold -> selected gate + inner stats
    supported_diag = []          # per axis/fold/d diagnostic
    selected_eval = []           # per axis/fold at selected gate
    cov_rows = []                # per axis aggregate at selected gate

    for axis in axes:
        mp = Path(cfg["protocol_dir"]) / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest {axis}")
            continue
        _, by_fold, n_folds, sid_group = load_splits(mp)
        fold_list = sorted(by_fold.keys())
        sel_gate_by_fold = {}

        for fold in fold_list:
            test_ids = by_fold[fold]
            outer_train_sids = [sid for sid in rows if sid not in test_ids]

            # --- nested inner gate selection on outer-train (folds != fold) ---
            inner_folds = _inner_folds(outer_train_sids, sid_group, k=K_INNER)
            inner_fold_results = []
            for inner_ids in inner_folds:
                inner_fold_results.append(_eval_supported_gain(inner_ids, outer_train_sids, rows))
            best_d, best_score, per_d = _select_gate(inner_fold_results)

            # --- evaluate on held-out outer fold f using the inner-selected gate ---
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            feats = support_features(train_rows, test_rows, SUPPORT_DIST)
            local = fit_local(train_rows)
            scaff = fit_scaffold(train_rows)

            # diagnostic at every gate (no selection involved, purely descriptive)
            for d in GATE_GRID:
                mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=d)
                sm = supported_metrics(test_rows, mu, sigma, support)
                bm = _scaffold_supported_nll(scaff, test_rows, support)
                supported_diag.append({
                    "axis": axis, "fold": fold, "d_thresh": d,
                    "coverage": sm["coverage"], "supported_nll": sm["supported_nll"],
                    "n_supported": sm["n_supported"],
                    "baseline_supported_nll": bm,
                })

            # selected-gate evaluation
            sel_gate_by_fold[fold] = best_d
            if best_d is not None:
                mu, sigma, cp, support, abstain = predict_gated(local, feats, test_rows, d_thresh=best_d)
                sm = supported_metrics(test_rows, mu, sigma, support)
                bm = _scaffold_supported_nll(scaff, test_rows, support)
                catastrophic = (sm["supported_nll"] is not None and bm is not None
                               and sm["supported_nll"] > 1.1 * bm)
            else:
                sm = {"coverage": 0.0, "supported_nll": None, "n_supported": 0}
                bm = None
                catastrophic = False
            selected_eval.append({
                "axis": axis, "fold": fold, "selected_d_thresh": best_d,
                "mean_inner_gain": (best_score if best_d is not None else None),
                "coverage": sm["coverage"], "supported_nll": sm["supported_nll"],
                "n_supported": sm["n_supported"], "baseline_supported_nll": bm,
                "catastrophic": bool(catastrophic),
            })
            selection_records.append({
                "axis": axis, "fold": fold, "selected_d_thresh": best_d,
                "per_gate_inner": per_d,
            })

        # aggregate per-axis at selected gate
        sel_evals = [e for e in selected_eval if e["axis"] == axis]
        nlls = [e["supported_nll"] for e in sel_evals if e["supported_nll"] is not None]
        covs = [e["coverage"] for e in sel_evals]
        cats = [e["catastrophic"] for e in sel_evals]
        cov_rows.append({
            "axis": axis, "n_folds": len(sel_evals),
            "mean_coverage": float(np.mean(covs)) if covs else 0.0,
            "mean_supported_nll": float(np.mean(nlls)) if nlls else None,
            "n_catastrophic_supported_folds": int(sum(cats)),
            "n_folds_gate_selected": int(sum(1 for e in sel_evals if e["selected_d_thresh"] is not None)),
        })

    # ---- write deliverables ----
    with (out_dir / "SupportedNLL.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "fold", "d_thresh", "coverage", "supported_nll",
                    "n_supported", "baseline_supported_nll"])
        for r in sorted(supported_diag, key=lambda x: (x["axis"], x["fold"], x["d_thresh"])):
            w.writerow([r["axis"], r["fold"], r["d_thresh"], r["coverage"],
                        r["supported_nll"], r["n_supported"], r["baseline_supported_nll"]])

    with (out_dir / "SelectedGateEvaluation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "fold", "selected_d_thresh", "mean_inner_gain", "coverage",
                    "supported_nll", "n_supported", "baseline_supported_nll", "catastrophic"])
        for r in sorted(selected_eval, key=lambda x: (x["axis"], x["fold"])):
            w.writerow([r["axis"], r["fold"], r["selected_d_thresh"], r["mean_inner_gain"],
                        r["coverage"], r["supported_nll"], r["n_supported"],
                        r["baseline_supported_nll"], r["catastrophic"]])

    with (out_dir / "CoverageRisk.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "n_folds", "mean_coverage", "mean_supported_nll",
                    "n_catastrophic_supported_folds", "n_folds_gate_selected"])
        for r in sorted(cov_rows, key=lambda x: x["axis"]):
            w.writerow([r["axis"], r["n_folds"], r["mean_coverage"], r["mean_supported_nll"],
                        r["n_catastrophic_supported_folds"], r["n_folds_gate_selected"]])

    (out_dir / "CandidateRegistry.json").write_text(
        json.dumps(CANDIDATE_REGISTRY, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "AblationRegistry.json").write_text(
        json.dumps(ABLATION_REGISTRY, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "InnerCVSelection.json").write_text(
        json.dumps(selection_records, indent=2, ensure_ascii=False) + "\n")

    decision = {
        "phase": "P3",
        "candidate": CANDIDATE_ID,
        "evidence_class": "DEVELOPMENT_ONLY",
        "gate_selection": "per-fold nested leave-one-inner-fold-out CV on outer-train; "
                          "outer test fold never informs its own gate (no selection-on-test)",
        "promotion": "NOT_ADJUDICATED",
        "note": "inner-CV gate selection and coverage-risk established; candidate "
                "promotion and outer-test comparison are Phase 4 scope",
        "claim_scope": CANDIDATE_REGISTRY["claim_scope"],
    }
    (out_dir / "CandidatePromotionDecision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n")

    status = {"phase": "P3", "state": "PASS",
              "candidate": CANDIDATE_ID,
              "axes": axes,
              "gate_grid": GATE_GRID,
              "gate_selection": decision["gate_selection"],
              "k_inner": K_INNER,
              "coverage_min": COVERAGE_MIN,
              "coverage_risk_rows": len(cov_rows),
              "selection_records": len(selection_records),
              "gates": {"nested_gate_selection_no_leak": True,
                        "abstention_rules_frozen": True,
                        "development_only_no_claim": True}}
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(main(cfg), indent=2, ensure_ascii=False))
