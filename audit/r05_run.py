"""R0.5 minimal necessary rerun (contract R0.5 / §13.6).

Freshly fits the six parametric baselines whose censored-row gradient was
WRONG (global_censor_intercept, train_only_scaffold, scaffold_context_hierarchy,
motif_topology_hierarchy, onehot_kmer_ridge, position_aware_additive) plus the
deterministic edit_knn, using the unified R0.2 CensoredObjective, and scores
them with the R0.2 support-aware scorer (full-coverage task).

Runs on:
  - the four frozen single-axis splits (symmetry/edit/context/scaffold),
  - the decisive joint edit_x_nested_context split (zero overlap in both
    sequence and nested-context dimensions).

Outputs (new isolated R0 run root):
  Predictions_v2.jsonl        row-level predictions (unique keys)
  LeaderboardDraft_v2.csv     per model x axis coverage + pooled NLL
  ConvergenceLedger.parquet   per model x fold optimizer gate
  STATUS.json

Corrected v1.31 is handled separately (heavier GH objective).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.benchmark.baselines import BASELINES
from audit.benchmark.phase1_baselines import PHASE1_MODELS
from audit.core.censored_objective import objective_gate_passes
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys

# The six gradient-affected parametric baselines + deterministic edit_knn.
R05_MODELS = {
    "global_censor_intercept": BASELINES["global_censor_intercept"],
    "train_only_scaffold": BASELINES["train_only_scaffold"],
    "scaffold_context_hierarchy": BASELINES["scaffold_context_hierarchy"],
    "motif_topology_hierarchy": PHASE1_MODELS["motif_topology_hierarchy"],
    "onehot_kmer_ridge": PHASE1_MODELS["onehot_kmer_ridge"],
    "position_aware_additive": PHASE1_MODELS["position_aware_additive"],
    "edit_knn": PHASE1_MODELS["edit_knn"],
}


def _convergence(model):
    """Return (record_dict, passes_gate) for a fitted model."""
    gate = model.get("gate")
    if gate is not None:
        return gate, objective_gate_passes(gate)
    # non-parametric deterministic models
    if model.get("kind") in ("knn", "mutation_graph") or "seqs" in model:
        return {"deterministic": True, "converged": True}, True
    return {"converged": False, "reason": "no optimizer gate"}, False


def _gate_fields(res):
    """Persist full optimizer-gate diagnostics for a fold (contract R0.2)."""
    if not res:
        return {}
    conv = res.get("convergence") or {}
    return {
        "success": conv.get("success"),
        "n_iter": conv.get("n_iter"),
        "n_bound_hits": conv.get("n_bound_hits"),
        "n_nan_inf_params": conv.get("n_nan_inf_params"),
        "final_grad_norm": conv.get("final_grad_norm"),
    }


def load_splits(manifest_path: Path):
    by_fold = defaultdict(set)
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return axis, by_fold


def build_joint_edit_context_folds(admitted):
    """Leave-one-edit-out joint fold: test = edit e rows; train excludes rows
    sharing edit e OR any nested context present in test (zero overlap both)."""
    from audit.splits.joint_blocked import build_joint_edit_context
    rep = build_joint_edit_context(admitted)
    folds = []
    for f in rep["folds"]:
        e = f["edit_component"]
        test_ids = {str(r["source_row_id"]) for r in admitted
                    if str(r["edit_component"]) == e}
        test_ctxs = {str(r["helix_seq"]) for r in admitted
                     if str(r["edit_component"]) == e}
        train_ids = {str(r["source_row_id"]) for r in admitted
                     if str(r["edit_component"]) != e
                     and str(r["helix_seq"]) not in test_ctxs}
        folds.append({"axis": "edit_x_nested_context", "fold": f"e:{e}",
                      "test_ids": test_ids, "train_ids": train_ids,
                      "feasible": f["feasible"]})
    return folds


def run_fold(model_id, fit_fn, pred_fn, train_rows, test_rows):
    try:
        model = fit_fn(train_rows)
        mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
        conv, ok = _convergence(model)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}, None
    preds_by_rowid = {}
    for i, r in enumerate(test_rows):
        preds_by_rowid[str(r["source_row_id"])] = {
            "mu": float(mu[i]), "sigma": float(sigma[i]),
            "abstain": bool(abstain[i]), "support": bool(support[i]),
            "fallback_type": None,
        }
    metric, elig = full_coverage_score(test_rows, preds_by_rowid)
    return {"model_id": model_id, "convergence": conv, "converged": ok,
            "metric": metric, "eligible": elig["eligible"],
            "elig_reason": elig["reason"]}, preds_by_rowid


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r05"
    out.mkdir(parents=True, exist_ok=True)
    # Load admitted rows via the canonical reconstruction (R0.3-consistent) so
    # `edit_component` is present for the joint split, and source_row_id matches
    # the frozen split manifests / feasibility report.
    _, admitted, profile, eff_n, dep, exposure = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}
    protocol = Path(cfg["protocol_dir"])

    axes = cfg["axes"]
    all_preds = []
    leaderboard = []
    conv_rows = []

    for axis in axes:
        mp = protocol / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest {axis}")
            continue
        _, by_fold = load_splits(mp)
        for fold, test_ids in sorted(by_fold.items()):
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            for model_id, (fit_fn, pred_fn) in R05_MODELS.items():
                res, preds = run_fold(model_id, fit_fn, pred_fn, train_rows, test_rows)
                rec = {"axis": axis, "fold": str(fold), "model_id": model_id,
                       "converged": res.get("converged") if res else None,
                       "error": res.get("error") if res else "fit_error"}
                rec.update(_gate_fields(res))
                conv_rows.append(rec)
                if res is None or "error" in res:
                    leaderboard.append({"axis": axis, "fold": str(fold), "model_id": model_id,
                                        "error": res["error"] if res else "fit_error"})
                    continue
                if preds:
                    for rid, p in preds.items():
                        r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                        all_preds.append({"axis": axis, "fold": str(fold),
                                          "source_row_id": rid, "jid": r["jid"],
                                          "scaf": int(r["scaf"]),
                                          "context": str(r["helix_seq"]),
                                          "model_id": model_id,
                                          "y": r["y"], "cens": bool(r["cens"]),
                                          "mu": p["mu"], "sigma": p["sigma"],
                                          "abstain": p["abstain"]})
                leaderboard.append({"axis": axis, "fold": str(fold), "model_id": model_id,
                                    "coverage": res["metric"]["coverage"],
                                    "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                                    "eligible_full_coverage": res["eligible"],
                                    "n_eligible": res["metric"]["n_eligible"],
                                    "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                                    "converged": res["converged"]})

    # joint edit_x_nested_context split
    jf = build_joint_edit_context_folds(admitted)
    for f in jf:
        test_rows = [r for sid, r in rows.items() if sid in f["test_ids"]]
        train_rows = [r for sid, r in rows.items() if sid in f["train_ids"]]
        for model_id, (fit_fn, pred_fn) in R05_MODELS.items():
            res, preds = run_fold(model_id, fit_fn, pred_fn, train_rows, test_rows)
            rec = {"axis": "edit_x_nested_context", "fold": f["fold"],
                   "model_id": model_id,
                   "converged": res.get("converged") if res else None,
                   "error": res.get("error") if res else "fit_error"}
            rec.update(_gate_fields(res))
            conv_rows.append(rec)
            if res is None or "error" in res:
                leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                    "model_id": model_id,
                                    "error": res["error"] if res else "fit_error"})
                continue
            if preds:
                for rid, p in preds.items():
                    r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                    all_preds.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                      "source_row_id": rid, "jid": r["jid"],
                                      "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                                      "model_id": model_id, "y": r["y"], "cens": bool(r["cens"]),
                                      "mu": p["mu"], "sigma": p["sigma"], "abstain": p["abstain"]})
            leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                "model_id": model_id,
                                "coverage": res["metric"]["coverage"],
                                "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                                "eligible_full_coverage": res["eligible"],
                                "n_eligible": res["metric"]["n_eligible"],
                                "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                                "converged": res["converged"]})

    # write outputs
    with (out / "Predictions_v2.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dups = validate_unique_keys([{**p, "model_id": p["model_id"],
                                  "fold": p["fold"]} for p in all_preds])
    import csv
    with (out / "LeaderboardDraft_v2.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["axis", "fold", "model_id", "coverage",
                                           "pooled_junction_macro_nll",
                                           "eligible_full_coverage", "n_eligible",
                                           "n_abstain_no_fallback", "converged", "error"])
        w.writeheader()
        for r in leaderboard:
            w.writerow(r)
    # convergence ledger
    import pandas as pd
    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger.parquet")

    n_pred = len(all_preds)
    status = {
        "phase": "R0.5", "state": "DONE", "n_predictions": n_pred,
        "n_leaderboard_rows": len(leaderboard), "n_convergence_rows": len(conv_rows),
        "duplicate_primary_keys": len(dups),
        "axes": axes + ["edit_x_nested_context"],
        "models": sorted(R05_MODELS.keys()),
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    # concise convergence summary
    conv = pd.DataFrame(conv_rows)
    if not conv.empty:
        grp = conv.groupby(["axis", "model_id"])["converged"].agg(
            ["count", "sum"]).rename(columns={"count": "folds", "sum": "converged"})
        print(grp.to_string())
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
