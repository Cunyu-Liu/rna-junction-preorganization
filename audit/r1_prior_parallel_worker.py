"""Parallel fold worker for the Denny-train-only and RNAMake/physical-prior
baselines.  Args: <cfg.json> <model_id> <worker_idx> <n_workers> <shard_dir>.

model_id: denny_train_only | physical_ensemble_prior
Replicates the frozen_lm worker sharding (same fold tasks, same record schemas,
same scorer) so the merge step can aggregate all three supplemental baselines
into r05_prior and R1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score
from audit.benchmark.denny_train_only import DENNY_TRAIN_ONLY
from audit.benchmark.physical_prior import (
    build_physical_cache, fit_physical_head, predict_physical_head)
from audit.r05_run import load_splits
from audit.r1_run import build_joint_edit_context_folds

LEADERBOARD_FIELDS = ["axis", "fold", "model_id", "coverage",
                      "pooled_junction_macro_nll", "eligible_full_coverage",
                      "n_eligible", "n_abstain_no_fallback", "converged", "error"]


def _denny_worker():
    return DENNY_TRAIN_ONLY["denny_train_only"]


def _phys_worker(feat_cache):
    def fit(train_rows):
        return fit_physical_head(train_rows, feat_cache)

    def pred(model, test_rows):
        return predict_physical_head(model, test_rows, feat_cache)

    return fit, pred


def run_fold(fit_fn, pred_fn, train_rows, test_rows):
    try:
        model = fit_fn(train_rows)
        mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
        conv = {"success": bool(model.get("success", model.get("gate", {}).get("success", True))),
                "n_iter": int(model.get("nit", model.get("gate", {}).get("n_iter", -1))),
                "n_bound_hits": int(model.get("gate", {}).get("n_bound_hits", -1)),
                "n_nan_inf_params": int(model.get("gate", {}).get("n_nan_inf_params", -1)),
                "final_grad_norm": float(model.get("final_grad_norm",
                                                    model.get("gate", {}).get("final_grad_norm", float("nan"))))}
        ok = bool(conv["success"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}, None, None
    preds_by_rowid = {}
    for i, r in enumerate(test_rows):
        preds_by_rowid[str(r["source_row_id"])] = {
            "mu": float(mu[i]), "sigma": float(sigma[i]),
            "abstain": bool(abstain[i]), "support": bool(support[i]),
            "fallback_type": None,
        }
    metric, elig = full_coverage_score(test_rows, preds_by_rowid)
    return {"convergence": conv, "converged": ok, "metric": metric,
            "eligible": elig["eligible"], "elig_reason": elig["reason"]}, preds_by_rowid, conv


def build_fold_tasks(cfg, admitted):
    from audit.r05_run import load_splits
    protocol = Path(cfg["protocol_dir"])
    tasks = []
    for axis in cfg["axes"]:
        mp = protocol / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            continue
        _, by_fold = load_splits(mp)
        for fold_key in sorted(by_fold.keys()):
            tasks.append({"kind": "axis", "axis": axis, "fold": str(fold_key),
                          "test_ids": by_fold[fold_key]})
    for f in build_joint_edit_context_folds(admitted):
        if f["feasible"]:
            tasks.append({"kind": "joint", "axis": "edit_x_nested_context",
                          "fold": f["fold"], "test_ids": f["test_ids"],
                          "train_ids": f["train_ids"]})
    return tasks


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    model_id = sys.argv[2]
    worker_idx = int(sys.argv[3])
    n_workers = int(sys.argv[4])
    shard_dir = Path(sys.argv[5])
    shard_dir.mkdir(parents=True, exist_ok=True)

    run_root = Path(cfg["run_root"])
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}

    if model_id == "denny_train_only":
        fit_fn, pred_fn = _denny_worker()
        feat_cache = None
    elif model_id == "physical_ensemble_prior":
        cache_fp = run_root / "r05_prior" / "PhysicalFeatureCache.npz"
        dat = np.load(cache_fp, allow_pickle=True)
        feat_cache = {str(k): v for k, v in zip(dat["seqs"], dat["feats"])}
        fit_fn, pred_fn = _phys_worker(feat_cache)
    else:
        raise ValueError(f"unknown model_id {model_id}")

    tasks = build_fold_tasks(cfg, admitted)
    mine = [t for i, t in enumerate(tasks) if i % n_workers == worker_idx]

    pred_shard = []
    lb_shard = []
    conv_shard = []
    for task in mine:
        test_ids = {str(x) for x in task["test_ids"]}
        test_rows = [r for sid, r in rows.items() if sid in test_ids]
        if task["kind"] == "joint":
            train_ids = {str(x) for x in task["train_ids"]}
            train_rows = [r for sid, r in rows.items() if sid in train_ids]
        else:
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
        res, preds, conv = run_fold(fit_fn, pred_fn, train_rows, test_rows)
        conv_shard.append({"axis": task["axis"], "fold": task["fold"],
                           "model_id": model_id,
                           **({} if conv is None else conv),
                           "error": res.get("error") if res else "fit_error"})
        if res is None or "error" in res:
            lb_shard.append({"axis": task["axis"], "fold": task["fold"],
                             "model_id": model_id,
                             "error": res["error"] if res else "fit_error"})
            continue
        if preds:
            for rid, p in preds.items():
                r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                pred_shard.append({"axis": task["axis"], "fold": task["fold"],
                                   "source_row_id": rid, "jid": r["jid"],
                                   "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                                   "model_id": model_id, "y": r["y"],
                                   "cens": bool(r["cens"]), "mu": p["mu"],
                                   "sigma": p["sigma"], "abstain": p["abstain"]})
        lb_shard.append({"axis": task["axis"], "fold": task["fold"],
                         "model_id": model_id,
                         "coverage": res["metric"]["coverage"],
                         "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                         "eligible_full_coverage": res["eligible"],
                         "n_eligible": res["metric"]["n_eligible"],
                         "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                         "converged": res["converged"]})

    with (shard_dir / f"{model_id}_worker_{worker_idx}.preds.jsonl").open("w") as fh:
        for rec in pred_shard:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pd.DataFrame(lb_shard).to_csv(shard_dir / f"{model_id}_worker_{worker_idx}.leaderboard.csv", index=False)
    pd.DataFrame(conv_shard).to_parquet(shard_dir / f"{model_id}_worker_{worker_idx}.conv.parquet")
    print(json.dumps({"model": model_id, "worker": worker_idx, "n_tasks_mine": len(mine),
                      "n_preds": len(pred_shard), "n_lb": len(lb_shard),
                      "n_conv": len(conv_shard)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
