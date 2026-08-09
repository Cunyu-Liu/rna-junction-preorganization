"""R1 unified correct-baseline leaderboard (contract §12.2 / §13.8).

Goal: one qualified baseline universe under identical rows, support policy,
splits, metric and budget.  R0.5 already emitted fresh predictions for the six
gradient-corrected parametric baselines + edit_knn (r05) and corrected v1.31
(r05_v131).  R1 adds the matched no-sequence latent operator (the R2 core
comparator, contract §9.2) on the SAME folds and merges everything into a
single Leaderboard_v2.csv + merged row predictions.

No old P1/P2 predictions are reused; everything comes from the new R0 lineage.

Outputs into RUN_ROOT/r1/:
  Leaderboard_v2.csv            per model x axis x fold coverage + pooled NLL
  Predictions_v2.jsonl          row-level predictions (unique primary keys)
  ConvergenceLedger.parquet     optimizer/convergence gate per model x fold
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.models.no_sequence_latent_operator import NO_SEQUENCE_LATENT_OPERATOR


def load_splits(manifest_path: Path):
    by_fold = defaultdict(set)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return by_fold


def build_joint_edit_context_folds(admitted):
    from audit.splits.joint_blocked import build_joint_edit_context
    rep = build_joint_edit_context(admitted)
    folds = []
    for f in rep["folds"]:
        e = f["edit_component"]
        test_ctxs = {str(r["helix_seq"]) for r in admitted
                     if str(r["edit_component"]) == e}
        test_ids = {str(r["source_row_id"]) for r in admitted
                    if str(r["edit_component"]) == e}
        train_ids = {str(r["source_row_id"]) for r in admitted
                     if str(r["edit_component"]) != e
                     and str(r["helix_seq"]) not in test_ctxs}
        folds.append({"axis": "edit_x_nested_context", "fold": f"e:{e}",
                      "test_ids": test_ids, "train_ids": train_ids,
                      "feasible": f["feasible"]})
    return folds


def run_fold(fit_fn, pred_fn, train_rows, test_rows):
    try:
        model = fit_fn(train_rows)
        mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
        conv = {"success": bool(model.get("success", True)),
                "n_iter": int(model.get("nit", -1)),
                "final_grad_norm": float(model.get("final_grad_norm", float("nan"))),
                "abstained_test": int(np.sum(abstain)),
                "supported_test": int(np.sum(support))}
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


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r1"
    out.mkdir(parents=True, exist_ok=True)
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}
    protocol = Path(cfg["protocol_dir"])

    axes = cfg["axes"]
    model_id = "no_sequence_latent_operator"
    fit_fn, pred_fn = NO_SEQUENCE_LATENT_OPERATOR[model_id]
    all_preds = []
    leaderboard = []
    conv_rows = []

    def eval_split(axis, folds, is_joint):
        for fold, test_ids in folds:
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            res, preds, conv = run_fold(fit_fn, pred_fn, train_rows, test_rows)
            conv_rows.append({"axis": axis, "fold": str(fold), "model_id": model_id,
                              **({} if conv is None else conv),
                              "error": res.get("error") if res else "fit_error"})
            if res is None or "error" in res:
                leaderboard.append({"axis": axis, "fold": str(fold),
                                    "model_id": model_id,
                                    "error": res["error"] if res else "fit_error"})
                continue
            if preds:
                for rid, p in preds.items():
                    r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                    all_preds.append({"axis": axis, "fold": str(fold),
                                      "source_row_id": rid, "jid": r["jid"],
                                      "scaf": int(r["scaf"]),
                                      "context": str(r["helix_seq"]),
                                      "model_id": model_id, "y": r["y"],
                                      "cens": bool(r["cens"]), "mu": p["mu"],
                                      "sigma": p["sigma"], "abstain": p["abstain"],
                                      "support": p["support"]})
            leaderboard.append({"axis": axis, "fold": str(fold), "model_id": model_id,
                                "coverage": res["metric"]["coverage"],
                                "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                                "eligible_full_coverage": res["eligible"],
                                "n_eligible": res["metric"]["n_eligible"],
                                "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                                "converged": res["converged"]})

    for axis in axes:
        mp = protocol / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest {axis}")
            continue
        eval_split(axis, sorted(load_splits(mp).items()), False)

    jf = build_joint_edit_context_folds(admitted)
    for f in jf:
        test_rows = [r for sid, r in rows.items() if sid in f["test_ids"]]
        train_rows = [r for sid, r in rows.items() if sid in f["train_ids"]]
        res, preds, conv = run_fold(fit_fn, pred_fn, train_rows, test_rows)
        conv_rows.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                          "model_id": model_id,
                          **({} if conv is None else conv),
                          "error": res.get("error") if res else "fit_error"})
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
                                  "mu": p["mu"], "sigma": p["sigma"], "abstain": p["abstain"],
                                  "support": p["support"]})
        leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                            "model_id": model_id,
                            "coverage": res["metric"]["coverage"],
                            "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                            "eligible_full_coverage": res["eligible"],
                            "n_eligible": res["metric"]["n_eligible"],
                            "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                            "converged": res["converged"]})

    ns_leaderboard = pd.DataFrame(leaderboard)

    # ---- merge with R0.5 baseline universe (parametric + edit_knn + v1.31) ----
    parts = []
    for sub, name in [("r05", "LeaderboardDraft_v2.csv"), ("r05_v131", "Leaderboard_v1_31.csv")]:
        p = run_root / sub / name
        if p.exists():
            parts.append(pd.read_csv(p))
    if parts:
        prior = pd.concat(parts, ignore_index=True)
        merged = pd.concat([prior, ns_leaderboard], ignore_index=True)
    else:
        merged = ns_leaderboard

    merged.to_csv(out / "Leaderboard_v2.csv", index=False)

    # merge row predictions
    pred_parts = [pd.DataFrame(all_preds)]
    for sub, name in [("r05", "Predictions_v2.jsonl"), ("r05_v131", "Predictions_v1_31.jsonl")]:
        p = run_root / sub / name
        if p.exists():
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            pred_parts.append(pd.DataFrame(recs))
    pred_merged = pd.concat(pred_parts, ignore_index=True)
    with (out / "Predictions_v2.jsonl").open("w") as fh:
        for rec in pred_merged.to_dict("records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger.parquet")

    status = {
        "phase": "R1", "state": "DONE",
        "n_models": int(merged["model_id"].nunique()),
        "n_leaderboard_rows": int(len(merged)),
        "n_no_sequence_preds": len(all_preds),
        "axes": axes + ["edit_x_nested_context"],
        "no_sequence_optimizer_success_folds": int((ns_leaderboard.get("converged", pd.Series(dtype=bool)).astype(bool)).sum())
            if "converged" in ns_leaderboard else None,
        "note": "Merged fresh R0-lineage predictions only; no old P1/P2 reuse.",
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
