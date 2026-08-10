"""R1 supplemental: mutation-graph Laplacian baseline (contract §11/§12.2).

The mutation-graph smoother (phase1_baselines.fit_mutation_graph /
predict_mutation_graph) was implemented and unit-tested but never emitted into
the R0.5/R1 lineage (flagged NOT_RUN).  This runner fits it on the SAME frozen
folds (4 single-axis splits + decisive edit_x_nested_context joint split) and
merges the fresh predictions into the unified R1 leaderboard WITHOUT recomputing
the other already-verified R1 models.

Outputs:
  r05_graph/Predictions.jsonl        mutation_graph_smoother row predictions
  r05_graph/LeaderboardDraft.csv     per model x fold coverage + pooled NLL
  r05_graph/ConvergenceLedger.parquet
  r05_graph/STATUS.json
  r1/Leaderboard_v2.csv              updated (9 -> 10 models)
  r1/Predictions_v2.jsonl            updated
  r1/ConvergenceLedger.parquet       updated with mutation_graph folds
  r1/STATUS.json                     updated
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.benchmark.phase1_baselines import fit_mutation_graph, predict_mutation_graph
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.r05_run import load_splits, build_joint_edit_context_folds

MODEL_ID = "mutation_graph_smoother"
LEADERBOARD_FIELDS = ["axis", "fold", "model_id", "coverage",
                      "pooled_junction_macro_nll", "eligible_full_coverage",
                      "n_eligible", "n_abstain_no_fallback", "converged", "error"]


def mutation_graph_already_in_leaderboard(leaderboard_fp):
    """True iff the mutation-graph baseline is already merged into the R1 set."""
    if not Path(leaderboard_fp).exists():
        return False
    models = set(pd.read_csv(leaderboard_fp)["model_id"].unique())
    return MODEL_ID in models


def run_fold(fit_fn, pred_fn, train_rows, test_rows):
    try:
        model = fit_fn(train_rows)
        mu, sigma, cp, support, abstain = pred_fn(model, test_rows)
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
    return {"converged": True, "metric": metric, "eligible": elig["eligible"],
            "elig_reason": elig["reason"]}, preds_by_rowid


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_graph"
    out.mkdir(parents=True, exist_ok=True)
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}
    protocol = Path(cfg["protocol_dir"])
    axes = cfg["axes"]

    # if mutation_graph already merged into R1, refuse to double-add
    leaderboard_fp = run_root / "r1" / "Leaderboard_v2.csv"
    if mutation_graph_already_in_leaderboard(leaderboard_fp):
        print(f"[skip] {MODEL_ID} already present in R1 leaderboard; remove first to re-run.")
        return {"phase": "R1_GRAPH", "state": "SKIPPED_ALREADY_PRESENT"}

    all_preds = []
    leaderboard = []
    conv_rows = []

    def eval_split(axis, folds):
        for fold, test_ids in folds:
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            res, preds = run_fold(fit_mutation_graph, predict_mutation_graph,
                                  train_rows, test_rows)
            conv_rows.append({"axis": axis, "fold": str(fold), "model_id": MODEL_ID,
                              "deterministic": True,
                              "converged": res.get("converged") if res else None,
                              "error": res.get("error") if res else "fit_error"})
            if res is None or "error" in res:
                leaderboard.append({"axis": axis, "fold": str(fold),
                                    "model_id": MODEL_ID,
                                    "error": res["error"] if res else "fit_error"})
                continue
            if preds:
                for rid, p in preds.items():
                    r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                    all_preds.append({"axis": axis, "fold": str(fold),
                                      "source_row_id": rid, "jid": r["jid"],
                                      "scaf": int(r["scaf"]),
                                      "context": str(r["helix_seq"]),
                                      "model_id": MODEL_ID, "y": r["y"],
                                      "cens": bool(r["cens"]),
                                      "mu": p["mu"], "sigma": p["sigma"],
                                      "abstain": p["abstain"]})
            leaderboard.append({"axis": axis, "fold": str(fold),
                                "model_id": MODEL_ID,
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
        _, by_fold = load_splits(mp)
        eval_split(axis, sorted(by_fold.items()))

    jf = build_joint_edit_context_folds(admitted)
    for f in jf:
        test_rows = [r for sid, r in rows.items() if sid in f["test_ids"]]
        train_rows = [r for sid, r in rows.items() if sid in f["train_ids"]]
        res, preds = run_fold(fit_mutation_graph, predict_mutation_graph,
                              train_rows, test_rows)
        conv_rows.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                          "model_id": MODEL_ID, "deterministic": True,
                          "converged": res.get("converged") if res else None,
                          "error": res.get("error") if res else "fit_error"})
        if res is None or "error" in res:
            leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                "model_id": MODEL_ID,
                                "error": res["error"] if res else "fit_error"})
            continue
        if preds:
            for rid, p in preds.items():
                r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                all_preds.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                  "source_row_id": rid, "jid": r["jid"],
                                  "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                                  "model_id": MODEL_ID, "y": r["y"],
                                  "cens": bool(r["cens"]), "mu": p["mu"],
                                  "sigma": p["sigma"], "abstain": p["abstain"]})
        leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                            "model_id": MODEL_ID,
                            "coverage": res["metric"]["coverage"],
                            "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                            "eligible_full_coverage": res["eligible"],
                            "n_eligible": res["metric"]["n_eligible"],
                            "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                            "converged": res["converged"]})

    # ---- r05_graph artifacts ----
    with (out / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "LeaderboardDraft.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        w.writeheader()
        for r in leaderboard:
            w.writerow(r)
    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger.parquet")

    # ---- merge into unified R1 artifacts ----
    r1 = run_root / "r1"
    r1_leaderboard = r1 / "Leaderboard_v2.csv"
    r1_preds = r1 / "Predictions_v2.jsonl"
    r1_conv = r1 / "ConvergenceLedger.parquet"

    lb = pd.read_csv(r1_leaderboard) if r1_leaderboard.exists() else pd.DataFrame(columns=LEADERBOARD_FIELDS)
    pd.concat([lb, pd.DataFrame(leaderboard)], ignore_index=True).to_csv(r1_leaderboard, index=False)

    if r1_preds.exists():
        recs = [json.loads(l) for l in r1_preds.read_text().splitlines() if l.strip()]
    else:
        recs = []
    recs.extend(all_preds)
    with r1_preds.open("w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if r1_conv.exists():
        old = pd.read_parquet(r1_conv)
        pd.concat([old, pd.DataFrame(conv_rows)], ignore_index=True).to_parquet(r1_conv)
    else:
        pd.DataFrame(conv_rows).to_parquet(r1_conv)

    # duplicate-key check across full merged predictions
    all_merged = [json.loads(l) for l in r1_preds.read_text().splitlines() if l.strip()]
    dups = validate_unique_keys(all_merged)

    n_models = int(pd.read_csv(r1_leaderboard)["model_id"].nunique())
    r1_status_fp = r1 / "STATUS.json"
    if r1_status_fp.exists():
        st = json.loads(r1_status_fp.read_text())
        st["n_models"] = n_models
        st["note"] = (st.get("note", "") + " Added mutation_graph_smoother (R1 supplemental).").strip()
        r1_status_fp.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    status = {
        "phase": "R1_GRAPH", "state": "DONE",
        "model_id": MODEL_ID,
        "n_predictions": len(all_preds),
        "n_leaderboard_rows": len(leaderboard),
        "n_r1_models_now": n_models,
        "duplicate_primary_keys": len(dups),
        "axes": axes + ["edit_x_nested_context"],
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    # concise per-axis summary
    conv = pd.DataFrame(conv_rows)
    if not conv.empty:
        print(conv.groupby(["axis", "model_id"])["converged"].agg(["count", "sum"]).to_string())
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
