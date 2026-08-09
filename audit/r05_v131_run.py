"""R0.5 corrected v1.31 fresh rerun (contract §13.6).

Because the old v1.31 lineage/convergence ledger is not closed, corrected v1.31
is freshly re-fit on every outer fold of the four single-axis splits AND the
decisive edit_x_nested_context joint split, under the same eligible rows and
the R0.2 support-aware full-coverage scorer.  No old P2 predictions are reused.

Outputs into RUN_ROOT/r05_v131/:
  Predictions_v1_31.jsonl         row-level predictions (unique keys)
  Leaderboard_v1_31.csv           per axis x fold coverage + pooled NLL
  ConvergenceLedger_v1_31.parquet per model x fold optimizer/success gate
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit.benchmark.legacy_adapters import LEGACY_MODELS
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.splits.joint_blocked import build_joint_edit_context


def load_splits(manifest_path: Path):
    by_fold = defaultdict(set)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return by_fold


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
    out = run_root / "r05_v131"
    out.mkdir(parents=True, exist_ok=True)
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}
    protocol = Path(cfg["protocol_dir"])

    axes = cfg["axes"]
    fit_fn, pred_fn = LEGACY_MODELS["corrected_v1_31"]
    all_preds = []
    leaderboard = []
    conv_rows = []

    def eval_split(axis, folds, is_joint):
        for fold, test_ids in folds:
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            res, preds, conv = run_fold(fit_fn, pred_fn, train_rows, test_rows)
            conv_rows.append({"axis": axis, "fold": str(fold), "model_id": "corrected_v1_31",
                              **({} if conv is None else conv),
                              "error": res.get("error") if res else "fit_error"})
            if res is None or "error" in res:
                leaderboard.append({"axis": axis, "fold": str(fold),
                                    "model_id": "corrected_v1_31",
                                    "error": res["error"] if res else "fit_error"})
                continue
            if preds:
                for rid, p in preds.items():
                    r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                    all_preds.append({"axis": axis, "fold": str(fold),
                                      "source_row_id": rid, "jid": r["jid"],
                                      "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                                      "model_id": "corrected_v1_31",
                                      "y": r["y"], "cens": bool(r["cens"]),
                                      "mu": p["mu"], "sigma": p["sigma"],
                                      "abstain": p["abstain"], "support": p["support"]})
            leaderboard.append({"axis": axis, "fold": str(fold),
                                "model_id": "corrected_v1_31",
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

    # joint edit_x_nested_context
    rep = build_joint_edit_context(admitted)
    jfolds = []
    for f in rep["folds"]:
        e = f["edit_component"]
        test_ids = {str(r["source_row_id"]) for r in admitted
                    if str(r["edit_component"]) == e}
        test_ctxs = {str(r["helix_seq"]) for r in admitted
                     if str(r["edit_component"]) == e}
        train_ids = {str(r["source_row_id"]) for r in admitted
                     if str(r["edit_component"]) != e
                     and str(r["helix_seq"]) not in test_ctxs}
        jfolds.append((f"e:{e}", test_ids))
    eval_split("edit_x_nested_context", jfolds, True)

    with (out / "Predictions_v1_31.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dups = validate_unique_keys([{**p} for p in all_preds])
    with (out / "Leaderboard_v1_31.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=["axis", "fold", "model_id", "coverage",
                                           "pooled_junction_macro_nll",
                                           "eligible_full_coverage", "n_eligible",
                                           "n_abstain_no_fallback", "converged", "error"])
        w.writeheader()
        for r in leaderboard:
            w.writerow(r)
    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger_v1_31.parquet")

    status = {
        "phase": "R0.5.v131", "state": "DONE",
        "n_predictions": len(all_preds), "n_leaderboard_rows": len(leaderboard),
        "n_convergence_rows": len(conv_rows), "duplicate_primary_keys": len(dups),
        "axes": axes + ["edit_x_nested_context"], "models": ["corrected_v1_31"],
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
