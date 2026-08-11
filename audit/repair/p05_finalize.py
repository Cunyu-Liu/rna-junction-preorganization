"""P0.5 finalize-only step (strict audit 2026-08-11).

The 37-fold joint rerun already wrote the row-level streams to disk
(Predictions_v3.jsonl, FoldSpec.jsonl, Leaderboard_full_coverage.csv,
ConvergenceLedger_v3.parquet).  The final aggregate step crashed on a numpy
generator bug (fixed).  This script recomputes ONLY the aggregate outputs
(GroupAwareGenuine.json, STATUS.json) from the existing on-disk artifacts,
so the expensive 37-fold fit is NOT re-run.

Usage: python p05_finalize.py <run_root>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from audit.data.audit_dataset import audit_dataset
from audit.repair import p05_run


def main(cfg):
    t0 = time.time()
    out = Path(cfg["run_root"]) / "r05_repair"
    out.mkdir(parents=True, exist_ok=True)

    _, admitted, profile, *_ = audit_dataset(Path(cfg["canonical_source"]))

    all_preds = []
    with (out / "Predictions_v3.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_preds.append(json.loads(line))

    df_lb = pd.read_csv(out / "Leaderboard_full_coverage.csv")
    conv = pd.read_parquet(out / "ConvergenceLedger_v3.parquet")

    dups = p05_run.validate_unique_keys([{**p} for p in all_preds])

    gen = p05_run._pooled_genuine(all_preds)
    cluster = p05_run._edit_cluster_ci(all_preds, admitted)
    n_eligible_folds = 0
    if len(df_lb):
        n_eligible_folds = int(df_lb[df_lb["eligible_full_coverage"] == True]["fold"].nunique())  # noqa: E712
    gen_report = {
        "axis": "edit_x_nested_context",
        "contrast": "corrected_v1_31 vs no_sequence_latent_operator",
        "statistic": "pooled-OOF junction-macro NLL delta (no_sequence - full)",
        "positive_means_full_better": True,
        "n_eligible_folds": n_eligible_folds,
        "genuine": gen,
        "edit_cluster": cluster,
    }
    (out / "GroupAwareGenuine.json").write_text(
        json.dumps(gen_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    model_ids = cfg.get("models") or []
    n_folds = int(df_lb["fold"].nunique()) if len(df_lb) else 0
    status = {
        "phase": "P0.5", "state": "DONE",
        "n_models": len(model_ids), "n_folds": n_folds,
        "n_predictions": len(all_preds),
        "n_leaderboard_rows": int(len(df_lb)),
        "n_convergence_rows": int(len(conv)),
        "duplicate_primary_keys": len(dups),
        "models": model_ids,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Finalize-only recompute of aggregate outputs from the on-disk "
                 "row-level streams; the 37-fold joint fit itself was not "
                 "re-run."),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    main(cfg)