"""Aggregate Denny-train-only and physical-prior worker shards into R1.

Reads shard patterns {model_id}_worker_*.{preds.jsonl,leaderboard.csv,conv.parquet}
from a shard dir and merges them into RUN_ROOT/r05_prior/ and the unified
RUN_ROOT/r1/ Leaderboard_v2.csv / Predictions_v2.jsonl / ConvergenceLedger.parquet.

Patched: predictions are deduplicated by (axis, fold, model_id, source_row_id)
keeping the LAST occurrence, so corrected (re-run) model predictions replace the
previous stale ones instead of being appended as duplicates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEADERBOARD_FIELDS = ["axis", "fold", "model_id", "coverage",
                      "pooled_junction_macro_nll", "eligible_full_coverage",
                      "n_eligible", "n_abstain_no_fallback", "converged", "error"]


def _load_preds(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    shard_dir = Path(sys.argv[2])
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_prior"
    out.mkdir(parents=True, exist_ok=True)

    for model_id in ["denny_train_only", "physical_ensemble_prior"]:
        lb_files = sorted(shard_dir.glob(f"{model_id}_worker_*.leaderboard.csv"))
        conv_files = sorted(shard_dir.glob(f"{model_id}_worker_*.conv.parquet"))
        if not lb_files:
            print(json.dumps({"model": model_id, "skipped": True}))
            continue
        lb = pd.concat([pd.read_csv(f) for f in lb_files], ignore_index=True)
        conv = pd.concat([pd.read_parquet(f) for f in conv_files], ignore_index=True)
        preds = []
        for f in sorted(shard_dir.glob(f"{model_id}_worker_*.preds.jsonl")):
            preds.extend(json.loads(l) for l in f.read_text().splitlines() if l.strip())
        with (out / f"Predictions_{model_id}.jsonl").open("w") as fh:
            for rec in preds:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lb.to_csv(out / f"Leaderboard_{model_id}.csv", index=False)
        conv.to_parquet(out / f"ConvergenceLedger_{model_id}.parquet")
        print(json.dumps({"model": model_id, "n_preds": len(preds),
                          "n_lb": len(lb), "n_conv": len(conv)}))

    # ---- merge into unified R1 ----
    r1 = run_root / "r1"
    r1_leaderboard = r1 / "Leaderboard_v2.csv"
    r1_preds = r1 / "Predictions_v2.jsonl"
    r1_conv = r1 / "ConvergenceLedger.parquet"

    new_lb_parts = []
    for m in ["denny_train_only", "physical_ensemble_prior"]:
        p = out / f"Leaderboard_{m}.csv"
        if p.exists():
            new_lb_parts.append(pd.read_csv(p))
    new_lb = pd.concat(new_lb_parts, ignore_index=True)
    old_lb = pd.read_csv(r1_leaderboard) if r1_leaderboard.exists() else pd.DataFrame(columns=LEADERBOARD_FIELDS)
    merged_lb = pd.concat([old_lb, new_lb], ignore_index=True)
    merged_lb = merged_lb.drop_duplicates(subset=["axis", "fold", "model_id"], keep="last")
    merged_lb.to_csv(r1_leaderboard, index=False)

    new_preds = []
    for m in ["denny_train_only", "physical_ensemble_prior"]:
        p = out / f"Predictions_{m}.jsonl"
        if p.exists():
            new_preds.extend(_load_preds(p))
    old_preds = _load_preds(r1_preds)
    all_preds = old_preds + new_preds
    by_key = {}
    for rec in all_preds:
        key = (rec["axis"], str(rec["fold"]), rec["model_id"], str(rec["source_row_id"]))
        by_key[key] = rec  # keep last => corrected overrides stale
    merged_preds = [by_key[k] for k in by_key]
    with r1_preds.open("w") as fh:
        for rec in merged_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    new_conv_parts = []
    for m in ["denny_train_only", "physical_ensemble_prior"]:
        p = out / f"ConvergenceLedger_{m}.parquet"
        if p.exists():
            new_conv_parts.append(pd.read_parquet(p))
    new_conv = pd.concat(new_conv_parts, ignore_index=True)
    old_conv = pd.read_parquet(r1_conv) if r1_conv.exists() else pd.DataFrame()
    merged_conv = pd.concat([old_conv, new_conv], ignore_index=True)
    merged_conv = merged_conv.drop_duplicates(subset=["axis", "fold", "model_id"], keep="last")
    merged_conv.to_parquet(r1_conv, index=False)
    print(json.dumps({"r1_leaderboard_rows": len(merged_lb),
                      "r1_preds": len(merged_preds),
                      "r1_conv": len(merged_conv)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
