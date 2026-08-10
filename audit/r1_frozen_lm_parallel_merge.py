"""Merge parallel frozen-LM fold shards into run-root + R1 artifacts.

Replicates the merge/aggregation block of audit/r1_frozen_lm_run.py exactly.

Args: <cfg.json> <shard_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.evaluation.scorer_v2 import validate_unique_keys

MODEL_ID = "frozen_rnafm_lm"
LEADERBOARD_FIELDS = ["axis", "fold", "model_id", "coverage",
                      "pooled_junction_macro_nll", "eligible_full_coverage",
                      "n_eligible", "n_abstain_no_fallback", "converged", "error"]
HF_MODEL = "multimolecule/rnafm-ss"


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    shard_dir = Path(sys.argv[2])
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_frozenlm"

    lb_files = sorted(shard_dir.glob("worker_*.leaderboard.csv"))
    lb = pd.concat([pd.read_csv(f) for f in lb_files], ignore_index=True)
    conv = pd.concat([pd.read_parquet(f) for f in sorted(shard_dir.glob("worker_*.conv.parquet"))],
                     ignore_index=True)
    all_preds = []
    for f in sorted(shard_dir.glob("worker_*.preds.jsonl")):
        all_preds.extend(json.loads(l) for l in f.read_text().splitlines() if l.strip())

    # normalize leaderboard columns to LEADERBOARD_FIELDS
    for col in LEADERBOARD_FIELDS:
        if col not in lb.columns:
            lb[col] = None
    lb = lb[LEADERBOARD_FIELDS]
    leaderboard = lb.to_dict("records")

    # ---- r05_frozenlm artifacts ----
    with (out / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lb.to_csv(out / "LeaderboardDraft.csv", index=False)
    conv.to_parquet(out / "ConvergenceLedger.parquet")

    # ---- merge into unified R1 artifacts ----
    r1 = run_root / "r1"
    r1_leaderboard = r1 / "Leaderboard_v2.csv"
    r1_preds = r1 / "Predictions_v2.jsonl"
    r1_conv = r1 / "ConvergenceLedger.parquet"

    old_lb = pd.read_csv(r1_leaderboard) if r1_leaderboard.exists() else pd.DataFrame(columns=LEADERBOARD_FIELDS)
    pd.concat([old_lb, lb], ignore_index=True).to_csv(r1_leaderboard, index=False)

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
        pd.concat([old, conv], ignore_index=True).to_parquet(r1_conv)
    else:
        conv.to_parquet(r1_conv)

    all_merged = [json.loads(l) for l in r1_preds.read_text().splitlines() if l.strip()]
    dups = validate_unique_keys(all_merged)
    n_models = int(pd.read_csv(r1_leaderboard)["model_id"].nunique())

    r1_status_fp = r1 / "STATUS.json"
    if r1_status_fp.exists():
        st = json.loads(r1_status_fp.read_text())
        st["n_models"] = n_models
        st["note"] = (st.get("note", "") + " Added frozen_rnafm_lm (R1 supplemental).").strip()
        r1_status_fp.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    status = {
        "phase": "R1_FROZEN_LM", "state": "DONE", "model_id": MODEL_ID,
        "foundation_model": HF_MODEL,
        "pretraining_exposure": "RNA-FM: 23M+ ncRNA sequences, MLM (Chen et al. 2022, arXiv:2204.00300)",
        "frozen_weights": True,
        "head": "single linear head, same censored objective + L-BFGS-B (maxiter=2000, gtol=1e-8, ridge=1.0)",
        "n_predictions": len(all_preds),
        "n_leaderboard_rows": len(leaderboard),
        "n_r1_models_now": n_models,
        "duplicate_primary_keys": len(dups),
        "axes": cfg["axes"] + ["edit_x_nested_context"],
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if not conv.empty:
        print(conv.groupby(["axis", "model_id"])["success"].agg(["count", "sum"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
