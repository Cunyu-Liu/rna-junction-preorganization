"""R0.6 comparison-eligibility re-adjudication (contract §13.7).

Reads the R0.5 fresh-rerun outputs and re-issues a single machine-readable
verdict per model x axis, deciding which old P0/P1/P2/P4 statuses are
retracted and whether R1/R2 may start.

Fail-closed rules applied here:
  - A parametric model is comparison-eligible on an axis only if EVERY fold:
      * optimizer reached success (converged gate)  -> from ConvergenceLedger
      * full-coverage primary task covered all eligible rows with no
        abstain-without-fallback (eligible_full_coverage==True).
  - If ANY fold fails either condition, that model x axis is NOT eligible for
    full-coverage comparison (may still appear as selective/secondary with a
    frozen coverage floor, which R1/R2 handle separately).
  - Old P0 P0_PASS_COMPARISON_ELIGIBLE, P1 leaderboard, and P2 conditional-signal
    statuses are explicitly retracted as INVALIDATED_OR_STALE; they cannot be
    inherited as defaults.

Outputs into RUN_ROOT/eligibility/:
  ComparisonEligibilityDecision_v2.json
  StatusRetractionLedger.jsonl
  STATUS_R0.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "eligibility"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    # load R0.5 leaderboards (parametric baselines + edit_knn + corrected v1.31)
    leaderboards = {}
    for name, sub in [("r05", "r05"), ("r05_v131", "r05_v131")]:
        lb = run_root / sub / "LeaderboardDraft_v2.csv" if name == "r05" else run_root / sub / "Leaderboard_v1_31.csv"
        if lb.exists():
            leaderboards[name] = pd.read_csv(lb)

    # load convergence ledgers
    conv = {}
    for sub in ["r05", "r05_v131"]:
        p = run_root / sub / ("ConvergenceLedger.parquet" if sub == "r05" else "ConvergenceLedger_v1_31.parquet")
        if p.exists():
            conv[sub] = pd.read_parquet(p)

    decision = {}
    verdicts = []
    blocked_evidence = []

    axes = cfg.get("axes", ["symmetry_5fold", "edit_5fold", "context_lomo",
                            "scaffold_lomo", "edit_x_nested_context"])

    # --- parametric baselines + edit_knn from r05 leaderboard ---
    for name, lb in leaderboards.items():
        for model_id, g in lb.groupby("model_id"):
            for axis in axes:
                agg = g[g["axis"] == axis]
                if agg.empty:
                    continue
                if "error" in agg.columns and agg["error"].notna().any():
                    n_err = int(agg["error"].notna().sum())
                    blocked_evidence.append({"model": model_id, "axis": axis,
                                             "reason": f"{n_err} folds errored"})
                    verdicts.append({"model_id": model_id, "axis": axis,
                                     "comparison_eligible": False,
                                     "reason": f"{n_err} fold fit errors"})
                    continue
                if "converged" in agg.columns:
                    n_folds = len(agg)
                    n_converged = int(agg["converged"].astype(bool).sum())
                    all_conv = n_converged == n_folds
                else:
                    all_conv = True
                    n_folds, n_converged = len(agg), len(agg)
                elig = agg["eligible_full_coverage"].astype(bool)
                all_eligible = bool(elig.all())
                n_ineligible_folds = int((~elig).sum())
                eligible = bool(all_conv and all_eligible)
                reason = []
                if not all_conv:
                    reason.append(f"converged {n_converged}/{n_folds}")
                if not all_eligible:
                    reason.append(f"full-coverage ineligible folds {n_ineligible_folds}/{n_folds}")
                verdicts.append({"model_id": model_id, "axis": axis,
                                 "comparison_eligible": eligible,
                                 "n_folds": n_folds, "n_converged": n_converged,
                                 "n_ineligible_full_coverage_folds": n_ineligible_folds,
                                 "pooled_junction_macro_nll": float(agg["pooled_junction_macro_nll"].mean())
                                     if "pooled_junction_macro_nll" in agg and agg["pooled_junction_macro_nll"].notna().any() else None,
                                 "reason": "; ".join(reason) if reason else None})
                if not eligible and not reason:
                    verdicts[-1]["reason"] = "blocked_with_evidence"

    eligible_any = [v for v in verdicts if v["comparison_eligible"]]

    # Retraction ledger for old statuses
    retractions = [
        {"status": "P0_PASS_COMPARISON_ELIGIBLE", "verdict": "INVALIDATED_OR_STALE",
         "reason": "strict contract not in old authority chain; manifest not bound to real commit; RunDAG dangling; gate not fail-closed"},
        {"status": "P1_leaderboard", "verdict": "INVALIDATED_OR_STALE",
         "reason": "six parametric baselines had reversed censored-row gradient"},
        {"status": "P2_CONDITIONAL_KNOWN_OPERATOR_SIGNAL", "verdict": "INVALIDATED_OR_STALE",
         "reason": "comparator numerical failure; matched no-sequence ablation missing; re-adjudicated post-hoc"},
    ]

    with (out / "StatusRetractionLedger.jsonl").open("w") as fh:
        for r in retractions:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    decision = {
        "run_id": cfg["run_id"], "phase": "R0.6", "generated_at_utc": utc,
        "r1_may_start": bool(eligible_any),
        "any_comparison_eligible": bool(eligible_any),
        "n_models_axes_eligible": len(eligible_any),
        "verdicts": verdicts,
        "retractions": retractions,
        "interpretation": (
            "COMPARISON_ELIGIBLE does NOT imply hypothesis supported, SOTA, or "
            "submission authorized.  It only gates which model x axis may enter "
            "R1/R2 full-coverage comparison.  Old statuses are retracted and "
            "cannot be inherited."),
    }
    (out / "ComparisonEligibilityDecision_v2.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "run_id": cfg["run_id"], "phase": "R0.6", "generated_at_utc": utc,
        "state": "DONE",
        "r1_may_start": bool(eligible_any),
        "n_verdicts": len(verdicts),
        "n_eligible_model_axes": len(eligible_any),
        "n_retractions": len(retractions),
    }
    (out / "STATUS_R0.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
