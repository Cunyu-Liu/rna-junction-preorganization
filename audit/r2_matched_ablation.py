"""R2 matched no-sequence ablation contrast (contract §12.3 / §9.2).

The core scientific contrast is:
    corrected v1.31 (full: sequence map X_j @ theta)
    vs
    no_sequence latent operator (matched: shared intercept, same scaffold
    machinery / censored likelihood / GH / budget).

Both are scored on the SAME frozen folds with the SAME support-aware
full-coverage scorer, so any difference is attributable to the sequence map.
This module reads the R1 unified leaderboard (which merges fresh R0-lineage
predictions) and emits the per-axis matched contrast + gate application.

Gain is defined relative to the matched no-sequence baseline (lower NLL better):
    relative_gain = (nll_no_sequence - nll_full) / nll_no_sequence
A positive gain means the full model beats its matched no-sequence comparator.

This is PART of R2 (matched ablation + gate application).  The 1000-axis-level
pairing null, joint holdout reporting and multiway cluster uncertainty are
separate modules that feed the same CoreHypothesisDecision_v3.json.

Outputs into RUN_ROOT/r2/:
  MatchedAblationContrast.json   per axis full/no-sequence NLL + relative gain
  GateApplication.json           FrozenGateSpec superiority/CI/fold checks
  CoreHypothesisDecision_v3.json preliminary verdict (null/multiway to be merged)
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_leaderboard(run_root: Path, sub, name):
    p = run_root / sub / name
    return pd.read_csv(p) if p.exists() else None


def _pooled_by_axis(lb, model_ids, axis):
    """Mean of pooled_junction_macro_nll over eligible full-coverage folds."""
    sub = lb[(lb["axis"] == axis) & (lb["model_id"].isin(model_ids))]
    elig = sub[sub["eligible_full_coverage"].astype(bool) &
               sub["pooled_junction_macro_nll"].notna()]
    return elig.groupby("model_id")["pooled_junction_macro_nll"].mean().to_dict()


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r2"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    full_lb = load_leaderboard(run_root, "r05_v131", "Leaderboard_v1_31.csv")
    ns_lb = load_leaderboard(run_root, "r1", "Leaderboard_v2.csv")
    if full_lb is None or ns_lb is None:
        status = {"phase": "R2", "state": "BLOCKED_PENDING_R1",
                  "reason": "need r05_v131 + r1 leaderboards"}
        (out / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return status

    # no-sequence rows only (Leaderboard_v2 also contains the r05 baselines)
    ns_rows = ns_lb[ns_lb["model_id"] == "no_sequence_latent_operator"]
    full_rows = full_lb[full_lb["model_id"] == "corrected_v1_31"]

    axes = cfg.get("axes", ["symmetry_5fold", "edit_5fold", "context_lomo",
                            "scaffold_lomo", "edit_x_nested_context"])
    contrasts = []
    gate_rows = []
    for axis in axes:
        f = full_rows[full_rows["axis"] == axis]
        n = ns_rows[ns_rows["axis"] == axis]
        if f.empty or n.empty:
            contrasts.append({"axis": axis, "available": False})
            continue
        # join on fold
        merged = f.merge(n, on="fold", suffixes=("_full", "_ns"))
        eligible = merged["eligible_full_coverage_full"].astype(bool) & \
                   merged["eligible_full_coverage_ns"].astype(bool) & \
                   merged["pooled_junction_macro_nll_full"].notna() & \
                   merged["pooled_junction_macro_nll_ns"].notna()
        m = merged[eligible]
        n_eligible_folds = int(len(m))
        if n_eligible_folds == 0:
            contrasts.append({"axis": axis, "available": True,
                              "n_eligible_folds": 0,
                              "verdict": "NOT_SCORABLE_NO_FULL_COVERAGE"})
            continue
        nll_full = float(m["pooled_junction_macro_nll_full"].mean())
        nll_ns = float(m["pooled_junction_macro_nll_ns"].mean())
        rel_gain = (nll_ns - nll_full) / nll_ns if nll_ns else None
        pos_folds = int((m["pooled_junction_macro_nll_ns"] >
                         m["pooled_junction_macro_nll_full"]).sum())
        n_folds = int(len(m))
        fold_ok = (pos_folds == n_folds) if n_folds > 0 else False
        superiority_ok = (rel_gain is not None and rel_gain >= 0.10)
        contrasts.append({
            "axis": axis, "available": True, "n_eligible_folds": n_eligible_folds,
            "pooled_junction_macro_nll_full": nll_full,
            "pooled_junction_macro_nll_no_sequence": nll_ns,
            "relative_gain": rel_gain,
            "positive_folds": f"{pos_folds}/{n_folds}",
            "gate_superiority_ge10pct": superiority_ok,
            "gate_fold_consistency_all_positive": bool(fold_ok),
            "verdict": ("SEQUENCE_INCREMENT_POSSIBLE" if superiority_ok else
                        "NOT_SUPPORTED_OR_INCONCLUSIVE"),
        })
        gate_rows.append({"axis": axis, "n_eligible_folds": n_eligible_folds,
                          "relative_gain": rel_gain,
                          "superiority_ge10pct": superiority_ok,
                          "fold_all_positive": bool(fold_ok)})

    decision = {
        "run_id": cfg["run_id"], "phase": "R2", "generated_at_utc": utc,
        "contrast": "corrected_v1_31 (full) vs no_sequence_latent_operator (matched)",
        "note": ("Matched ablation only.  Null separation, multiway cluster "
                 "uncertainty and interval/calibration gates are merged by "
                 "separate R2 modules before a final CoreHypothesisDecision_v3 "
                 "is frozen.  No gate threshold is moved post-hoc."),
        "axes": contrasts,
    }
    (out / "MatchedAblationContrast.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    pd.DataFrame(gate_rows).to_csv(out / "GateApplication.csv", index=False)

    status = {"phase": "R2", "state": "MATCHED_ABLATION_DONE",
              "generated_at_utc": utc,
              "n_axes_contrasted": sum(1 for c in contrasts if c.get("available"))}
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
