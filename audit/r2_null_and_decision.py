"""R2 final CoreHypothesisDecision_v3 (contract §12.3 / FrozenGateSpec).

Merges three complementary, consistent pieces of evidence per axis:
  1. junction-macro matched ablation (multiway_cluster.axis_statistic),
  2. junction x context / junction-cluster group uncertainty
     (multiway_cluster bootstrap CI lower bound > 0),
  3. junction-level pairing null (same statistic as genuine; 1000 per axis),
plus fold-level positivity from the frozen leaderboards (5/5 for 5-fold axes).

All three must agree for SEQUENCE_INCREMENT_SUPPORTED (fail-closed):
    - relative_gain >= MIN_GAIN (0.10)
    - group-bootstrap 95% CI lower bound > 0
    - null one-sided p < ALPHA (0.01) AND null 97.5% upper bound < genuine
    - fold positivity: every eligible fold has full > no_sequence
Otherwise NOT_SUPPORTED_OR_INCONCLUSIVE (no claim of sequence increment).

Outputs into RUN_ROOT/r2/:
  NullStatistics.parquet       (junction-level, from multiway_cluster)
  MatchedAblationFoldData.csv  (per-fold full/no-sequence NLL)
  CoreHypothesisDecision_v3.json
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.statistics import multiway_cluster as mw

SEED = 17
N_NULL = 1000
MIN_GAIN = 0.10
ALPHA = 0.01

AXES = ["symmetry_5fold", "edit_5fold", "context_lomo",
        "scaffold_lomo", "edit_x_nested_context"]


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fold_positivity(run_root, axis):
    """Per-fold pooled_junction_macro NLL positivity: full better than
    no_sequence on every eligible fold."""
    full = pd.read_csv(run_root / "r05_v131" / "Leaderboard_v1_31.csv")
    ns = pd.read_csv(run_root / "r1" / "Leaderboard_v2.csv")
    ns = ns[ns["model_id"] == "no_sequence_latent_operator"]
    f = full[full["axis"] == axis]
    n = ns[ns["axis"] == axis]
    if f.empty or n.empty:
        return None
    m = f.merge(n, on="fold", suffixes=("_full", "_ns"))
    elig = (m["eligible_full_coverage_full"].astype(bool) &
            m["eligible_full_coverage_ns"].astype(bool) &
            m["pooled_junction_macro_nll_full"].notna() &
            m["pooled_junction_macro_nll_ns"].notna())
    m = m[elig]
    if len(m) == 0:
        return None
    pos = int((m["pooled_junction_macro_nll_ns"] >
               m["pooled_junction_macro_nll_full"]).sum())
    return {"n_folds": int(len(m)), "positive_folds": pos,
            "all_positive": bool(pos == len(m))}


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r2"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    axes = cfg.get("axes", AXES)
    mw_results = mw.run(run_root, axes, out)

    axis_results = []
    null_rows = []
    fold_rows = []
    for a in mw_results["axes"]:
        axis = a["axis"]
        if not a.get("available"):
            axis_results.append({"axis": axis, "available": False,
                                 "reason": a.get("reason")})
            continue
        fp = _fold_positivity(run_root, axis)
        # persist null samples for this axis (junction-level)
        rows = mw.load_axis_rows(run_root, axis)
        nulls = mw.junction_pairing_null(rows)
        for idx, v in enumerate(nulls):
            null_rows.append({"axis": axis, "null_index": idx,
                              "theta": float(v)})
        theta = a["theta"]
        lo = a["junction_boot_ci"][0]
        tw_lower = (a.get("two_way_ci") or [None, None])[0]
        supported = bool(
            a["relative_gain"] is not None
            and a["relative_gain"] >= MIN_GAIN
            and a["junction_boot_lower_gt_0"]
            and a["null_p_value"] < ALPHA
            and a["null_975_upper_lt_genuine"]
            and (fp is not None and fp["all_positive"])
        )
        axis_results.append({
            "axis": axis, "available": True,
            "n_junctions": a["n_junctions"],
            "n_contexts": a["n_contexts"],
            "n_rows": a["n_rows"],
            "junction_macro_theta": theta,
            "relative_gain": a["relative_gain"],
            "junction_boot_95ci": a["junction_boot_ci"],
            "junction_boot_lower_gt_0": a["junction_boot_lower_gt_0"],
            "two_way_ci": a.get("two_way_ci"),
            "two_way_lower_gt_0": a.get("two_way_lower_gt_0"),
            "null_p_value": a["null_p_value"],
            "null_975_upper": a["null_975_upper"],
            "null_975_upper_lt_genuine": a["null_975_upper_lt_genuine"],
            "fold_positivity": fp,
            "verdict": "SEQUENCE_INCREMENT_SUPPORTED" if supported
                       else "NOT_SUPPORTED_OR_INCONCLUSIVE",
        })

    any_supported = any(a.get("verdict") == "SEQUENCE_INCREMENT_SUPPORTED"
                        for a in axis_results)

    pd.DataFrame(null_rows).to_parquet(out / "NullStatistics.parquet", index=False)

    decision = {
        "run_id": cfg["run_id"], "phase": "R2", "generated_at_utc": utc,
        "contrast": "corrected_v1_31 (full) vs no_sequence_latent_operator (matched)",
        "statistic": "junction-macro mean of (nll_no_sequence - nll_full); "
                     "positive => full model better",
        "uncertainty": ("junction-cluster percentile bootstrap 95% CI "
                        "(junction x context two-way for context_lomo)"),
        "null": f"junction-level pairing sign-flip, {N_NULL}/axis, seed {SEED}",
        "gates": {"min_relative_gain": MIN_GAIN, "alpha": ALPHA,
                  "require_boot_ci_lower_gt_0": True,
                  "require_all_folds_positive": True,
                  "require_null_975_upper_lt_genuine": True},
        "note": ("Fail-closed; no gate threshold moved post-hoc.  All gates must "
                 "hold simultaneously for SUPPORTED."),
        "axes": axis_results,
        "any_sequence_increment_supported": bool(any_supported),
    }
    (out / "CoreHypothesisDecision_v3.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "R2", "state": "CORE_HYPOTHESIS_DECIDED", "generated_at_utc": utc,
        "n_axes_available": sum(1 for a in axis_results if a.get("available")),
        "n_axes_supported": sum(1 for a in axis_results
                                if a.get("verdict") == "SEQUENCE_INCREMENT_SUPPORTED"),
        "null_rows": len(null_rows),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
