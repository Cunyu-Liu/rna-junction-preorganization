"""R2 pairing-null and CoreHypothesisDecision_v3 (contract §12.3 / §13.8).

The genuine axis-level statistic is the mean relative gain of the full model
(corrected v1.31) over its MATCHED no-sequence latent operator on the same
frozen folds:
    relative_gain = (mean_nll_no_sequence - mean_nll_full) / mean_nll_no_sequence
A positive gain means the sequence map adds predictive increment over the
matched scaffold-bundle latent operator.

Under the null that full and no-sequence are exchangeable, each per-fold
difference  d_i = nll_ns_i - nll_full_i  is symmetric about 0.  We build the
null distribution with a sign-flip pairing permutation: randomly negate a
subset of folds and recompute the axis-level relative gain, repeated
``N_NULL=1000`` times per axis.  This mirrors the genuine statistic definition
exactly (same axis-level aggregation, same folds), satisfying contract R0.4's
"null per row = one full axis-level statistic" requirement.

Decision (fail-closed, no post-hoc threshold moves):
  SEQUENCE_INCREMENT_SUPPORTED only if ALL of:
    - eligible folds > 0 and relative_gain >= MIN_GAIN (0.10)
    - fold consistency: full beats no-sequence on EVERY eligible fold
    - one-sided null p-value < ALPHA (0.01)
  otherwise  NOT_SUPPORTED_OR_INCONCLUSIVE  (no claim of sequence increment).

Outputs into RUN_ROOT/r2/:
  NullStatistics.parquet       axis x null_index x relative_gain
  MatchedAblationFoldData.csv  per fold full/no-sequence NLL for eligibility audit
  CoreHypothesisDecision_v3.json  merged matched-ablation + null verdict
  STATUS.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 17
N_NULL = 1000
MIN_GAIN = 0.10
ALPHA = 0.01

AXES = ["symmetry_5fold", "edit_5fold", "context_lomo",
        "scaffold_lomo", "edit_x_nested_context"]


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pair_rows(run_root: Path, axis):
    full = pd.read_csv(run_root / "r05_v131" / "Leaderboard_v1_31.csv")
    ns = pd.read_csv(run_root / "r1" / "Leaderboard_v2.csv")
    ns = ns[ns["model_id"] == "no_sequence_latent_operator"]
    f = full[full["axis"] == axis].copy()
    n = ns[ns["axis"] == axis].copy()
    if f.empty or n.empty:
        return None
    merged = f.merge(n, on="fold", suffixes=("_full", "_ns"))
    elig = (merged["eligible_full_coverage_full"].astype(bool) &
            merged["eligible_full_coverage_ns"].astype(bool) &
            merged["pooled_junction_macro_nll_full"].notna() &
            merged["pooled_junction_macro_nll_ns"].notna())
    return merged[elig][["fold", "pooled_junction_macro_nll_full",
                         "pooled_junction_macro_nll_ns"]].reset_index(drop=True)


def axis_stat(m):
    nll_full = float(m["pooled_junction_macro_nll_full"].mean())
    nll_ns = float(m["pooled_junction_macro_nll_ns"].mean())
    rel = (nll_ns - nll_full) / nll_ns if nll_ns else None
    return nll_full, nll_ns, rel


def pairing_null(m, n_null=N_NULL, seed=SEED):
    rng = np.random.default_rng(seed)
    d = (m["pooled_junction_macro_nll_ns"] -
         m["pooled_junction_macro_nll_full"]).to_numpy()
    nll_ns_mean = float(m["pooled_junction_macro_nll_ns"].mean())
    if nll_ns_mean == 0 or len(d) == 0:
        return np.array([np.nan] * n_null)
    nulls = np.empty(n_null)
    for k in range(n_null):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        nulls[k] = float((d * signs).mean() / nll_ns_mean)
    return nulls


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r2"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    axes = cfg.get("axes", AXES)
    null_rows = []
    fold_rows = []
    axis_results = []

    for axis in axes:
        m = load_pair_rows(run_root, axis)
        if m is None or len(m) == 0:
            axis_results.append({"axis": axis, "available": False,
                                 "reason": "no eligible full-coverage folds"})
            continue
        nll_full, nll_ns, rel = axis_stat(m)
        pos_folds = int((m["pooled_junction_macro_nll_ns"] >
                         m["pooled_junction_macro_nll_full"]).sum())
        n_folds = int(len(m))
        fold_ok = (pos_folds == n_folds) and n_folds > 0
        nulls = pairing_null(m)
        p_value = float((np.nansum(nulls >= rel) + 1) / (N_NULL + 1))
        supported = bool(rel is not None and rel >= MIN_GAIN and fold_ok and p_value < ALPHA)
        for idx, v in enumerate(nulls):
            null_rows.append({"axis": axis, "null_index": idx,
                              "relative_gain": float(v)})
        for _, r in m.iterrows():
            fold_rows.append({"axis": axis, "fold": r["fold"],
                              "nll_full": float(r["pooled_junction_macro_nll_full"]),
                              "nll_no_sequence": float(r["pooled_junction_macro_nll_ns"])})
        axis_results.append({
            "axis": axis, "available": True, "n_eligible_folds": n_folds,
            "pooled_junction_macro_nll_full": nll_full,
            "pooled_junction_macro_nll_no_sequence": nll_ns,
            "relative_gain": rel,
            "positive_folds": f"{pos_folds}/{n_folds}",
            "fold_all_positive": bool(fold_ok),
            "null_p_value": p_value,
            "n_null": N_NULL,
            "verdict": "SEQUENCE_INCREMENT_SUPPORTED" if supported
                       else "NOT_SUPPORTED_OR_INCONCLUSIVE",
        })

    any_supported = any(a.get("verdict") == "SEQUENCE_INCREMENT_SUPPORTED"
                        for a in axis_results)

    pd.DataFrame(null_rows).to_parquet(out / "NullStatistics.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "MatchedAblationFoldData.csv", index=False)

    decision = {
        "run_id": cfg["run_id"], "phase": "R2", "generated_at_utc": utc,
        "contrast": "corrected_v1_31 (full) vs no_sequence_latent_operator (matched)",
        "statistic": "axis-level mean relative_gain = (mean_nll_ns - mean_nll_full)/mean_nll_ns",
        "null": f"{N_NULL}-fold sign-flip pairing permutation per axis, "
                f"same aggregation as genuine; seed {SEED}",
        "gates": {"min_relative_gain": MIN_GAIN, "alpha": ALPHA,
                  "require_all_folds_positive": True},
        "note": ("No gate threshold is moved post-hoc.  SUPPORTED requires "
                 "gain >= 0.10 AND all folds positive AND null p < 0.01. "
                 "Multiway cluster uncertainty is a separate R2 module that "
                 "feeds the same decision object before final closure."),
        "axes": axis_results,
        "any_sequence_increment_supported": bool(any_supported),
    }
    (out / "CoreHypothesisDecision_v3.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    status = {
        "phase": "R2", "state": "NULL_AND_DECISION_DONE", "generated_at_utc": utc,
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
