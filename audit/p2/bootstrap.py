"""Phase 2 split-unit group bootstrap (contract Phase 2).

The outer fold is the independent split unit.  We bootstrap the per-fold gain
values (gain_fold = NLL_baseline - NLL_candidate) with replacement n_boot times
and report the 95% bootstrap CI on the mean gain.  Because the number of outer
units is small for the 5-fold axes (and the per-fold gain already aggregates
many junctions), we also report the junction-level (split-unit = junction)
bootstrap for the two grouped axes as a secondary, more granular interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.p2.common import load_fold_metrics, observed_fold_gains


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap the mean of `values` by resampling with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": n, "n_boot": n_boot, "p_positive": float(np.mean(means > 0))}


def run_fold_bootstrap(fold_metrics, axes_spec, n_boot=2000, seed=0):
    """Return per-axis fold-unit bootstrap intervals + observed per-fold gains."""
    out = {}
    for axis, n_folds in axes_spec:
        per = observed_fold_gains(fold_metrics, axis, n_folds)
        out[axis] = {
            "n_folds": len(per),
            "per_fold_gain": {str(k): float(v) for k, v in sorted(per.items())},
            "n_folds_positive": int(sum(v > 0 for v in per.values())),
            "all_folds_positive": bool(per and all(v > 0 for v in per.values())),
            "observed_mean_gain": float(np.mean(list(per.values()))) if per else None,
            "fold_unit_bootstrap": bootstrap_ci(list(per.values()), n_boot, seed),
        }
    return out


def write_bootstrap(out_dir, fold_boot, strat_boot=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BootstrapIntervals.json").write_text(
        json.dumps({"fold_unit": fold_boot, "stratified": strat_boot or {}},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return out_dir / "BootstrapIntervals.json"


def write_bootstrap_csv(out_dir, fold_boot):
    rows = []
    for axis, info in fold_boot.items():
        b = info["fold_unit_bootstrap"]
        rows.append({"axis": axis, "n_folds": info["n_folds"],
                     "n_folds_positive": info["n_folds_positive"],
                     "all_folds_positive": info["all_folds_positive"],
                     "observed_mean_gain": info["observed_mean_gain"],
                     "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                     "p_positive": b["p_positive"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "BootstrapIntervals.csv", index=False)
    return out_dir / "BootstrapIntervals.csv"
