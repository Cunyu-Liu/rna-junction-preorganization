"""P0.4 metric specification + right-censored NLL implementation.

Defines the single grouped-macro right-censored NLL used across the audit
and the MetricSpec (formulas, aggregation order, censor direction, epsilon,
degenerate handling).  This is the metric that all qualified models must be
re-evaluated against in P0.5.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

CAP = -7.1
EPS = 1e-8

METRIC_SPEC = {
    "name": "right_censored_macro_junction_nll",
    "observation_scale": "kcal/mol dg10",
    "censor_direction": "right (Y>=CAP recorded as CAP=-7.1)",
    "row_likelihood": {
        "measured": "-(0.5*log(2*pi) + log(sigma) + 0.5*((y-mu)/sigma)^2)",
        "censored": "log_Phi((mu-CAP)/sigma)   [P(Y>=CAP)]",
    },
    "aggregation": "per-junction mean of row NLL, then macro mean across junctions (equally weighted)",
    "mac_aggregation_note": "legacy used per-junction relative-ratio mean; this spec uses macro mean of per-junction NLL",
    "epsilon": "sigma floored at 0.05; log survival clipped at -50 to avoid -inf",
    "degenerate_handling": "abstention/unsupported rows are scored separately as support strata, not zeroed",
    "strata": ["measured", "censored", "coverage", "support/abstention"],
}


def row_nll(y, cens, mu, sigma):
    """Vectorized right-censored NLL. Returns per-row NLL (positive loss)."""
    y = np.asarray(y, dtype=float)
    cens = np.asarray(cens, dtype=bool)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.05)
    out = np.zeros(len(y), dtype=float)
    measured = ~cens
    if measured.any():
        z = (y[measured] - mu[measured]) / sigma[measured]
        out[measured] = 0.5 * math.log(2.0 * math.pi) + np.log(sigma[measured]) + 0.5 * z * z
    if cens.any():
        a = (mu[cens] - CAP) / sigma[cens]
        out[cens] = -np.clip(log_ndtr(a), -50.0, 50.0)
    return out


def junction_macro_nll(rows, mu, sigma):
    """Equally-weighted macro mean of per-junction mean NLL."""
    losses = row_nll([r["y"] for r in rows], [r["cens"] for r in rows],
                     np.asarray(mu), np.asarray(sigma))
    by = defaultdict(list)
    for r, loss in zip(rows, losses):
        by[str(r["jid"])].append(float(loss))
    if not by:
        return None
    return float(np.mean([np.mean(v) for v in by.values()]))


def strata_nll(rows, mu, sigma):
    losses = row_nll([r["y"] for r in rows], [r["cens"] for r in rows],
                     np.asarray(mu), np.asarray(sigma))
    out = {}
    for name, mask in (("measured", [not r["cens"] for r in rows]),
                       ("censored", [bool(r["cens"]) for r in rows])):
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            out[name] = float(np.mean([losses[i] for i in idx]))
        else:
            out[name] = None
    return out


def coverage_width_ranking(rows, mu, sigma):
    """Interval diagnostics on measured rows: coverage/width/ranking."""
    measured = [r for r in rows if not r["cens"]]
    if not measured:
        return {"n_measured": 0}
    sig = np.asarray([sigma[i] for i, r in enumerate(rows) if not r["cens"]])
    mu_m = np.asarray([mu[i] for i, r in enumerate(rows) if not r["cens"]])
    y_m = np.asarray([r["y"] for r in measured])
    width = 2.0 * 1.96 * sig
    covered = np.abs(y_m - mu_m) <= 1.96 * sig
    # ranking: spearman of predicted vs observed on measured
    rho = float(np.corrcoef(np.argsort(np.argsort(mu_m)),
                            np.argsort(np.argsort(y_m)))[0, 1]) if len(y_m) > 2 else None
    return {
        "n_measured": len(measured),
        "mean_interval_width_kcal": float(np.mean(width)) if len(width) else None,
        "coverage_95": float(np.mean(covered)) if len(covered) else None,
        "rank_correlation_spearman": rho,
    }


def write_metric_spec(out_dir: Path):
    (out_dir / "MetricSpec.json").write_text(json.dumps(METRIC_SPEC, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    import sys
    write_metric_spec(Path(sys.argv[1]))
    print("MetricSpec written")
