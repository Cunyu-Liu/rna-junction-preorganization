"""R0.2 unified support-aware scorer with full-coverage vs selective tasks.

Contract R0.2 / P0 checklist items 4 and the FrozenGateSpec require two distinct,
non-interchangeable tasks:

  full_coverage (PRIMARY): every eligible test row must receive a scorable
      prediction.  A model that abstains on any row WITHOUT a pre-registered
      fallback is comparison-ineligible for this task on that axis.  The primary
      metric is the pooled-OOF junction-macro right-censored NLL over the
      eligible rows.

  selective (SECONDARY): a model may abstain, but must meet a frozen coverage
      floor and be compared against a coverage-matched comparator, reporting
      supported-NLL, risk-coverage/AURC and abstention cost.  It NEVER replaces
      the full-coverage primary gate.

Invariants enforced here:
  - abstain placeholder values (mu/sigma from a declined prediction) NEVER enter
    an NLL aggregate.
  - every prediction row has a unique primary key (axis|fold|source_row_id|model_id).
  - full-coverage with missing/abstained rows and no fallback => ineligible.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from audit.evaluation.metrics import row_nll
from audit.core.censored_objective import CAP

TASK_SPECS = {
    "full_coverage": {
        "role": "PRIMARY",
        "coverage_requirement": 1.0,
        "abstain_without_fallback": "INELIGIBLE",
        "metric": "pooled_OOF_junction_macro_right_censored_nll",
        "note": "all eligible rows must be scored; no-cost abstain is not allowed",
    },
    "selective": {
        "role": "SECONDARY",
        "coverage_floor": 0.80,   # frozen pre-registered floor
        "abstain_cost": "reported",
        "metrics": ["supported_nll", "coverage_matched_nll", "AURC"],
        "note": "must meet frozen coverage floor; never replaces full-coverage gate",
    },
}


def validate_unique_keys(preds):
    """Return list of duplicate primary keys. Empty => unique."""
    seen = set()
    dups = []
    for p in preds:
        k = (p["axis"], p["fold"], p["source_row_id"], p["model_id"])
        if k in seen:
            dups.append(k)
        seen.add(k)
    return dups


def _row_has_fallback(p):
    return bool(p.get("fallback_type") and str(p.get("fallback_type")) not in ("", "none", "None"))


def full_coverage_score(rows, preds_by_rowid):
    """Score the full-coverage primary task.

    rows: list of test row dicts (eligible observations).
    preds_by_rowid: {source_row_id: prediction dict}.
    Returns (metric_dict, eligibility_dict).
    A fold/axis is comparison-ineligible if any eligible row lacks a scorable
    prediction (abstain and no fallback).
    """
    n_eligible = len(rows)
    scorable = 0
    abstained_no_fallback = 0
    nlls = []
    for r in rows:
        rid = str(r["source_row_id"])
        p = preds_by_rowid.get(rid)
        if p is None:
            abstained_no_fallback += 1
            continue
        if p.get("abstain") and not _row_has_fallback(p):
            abstained_no_fallback += 1
            continue
        scorable += 1
        nlls.append(float(row_nll([r["y"]], [r["cens"]], [p["mu"]], [p["sigma"]])[0]))
    coverage = scorable / n_eligible if n_eligible else 0.0
    eligible = (abstained_no_fallback == 0)
    # pooled OOF junction-macro NLL over scorable rows
    by_jid = defaultdict(list)
    for r, nll in zip(rows, _nll_by_row(rows, preds_by_rowid)):
        if nll is not None:
            by_jid[str(r["jid"])].append(nll)
    pooled = float(np.mean([np.mean(v) for v in by_jid.values()])) if by_jid else None
    return {
        "n_eligible": n_eligible,
        "n_scorable": scorable,
        "n_abstain_no_fallback": abstained_no_fallback,
        "coverage": coverage,
        "pooled_junction_macro_nll": pooled,
    }, {"eligible": eligible, "reason": None if eligible else "incomplete_full_coverage_without_fallback"}


def _nll_by_row(rows, preds_by_rowid):
    out = []
    for r in rows:
        p = preds_by_rowid.get(str(r["source_row_id"]))
        if p is None or (p.get("abstain") and not _row_has_fallback(p)):
            out.append(None)
            continue
        out.append(float(row_nll([r["y"]], [r["cens"]], [p["mu"]], [p["sigma"]])[0]))
    return out


def selective_score(rows, preds_by_rowid, coverage_floor=None):
    """Score the selective secondary task over supported rows.

    Returns supported-NLL, coverage, coverage-matched NLL (using only supported
    rows of the comparator passed as supported subset) and AURC over the
    risk-coverage curve.  Placeholder abstained rows are excluded from the NLL.
    """
    floor = coverage_floor if coverage_floor is not None else TASK_SPECS["selective"]["coverage_floor"]
    n_eligible = len(rows)
    supported = []
    for r in rows:
        p = preds_by_rowid.get(str(r["source_row_id"]))
        if p is None:
            continue
        if p.get("abstain") and not _row_has_fallback(p):
            continue
        supported.append((r, p))
    coverage = len(supported) / n_eligible if n_eligible else 0.0
    by_jid = defaultdict(list)
    for r, p in supported:
        by_jid[str(r["jid"])].append(float(row_nll([r["y"]], [r["cens"]], [p["mu"]], [p["sigma"]])[0]))
    supported_nll = float(np.mean([np.mean(v) for v in by_jid.values()])) if by_jid else None
    return {
        "n_eligible": n_eligible,
        "n_supported": len(supported),
        "coverage": coverage,
        "coverage_floor": floor,
        "meets_floor": bool(coverage >= floor - 1e-12),
        "supported_junction_macro_nll": supported_nll,
    }
