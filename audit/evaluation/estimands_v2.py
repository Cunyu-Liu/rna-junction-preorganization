"""R0.3 multi-estimand scorer (contract R0.3 / FrozenGateSpec).

The word "NLL" must never conflate three different aggregations:
  pooled_junction_macro       : per-junction mean, then equal-weight macro across junctions
  nested_context_macro        : per-junction mean within a context, then equal-weight across contexts
  scaffold_bundle_macro       : per-junction mean within a scaffold bundle, then equal-weight across scaffolds

All are computed over the FULL-COVERAGE scorable rows (abstain-without-fallback
rows are excluded; see scorer_v2).
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from audit.evaluation.metrics import row_nll


def _nll_by_jid(rows, mu, sigma):
    out = {}
    for i, r in enumerate(rows):
        jid = str(r["jid"])
        out.setdefault(jid, []).append(float(row_nll([r["y"]], [r["cens"]], [mu[i]], [sigma[i]])[0]))
    return {j: float(np.mean(v)) for j, v in out.items()}


def pooled_junction_macro(rows, mu, sigma):
    by = _nll_by_jid(rows, mu, sigma)
    return float(np.mean(list(by.values()))) if by else None


def nested_context_macro(rows, mu, sigma):
    """Equal-weight macro across contexts of the per-junction mean NLL."""
    by_jid = _nll_by_jid(rows, mu, sigma)
    ctx = defaultdict(list)
    for r in rows:
        jid = str(r["jid"])
        if jid in by_jid:
            ctx[str(r["helix_seq"])].append(by_jid[jid])
    if not ctx:
        return None
    per_ctx = {c: float(np.mean(v)) for c, v in ctx.items()}
    return float(np.mean(list(per_ctx.values()))), per_ctx


def scaffold_bundle_macro(rows, mu, sigma):
    """Equal-weight macro across scaffolds of the per-junction mean NLL."""
    by_jid = _nll_by_jid(rows, mu, sigma)
    scaf = defaultdict(list)
    for r in rows:
        jid = str(r["jid"])
        if jid in by_jid:
            scaf[int(r["scaf"])].append(by_jid[jid])
    if not scaf:
        return None
    per_scaf = {s: float(np.mean(v)) for s, v in scaf.items()}
    return float(np.mean(list(per_scaf.values()))), per_scaf
