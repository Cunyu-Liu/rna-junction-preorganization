"""r59b: r56b emitted sigma vs actual residual spread - is sigma_m too large?

The measured kurtosis is +1.117 (heavy tails) under Gaussian scoring.  The
frozen sigma_m is a per-scaf x stratum CONSTANT.  If within a scaf x stratum
the residual sd varies by context (or by row feature), a per-row sigma could
be smaller on most rows and the Gaussian NLL drops even though the marginal
kurtosis stays heavy.  Check:
  1. r56b emitted sigma_m vs realized residual sd per scaf x stratum.
  2. per-context residual sd within a scaf x stratum - is it constant or
     heterogeneous (if heterogeneous and stable, per-context sigma helps).
  3. does per-context sigma (r54-like) reduce NLL under the CURRENT r56b mu?
     (r54 tested context sigma on r51 mu and failed; retest on r56b mu since
     mu is now better calibrated -> smaller residuals -> context sigma signal
     may appear)
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b, _pooled, _scan_sigma, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
    GBDT, MLP,
)

elig33 = _elig([R33_LEDGER])
elig34 = _elig([R34_LEDGER])
elig35 = _elig([R35_LEDGER])
elig24 = _elig(R24_LEDGERS)
members = {}
members[XGB] = _by_rid(_load(R33), XGB, elig33)
members[XGB_S99] = _by_rid(_load(R34), XGB_S99, elig34)
members[XGB_S2026] = _by_rid(_load(R34), XGB_S2026, elig34)
members[XGB_LR03] = _by_rid(_load(R35), XGB_LR03, elig35)
members[T7] = _by_rid(_load(R24), T7, elig24)
members[T7_S99] = _by_rid(_load(R24), T7_S99, elig24)
members[T7_S2026] = _by_rid(_load(R24), T7_S2026, elig24)
common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
ref = members[ALL_MEMBERS[0]]
ens = {}
for rid in common:
    p0 = ref[rid]
    gmu = float(np.mean([members[m][rid]["mu"] for m in GBDT]))
    mmu = float(np.mean([members[m][rid]["mu"] for m in MLP]))
    ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": int(p0["scaf"]),
                "context": str(p0.get("context", "?")), "y": p0["y"],
                "cens": p0["cens"], "mu": 0.5 * gmu + 0.5 * mmu}
folds = sorted(set(ens[r]["fold"] for r in ens))
cal56, _ = _calibrate_r56b(ens, folds, kappa=2.0, min_meas=3)
print("r56b baseline =", round(_pooled(cal56), 4))

# 1. emitted sigma_m vs residual sd per scaf
by_scaf = defaultdict(lambda: {"sd": [], "sig": []})
for rid, p in cal56.items():
    if not p["cens"]:
        by_scaf[int(p["scaf"])]["sd"].append(p["y"] - p["mu"])
        by_scaf[int(p["scaf"])]["sig"].append(p["sigma"])
print("\n=== emitted sigma_m vs residual sd (measured) ===")
for sc in sorted(by_scaf):
    d = by_scaf[sc]
    print(f"  scaf{sc}: resid_sd={np.std(d['sd']):.3f}  emitted_sigma_m={np.mean(d['sig']):.3f}")

# 2. within scaf x stratum, per-context residual sd spread
ctx_sd = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        ctx_sd[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
print("\n=== per-context residual sd within scaf (contexts >=10 measured rows) ===")
for sc in range(1, 10):
    sds = [float(np.std(v)) for (sc2, c), v in ctx_sd.items() if sc2 == sc and len(v) >= 10]
    if len(sds) >= 3:
        print(f"  scaf{sc}: n_ctx={len(sds)} sd min={min(sds):.3f} med={np.median(sds):.3f} "
              f"max={max(sds):.3f} spread={max(sds)-min(sds):.3f}")

# 3. retest per-context sigma on r56b mu (r54 tested on r51 mu and failed)
#    Here: for each held-out fold, fit per-context sigma on OTHER folds' r56b
#    residuals (shrink to scaf), apply to held-out.  Quick implementation.
by_fold = defaultdict(dict)
for rid, p in cal56.items():
    by_fold[p["fold"]][rid] = p
cal_ctxsig = {}
for f in folds:
    other = {}
    for ff in folds:
        if ff != f:
            other.update(by_fold[ff])
    by_scaf_rows = defaultdict(dict)
    for rid, p in other.items():
        by_scaf_rows[int(p["scaf"])][rid] = p
    s_global, _ = _scan_sigma(other, cens_mask=False, grid=GRID)
    scaf_sm = {}
    for sc, rows in by_scaf_rows.items():
        scaf_sm[sc] = _scan_sigma(rows, cens_mask=False, grid=GRID)[0]
    by_ctx = defaultdict(dict)
    for rid, p in other.items():
        if not p["cens"]:
            by_ctx[(int(p["scaf"]), str(p["context"]))][rid] = p
    ctx_sig = {}
    for (sc, ctx), rows in by_ctx.items():
        if len(rows) >= 8:
            s_ctx, _ = _scan_sigma(rows, cens_mask=False, grid=GRID)
            w = len(rows) / (len(rows) + 10.0)
            ctx_sig[(sc, ctx)] = w * s_ctx + (1 - w) * scaf_sm.get(sc, s_global)
    for rid, p in by_fold[f].items():
        if p["cens"]:
            cal_ctxsig[rid] = p
        else:
            sc = int(p["scaf"])
            sig = ctx_sig.get((sc, str(p["context"])), scaf_sm.get(sc, s_global))
            cal_ctxsig[rid] = {**p, "sigma": float(sig)}
print("\n=== per-context sigma on r56b mu ===")
print("r56b + ctx sigma =", round(_pooled(cal_ctxsig), 4), "vs r56b", round(_pooled(cal56), 4))
