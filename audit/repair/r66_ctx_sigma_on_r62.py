"""r66: per-context EB sigma on TOP of the r62 decoupled frozen method.

r60 scanned per-context EB sigma on r56b mu (coupled), best 0.7282 at
min_ctx=5,kappa=50 (vs r56b 0.7314).  r62 then decoupled the scaf x stratum
sigma scan and reached 0.7243.  The untested combination: apply per-context EB
sigma to the r62-corrected mu, shrinking toward r62's DECOUPLED per-scaf
stratum sigma_m (not r56b's coupled per-scaf sigma).  The decoupled scaf sigma
is the honest sigma for the corrected mu, so context-level heteroscedasticity
(if real) should add on top.

Estimands: r62 (frozen) vs r66 (r62 + per-context EB sigma).
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r62_decoupled_frozen import (
    _load, _elig, _by_rid, _calibrate_r62, _calibrate_r56b, _pooled,
    _scan_sigma, GRID, R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER,
    R24_LEDGERS, XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026,
    ALL_MEMBERS, GBDT, MLP,
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
cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)
print("r62 baseline =", round(_pooled(cal62), 4))


def cal_ctx_sigma(kappa, min_ctx):
    """r62 mu + per-context EB sigma, shrinking toward r62 decoupled scaf sigma_m."""
    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        by_fold[p["fold"]][rid] = p
    out = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        # r62 decoupled scaf sigma_m (measured rows only): re-derive per-scaf
        # by scanning on other folds' corrected mu.
        by_scaf_rows = defaultdict(dict)
        for rid, p in other.items():
            if not p["cens"]:
                by_scaf_rows[int(p["scaf"])][rid] = p
        s_global_m, _ = _scan_sigma(
            {r: p for r, p in other.items() if not p["cens"]},
            cens_mask=False, grid=GRID)
        scaf_sm = {}
        for sc, rows in by_scaf_rows.items():
            scaf_sm[sc] = _scan_sigma(rows, cens_mask=False, grid=GRID)[0]
        # per-context measured sigma
        by_ctx = defaultdict(dict)
        for rid, p in other.items():
            if not p["cens"]:
                by_ctx[(int(p["scaf"]), str(p["context"]))][rid] = p
        ctx_sig = {}
        for (sc, ctx), rows in by_ctx.items():
            if len(rows) >= min_ctx:
                s_ctx, _ = _scan_sigma(rows, cens_mask=False, grid=GRID)
                w = len(rows) / (len(rows) + kappa)
                target = scaf_sm.get(sc, s_global_m)
                ctx_sig[(sc, ctx)] = w * s_ctx + (1 - w) * target
        for rid, p in by_fold[f].items():
            if p["cens"]:
                out[rid] = p
            else:
                sc = int(p["scaf"])
                sig = ctx_sig.get((sc, str(p["context"])),
                                  scaf_sm.get(sc, s_global_m))
                out[rid] = {**p, "sigma": float(sig)}
    return out


results = {}
for min_ctx in (5, 8, 12, 20):
    for kappa in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        cal = cal_ctx_sigma(kappa, min_ctx)
        nll = round(_pooled(cal), 4)
        results[(min_ctx, kappa)] = nll
        print(f"min_ctx={min_ctx} kappa={kappa:g}: {nll}")

best = min(results, key=results.get)
print(f"\nBEST: min_ctx={best[0]} kappa={best[1]} -> {results[best]}")
print(f"vs r62 {0.7243}: delta={results[best] - 0.7243:+.4f}")
