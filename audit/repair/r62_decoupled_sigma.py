"""r62: decoupled r56b - fix mu with r56b, then independently re-scan sigma.

r61 showed pure-scaf sigma scanned on r56b's corrected mu = 0.725 vs r56b's
internal 0.7314.  The likely cause: r56b couples the context-mu correction and
sigma scan inside one LOO loop, and the per-scaf sigma there is scanned on the
OTHER-fold corrected rows -- which should be equivalent, but the min_ctx/kappa
context correction may over-shift sparse contexts, inflating sigma.

r62 cleanly separates the two stages:
  Stage 1: r56b mu (context EB correction) -- frozen as-is.
  Stage 2: on the r56b-corrected mu of the OTHER folds, scan sigma_m per scaf
           (and sigma_c per scaf) with a fine grid, apply to held-out.
This avoids any coupling artifact and gives the honest sigma for the corrected
mu.  Report pooled NLL + edit-cluster CI vs r56b and vs nuisance.
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
print("r56b (coupled) =", round(_pooled(cal56), 4))

# Stage 1: get r56b mu (already in cal56).  Stage 2: independent sigma re-scan.
by_fold = defaultdict(dict)
for rid, p in cal56.items():
    by_fold[p["fold"]][rid] = p

cal62 = {}
for f in folds:
    other = {}
    for ff in folds:
        if ff != f:
            other.update(by_fold[ff])
    s_global, _ = _scan_sigma(other, grid=GRID)
    sm_global, _ = _scan_sigma(other, cens_mask=False, grid=GRID)
    sc_global, _ = _scan_sigma(other, cens_mask=True, grid=GRID)
    by_scaf = defaultdict(dict)
    for rid, p in other.items():
        by_scaf[int(p["scaf"])][rid] = p
    strat_sigma = {}
    for sc, rows_sc in by_scaf.items():
        n = len(rows_sc)
        n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
        entry = {}
        if n - n_c >= 15:
            sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=GRID)
            entry["sigma_m"] = sm
        else:
            entry["sigma_m"] = sm_global if sm_global is not None else s_global
        if n_c >= 15:
            sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=GRID)
            entry["sigma_c"] = sc_
        else:
            entry["sigma_c"] = sc_global if sc_global is not None else s_global
        strat_sigma[sc] = entry
    for rid, p in by_fold[f].items():
        sc = int(p["scaf"])
        e = strat_sigma.get(sc, {})
        sig = e.get("sigma_c" if p["cens"] else "sigma_m", s_global)
        cal62[rid] = {**p, "sigma": float(sig)}

print("r62 (decoupled sigma re-scan on r56b mu) =", round(_pooled(cal62), 4))
print("delta vs r56b =", round(_pooled(cal62) - _pooled(cal56), 4))

# per-scaf sigma compare
sig56 = defaultdict(list)
sig62 = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        sig56[int(p["scaf"])].append(p["sigma"])
for rid, p in cal62.items():
    if not p["cens"]:
        sig62[int(p["scaf"])].append(p["sigma"])
print("\nper-scaf sigma_m:")
for sc in sorted(sig56):
    print(f"  scaf{sc}: r56b={np.mean(sig56[sc]):.3f}  r62={np.mean(sig62[sc]):.3f}")
