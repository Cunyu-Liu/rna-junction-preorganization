"""r65: verify r62's censored sigma_c - should it also be decoupled/re-scanned?

r62 decoupled ONLY the measured sigma_m (independent re-scan on r56b mu).  The
censored sigma_c was left at r56b's coupled value.  In r56b, sigma_c is scanned
on corr_other whose censored mu is UNCHANGED (context correction applies to
measured rows only), so sigma_c should already be clean.  But verify: does an
independent sigma_c re-scan (same decoupling as r62's measured side) change
pooled NLL?  Also check: is there a separate gain from a joint (mu, sigma)
optimization on the censored side (survival likelihood)?
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

# r65: r62 mu + independent re-scan of BOTH sigma_m and sigma_c (decouple censored too)
by_fold = defaultdict(dict)
for rid, p in cal62.items():
    by_fold[p["fold"]][rid] = p
cal65 = {}
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
        cal65[rid] = {**p, "sigma": float(sig)}

print("r65 (decouple sigma_c too) =", round(_pooled(cal65), 4))
print("delta vs r62 =", round(_pooled(cal65) - _pooled(cal62), 4))

# compare censored sigma: r62 vs r65
sigc62 = defaultdict(list)
sigc65 = defaultdict(list)
for rid, p in cal62.items():
    if p["cens"]:
        sigc62[int(p["scaf"])].append(p["sigma"])
for rid, p in cal65.items():
    if p["cens"]:
        sigc65[int(p["scaf"])].append(p["sigma"])
print("\nper-scaf sigma_c:")
for sc in sorted(sigc62):
    print(f"  scaf{sc}: r62={np.mean(sigc62[sc]):.3f}  r65={np.mean(sigc65[sc]):.3f}")
