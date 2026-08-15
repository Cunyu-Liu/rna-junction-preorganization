"""r64: verify r62 gain mechanism - is larger sigma_m the cause?

r62 best (0.7243) < r56b (0.7314) with measured 0.8182 < 0.8265.  Hypothesis:
r56b's coupled sigma is too small (scaf8 0.530 vs 0.599 decoupled), and under
heavy tails the optimal Gaussian sigma is LARGER, so decoupling helps.  Verify
per-fold: does r62 sigma_m > r56b sigma_m on most folds, and does the measured
NLL drop follow sigma increase?  Also confirm r62's sigma is the argmin (not
just larger).
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r62_decoupled_frozen import (
    _load, _elig, _by_rid, _pooled, _calibrate_r62, _calibrate_r56b,
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
cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)

# per-fold: sigma_m ratio and NLL delta (measured only)
byf56 = defaultdict(list)
byf62 = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        byf56[p["fold"]].append((p["sigma"], p["y"], p["mu"]))
for rid, p in cal62.items():
    if not p["cens"]:
        byf62[p["fold"]].append((p["sigma"], p["y"], p["mu"]))

print("per-fold measured: sigma56 vs sigma62, NLL56 vs NLL62 (positive delta = r62 better)")
n_sig_up = 0
for f in folds:
    s56 = np.mean([x[0] for x in byf56[f]])
    s62 = np.mean([x[0] for x in byf62[f]])
    if s62 > s56:
        n_sig_up += 1
    n56 = np.mean([0.5*np.log(2*np.pi) + np.log(x[0]) + 0.5*((x[1]-x[2])/x[0])**2 for x in byf56[f]])
    n62 = np.mean([0.5*np.log(2*np.pi) + np.log(x[0]) + 0.5*((x[1]-x[2])/x[0])**2 for x in byf62[f]])
    tag = "+" if n62 < n56 else "-"
    print(f"  {str(f)[-12:]:12s} sig {s56:.3f}->{s62:.3f}  nll {n56:.4f}->{n62:.4f} ({tag})")

print(f"\nsigma_m increased on {n_sig_up}/{len(folds)} folds")
