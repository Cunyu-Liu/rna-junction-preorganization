"""r59 diagnostic: r56b measured-residual distribution shape.

If the post-r56b measured residuals have heavy tails / skew that a Student-t
predictive would fit better than Gaussian, a distributional upgrade (not just
sigma calibration) could further lower NLL.  Also check kurtosis per scaffold
and whether the fixed Student-t df=7 head matches the actual residual shape.
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
    GBDT, MLP,
)
from scipy import stats

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

# standardized residuals for measured rows: z = (y - mu)/sigma
z = []
for rid, p in cal56.items():
    if not p["cens"]:
        z.append((p["y"] - p["mu"]) / p["sigma"])
z = np.asarray(z)
print(f"n measured = {len(z)}")
print(f"z mean = {np.mean(z):+.3f}  z sd = {np.std(z):.3f}  (ideal 0/1)")
print(f"z skew = {stats.skew(z):+.3f}  z kurtosis = {stats.kurtosis(z):+.3f}")
print(f"Gaussian kurtosis = 0, heavy tails => kurtosis > 0")

# per-scaf standardized residual kurtosis
by_scaf = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        by_scaf[int(p["scaf"])].append((p["y"] - p["mu"]) / p["sigma"])
print("\nper-scaf standardized residual stats:")
for sc in sorted(by_scaf):
    v = np.asarray(by_scaf[sc])
    print(f"  scaf{sc}: n={len(v)} sd={np.std(v):.3f} skew={stats.skew(v):+.3f} "
          f"kurt={stats.kurtosis(v):+.3f}")

# Would a heavier/lighter tail (Student-t df) fit better?  Fit df to residuals
# via MLE on the standardized residuals (t location-scale).
from scipy.stats import t as tdist
df_hat, loc_hat, scale_hat = tdist.fit(z)
print(f"\nMLE Student-t on standardized residuals: df={df_hat:.2f} "
      f"loc={loc_hat:+.3f} scale={scale_hat:.3f}")
# Gaussian NLL vs t NLL at their MLEs
nll_g = np.mean(-stats.norm.logpdf(z, 0, 1))
nll_t = np.mean(-tdist.logpdf(z, df_hat, loc_hat, scale_hat))
print(f"mean NLL: Gaussian={nll_g:.4f}  Student-t(MLE)={nll_t:.4f}")
print(f"  delta (lower=better): {nll_t - nll_g:+.4f}")
