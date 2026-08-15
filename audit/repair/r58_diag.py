"""r58 diagnostic: is there context x stratum or context x scaf residual bias
after r56b that r56b's additive context correction cannot reach?

r56b applies the SAME additive context bias alpha_ctx to all measured rows of a
context, regardless of stratum/scaf (context is nested in scaf so scaf is fixed
per context).  Two unexplored interaction structures:
  1. context x censoring-stratum: measured rows corrected, censored untouched.
     If censored rows of a context also carry bias, we can't fix mu (no y) but
     the sigma_c might be off.
  2. context x jid-within-context: not extractable (jid single-fold).
  3. fold-level residual within context: is the per-context bias fold-stable
     (should be, split-half 0.986 pre-r56b) and post-r56b is the residual
     still fold-structured?

We check: per-context post-r56b residual bias by stratum, and whether residual
fold-structure remains after the context correction.
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
import statistics
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b,
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

# per-context residual by stratum (measured only - censored has no y)
ctx_meas = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        ctx_meas[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])

# 1. Is per-context residual fold-stable post-r56b? (split-half on >=16)
stab = []
for (sc, ctx), v in sorted(ctx_meas.items()):
    if len(v) >= 16:
        h1, h2 = v[::2], v[1::2]
        stab.append((float(np.mean(h1)), float(np.mean(h2))))
m1 = np.array([s[0] for s in stab])
m2 = np.array([s[1] for s in stab])
print(f"Q1 post-r56b per-context residual split-half corr = {np.corrcoef(m1, m2)[0,1]:+.3f} (n={len(stab)})")

# 2. context residual vs context properties - is it explained by context
#    sequence features (which a training-side context feature could capture)?
#    context name like 'AAA_CCG&GCG_CUU' = flanks.  Use GC/length.
def ctx_feat(name):
    parts = name.split("&")
    s = "".join(parts)
    gc = (s.count("G") + s.count("C")) / len(s) if s else 0
    return [len(s), gc]

ctx_mean = {k: float(np.mean(v)) for k, v in ctx_meas.items() if len(v) >= 10}
X = np.array([ctx_feat(k[1]) for k in ctx_mean])
y = np.array(list(ctx_mean.values()))
for i, nm in enumerate(["len", "gc"]):
    print(f"  corr(ctx_resid, {nm}) = {np.corrcoef(X[:,i], y)[0,1]:+.3f}")

# 3. within-context residual sd (noise floor) vs between-context sd
within = []
ctx_means_vals = []
for (sc, ctx), v in ctx_meas.items():
    if len(v) >= 10:
        within.append(float(np.std(v)))
        ctx_means_vals.append(float(np.mean(v)))
between_sd = float(np.std(ctx_means_vals))
print(f"Q3 within-context residual sd mean={float(np.mean(within)):.3f}")
print(f"Q3 between-context residual sd={between_sd:.3f}")
print(f"  ratio (signal that context explains): {1 - float(np.mean(within))/between_sd:.3f}")

# 4. per-fold residual within measured after r56b - fold structure?
fold_res = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        fold_res[str(p["fold"])].append(p["y"] - p["mu"])
fr = {k: float(np.mean(v)) for k, v in fold_res.items()}
frv = list(fr.values())
print(f"Q4 per-fold measured residual after r56b: sd={float(np.std(frv)):.3f} "
      f"min={min(frv):+.3f} max={max(frv):+.3f}")
