"""r57 diagnostic: residual structure after r56b + jid-vs-fold geometry.

Questions:
1. Does a jid (junction) span multiple folds?  If YES, a jid-level EB mu random
   effect is legal (leakage-free); if NO (each jid lives in exactly one fold),
   jid-level correction would leak and is forbidden.
2. After r56b, is the context bias fully removed, or is there residual
   structure (fold-level / jid-level / context-by-stratum)?
3. Is the r56b residual bias stable under split-half, i.e. is there MORE signal
   to extract?
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
import statistics
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b, _calibrate_r51,
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

# --- Q1: jid vs fold geometry ---
jid_folds = defaultdict(set)
for rid, p in ens.items():
    jid_folds[str(p["jid"])].add(str(p["fold"]))
spans = [len(v) for v in jid_folds.values()]
print("=== Q1: jid span across folds ===")
print(f"n jids={len(jid_folds)}  span: min={min(spans)} med={statistics.median(spans)} max={max(spans)}")
print(f"jids spanning 1 fold: {sum(1 for s in spans if s == 1)}  "
      f">=2 folds: {sum(1 for s in spans if s >= 2)}")

# --- Q2/Q3: residual context bias after r56b, split-half stability ---
cal56, _ = _calibrate_r56b(ens, folds, kappa=2.0, min_meas=3)
cal51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)

ctx_res_56 = defaultdict(list)
ctx_res_51 = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        ctx_res_56[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
for rid, p in cal51.items():
    if not p["cens"]:
        ctx_res_51[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])

print("\n=== Q2: context bias after r56b (>=10 measured rows) ===")
b51 = [np.mean(v) for v in ctx_res_51.values() if len(v) >= 10]
b56 = [np.mean(v) for v in ctx_res_56.values() if len(v) >= 10]
print(f"r51:  n={len(b51)} sd={float(np.std(b51)):.3f} mean={float(np.mean(b51)):+.3f}")
print(f"r56b: n={len(b56)} sd={float(np.std(b56)):.3f} mean={float(np.mean(b56)):+.3f}")
print(f"r56b mean |bias|={float(np.mean(np.abs(b56))):.3f} vs r51 {float(np.mean(np.abs(b51))):.3f}")

# split-half stability of r56b residual (is there MORE signal?)
stab = []
for (sc, ctx), v in sorted(ctx_res_56.items()):
    if len(v) >= 16:
        h1, h2 = v[::2], v[1::2]
        if len(h1) >= 8 and len(h2) >= 8:
            stab.append((float(np.mean(h1)), float(np.mean(h2))))
if len(stab) >= 5:
    m1 = np.array([s[0] for s in stab])
    m2 = np.array([s[1] for s in stab])
    print(f"\n=== Q3: r56b residual context bias split-half corr = "
          f"{np.corrcoef(m1, m2)[0, 1]:+.3f} (n={len(stab)}) ===")
    print(f"  if corr high (>0.5): MORE context signal remains (r56b incomplete)")
    print(f"  if corr ~0: residual is noise, context mu fully extracted")

# Q3b: is residual fold-level bias stable? (per-fold mean of measured residual)
fold_res = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        fold_res[str(p["fold"])].append(p["y"] - p["mu"])
fr = {k: float(np.mean(v)) for k, v in fold_res.items()}
print(f"\n=== Q3b: per-fold measured residual bias after r56b ===")
vals = list(fr.values())
print(f"  n_folds={len(vals)} sd={float(np.std(vals)):.3f} "
      f"mean={float(np.mean(vals)):+.3f} min={min(vals):+.3f} max={max(vals):+.3f}")
