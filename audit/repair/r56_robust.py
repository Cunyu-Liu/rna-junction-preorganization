"""r56 robustness: leave-one-edit-out stability + worst-fold context audit."""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r56_per_ctx_eb_mu import (
    _load, _elig, _by_rid, _calibrate_r56, _calibrate_r51, _pooled,
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
cal51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)
cal56, _ = _calibrate_r56(ens, folds, kappa=10.0)

# per-edit delta (r51 - r56, positive = r56 better)
byed = defaultdict(list)
for rid, p in cal56.items():
    e = str(p["fold"]).split(":", 1)[1]
    n51 = float(row_nll([cal51[rid]["y"]], [cal51[rid]["cens"]],
                        [cal51[rid]["mu"]], [cal51[rid]["sigma"]])[0])
    n56 = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
    byed[e].append(n51 - n56)
means = {k: float(np.mean(v)) for k, v in byed.items()}
print("per-edit delta (r51-r56):")
for k in sorted(means, key=lambda x: means[x]):
    print(f"  {k:15s} {means[k]:+.4f}")

# worst fold context audit
worst = min(means, key=means.get)
print(f"\nWORST edit = {worst}, delta={means[worst]:.4f}")
# how many rows does the worst edit have per context, and do its contexts appear
# in other folds with enough rows for a stable bias estimate?
byfold = defaultdict(dict)
for rid, p in cal56.items():
    byfold[p["fold"]][rid] = p
worst_ctx = defaultdict(list)
for rid, p in byfold[f"e:{worst}"].items():
    if not p["cens"]:
        worst_ctx[str(p["context"])].append(p["y"] - p["mu"])
for ctx, v in sorted(worst_ctx.items(), key=lambda kv: -len(kv[1])):
    print(f"  ctx {ctx}: n_meas={len(v)} bias={np.mean(v):+.3f}")

# leave-one-edit-out total
total = float(np.mean(list(means.values())))
leave1 = sorted(means.items(), key=lambda kv: kv[1])
print(f"\ntotal delta mean={total:.4f}")
print(f"leave_one_worst_mean={float(np.mean([v for k, v in leave1[1:]])):.4f}")
print(f"n_positive={sum(1 for v in means.values() if v > 0)}/{len(means)}")
