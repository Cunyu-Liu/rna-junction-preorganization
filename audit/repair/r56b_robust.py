"""r56b best-config robustness: per-edit deltas + leave-one-edit-out."""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b, _calibrate_r51, _pooled,
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
cal56, _ = _calibrate_r56b(ens, folds, kappa=2.0, min_meas=3)

byed = defaultdict(list)
for rid, p in cal56.items():
    e = str(p["fold"]).split(":", 1)[1]
    n51 = float(row_nll([cal51[rid]["y"]], [cal51[rid]["cens"]],
                        [cal51[rid]["mu"]], [cal51[rid]["sigma"]])[0])
    n56 = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
    byed[e].append(n51 - n56)
means = {k: float(np.mean(v)) for k, v in byed.items()}
print("per-edit delta (r51-r56b), sorted:")
for k in sorted(means, key=lambda x: means[x]):
    print(f"  {k:15s} {means[k]:+.4f}")
total = float(np.mean(list(means.values())))
print(f"\ntotal={total:.4f}  n_pos={sum(1 for v in means.values() if v > 0)}/{len(means)}")
print(f"median={float(np.median(list(means.values()))):.4f}")
worst = min(means.items(), key=lambda kv: kv[1])
keep = {k: v for k, v in means.items() if k != worst[0]}
print(f"worst={worst[0]} ({worst[1]:+.4f})  leave_one_worst={float(np.mean(list(keep.values()))):.4f}")

# context bias after r56b
ctx_bias = defaultdict(list)
for rid, p in cal56.items():
    if not p["cens"]:
        ctx_bias[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
big = [np.mean(v) for v in ctx_bias.values() if len(v) >= 10]
print(f"\ncontext bias after r56b (>=10 rows): n={len(big)} sd={float(np.std(big)):.3f}")
