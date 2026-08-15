"""r56 diagnostic: compare residual sd per scaffold between r51 and r56."""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.repair.r56_per_ctx_eb_mu import (
    _load, _elig, _by_rid, _calibrate_r56, _calibrate_r51,
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


def per_scaf_sd(cal):
    out = defaultdict(list)
    for rid, p in cal.items():
        if not p["cens"]:
            out[int(p["scaf"])].append(p["y"] - p["mu"])
    return {sc: float(np.std(v)) for sc, v in out.items()}


def per_scaf_emitted_sigma(cal):
    out = defaultdict(list)
    for rid, p in cal.items():
        if not p["cens"]:
            out[int(p["scaf"])].append(p["sigma"])
    return {sc: float(np.mean(v)) for sc, v in out.items()}


print("=== measured residual sd per scaf ===")
sd51, sd56 = per_scaf_sd(cal51), per_scaf_sd(cal56)
for sc in sorted(sd51):
    print(f"  scaf{sc}: r51 sd={sd51[sc]:.3f}  r56 sd={sd56[sc]:.3f}")
print("=== emitted sigma_m per scaf ===")
em51, em56 = per_scaf_emitted_sigma(cal51), per_scaf_emitted_sigma(cal56)
for sc in sorted(em51):
    print(f"  scaf{sc}: r51 sigma_m={em51[sc]:.3f}  r56 sigma_m={em56[sc]:.3f}")
