"""r63: paired edit-cluster CI for r62 (decoupled sigma) vs r56b and vs r51.

Also verify r62 sigma is scanned on r56b's corrected mu (not a regression to
r51).  Report the frozen-method decision numbers.
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
from audit.evaluation.metrics import row_nll
from audit.repair.r56b_per_ctx_eb_mu_floor import (
    _load, _elig, _by_rid, _calibrate_r56b, _calibrate_r51, _pooled,
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
cal56, _ = _calibrate_r56b(ens, folds, kappa=2.0, min_meas=3)
cal51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)

# r62 decoupled sigma
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

print("r51 =", round(_pooled(cal51), 4), " r56b =", round(_pooled(cal56), 4),
      " r62 =", round(_pooled(cal62), 4))


def paired_ci(calB, calA, seed=23):
    jid_edit = {}
    for rid, p in calB.items():
        jid_edit.setdefault(p["jid"], str(p["fold"]).split(":", 1)[1])
    jid_d = defaultdict(list)
    for rid, p in calB.items():
        if rid not in calA:
            continue
        nB = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        nA = float(row_nll([calA[rid]["y"]], [calA[rid]["cens"]],
                           [calA[rid]["mu"]], [calA[rid]["sigma"]])[0])
        jid_d[p["jid"]].append(nA - nB)
    by_edit = defaultdict(list)
    for j, vals in jid_d.items():
        by_edit[jid_edit.get(j, "?")].append(float(np.mean(vals)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(2000):
        ch = rng.choice(list(by_edit), size=len(by_edit), replace=True)
        boots.append(float(np.mean([v for e in ch for v in by_edit[e]])))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    sizes = {e: len(v) for e, v in by_edit.items()}
    largest = max(sizes, key=sizes.get)
    keep = [v for e, v in by_edit.items() if e != largest]
    leave1 = float(np.mean([x for g in keep for x in g]))
    return {"ci": [round(lo, 4), round(hi, 4)], "ci_lower_gt_0": bool(lo > 0),
            "leave_one_largest": round(leave1, 4), "largest": largest,
            "n_edit": len(by_edit)}


print("\npaired CI (positive = second better):")
print("  r62 vs r56b:", paired_ci(cal62, cal56))
print("  r62 vs r51:", paired_ci(cal62, cal51))
print("  r56b vs r51:", paired_ci(cal56, cal51))

# strata
def strata(cal):
    jd_m, jd_c = defaultdict(list), defaultdict(list)
    for rid, p in cal.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        (jd_c if p["cens"] else jd_m)[p["jid"]].append(nll)
    return {"measured": round(float(np.mean([np.mean(v) for v in jd_m.values()])), 4),
            "censored": round(float(np.mean([np.mean(v) for v in jd_c.values()])), 4)}
print("\nstrata: r56b =", strata(cal56), " r62 =", strata(cal62))
