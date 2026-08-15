"""r57b: r56b residual re-diagnosis - why is context bias not fully removed?

r56b reduced context bias sd 0.341 -> 0.313 but split-half corr of the
RESIDUAL is still +0.622 (n=18 contexts with >=16 measured rows), i.e. real
signal remains.  Hypotheses to test:
  H1: r56b's EB shrink (kappa=2, min_meas=3) is too weak -> more shrinkage /
      higher min_meas extracts more.
  H2: the correction should be per-context but on the RESIDUAL OF r51 (not
      directly), i.e. iterate r56b (a second round on the already-corrected mu).
  H3: the residual signal is context x stratum interaction (measured bias !=
      censored bias) - but censored rows have no y, can't correct directly.
  H4: the +0.622 corr is fragile (only 18 contexts) and the remaining bias is
      actually jid/within-context noise that is NOT extractable without leak.

We test H1 (stronger shrink) and H2 (iterate) on the same LOO protocol.
"""
import sys
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
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
print("r51 baseline =", round(_pooled(cal51), 4))


def per_ctx_bias(cal):
    out = defaultdict(list)
    for rid, p in cal.items():
        if not p["cens"]:
            out[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
    return out


def ctx_bias_sd(cal):
    b = [np.mean(v) for v in per_ctx_bias(cal).values() if len(v) >= 10]
    return float(np.std(b)), float(np.mean(np.abs(b)))


def calibrate_generic(ens, folds, kappa=2.0, min_meas=3, n_iter=1, grid=None):
    """Iterative per-context EB mu correction on top of r51.

    n_iter=1 == r56b.  n_iter>1 re-fits the context bias on the ALREADY
    corrected mu from the previous round (on other folds), applying the second
    round correction to the held-out fold.  This tests H2.
    """
    from audit.evaluation.metrics import row_nll
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma, GRID as _G
    grid = grid if grid is not None else _G
    # start from r51 corrected mu
    cur = {}
    for rid, p in cal51.items():
        cur[rid] = {"jid": p["jid"], "fold": p["fold"], "scaf": int(p["scaf"]),
                    "context": str(p.get("context", "?")), "y": p["y"],
                    "cens": p["cens"], "mu": p["mu"]}
    by_fold = defaultdict(dict)
    for rid, p in cur.items():
        by_fold[p["fold"]][rid] = p

    cal = {}
    for it in range(n_iter):
        cal = {}
        fit_log = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            ctx_res = defaultdict(list)
            scaf_res = defaultdict(list)
            for rid, p in other.items():
                if not p["cens"]:
                    ctx_res[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
                    scaf_res[int(p["scaf"])].append(p["y"] - p["mu"])
            b_scaf = {sc: float(np.mean(v)) for sc, v in scaf_res.items()}
            alpha_ctx = {}
            for (sc, ctx), v in ctx_res.items():
                if len(v) >= min_meas:
                    b_ctx = float(np.mean(v))
                    w = float(len(v) / (len(v) + kappa))
                    alpha_ctx[(sc, ctx)] = float(w * b_ctx + (1.0 - w) * b_scaf.get(sc, 0.0))
            corr_other = {}
            for rid, p in other.items():
                if p["cens"]:
                    corr_other[rid] = p
                else:
                    sc = int(p["scaf"])
                    alpha = alpha_ctx.get((sc, str(p["context"])), b_scaf.get(sc, 0.0))
                    corr_other[rid] = {**p, "mu": float(p["mu"] + alpha)}
            s_global, _ = _scan_sigma(corr_other, grid=grid)
            sm_global, _ = _scan_sigma(corr_other, cens_mask=False, grid=grid)
            sc_global, _ = _scan_sigma(corr_other, cens_mask=True, grid=grid)
            by_scaf = defaultdict(dict)
            for rid, p in corr_other.items():
                by_scaf[int(p["scaf"])][rid] = p
            strat_sigma = {}
            for sc, rows_sc in by_scaf.items():
                n = len(rows_sc)
                n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
                entry = {}
                if n - n_c >= 15:
                    sm, _ = _scan_sigma(rows_sc, cens_mask=False, grid=grid)
                    entry["sigma_m"] = sm
                else:
                    entry["sigma_m"] = sm_global if sm_global is not None else s_global
                if n_c >= 15:
                    sc_, _ = _scan_sigma(rows_sc, cens_mask=True, grid=grid)
                    entry["sigma_c"] = sc_
                else:
                    entry["sigma_c"] = sc_global if sc_global is not None else s_global
                strat_sigma[sc] = entry
            for rid, p in by_fold[f].items():
                sc = int(p["scaf"])
                e = strat_sigma.get(sc, {})
                if p["cens"]:
                    sig = e.get("sigma_c", s_global)
                    mu_new = p["mu"]
                else:
                    sig = e.get("sigma_m", s_global)
                    alpha = alpha_ctx.get((sc, str(p["context"])), b_scaf.get(sc, 0.0))
                    mu_new = float(p["mu"] + alpha)
                cal[rid] = {"jid": p["jid"], "fold": p["fold"], "scaf": int(p["scaf"]),
                            "context": str(p.get("context", "?")), "y": p["y"],
                            "cens": p["cens"], "mu": float(mu_new), "sigma": float(sig)}
            fit_log[f] = {"n_ctx": len(alpha_ctx)}
        # update cur for next iteration
        for rid, p in cal.items():
            by_fold[p["fold"]][rid] = p
    return cal, fit_log


# H1: stronger shrink / higher min_meas (r56b variants beyond current scan)
print("\n=== H1: stronger shrink variants ===")
for mm, kp in [(5, 1.0), (8, 1.0), (3, 0.5), (5, 0.5), (10, 0.5), (12, 0.2)]:
    cal, _ = _calibrate_r56b(ens, folds, kappa=kp, min_meas=mm)
    nll = round(_pooled(cal), 4)
    sd, mab = ctx_bias_sd(cal)
    print(f"  mm={mm} kappa={kp}: nll={nll}  ctx_bias_sd={sd:.3f} mean|b|={mab:.3f}")

# H2: iterate (n_iter=2)
print("\n=== H2: iterate r56b correction ===")
for it in (1, 2, 3):
    cal, _ = calibrate_generic(ens, folds, kappa=2.0, min_meas=3, n_iter=it)
    nll = round(_pooled(cal), 4)
    sd, mab = ctx_bias_sd(cal)
    print(f"  n_iter={it}: nll={nll}  ctx_bias_sd={sd:.3f} mean|b|={mab:.3f}")
