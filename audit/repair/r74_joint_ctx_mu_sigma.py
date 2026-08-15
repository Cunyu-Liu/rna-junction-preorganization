"""r74: joint per-context (mu, sigma) 2-parameter EB calibration.

r56b fits per-context mu EB, then r62 re-scans sigma at scaf x stratum level
SEQUENTIALLY (mu first, sigma second, decoupled).  A genuinely untested lever:
fit per-context (alpha_mu, sigma) TOGETHER as a 2-parameter EB per context on
OTHER folds, then emit both for the held-out fold.  The interaction (a context
with large bias may also want a different sigma) was never jointly optimized.

Design (honest LOO):
  - mu base = r62-corrected mu (frozen).
  - For each held-out fold: on OTHER folds' measured rows, for each (scaf, ctx)
    with >= min_meas rows, grid-search (alpha, sigma) over a small EB-constrained
    grid minimizing that context's macro NLL; shrink alpha toward scaf-mean
    residual (r56b prior ~0) and sigma toward the r62 decoupled scaf sigma.
  - Emit per-context (mu+alpha, sigma) for held-out measured rows; censored
    unchanged.
  - Compare pooled NLL vs r62 (0.7243) + edit-cluster CI.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _pooled, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)
from audit.repair.r62_decoupled_frozen import _calibrate_r62

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def _edit_ci(ens_dict, base_dict):
    jid_edit = {}
    for rid, p in ens_dict.items():
        jid_edit.setdefault(p["jid"], str(p["fold"]).split(":", 1)[1])
    jid_d = defaultdict(list)
    for rid, p in ens_dict.items():
        if rid not in base_dict:
            continue
        nll_e = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        nll_b = float(row_nll([base_dict[rid]["y"]], [base_dict[rid]["cens"]],
                              [base_dict[rid]["mu"]], [base_dict[rid]["sigma"]])[0])
        jid_d[p["jid"]].append(nll_b - nll_e)
    by_edit = defaultdict(list)
    for j, vals in jid_d.items():
        by_edit[jid_edit.get(j, "?")].append(float(np.mean(vals)))
    rng = np.random.default_rng(17)
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


def main():
    elig33 = _elig([R33_LEDGER])
    elig34 = _elig([R34_LEDGER])
    elig35 = _elig([R35_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows24 = _load(R24)
    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[T7] = _by_rid(rows24, T7, elig24)
    members[T7_S99] = _by_rid(rows24, T7_S99, elig24)
    members[T7_S2026] = _by_rid(rows24, T7_S2026, elig24)
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

    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        by_fold[p["fold"]][rid] = {**p}

    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma

    # alpha grid (shrink toward scaf-mean residual, which is ~0 after r62)
    a_grid = np.round(np.arange(-0.30, 0.31, 0.05), 2)
    # sigma grid around the r62 decoupled scaf sigma_m
    out = {}
    for min_meas in (3, 5):
        cal = {}
        fit_log = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            # r62 decoupled scaf sigma_m (fit on other folds)
            by_scaf = defaultdict(dict)
            for rid, p in other.items():
                if not p["cens"]:
                    by_scaf[int(p["scaf"])][rid] = p
            scaf_sigma = {}
            for sc, rows_sc in by_scaf.items():
                if len(rows_sc) >= 15:
                    s, _ = _scan_sigma(rows_sc, cens_mask=False, grid=GRID)
                    scaf_sigma[sc] = s
            sm_global, _ = _scan_sigma(
                {r: p for r, p in other.items() if not p["cens"]},
                cens_mask=False, grid=GRID)
            # scaf-mean measured residual (prior for alpha ~ 0)
            b_scaf = {}
            for sc, rows_sc in by_scaf.items():
                b_scaf[sc] = float(np.mean([p["y"] - p["mu"] for p in rows_sc.values()]))
            # per-context rows
            by_ctx = defaultdict(dict)
            for rid, p in other.items():
                if not p["cens"]:
                    by_ctx[(int(p["scaf"]), str(p["context"]))][rid] = p
            ctx_params = {}
            for (sc, ctx), rows_ctx in by_ctx.items():
                n = len(rows_ctx)
                if n < min_meas:
                    continue
                y = np.asarray([p["y"] for p in rows_ctx.values()])
                mu = np.asarray([p["mu"] for p in rows_ctx.values()])
                jid = np.asarray([p["jid"] for p in rows_ctx.values()])
                # shrink alpha toward b_scaf (r56b prior ~0) with EB weight
                w = n / (n + 2.0)  # kappa=2 as r56b
                a_prior = b_scaf.get(sc, 0.0)
                sig_base = scaf_sigma.get(sc, sm_global or 0.5)
                s_grid = np.round(np.arange(max(0.15, sig_base - 0.25),
                                            min(1.6, sig_base + 0.35) + 0.001, 0.05), 2)
                best = None
                best_n = np.inf
                for a_raw in a_grid:
                    a = float(w * a_raw + (1 - w) * a_prior)
                    mu_c = mu + a
                    for s in s_grid:
                        losses = row_nll(y, np.zeros(n, dtype=bool), mu_c,
                                         np.full(n, float(s)))
                        uniq, jc = np.unique(jid, return_inverse=True)
                        sums = np.bincount(jc, weights=losses, minlength=len(uniq))
                        cnt = np.bincount(jc, minlength=len(uniq))
                        nll = float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))
                        if nll < best_n:
                            best_n, best = nll, (a, float(s))
                ctx_params[(sc, ctx)] = best
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    cal[rid] = p
                else:
                    sc = int(p["scaf"])
                    prm = ctx_params.get((sc, str(p["context"])))
                    if prm is None:
                        cal[rid] = p
                    else:
                        a, s = prm
                        cal[rid] = {**p, "mu": float(p["mu"] + a), "sigma": s}
            fit_log[f] = {"n_ctx_joint": len(ctx_params)}
        nll = _pooled(cal)
        out[f"mm{min_meas}"] = round(nll, 4)
        print(f"r74 joint (mu,sigma) per-context mm{min_meas}: {nll:.4f} "
              f"(delta vs r62 {nll-0.7243:+.4f})")
        if min_meas == 3:
            ci = _edit_ci(cal, cal62)
            print(f"  edit CI vs r62 = {ci['ci']} lower_gt_0={ci['ci_lower_gt_0']} "
                  f"leave1={ci['leave_one_largest']}")

    out["r62"] = round(_pooled(cal62), 4)
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r74_joint_ctx_mu_sigma.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
