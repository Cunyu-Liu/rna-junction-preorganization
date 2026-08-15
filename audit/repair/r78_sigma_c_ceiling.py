"""r78: extend sigma_c grid ceiling for censored layer (scaf8 hits 1.6 ceiling).

r62's sigma grid ceiling is 1.6.  The extended per-row scan shows scaf8's
censored rows want sigma_c well beyond 1.6 (optimum at 5.0 in a per-row scan).
The frozen censored layer NLL is 0.199.  If extending the sigma_c ceiling lowers
censored NLL under the TRUE junction-macro aggregation, the frozen method could
improve.  Test: re-run the r62 decoupled sigma_c scan with ceiling {1.6, 3.0,
5.0} on the SAME r62 mu, LOO, junction-macro.

Design:
  - mu = r62-corrected mu (frozen).
  - Per-fold: re-scan sigma_c with extended grid on OTHER folds' censored rows
    (per-scaf with global fallback, n_c >= 15), apply to held-out fold.
  - Measured sigma_m unchanged (r62 value).  Compare pooled NLL.
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


def _pooled_strata(ens):
    jd_m = defaultdict(list)
    jd_c = defaultdict(list)
    for rid, p in ens.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        (jd_c if p["cens"] else jd_m)[p["jid"]].append(nll)
    out = {}
    for name, d in (("measured", jd_m), ("censored", jd_c)):
        out[name] = float(np.mean([np.mean(v) for v in d.values()])) if d else None
    return out


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
    print("r62 baseline =", round(_pooled(cal62), 4),
          "strata=", {k: round(v, 4) for k, v in _pooled_strata(cal62).items()})

    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        by_fold[p["fold"]][rid] = {**p}

    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma

    def _scan_c(rows, grid):
        items = list(rows.values())
        y = np.asarray([p["y"] for p in items])
        cens = np.asarray([p["cens"] for p in items])
        mu = np.asarray([p["mu"] for p in items])
        jid = np.asarray([p["jid"] for p in items])
        sel = cens
        if not sel.any():
            return None, np.inf
        y, cens, mu, jid = y[sel], cens[sel], mu[sel], jid[sel]
        uniq, jcode = np.unique(jid, return_inverse=True)
        jcounts = np.bincount(jcode, minlength=len(uniq))
        best_s, best_n = None, np.inf
        for s in grid:
            losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
            sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
            jm = sums / jcounts
            nll = float(np.mean(jm[jcounts > 0]))
            if nll < best_n:
                best_n, best_s = nll, s
        return float(best_s), float(best_n)

    results = {}
    for ceiling in (1.6, 3.0, 5.0):
        grid_c = np.round(np.arange(0.05, ceiling + 0.001, 0.01), 2)
        cal = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            sc_global, _ = _scan_c(other, grid_c)
            by_scaf = defaultdict(dict)
            for rid, p in other.items():
                by_scaf[int(p["scaf"])][rid] = p
            scaf_c = {}
            for sc, rows_sc in by_scaf.items():
                n_c = int(sum(1 for p in rows_sc.values() if p["cens"]))
                if n_c >= 15:
                    s, _ = _scan_c(rows_sc, grid_c)
                    scaf_c[sc] = s
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    sc = int(p["scaf"])
                    sig = scaf_c.get(sc, sc_global or p["sigma"])
                    cal[rid] = {**p, "sigma": float(sig)}
                else:
                    cal[rid] = p
        nll = _pooled(cal)
        strata = _pooled_strata(cal)
        results[str(ceiling)] = {"nll": round(nll, 4),
                                 "strata": {k: round(v, 4) for k, v in strata.items()},
                                 "delta": round(nll - 0.7243, 4)}
        print(f"ceiling={ceiling}: pooled={nll:.4f} "
              f"strata={ {k: round(v,4) for k,v in strata.items()} } "
              f"delta={nll-0.7243:+.4f}")

    out = {"r62": round(_pooled(cal62), 4), "results": results,
           "note": "extend sigma_c ceiling for censored layer (scaf8 wants >1.6)"}
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r78_sigma_c_ceiling.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
