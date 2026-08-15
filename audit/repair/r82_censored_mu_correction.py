"""r82: censored-specific per-scaf mu correction (fix scaf8 censored mu error).

r78/r79 showed scaf8 censored rows want sigma_c->inf, which masks a REAL mu
error: the model predicts mu median -8.0 for scaf8 censored rows, but right
censoring means true value >= CAP=-7.1 (LESS stable).  The model over-predicts
stability for these rows.  r80 tried applying the MEASURED-row context alpha to
censored rows (wrong signal, NEGATIVE).  The untested fix: a censored-specific
per-scaf mu shift, fit on OTHER folds' censored rows by maximizing the SURVIVAL
likelihood (log Phi((mu-CAP)/sigma)), with EB shrinkage so mu doesn't blow up.

Design (honest LOO):
  - mu = r62-corrected mu (frozen).
  - For each held-out fold: on OTHER folds' censored rows, fit per-scaf delta_c
    (mu shift) and sigma_c together via grid search maximizing survival NLL with
    shrinkage toward the measured-mu scale.  Apply to held-out censored rows.
  - Measured rows unchanged (r62 mu/sigma).  Compare pooled NLL vs r62.
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
    _load, _elig, _by_rid, _pooled, _scan_sigma, GRID,
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


def _surv_nll(y, cens, mu, sigma):
    """Right-censored NLL for the given rows (censored only)."""
    losses = row_nll(y, cens, mu, sigma)
    uniq, jcode = np.unique(np.asarray(y, dtype=object), return_inverse=True)
    return float(np.mean(losses))


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

    # censored-specific per-scaf mu+sigma fit on OTHER folds, applied to held-out
    d_grid = np.round(np.arange(-0.8, 0.81, 0.05), 2)   # mu shift toward/away from CAP
    s_grid = np.round(np.arange(0.05, 1.61, 0.05), 2)   # sigma_c grid
    out = {}
    for kappa in (5.0, 20.0, 50.0):
        cal = {}
        for f in folds:
            other = {}
            for ff in folds:
                if ff != f:
                    other.update(by_fold[ff])
            cens_rows = [p for p in other.values() if p["cens"]]
            # global censored mean mu (shrinkage anchor)
            gmu_c = float(np.mean([p["mu"] for p in cens_rows])) if cens_rows else -8.0
            # per-scaf fit
            by_scaf_c = defaultdict(list)
            for p in cens_rows:
                by_scaf_c[int(p["scaf"])].append(p)
            scaf_params = {}
            for sc, rows in by_scaf_c.items():
                n = len(rows)
                if n < 10:
                    continue
                mu = np.asarray([p["mu"] for p in rows])
                y = np.asarray([p["y"] for p in rows])
                cens = np.ones(n, dtype=bool)
                # anchor: current sigma_c from r62 (fallback global)
                best = None
                best_n = np.inf
                for d in d_grid:
                    mu_c = mu + d
                    for s in s_grid:
                        losses = row_nll(y, cens, mu_c, np.full(n, float(s)))
                        nll = float(np.mean(losses))
                        if nll < best_n:
                            best_n, best = nll, (float(d), float(s))
                # EB shrink d toward 0 with kappa (n/(n+kappa) weight)
                d_opt, s_opt = best
                w = n / (n + kappa)
                d_sh = w * d_opt  # shrink shift toward 0 (no change)
                scaf_params[sc] = (d_sh, s_opt)
            # global fallback params (from all censored rows)
            if cens_rows:
                mu_all = np.asarray([p["mu"] for p in cens_rows])
                y_all = np.asarray([p["y"] for p in cens_rows])
                cens_all = np.ones(len(cens_rows), dtype=bool)
                best_g = None
                best_gn = np.inf
                for d in d_grid:
                    for s in s_grid:
                        losses = row_nll(y_all, cens_all, mu_all + d, np.full(len(y_all), float(s)))
                        nll = float(np.mean(losses))
                        if nll < best_gn:
                            best_gn, best_g = nll, (float(d), float(s))
            else:
                best_g = (0.0, 0.5)
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    sc = int(p["scaf"])
                    prm = scaf_params.get(sc, (best_g[0] * 0.5, best_g[1]))
                    cal[rid] = {**p, "mu": float(p["mu"] + prm[0]),
                                "sigma": float(prm[1])}
                else:
                    cal[rid] = p
        nll = _pooled(cal)
        strata = _pooled_strata(cal)
        out[f"kappa{kappa:g}"] = {"nll": round(nll, 4),
                                  "strata": {k: round(v, 4) for k, v in strata.items()},
                                  "delta": round(nll - 0.7243, 4)}
        print(f"r82 censored-mu kappa={kappa}: pooled={nll:.4f} "
              f"strata={ {k: round(v,4) for k,v in strata.items()} } "
              f"delta={nll-0.7243:+.4f}")
        if kappa == 20.0:
            ci = _edit_ci(cal, cal62)
            print(f"  edit CI vs r62 = {ci['ci']} lower_gt_0={ci['ci_lower_gt_0']} "
                  f"leave1={ci['leave_one_largest']}")

    out["r62"] = round(_pooled(cal62), 4)
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r82_censored_mu_correction.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
