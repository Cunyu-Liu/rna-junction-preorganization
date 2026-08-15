"""r80: apply r56b context EB mu alpha to CENSORED rows too (untested).

r56b's per-context EB mu correction applies ONLY to measured rows; censored rows
keep the raw ensemble mu.  r78/r79 discovered scaf8 censored rows have mu
systematically BELOW CAP (median -8.01 vs CAP -7.1) -- the model believes these
junctions are stable (below CAP) but the data censored them.  If contexts have a
real bias that r56b corrects for measured rows, the SAME context bias should
apply to censored rows of that context (their mu is just as biased).  This is
the model-side fix (better mu) vs r78's sigma inflation (abstention hedge).

Design (honest LOO):
  - Stage 1: r56b per-context EB mu on OTHER folds (kappa=1, min_meas=3),
    now applied to BOTH measured and censored rows of the held-out fold.
  - Stage 2: r62 decoupled sigma re-scan (both strata) on corrected mu.
  - Compare pooled NLL vs r62 (0.7243) + edit-cluster CI + censored stratum.
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
    _load, _elig, _by_rid, _pooled, GRID, _scan_sigma,
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


def _calibrate_r80(ens, folds, kappa=1.0, min_meas=3, grid=GRID):
    """r56b mu (applied to BOTH strata) + r62 decoupled sigma re-scan."""
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _calibrate_r51
    cal_r51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)
    by_fold = defaultdict(dict)
    for rid, p in cal_r51.items():
        by_fold[p["fold"]][rid] = p

    cal = {}
    fit_log = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        # per-context residual (measured rows only -> alpha estimand)
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
        # apply alpha to BOTH measured and censored rows of held-out fold
        corr_other = {}
        for rid, p in other.items():
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
                entry["sigma_m"] = _scan_sigma(rows_sc, cens_mask=False, grid=grid)[0]
            else:
                entry["sigma_m"] = sm_global if sm_global is not None else s_global
            if n_c >= 15:
                entry["sigma_c"] = _scan_sigma(rows_sc, cens_mask=True, grid=grid)[0]
            else:
                entry["sigma_c"] = sc_global if sc_global is not None else s_global
            strat_sigma[sc] = entry
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            e = strat_sigma.get(sc, {})
            if p["cens"]:
                sig = e.get("sigma_c", s_global)
            else:
                sig = e.get("sigma_m", s_global)
            alpha = alpha_ctx.get((sc, str(p["context"])), b_scaf.get(sc, 0.0))
            mu_new = float(p["mu"] + alpha)
            cal[rid] = {**p, "mu": mu_new, "sigma": float(sig)}
        fit_log[f] = {"n_ctx_alpha": len(alpha_ctx)}
    return cal, fit_log


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

    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma as _ss  # noqa: F401

    for kappa in (1.0, 2.0):
        cal, _ = _calibrate_r80(ens, folds, kappa=kappa, min_meas=3)
        nll = _pooled(cal)
        strata = _pooled_strata(cal)
        print(f"r80 (ctx alpha on BOTH strata) kappa={kappa}: {nll:.4f} "
              f"strata={ {k: round(v,4) for k,v in strata.items()} } "
              f"delta={nll-0.7243:+.4f}")
        if kappa == 1.0:
            ci = _edit_ci(cal, cal62)
            print(f"  edit CI vs r62 = {ci['ci']} lower_gt_0={ci['ci_lower_gt_0']} "
                  f"leave1={ci['leave_one_largest']}")
            # scaf8 censored mu before/after
            for tag, cc in (("r62", cal62), ("r80", cal)):
                mu8 = [p["mu"] for p in cc.values() if p["cens"] and int(p["scaf"]) == 8]
                print(f"  {tag} scaf8 censored mu: med={np.median(mu8):.3f} "
                      f"(CAP={-7.1})")

    out = {"r62": round(_pooled(cal62), 4),
           "note": "r80: apply r56b context EB mu to censored rows too (model-side fix vs r78 sigma hedge)"}
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r80_ctx_mu_on_censored.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
