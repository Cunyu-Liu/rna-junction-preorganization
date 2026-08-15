"""r70: per-row err10-driven sigma calibration (genuinely untested lever).

r68/r69 showed the measured residual (sd 0.548) exceeds err10 (rms 0.248), and
the gap is dominated by per-context random effects (r67: NOT feature-
predictable).  BUT err10 itself is a per-row measurement error available at
inference time, and it shows a real monotonic relationship with |residual|
(mean|r|: 0.36 at err10~0.07 -> 0.49 at err10>0.3).  The frozen r62 emits a
per-scaf x stratum sigma_m that is CONSTANT within (scaf, stratum) -- it never
uses per-row err10.  If err10 predicts which rows have genuinely larger
predictive uncertainty, a per-row sigma = f(err10) should improve Gaussian NLL.

Design (honest, LOO):
  - Base mu = r62-corrected mu (frozen).
  - For each held-out fold: on OTHER folds' measured rows, fit sigma_m(row) as
    a function of err10, two families:
      (A) affine:      sigma = clip(a + b*err10, 0.05, 2.0)
      (B) quadrature:  sigma = sqrt(sigma_scaf^2 + (c*err10)^2) with sigma_scaf
                       = r62 decoupled per-scaf sigma (kept), only c fitted.
  - (A)/(B) coefficients fit on OTHER folds by minimizing pooled junction-macro
    NLL (grid search over small parameter grids), then applied to held-out fold.
  - Compare pooled NLL vs r62 (0.7243).  Report edit-cluster CI.
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
CANON = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"


def _pooled_macro(y, cens, mu, sigma, jid):
    """Pooled junction-macro NLL (same aggregation as the MetricSpec)."""
    losses = row_nll(y, cens, mu, sigma)
    uniq, jcode = np.unique(jid, return_inverse=True)
    sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
    cnt = np.bincount(jcode, minlength=len(uniq))
    return float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))


def _fit_affine(errs, res, cens, jid, grid_a, grid_b):
    """Fit sigma=clip(a+b*err) minimizing macro NLL on the fit rows."""
    best = None
    best_n = np.inf
    for a in grid_a:
        for b in grid_b:
            sig = np.clip(a + b * errs, 0.05, 2.0)
            n = _pooled_macro(res, cens, np.zeros_like(res), sig, jid)
            if n < best_n:
                best_n, best = n, (a, b)
    return best, best_n


def _fit_quad(errs, scaf, res, cens, jid, scaf_sigma, grid_c):
    """Fit sigma=sqrt(scaf_sigma[scaf]^2 + (c*err)^2), only c fitted."""
    best = None
    best_n = np.inf
    base = np.asarray([scaf_sigma[s] for s in scaf], dtype=float)
    for c in grid_c:
        sig = np.sqrt(base ** 2 + (c * errs) ** 2)
        sig = np.clip(sig, 0.05, 2.0)
        n = _pooled_macro(res, cens, np.zeros_like(res), sig, jid)
        if n < best_n:
            best_n, best = n, c
    return best, best_n


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

    canon_lines = [json.loads(l) for l in Path(CANON).read_text().splitlines() if l.strip()]

    # per-row err10
    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        idx = int(rid)
        e = None
        if 0 <= idx < len(canon_lines):
            try:
                e = float(canon_lines[idx]["err10"]) if canon_lines[idx].get("err10") not in (None, "") else None
            except (TypeError, ValueError):
                e = None
        by_fold[p["fold"]][rid] = {**p, "err10": e}

    grid_a = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
    grid_b = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
    grid_c = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    out_aff = {}
    out_quad = {}
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        # r62 decoupled per-scaf sigma_m for measured rows (fit on other folds)
        from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            if not p["cens"]:
                by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 15:
                s, _ = _scan_sigma(rows_sc, cens_mask=False, grid=GRID)
                scaf_sigma[sc] = s
        sm_global, _ = _scan_sigma({r: p for r, p in other.items() if not p["cens"]},
                                   cens_mask=False, grid=GRID)
        # fit rows = measured with err10
        fit = [p for p in other.values() if (not p["cens"]) and p["err10"] is not None]
        if len(fit) < 50:
            continue
        errs = np.asarray([p["err10"] for p in fit], dtype=float)
        res = np.asarray([p["y"] - p["mu"] for p in fit], dtype=float)
        jid = np.asarray([p["jid"] for p in fit])
        cens = np.zeros(len(fit), dtype=bool)
        scaf = np.asarray([int(p["scaf"]) for p in fit])
        ss = {s: scaf_sigma.get(s, sm_global or 0.5) for s in set(scaf.tolist())}
        # A: affine
        (a, b), _ = _fit_affine(errs, res, cens, jid, grid_a, grid_b)
        # B: quadrature on r62 scaf sigma
        c, _ = _fit_quad(errs, scaf, res, cens, jid, ss, grid_c)
        out_aff[f] = (a, b)
        out_quad[f] = c

    def _apply(scheme):
        cal = {}
        for f in folds:
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    cal[rid] = {k: v for k, v in p.items() if k != "err10"}
                    continue
                if scheme == "aff":
                    a, b = out_aff.get(f, (0.55, 0.0))
                    sig = float(np.clip(a + b * (p["err10"] or 0.0), 0.05, 2.0))
                else:  # quad
                    c = out_quad.get(f, 0.0)
                    base = p["sigma"]
                    sig = float(np.clip(np.sqrt(base ** 2 + (c * (p["err10"] or 0.0)) ** 2),
                                        0.05, 2.0))
                cal[rid] = {**{k: v for k, v in p.items() if k != "err10"},
                            "sigma": sig}
        return cal

    cal_aff = _apply("aff")
    cal_quad = _apply("quad")
    nll_aff = _pooled(cal_aff)
    nll_quad = _pooled(cal_quad)
    print(f"r70 affine err10 sigma  = {nll_aff:.4f} (vs r62 delta {nll_aff-0.7243:+.4f})")
    print(f"r70 quadrature err10 sigma = {nll_quad:.4f} (vs r62 delta {nll_quad-0.7243:+.4f})")

    # per-scaf sigma comparison for affine scheme
    bysc = defaultdict(list)
    for rid, p in cal_aff.items():
        if not p["cens"]:
            bysc[int(p["scaf"])].append(p["sigma"])
    print("\nper-scaf affine sigma_m:")
    for sc in sorted(bysc):
        print(f"  scaf{sc}: mean={np.mean(bysc[sc]):.3f} (err10-driven, row-varying)")

    out = {
        "r62_baseline": round(_pooled(cal62), 4),
        "r70_affine_err10": round(nll_aff, 4),
        "r70_quadrature_err10": round(nll_quad, 4),
        "aff_delta_vs_r62": round(nll_aff - 0.7243, 4),
        "quad_delta_vs_r62": round(nll_quad - 0.7243, 4),
        "grids": {"affine_a": grid_a, "affine_b": grid_b, "quad_c": grid_c},
        "note": ("per-row err10 sigma; affine = clip(a+b*err10), quad = "
                 "sqrt(sigma_scaf^2+(c*err10)^2) with r62 decoupled scaf sigma"),
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r70_err10_sigma.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
