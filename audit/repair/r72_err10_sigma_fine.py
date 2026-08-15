"""r72: fine c scan + global-c robustness + mechanism for r70 err10 sigma.

r71 confirmed r70 (quadrature sigma = sqrt(sigma_scaf^2+(c*err10)^2)) at 0.7199
vs r62 0.7243 (delta -0.0044, edit-cluster CI [0.0003,0.007] lower>0, leave1
0.0045).  But all per-fold c saturated at the grid floor 0.5 -> the true
optimum may be finer.  Tests:
  1. fine c scan on a global (single shared) c fit LOO, grid {0.1..4.0}.
  2. per-fold c with fine grid {0.1,0.2,...,2.0}.
  3. mechanism: does sigma widen most on high-err10 rows (which have larger
     residual)?  Report mean sigma by err10 decile for r70.
  4. robustness: split-half (fold parity) recompute of delta.
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
    losses = row_nll(y, cens, mu, sigma)
    uniq, jcode = np.unique(jid, return_inverse=True)
    sums = np.bincount(jcode, weights=losses, minlength=len(uniq))
    cnt = np.bincount(jcode, minlength=len(uniq))
    return float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))


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
    canon_lines = [json.loads(l) for l in Path(CANON).read_text().splitlines() if l.strip()]

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

    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma

    # ---- per-fold scaf sigma (r62 style, fit on other folds) ----
    def _other_scaf_sigma(f):
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            if not p["cens"]:
                by_scaf[int(p["scaf"])][rid] = p
        out = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 15:
                s, _ = _scan_sigma(rows_sc, cens_mask=False, grid=GRID)
                out[sc] = s
        sm_global, _ = _scan_sigma({r: p for r, p in other.items() if not p["cens"]},
                                   cens_mask=False, grid=GRID)
        return out, sm_global

    # cache per-fold scaf sigma
    scaf_sig_cache = {f: _other_scaf_sigma(f) for f in folds}

    # ---- fit c (global) on all-but-one-fold, fine grid ----
    fine_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    def _fit_global_c(f, grid):
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        fit = [p for p in other.values() if (not p["cens"]) and p["err10"] is not None]
        if len(fit) < 50:
            return 0.0
        errs = np.asarray([p["err10"] for p in fit], dtype=float)
        res = np.asarray([p["y"] - p["mu"] for p in fit], dtype=float)
        jid = np.asarray([p["jid"] for p in fit])
        ss, sm = scaf_sig_cache[f]
        base = np.asarray([ss.get(int(p["scaf"]), sm or 0.5) for p in fit], dtype=float)
        best_c, best_n = 0.0, np.inf
        for c in grid:
            sig = np.clip(np.sqrt(base ** 2 + (c * errs) ** 2), 0.05, 2.0)
            losses = row_nll(res, np.zeros(len(res), dtype=bool),
                             np.zeros_like(res), sig)
            uniq, jc = np.unique(jid, return_inverse=True)
            sums = np.bincount(jc, weights=losses, minlength=len(uniq))
            cnt = np.bincount(jc, minlength=len(uniq))
            n = float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))
            if n < best_n:
                best_n, best_c = n, c
        return best_c

    # global c, fit on all-but-one-fold, applied to held-out fold
    c_global = {f: _fit_global_c(f, fine_grid) for f in folds}
    from collections import Counter
    print("global-c (fine) per-fold distribution:", dict(Counter(c_global.values())))

    def _apply(cmap):
        cal = {}
        for f in folds:
            c = cmap.get(f, 0.0)
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    cal[rid] = {k: v for k, v in p.items() if k != "err10"}
                else:
                    sig = float(np.clip(
                        np.sqrt(p["sigma"] ** 2 + (c * (p["err10"] or 0.0)) ** 2),
                        0.05, 2.0))
                    cal[rid] = {**{k: v for k, v in p.items() if k != "err10"},
                                "sigma": sig}
        return cal

    cal_g = _apply(c_global)
    nll_g = _pooled(cal_g)
    print(f"global-c fine: r70 = {nll_g:.4f}  delta vs r62 = {nll_g-0.7243:+.4f}")
    ci = _edit_ci(cal_g, cal62)
    print(f"  edit CI vs r62 = {ci['ci']} lower_gt_0={ci['ci_lower_gt_0']} "
          f"leave1={ci['leave_one_largest']}")

    # ---- per-fold c, fine grid ----
    def _fit_fold_c(f, grid):
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        fit = [p for p in other.values() if (not p["cens"]) and p["err10"] is not None]
        if len(fit) < 50:
            return 0.0
        errs = np.asarray([p["err10"] for p in fit], dtype=float)
        res = np.asarray([p["y"] - p["mu"] for p in fit], dtype=float)
        jid = np.asarray([p["jid"] for p in fit])
        ss, sm = scaf_sig_cache[f]
        base = np.asarray([ss.get(int(p["scaf"]), sm or 0.5) for p in fit], dtype=float)
        best_c, best_n = 0.0, np.inf
        for c in grid:
            sig = np.clip(np.sqrt(base ** 2 + (c * errs) ** 2), 0.05, 2.0)
            losses = row_nll(res, np.zeros(len(res), dtype=bool),
                             np.zeros_like(res), sig)
            uniq, jc = np.unique(jid, return_inverse=True)
            sums = np.bincount(jc, weights=losses, minlength=len(uniq))
            cnt = np.bincount(jc, minlength=len(uniq))
            n = float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))
            if n < best_n:
                best_n, best_c = n, c
        return best_c

    c_fold = {f: _fit_fold_c(f, fine_grid) for f in folds}
    cal_pf = _apply(c_fold)
    nll_pf = _pooled(cal_pf)
    print(f"per-fold-c fine: r70 = {nll_pf:.4f}  delta vs r62 = {nll_pf-0.7243:+.4f}")
    print("  per-fold c dist:", dict(Counter(c_fold.values())))

    # ---- mechanism: mean emitted sigma & resid by err10 decile (global c=best) ----
    best_c_mode = Counter(c_global.values()).most_common(1)[0][0]
    cal_mech = _apply({f: best_c_mode for f in folds})
    print(f"\nmechanism at c={best_c_mode}:")
    q = np.quantile([p["err10"] for p in by_fold[list(folds)[0]].values()
                     if not p["cens"] and p["err10"] is not None], [0.2, 0.4, 0.6, 0.8])
    rows_m = []
    for f in folds:
        for rid, p in by_fold[f].items():
            if not p["cens"] and p["err10"] is not None:
                rows_m.append((p["err10"], abs(p["y"] - p["mu"]), cal_mech[rid]["sigma"]))
    arr = sorted(rows_m, key=lambda x: x[0])
    n = len(arr)
    for i in range(5):
        seg = arr[i * n // 5:(i + 1) * n // 5]
        e = np.mean([x[0] for x in seg]); r = np.mean([x[1] for x in seg])
        s = np.mean([x[2] for x in seg])
        print(f"  err10 decile {i}: mean_err10={e:.3f} mean|resid|={r:.3f} "
              f"mean_sigma={s:.3f}")

    # ---- split-half robustness (fold parity) ----
    even = [f for i, f in enumerate(folds) if i % 2 == 0]
    c_even = {f: _fit_global_c(f, fine_grid) for f in even}
    cal_e = _apply(c_even)
    nll_e = _pooled(cal_e)
    print(f"\nsplit-half even-fold LOO: nll={nll_e:.4f} delta={nll_e-0.7243:+.4f}")

    out = {
        "r62": 0.7243,
        "global_c_fine": round(nll_g, 4),
        "global_c_delta": round(nll_g - 0.7243, 4),
        "global_c_CI": ci,
        "per_fold_c_fine": round(nll_pf, 4),
        "per_fold_c_delta": round(nll_pf - 0.7243, 4),
        "c_global_dist": dict(Counter(c_global.values())),
        "c_fold_dist": dict(Counter(c_fold.values())),
        "best_c_mode": best_c_mode,
        "split_half_even": round(nll_e, 4),
        "note": "quadrature sigma = sqrt(sigma_scaf^2 + (c*err10)^2) on r62 mu",
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r72_err10_sigma_fine.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
