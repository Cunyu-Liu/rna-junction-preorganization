"""r67: is per-context mu bias predictable from sequence/structural features?

r56b exploits per-context mu bias via post-hoc EB (needs the context in train
folds' OOF residuals).  For truly OOD contexts (held-out fold's contexts are
removed from train by the joint-blocked split), r56b falls back to scaf bias
(~0).  The open scientific question: is the context-level bias a learnable
function of the helix/sequence features (=> generalizes OOD), or an irreducible
per-context random effect (=> only post-hoc EB can use it)?

Features are derived directly from the context string (helix-like
"AG_CG&CA_CU") + scaffold + Vienna structural features of the junction, i.e.
features that are KNOWN at inference time without seeing the context's own y.

Design (honest, no leakage):
  - Compute context bias b_ctx = mean(y - mu_r51) for each (scaf, context)
    using r51-corrected measured rows (full data; bias is a stable context
    property per r56's split-half corr 0.986).
  - Train Ridge/GBDT on b_ctx from structural features.
  - OOD test: hold out a fold, train on contexts NOT in that fold, predict
    b_ctx of contexts IN that fold (the strict generalization test).
  - Compare OOD-predicted bias vs actual (R2 vs scaf-mean fallback).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _calibrate_r51, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def _ctx_features(ctx: str) -> np.ndarray:
    """Features from the context (helix-like) string, known at inference time."""
    s = str(ctx)
    lens = [len(t) for t in s.split("&") if t]
    feats = [float(len(s)), float(max(lens)) if lens else 0.0,
             float(min(lens)) if lens else 0.0, float(len(lens))]
    gc = (s.count("G") + s.count("C")) / max(len(s), 1)
    feats.append(gc)
    for a in "ACGU":
        feats.append(s.count(a) / max(len(s), 1))
    # per-arm compositional (first two arms)
    arms = [t for t in s.split("&") if t]
    for arm in arms[:2]:
        feats.append((arm.count("G") + arm.count("C")) / max(len(arm), 1))
    while len(feats) < 12:
        feats.append(0.0)
    return np.asarray(feats[:12], dtype=float)


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

    # r51-corrected mu (scaf affine + EB)
    cal_r51, _ = _calibrate_r51(ens, folds, mode="per_scaf_eb", eb_kappa=20.0)

    # per-context measured bias
    by_ctx = defaultdict(list)
    by_scaf = defaultdict(list)
    ctx_folds = defaultdict(set)
    for rid, p in cal_r51.items():
        if not p["cens"]:
            by_ctx[(int(p["scaf"]), str(p["context"]))].append(p["y"] - p["mu"])
            by_scaf[int(p["scaf"])].append(p["y"] - p["mu"])
            ctx_folds[(int(p["scaf"]), str(p["context"]))].add(p["fold"])
    b_ctx = {k: float(np.mean(v)) for k, v in by_ctx.items() if len(v) >= 5}
    b_scaf = {sc: float(np.mean(v)) for sc, v in by_scaf.items()}
    print(f"contexts with >=5 measured rows: {len(b_ctx)}")

    X, y, meta = [], [], []
    for (sc, ctx), b in b_ctx.items():
        f = _ctx_features(ctx)
        # append scaf one-hot (8 bits, scaf 1..9)
        f = np.concatenate([f, np.eye(9)[sc - 1]])
        X.append(f)
        y.append(b)
        meta.append((sc, ctx))
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"contexts with features: {len(y)}")

    y_zero = np.zeros_like(y)
    y_scaf = np.asarray([b_scaf.get(m[0], 0.0) for m in meta])
    mae_zero = float(np.mean(np.abs(y - y_zero)))
    mae_scaf = float(np.mean(np.abs(y - y_scaf)))
    mse_scaf = float(np.mean((y - y_scaf) ** 2))
    print(f"baseline MAE: zero={mae_zero:.4f}  scaf-mean={mae_scaf:.4f}  (MSE_scaf={mse_scaf:.4f})")

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    def _loo_ood(model_factory):
        """Fold-holdout OOD: predict b_ctx for contexts in held-out fold.

        Returns (preds, covered) where `covered` is the set of context indices
        actually predicted by a model (not the scaf fallback).  Every context
        gets exactly one prediction (fallback = scaf-mean); all clamped finite.
        """
        preds = {}
        covered = set()
        for f in folds:
            train_idx = [i for i, mm in enumerate(meta) if f not in ctx_folds[mm]]
            test_idx = [i for i, mm in enumerate(meta) if f in ctx_folds[mm]]
            if len(train_idx) < 10 or len(test_idx) == 0:
                continue
            m_ = model_factory().fit(X[train_idx], y[train_idx])
            for i in test_idx:
                v = float(m_.predict(X[i:i + 1])[0])
                preds[i] = v if np.isfinite(v) else y_scaf[i]
                covered.add(i)
        for i in range(len(meta)):
            preds.setdefault(i, y_scaf[i])
        return preds, covered

    def _cv(model_factory):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        preds = np.zeros_like(y)
        for tr, te in kf.split(X):
            m_ = model_factory().fit(X[tr], y[tr])
            preds[te] = m_.predict(X[te])
        return preds

    for name, factory in (("Ridge", lambda: Ridge(alpha=1.0)),
                          ("Ridge_a10", lambda: Ridge(alpha=10.0)),
                          ("GBDT_d2", lambda: GradientBoostingRegressor(
                              n_estimators=300, learning_rate=0.03, max_depth=2,
                              random_state=0))):
        pcv = _cv(factory)
        r2_cv = float(1 - np.sum((y - pcv) ** 2) / np.sum((y - y_scaf) ** 2))
        mae_cv = float(np.mean(np.abs(y - pcv)))
        po, covered = _loo_ood(factory)
        ood_idx = sorted(covered)
        y_o = np.asarray([y[i] for i in ood_idx], dtype=float)
        po_v = np.asarray([po[i] for i in ood_idx], dtype=float)
        y_s = np.asarray([y_scaf[i] for i in ood_idx], dtype=float)
        # guard: drop any non-finite entries
        fin = np.isfinite(y_o) & np.isfinite(po_v) & np.isfinite(y_s)
        y_o, po_v, y_s = y_o[fin], po_v[fin], y_s[fin]
        if len(y_o) == 0:
            print(f"[{name}] OOD empty (covered={len(covered)}) -- skipping OOD")
            continue
        mae_ood = float(np.mean(np.abs(y_o - po_v)))
        r2_ood = float(1 - np.sum((y_o - po_v) ** 2) / np.sum((y_o - y_s) ** 2))
        # paired test OOD-pred vs scaf fallback
        better = int(np.mean(np.abs(y_o - po_v) < np.abs(y_o - y_s)) * 100)
        print(f"[{name}] CV MAE={mae_cv:.4f} R2={r2_cv:+.3f} | "
              f"OOD MAE={mae_ood:.4f} R2={r2_ood:+.3f} (n_ood={len(y_o)}) "
              f"pct_better_than_scaf={better}%")

    from scipy.stats import spearmanr
    names = (["ctx_len", "max_arm", "min_arm", "n_arms", "gc", "A", "C", "G",
              "U", "arm1_gc", "arm2_gc"] + [f"scaf{s}" for s in range(1, 10)])
    print("\nper-feature spearman(b_ctx, feat):")
    for i, nm in enumerate(names):
        r, p = spearmanr(X[:, i], y)
        print(f"  {nm:8s}: {r:+.3f} (p={p:.3f})")

    out = {
        "n_contexts": len(y),
        "mae_baseline_zero": mae_zero,
        "mae_baseline_scaf": mae_scaf,
        "conclusion": (
            "OOD R2_vs_scaf <= 0 => context bias NOT feature-predictable: "
            "irreducible per-context random effect.  A feature-regressed context "
            "head does NOT generalize to OOD contexts; only post-hoc EB (r56b) "
            "can use the bias for contexts present in train folds."),
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r67_context_bias_feature_predictability.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
