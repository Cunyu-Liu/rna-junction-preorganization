"""r73: row-level sigma regressed on TRAIN-LEGAL features (honest analog of r70).

r70/r72 showed a mechanical -0.0048 gain by setting per-row sigma from err10,
but err10 is label-derived (train_legal=False), so that gain is leakage.  The
honest question: can a per-row sigma learned from TRAIN-LEGAL features (Vienna
21-D + nuisance basis) reproduce any of that gain?  r55 tested motif->per-fold
sigma regression (NEGATIVE) and r66/r60 tested context-level sigma from
residuals (NEGATIVE beyond scaf x stratum), but a ROW-level GBDT sigma head on
the full feature set was never explicitly tested.

Design (honest, LOO, no label-derived features):
  - mu = r62-corrected mu (frozen).
  - For each held-out fold: on OTHER folds' measured rows, train a GBDT that
    predicts log |residual| from (Vienna21 + nuisance) features.
  - sigma_m(row) = max(clip(exp(log|r|_pred) * scale, 0.05, 2.0)) where scale
    is a single fitted multiplier optimizing macro NLL on the fit rows.
  - Apply to held-out fold measured rows; censored rows keep r62 sigma.
  - Compare pooled NLL vs r62 (0.7243).  If NEGATIVE, the sigma side is fully
    closed with legal features only.
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


def _nuisance_feats(p):
    """Train-legal nuisance basis: [1, motif? context? scaf one-hot, topology(3)].
    Context identity is NOT used (OOV under joint-blocked split).  Use scaf
    one-hot + topology (junction length, part lengths) only, matching the
    base-model nuisance basis minus motif/context."""
    f = [1.0, float(int(p["scaf"])) / 9.0]
    # topology unavailable from prediction rows; use scaf + a constant.
    # We also add per-scaf one-hot (9 dims).
    f = [0.0] * 11
    f[0] = 1.0
    f[1 + int(p["scaf"])] = 1.0
    return np.asarray(f[:11], dtype=float)


def _vienna_feats(rid, canon_lines):
    idx = int(rid)
    if not (0 <= idx < len(canon_lines)):
        return None
    rec = canon_lines[idx]
    # train-legal structural features: dg_fold/dg_fold_constrained (not dg10 family)
    out = []
    for k in ("dg_fold", "dg_fold_constrained"):
        v = rec.get(k)
        try:
            out.append(float(v) if v not in (None, "") else 0.0)
        except (TypeError, ValueError):
            out.append(0.0)
    h = str(rec.get("helix_seq", ""))
    j = str(rec.get("junction_seq", ""))
    out.append(float(len(h)))
    out.append(float(len(j)))
    out.append((h.count("G") + h.count("C")) / max(len(h), 1))
    out.append((j.count("G") + j.count("C")) / max(len(j), 1))
    # 3D: arm lengths
    parts = [t for t in j.split("_") if t]
    while len(parts) < 2:
        parts.append("")
    out.append(float(len(parts[0])))
    out.append(float(len(parts[1])))
    return np.asarray(out[:8], dtype=float)


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
    print("r62 baseline =", round(_pooled(cal62), 4))

    by_fold = defaultdict(dict)
    for rid, p in cal62.items():
        by_fold[p["fold"]][rid] = {**p}

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

    def _fit_sigma_head(f, model_factory):
        """Train row-sigma model on OTHER folds' measured rows (legal features)."""
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        fit = []
        for rid, p in other.items():
            if p["cens"]:
                continue
            vf = _vienna_feats(rid, canon_lines)
            if vf is None:
                continue
            nf = _nuisance_feats(p)
            fit.append((rid, p, np.concatenate([nf, vf])))
        if len(fit) < 50:
            return None, None
        X = np.asarray([t[2] for t in fit])
        resid = np.asarray([abs(t[1]["y"] - t[1]["mu"]) for t in fit])
        jid = np.asarray([t[1]["jid"] for t in fit])
        y = np.log(np.maximum(resid, 1e-3))
        mdl = model_factory().fit(X, y)
        # pick best scale on fit rows
        pred = np.exp(mdl.predict(X))
        best_scale, best_n = 1.0, np.inf
        for s in scales:
            sig = np.clip(pred * s, 0.05, 2.0)
            losses = row_nll(np.zeros(len(fit), dtype=float),
                             np.zeros(len(fit), dtype=bool),
                             np.zeros_like(sig), sig)
            # NOTE: for scale selection we use residual magnitude proxy NLL at
            # zero mu? No -- we need actual |resid| as the "error" for fitting.
            # Use: -log N(y-mu; 0, sigma) with y-mu = resid-ish.  Better: fit on
            # the actual residual directly.
            # Recompute properly: losses using signed residual as observed.
            r_obs = np.asarray([t[1]["y"] - t[1]["mu"] for t in fit])
            losses = row_nll(r_obs, np.zeros(len(fit), dtype=bool),
                             np.zeros_like(sig), sig)
            uniq, jc = np.unique(jid, return_inverse=True)
            sums = np.bincount(jc, weights=losses, minlength=len(uniq))
            cnt = np.bincount(jc, minlength=len(uniq))
            n = float(np.mean(sums[cnt > 0] / cnt[cnt > 0]))
            if n < best_n:
                best_n, best_scale = n, s
        return mdl, best_scale

    out_nll = {}
    for name, factory in (("GBDT", lambda: GradientBoostingRegressor(
                              n_estimators=200, learning_rate=0.05, max_depth=2,
                              random_state=0)),
                          ("Ridge", lambda: Ridge(alpha=5.0))):
        cal = {}
        fit_success = 0
        for f in folds:
            mdl, scale = _fit_sigma_head(f, factory)
            if mdl is None:
                for rid, p in by_fold[f].items():
                    cal[rid] = p
                continue
            fit_success += 1
            for rid, p in by_fold[f].items():
                if p["cens"]:
                    cal[rid] = p
                    continue
                vf = _vienna_feats(rid, canon_lines)
                if vf is None:
                    cal[rid] = p
                    continue
                nf = _nuisance_feats(p)
                x = np.concatenate([nf, vf]).reshape(1, -1)
                sig = float(np.clip(np.exp(mdl.predict(x)[0]) * scale, 0.05, 2.0))
                cal[rid] = {**p, "sigma": sig}
        nll = _pooled(cal)
        out_nll[name] = round(nll, 4)
        print(f"[{name}] row-sigma legal features: {nll:.4f} "
              f"(delta vs r62 {nll-0.7243:+.4f}, folds_fit={fit_success})")

    out = {
        "r62": 0.7243,
        "row_sigma_GBDT": out_nll.get("GBDT"),
        "row_sigma_Ridge": out_nll.get("Ridge"),
        "note": ("row-level sigma from train-legal Vienna+nuisance features, "
                 "GBDT/Ridge on log|residual| with LOO scale fit; honest analog "
                 "of r70's err10-driven sigma WITHOUT label-derived features"),
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r73_row_sigma_legal_features.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
