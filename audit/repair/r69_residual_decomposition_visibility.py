"""r69: decompose the measured-layer residual by context visibility + err10.

r68 showed residual sd (0.548) = 2.2x err10 (0.248).  But r67 proved context
bias is NOT feature-predictable (OOD R2 < 0), so the "0.49 implied feature sd"
is NOT reachable by a better predictor for OOD contexts.  The honest question:
how much of the residual is (a) reported measurement error (err10), (b) context
structure that r56b DOES recover for train-visible contexts, (c) irreducible
context structure for truly OOD contexts (held-out fold's contexts removed from
train by the joint-blocked split), and (d) remaining per-row noise?

Design:
  - For each measured row, tag context visibility: a context is "train-visible"
    if it appears in ANY fold other than the row's own fold (so r56b's EB alpha
    is estimable), else "OOD".
  - Compare residual sd after r62 for train-visible vs OOD contexts.
  - Compare each to err10 rms.  If OOD residual >> train-visible residual and
    both >> err10, the gap is irreducible context structure (only reachable via
    cross-fold EB, i.e. r56b) -- NOT feature-predictable headroom.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]

CANON = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"


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

    from audit.repair.r62_decoupled_frozen import _calibrate_r62
    cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)

    # canon err10 by line index
    canon_lines = [json.loads(l) for l in Path(CANON).read_text().splitlines() if l.strip()]

    # context -> set of folds
    ctx_folds = defaultdict(set)
    for rid, p in cal62.items():
        if not p["cens"]:
            ctx_folds[(int(p["scaf"]), str(p["context"]))].add(p["fold"])

    rows = []
    for rid, p in cal62.items():
        if p["cens"]:
            continue
        idx = int(rid)
        e = None
        if 0 <= idx < len(canon_lines):
            try:
                e = float(canon_lines[idx]["err10"]) if canon_lines[idx].get("err10") not in (None, "") else None
            except (TypeError, ValueError):
                e = None
        key = (int(p["scaf"]), str(p["context"]))
        n_folds = len(ctx_folds.get(key, set()))
        visible = n_folds >= 2 and len(ctx_folds[key] - {p["fold"]}) >= 1
        rows.append({"rid": rid, "jid": p["jid"], "scaf": int(p["scaf"]),
                     "context": str(p["context"]), "y": p["y"], "mu": p["mu"],
                     "sigma_m": p["sigma"], "err10": e,
                     "n_ctx_folds": n_folds, "visible": visible})
    print(f"measured rows: {len(rows)}; with err10: "
          f"{sum(1 for r in rows if r['err10'] is not None)}")
    print(f"train-visible contexts (r56b alpha estimable): "
          f"{sum(1 for r in rows if r['visible'])} rows, "
          f"{len({(r['scaf'], r['context']) for r in rows if r['visible']})} contexts")
    print(f"OOD contexts (held-out fold's own contexts): "
          f"{sum(1 for r in rows if not r['visible'])} rows, "
          f"{len({(r['scaf'], r['context']) for r in rows if not r['visible']})} contexts")

    def _stats(rr, tag):
        rs = np.asarray([r["y"] - r["mu"] for r in rr], dtype=float)
        errs = np.asarray([r["err10"] for r in rr if r["err10"] is not None], dtype=float)
        sig = np.asarray([r["sigma_m"] for r in rr], dtype=float)
        print(f"[{tag}] n={len(rs):5d} resid_sd={np.std(rs):.4f} "
              f"err_rms={np.sqrt(np.mean(errs**2)):.4f} (n_err={len(errs)}) "
              f"resid/err={np.std(rs)/max(np.sqrt(np.mean(errs**2)),1e-8):.2f} "
              f"emitted_sigma_m={np.mean(sig):.4f}")
        # NLL contribution using emitted sigma
        from audit.evaluation.metrics import row_nll
        z = np.zeros(len(rs))
        cens = np.zeros(len(rs), dtype=bool)
        nlls = row_nll(rs + z, cens, z, np.full(len(rs), float(np.mean(sig))))
        return {"n": len(rs), "resid_sd": float(np.std(rs)),
                "err_rms": float(np.sqrt(np.mean(errs**2))),
                "resid_over_err": float(np.std(rs) / max(np.sqrt(np.mean(errs**2)), 1e-8)),
                "mean_nll_at_emitted_sigma": float(np.mean(nlls))}

    vis = [r for r in rows if r["visible"]]
    ood = [r for r in rows if not r["visible"]]
    sv = _stats(vis, "train-visible")
    so = _stats(ood, "OOD-context")
    sall = _stats(rows, "ALL")

    out = {
        "n_rows": len(rows),
        "train_visible": sv,
        "ood_context": so,
        "all": sall,
        "conclusion": (
            "If OOD residual ~ train-visible residual (and both >> err10), the "
            "gap is irreducible context structure, NOT feature-predictable "
            "headroom: a better predictor cannot reach err10 because per-context "
            "bias is a random effect (r67) only reachable by cross-fold EB (r56b) "
            "for train-visible contexts.  The frozen method is at the data limit."),
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r69_residual_decomposition_visibility.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
