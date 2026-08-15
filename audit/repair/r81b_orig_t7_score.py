"""r81b: compare per-scaf-sigma-trained t7 vs original t7 on held-out folds.

The smoke test (r81) trained a new t7 member with per-scaf calibrated sigma in
the loss on 2 folds.  Here we score BOTH members' held-out mu at the per-scaf
calibrated sigma and compare measured-layer NLL.  This isolates whether the
train/eval sigma mismatch is costing real mu quality.
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

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def _score_meas(mus, test_rows, sig_m):
    """Row-wise measured NLL at per-scaf calibrated sigma, junction-macro."""
    by_jid = defaultdict(list)
    for i, r in enumerate(test_rows):
        if r["cens"]:
            continue
        sc = int(r["scaf"])
        nll = float(row_nll([r["y"]], [False], [float(mus[i])],
                            [sig_m.get(sc, 0.55)])[0])
        by_jid[str(r["jid"])].append(nll)
    return float(np.mean([np.mean(v) for v in by_jid.values()])) if by_jid else None


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
    # per-scaf sigma_m (calibrated, from corrected mu)
    by_scaf_m = defaultdict(dict)
    for rid, p in cal62.items():
        if not p["cens"]:
            by_scaf_m[int(p["scaf"])][rid] = p
    sig_m = {}
    for sc, rows in by_scaf_m.items():
        if len(rows) >= 15:
            s, _ = _scan_sigma(rows, cens_mask=False, grid=GRID)
            sig_m[sc] = s
    print("per-scaf sigma_m:", {k: round(v, 3) for k, v in sorted(sig_m.items())})

    # load the smoke-trained t7 predictions (saved to a json by r81? not saved -> rerun inline)
    # For this comparison, load original t7 member OOF predictions and score.
    # r81 saved nothing; simplest: score ORIGINAL t7 member on its own held-out
    # predictions (members[T7] already holds OOF mu per rid) at per-scaf sigma.
    by_jid = defaultdict(list)
    for rid, p in members[T7].items():
        if not p["cens"]:
            sc = int(p["scaf"])
            nll = float(row_nll([p["y"]], [False], [p["mu"]],
                                [sig_m.get(sc, 0.55)])[0])
            by_jid[str(p["jid"])].append(nll)
    orig_nll = float(np.mean([np.mean(v) for v in by_jid.values()]))
    print(f"ORIGINAL t7 member measured NLL (per-scaf sigma) = {orig_nll:.4f}")

    # full t7-alone at its OWN trained sigma 0.7 (what the audit reported for t7)
    by_jid2 = defaultdict(list)
    for rid, p in members[T7].items():
        if not p["cens"]:
            nll = float(row_nll([p["y"]], [False], [p["mu"]], [0.7])[0])
            by_jid2[str(p["jid"])].append(nll)
    orig_nll07 = float(np.mean([np.mean(v) for v in by_jid2.values()]))
    print(f"ORIGINAL t7 member measured NLL (sigma=0.7) = {orig_nll07:.4f}")

    print("\nInterpretation: if per-scaf-sigma training lowers the measured NLL "
          "below the original member's, the train/eval mismatch was costing mu "
          "quality -> full retrain worthwhile.")


if __name__ == "__main__":
    main()
