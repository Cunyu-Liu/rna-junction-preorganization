"""r75: re-verify equal-family weight (wg) under the r62 frozen method.

r50/r53 confirmed equal-family wg=0.5 optimal under r45/r51.  r62 changed the
emitted mu/sigma (context mu + decoupled sigma).  The ensemble weight was never
re-swept UNDER r62: wg=0.5 is the frozen choice.  Test wg in {0,0.25,0.5,0.75,1}
under the full r62 calibration to confirm 0.5 remains optimal.
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

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


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

    from audit.repair.r62_decoupled_frozen import _calibrate_r62

    results = {}
    for wg in (0.0, 0.25, 0.5, 0.75, 1.0):
        ens = {}
        for rid in common:
            p0 = ref[rid]
            gmu = float(np.mean([members[m][rid]["mu"] for m in GBDT]))
            mmu = float(np.mean([members[m][rid]["mu"] for m in MLP]))
            ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": int(p0["scaf"]),
                        "context": str(p0.get("context", "?")), "y": p0["y"],
                        "cens": p0["cens"],
                        "mu": wg * gmu + (1.0 - wg) * mmu}
        folds = sorted(set(ens[r]["fold"] for r in ens))
        cal, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)
        nll = _pooled(cal)
        results[f"wg{wg:g}"] = round(nll, 4)
        print(f"wg={wg}: {nll:.4f}")

    best = min(results, key=results.get)
    print(f"\nBEST: {best} -> {results[best]}")
    print(f"vs frozen wg=0.5 ({results['wg0.5']}): "
          f"delta={results[best]-results['wg0.5']:+.4f}")

    out = {"results": results, "best": best, "frozen_wg0_5": results["wg0.5"],
           "note": "ensemble weight sweep under r62 calibration (kappa=1,mm3)"}
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r75_family_weight_under_r62.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
