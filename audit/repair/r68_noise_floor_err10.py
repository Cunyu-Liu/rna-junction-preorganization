"""r68: measured-layer noise floor vs canonical per-junction measurement error.

The canonical records carry `err10` (per-junction standard error of dg10, the
label).  If the ensemble's measured-layer residuals are dominated by the
reported measurement error, the model is at the irreducible noise floor and no
base-model lever can help.  If residuals FAR exceed err10, there is real,
feature-predictable structure left -> base-model levers are NOT exhausted and
we should keep pushing the predictor (not just calibration).

Estimands:
  - per-junction err10 (label measurement error) distribution, mapped by
    source_row_id.
  - ensemble measured residual sd (after r62-corrected mu) vs err10 per row.
  - "noise-floor NLL": 0.5*log(2pi) + log(err10) + 0.5 (mean over measured
    rows/junction-macro), i.e. the best achievable Gaussian NLL if the model
    mu exactly equals the true value and sigma = err10.
  - residual decomposition: sd_total^2 vs sd_err^2 -> sd_signal^2.
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

    # r62-corrected mu (frozen method)
    from audit.repair.r62_decoupled_frozen import _calibrate_r62
    cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)

    # canon err10 by line index (source_row_id == f"{i:06d}" of the source file)
    canon_lines = []
    for line in Path(CANON).read_text().splitlines():
        if not line.strip():
            continue
        canon_lines.append(json.loads(line))

    # assemble measured rows with err10
    rows = []
    for rid, p in cal62.items():
        if p["cens"]:
            continue
        idx = int(rid)
        e = None
        if 0 <= idx < len(canon_lines):
            rec = canon_lines[idx]
            try:
                e = float(rec["err10"]) if rec.get("err10") not in (None, "") else None
            except (TypeError, ValueError):
                e = None
        rows.append({"rid": rid, "jid": p["jid"], "scaf": int(p["scaf"]),
                     "y": p["y"], "mu": p["mu"], "sigma_m": p["sigma"],
                     "err10": e})
    print(f"measured rows with err10: {sum(1 for r in rows if r['err10'] is not None)} "
          f"/ {len(rows)}")

    rr = [r for r in rows if r["err10"] is not None]
    err = np.asarray([r["err10"] for r in rr], dtype=float)
    resid = np.asarray([r["y"] - r["mu"] for r in rr], dtype=float)
    sig_m = np.asarray([r["sigma_m"] for r in rr], dtype=float)

    sd_resid = float(np.std(resid))
    mean_err = float(np.mean(err))
    med_err = float(np.median(err))
    sd_err = float(np.sqrt(np.mean(err ** 2)))
    print(f"\nresidual sd (r62 corrected) = {sd_resid:.4f}")
    print(f"err10: mean={mean_err:.4f} med={med_err:.4f} rms={sd_err:.4f}")

    # residual sd / err10 ratio
    ratio = sd_resid / max(sd_err, 1e-8)
    print(f"residual_sd / err_rms = {ratio:.2f}")

    # decomposition sd_total^2 = sd_signal^2 + sd_err^2  (if independent)
    sd_sig2 = max(sd_resid ** 2 - sd_err ** 2, 0.0)
    print(f"implied feature-predictable sd (sqrt(max(0, sd_res^2 - sd_err^2))) = "
          f"{np.sqrt(sd_sig2):.4f}")

    # per-scaf breakdown
    by_scaf = defaultdict(list)
    for r, e, res in zip(rr, err, resid):
        by_scaf[r["scaf"]].append((e, res))
    print("\nper-scaf residual sd vs err:")
    for sc in sorted(by_scaf):
        es = np.asarray([v[0] for v in by_scaf[sc]])
        rs = np.asarray([v[1] for v in by_scaf[sc]])
        print(f"  scaf{sc}: n={len(rs):5d} resid_sd={np.std(rs):.4f} "
              f"err_rms={np.sqrt(np.mean(es**2)):.4f} "
              f"ratio={np.std(rs)/max(np.sqrt(np.mean(es**2)),1e-8):.2f}")

    # noise-floor NLL (junction-macro): perfect mu + sigma=err10 per row
    # Gaussian measured NLL = 0.5*log(2pi) + log(sigma) + 0.5*(r/sigma)^2
    # at perfect mu r=0 -> 0.5*log(2pi)+log(sigma)+0.5
    jid_err = defaultdict(list)
    jid_r = defaultdict(list)
    for r, e, res in zip(rr, err, resid):
        jid_err[r["jid"]].append(e)
        jid_r[r["jid"]].append(res)
    per_jid_floor = []
    per_jid_actual = []
    for j, es in jid_err.items():
        em = float(np.mean(es))
        per_jid_floor.append(0.5 * np.log(2 * np.pi) + np.log(em) + 0.5)
        # actual using emitted sigma_m
        rs_ = np.asarray([r["y"] - r["mu"] for r in rr if r["jid"] == j])
        sm_ = np.asarray([r["sigma_m"] for r in rr if r["jid"] == j])
        from audit.evaluation.metrics import row_nll
        nlls = row_nll(rs_ + np.zeros_like(rs_), np.zeros(len(rs_), dtype=bool),
                       np.zeros_like(rs_), sm_)
        per_jid_actual.append(float(np.mean(nlls)))
    floor_nll = float(np.mean(per_jid_floor))
    actual_nll = float(np.mean(per_jid_actual))
    print(f"\nnoise-floor measured NLL (perfect mu, sigma=err10): {floor_nll:.4f}")
    print(f"r62 measured NLL: {actual_nll:.4f}")
    print(f"gap (headroom): {actual_nll - floor_nll:+.4f}")

    # also with sigma = rms of residual (no per-row err): the sd-equivalent floor
    jid_sigma_global = float(np.mean(
        [np.log(np.sqrt(np.mean([r["sigma_m"] ** 2 for r in rr])))
         for _ in [0]]))
    floor_global = 0.5 * np.log(2 * np.pi) + jid_sigma_global + 0.5
    print(f"sd-equivalent floor (sigma=emitted rms): {floor_global:.4f}")

    out = {
        "n_measured_with_err": len(rr),
        "residual_sd": round(sd_resid, 4),
        "err10_mean": round(mean_err, 4),
        "err10_med": round(med_err, 4),
        "err10_rms": round(sd_err, 4),
        "ratio_resid_over_err": round(ratio, 2),
        "implied_feature_sd": round(float(np.sqrt(sd_sig2)), 4),
        "noise_floor_nll_perfect_mu_sigma_err10": round(floor_nll, 4),
        "r62_measured_nll": round(actual_nll, 4),
        "headroom_nll": round(actual_nll - floor_nll, 4),
        "conclusion": (
            "If residual sd ~ err_rms (ratio ~1), the model is at the noise "
            "floor; base-model levers cannot help.  If ratio >> 1, residual is "
            "dominated by feature-predictable structure -> push the predictor."),
    }
    Path("/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/"
         "r68_noise_floor_err10.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
