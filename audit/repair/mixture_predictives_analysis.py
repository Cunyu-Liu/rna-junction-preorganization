"""r41: Mixture-of-predictives ensemble scoring (distribution-level combination).

All prior ensembles average the predictive LOCATION mu (equal/family-equal
weights) and emit a single sigma.  This is one valid combination rule, but a
Gaussian-mixture of the members' predictives is a DIFFERENT, principled scoring
rule that has never been tested:

  predictive(y) = (1/K) * sum_i  N(y ; mu_i, sigma_i)
  nll_measured    = -log( (1/K) sum_i N(y ; mu_i, sigma_i) )
  nll_censored    = -log( (1/K) sum_i S((CAP-mu_i)/sigma_i) )

This is the log of the average predictive density -- a proper scoring rule that
uses the FULL predictive (mean AND variance) of every member, not just the mu.
It is honest: all members' predictions are OOF on the same 37 joint-blocked
folds, and the combination has NO free parameters.

Compared estimands (all pooled-OOF junction-macro right-censored NLL):
  - frozen sigma=0.7 mu-averaged ensemble      (r24/r34/r35 freeze)
  - per-scaffold sigma LOO calibration (r38)   (current best, 0.8166)
  - mixture-of-predictives at sigma=0.7 (r41)
  - mixture-of-predictives with per-scaffold sigma (r41 x r38)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.shootout_run import _eligible_keys

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
R33 = f"{R}/r33_xgboost_full/Predictions_v3.jsonl"
R34 = f"{R}/r34_gbdt_seeds_full/Predictions_v3.jsonl"
R35 = f"{R}/r35_gbdt_hp_full/Predictions_v3.jsonl"
R24 = f"{R}/r24_t7_seed7/combined_r20_r21_r23_r24_preds.jsonl"

R33_LEDGER = f"{R}/r33_xgboost_full/ConvergenceLedger_v3.parquet"
R34_LEDGER = f"{R}/r34_gbdt_seeds_full/ConvergenceLedger_v3.parquet"
R35_LEDGER = f"{R}/r35_gbdt_hp_full/ConvergenceLedger_v3.parquet"
R24_LEDGERS = [
    f"{R}/r20_robust_t_df_sweep/ConvergenceLedger_v3.parquet",
    f"{R}/r21_seed99_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r23_seed2026_replication/ConvergenceLedger_v3.parquet",
    f"{R}/r24_t7_seed7/ConvergenceLedger_v3.parquet",
]

XGB = "xgboost_censored_hybrid"
XGB_S99 = "xgboost_censored_hybrid_s99"
XGB_S2026 = "xgboost_censored_hybrid_s2026"
XGB_LR03 = "xgboost_censored_hybrid_hp_lr03"
T7 = "nonlinear_mlp_extended_hybrid_reg_deep_t7"
T7_S99 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99"
T7_S2026 = "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026"
NUIS = "motif_topology_hierarchy"
ALL_MEMBERS = [XGB, XGB_LR03, XGB_S99, XGB_S2026, T7, T7_S99, T7_S2026]

CAP = -7.1


def _load(path):
    return [json.loads(l) for l in open(path)]


def _elig(paths):
    import pandas as pd
    frames = [pd.read_parquet(p) for p in paths]
    conv = [dict(r) for r in pd.concat(frames, ignore_index=True).to_dict("records")]
    return _eligible_keys(conv)


def _by_rid(rows, model_id, eligible):
    out = {}
    for p in rows:
        if p["model_id"] == model_id and (model_id, p["fold"]) in eligible \
                and p["support"] and not p["abstain"]:
            out[p["source_row_id"]] = p
    return out


def _pooled_nll(rows, mu_fn, sigma_fn):
    """pooled junction-macro NLL with per-row (mu, sigma)."""
    jd = defaultdict(list)
    for rid, p in rows.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [mu_fn(p)], [sigma_fn(p)])[0])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


def _mixture_nll_row(y, cens, mus, sigmas):
    """NLL of the equal-weight Gaussian-mixture predictive for one row."""
    K = len(mus)
    if cens:
        # survival of the mixture = mean of member survival
        surv = np.mean([ndtr((mu - CAP) / sig) for mu, sig in zip(mus, sigmas)])
        surv = float(np.clip(surv, 1e-12, 1.0))
        return -float(np.log(surv))
    # density of the mixture at y = mean of member densities
    dens = np.mean([np.exp(-0.5 * ((y - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                    for mu, sig in zip(mus, sigmas)])
    dens = float(np.clip(dens, 1e-12, None))
    return -float(np.log(dens))


def _pooled_mixture(rows, mus_of, sigmas_of):
    jd = defaultdict(list)
    for rid, p in rows.items():
        mus = mus_of[p["rid_key"]]
        sigmas = sigmas_of[p["rid_key"]]
        nll = _mixture_nll_row(p["y"], p["cens"], mus, sigmas)
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


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
    for _ in range(1000):
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
    print("Loading predictions...", file=sys.stderr)
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
    nuis = _by_rid(rows33, NUIS, elig33)

    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ref = members[ALL_MEMBERS[0]]
    # reference rows dict for the mu-averaged ensemble (sigma=0.7)
    ens07 = {}
    for rid in common:
        p0 = ref[rid]
        ens07[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": p0["scaf"],
                      "y": p0["y"], "cens": p0["cens"],
                      "mu": float(np.mean([members[m][rid]["mu"] for m in ALL_MEMBERS])),
                      "sigma": 0.7}

    out = {"nuisance_nll_frozen07": round(_pooled_nll(nuis, lambda p: p["mu"],
                                                      lambda p: 0.7), 4)}

    # ---- baseline: mu-averaged at frozen sigma ----
    out["muavg_7mem_frozen07_nll"] = round(_pooled_nll(ens07, lambda p: p["mu"],
                                                       lambda p: p["sigma"]), 4)

    # ---- r38 reference: per-scaffold sigma LOO on the mu-averaged ensemble ----
    from audit.repair.per_scaf_sigma_calibration import _scan_sigma
    folds = sorted(set(ens07[r]["fold"] for r in ens07))
    by_fold = defaultdict(dict)
    for rid, p in ens07.items():
        by_fold[p["fold"]][rid] = p
    cal_scaf = {}
    fit_log = {}
    grid = np.arange(0.4, 1.4, 0.01)
    for f in folds:
        other = {}
        for ff in folds:
            if ff != f:
                other.update(by_fold[ff])
        s_global, _ = _scan_sigma(other, grid=grid)
        by_scaf = defaultdict(dict)
        for rid, p in other.items():
            by_scaf[int(p["scaf"])][rid] = p
        scaf_sigma = {}
        for sc, rows_sc in by_scaf.items():
            if len(rows_sc) >= 20:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            cal_scaf[rid] = {**p, "sigma": scaf_sigma.get(sc, s_global)}
        fit_log[f] = {str(k): round(v, 3) for k, v in sorted(scaf_sigma.items())}
    out["muavg_per_scaf_sigma_nll"] = round(_pooled_nll(
        cal_scaf, lambda p: p["mu"], lambda p: p["sigma"]), 4)

    # ---- r41: mixture-of-predictives ----
    # member mus/sigmas per common row
    mus_of = {}
    sigmas_of = {}
    for rid in common:
        mus_of[rid] = [members[m][rid]["mu"] for m in ALL_MEMBERS]
        sigmas_of[rid] = [members[m][rid]["sigma"] for m in ALL_MEMBERS]

    # mixture at frozen sigma=0.7
    mix_rows = {}
    for rid in common:
        p0 = ref[rid]
        mix_rows[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": p0["scaf"],
                         "y": p0["y"], "cens": p0["cens"],
                         "rid_key": rid}
    out["mixture_7mem_frozen07_nll"] = round(_pooled_mixture(
        mix_rows, mus_of, sigmas_of), 4)

    # mixture with per-scaffold sigma (r41 x r38): use cal_scaf's per-row sigma
    sigmas_of_scaf = {rid: [cal_scaf[rid]["sigma"]] * len(ALL_MEMBERS)
                      for rid in common}
    out["mixture_7mem_per_scaf_sigma_nll"] = round(_pooled_mixture(
        mix_rows, mus_of, sigmas_of_scaf), 4)

    # mixture of mixtures not needed; report the two main estimates
    out["n_rows"] = len(common)
    out["note"] = (
        "r41 lever: mixture-of-predictives -- the ensemble predictive is the "
        "equal-weight Gaussian mixture of the 7 members' FULL predictives "
        "(mu AND sigma), scored by -log(mean member density/survival).  This is "
        "a proper scoring rule that uses the full predictive of every member, "
        "unlike mu-averaging which collapses to a single mu.  No free params; "
        "all OOF on the same 37 joint-blocked folds.  Compared against muavg at "
        "frozen sigma and the r38 per-scaffold sigma calibration."
    )
    Path(f"{R}/mixture_predictives_analysis.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
