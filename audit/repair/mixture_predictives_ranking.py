"""r41b: ranking robustness under the mixture-of-predictives scoring rule.

The frozen primary is the right-censored Gaussian NLL of a single (mu, sigma)
per row.  The mixture-of-predictives scores the equal-weight Gaussian mixture
of the members' full predictives instead (-log mean member density/survival) --
a DIFFERENT proper scoring rule.  If the model ranking (nuisance < single < 3x
t7 < 7-mem) is preserved under this orthogonal scoring rule, the conclusion is
robust to the combination/aggregation choice, which strengthens the claim.

Compute the mixture score for every comparator at:
  - frozen sigma=0.7
  - per-scaffold sigma LOO (r38 calibration applied per member then mixed)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr

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
MLP = [T7, T7_S99, T7_S2026]

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


def _mixture_nll_row(y, cens, mus, sigmas):
    K = len(mus)
    if cens:
        surv = np.mean([ndtr((mu - CAP) / sig) for mu, sig in zip(mus, sigmas)])
        surv = float(np.clip(surv, 1e-12, 1.0))
        return -float(np.log(surv))
    dens = np.mean([np.exp(-0.5 * ((y - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                    for mu, sig in zip(mus, sigmas)])
    dens = float(np.clip(dens, 1e-12, None))
    return -float(np.log(dens))


def _pooled_mixture(rows_by_rid, mus_of, sigmas_of):
    jd = defaultdict(list)
    for rid, p in rows_by_rid.items():
        nll = _mixture_nll_row(p["y"], p["cens"], mus_of[rid], sigmas_of[rid])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


def _single_mixture(preds, sigmas_of):
    """mixture score of a single model = its own Gaussian NLL (degenerate mix)."""
    jd = defaultdict(list)
    for rid, p in preds.items():
        nll = _mixture_nll_row(p["y"], p["cens"], [p["mu"]], [sigmas_of[rid]])
        jd[p["jid"]].append(nll)
    return float(np.mean([np.mean(v) for v in jd.values()]))


def _scan_sigma(rows, grid=None):
    if not rows:
        return None, np.inf
    items = list(rows.values())
    y = np.asarray([p["y"] for p in items], dtype=float)
    cens = np.asarray([p["cens"] for p in items], dtype=bool)
    mu = np.asarray([p["mu"] for p in items], dtype=float)
    jid = np.asarray([p["jid"] for p in items])
    grid = grid if grid is not None else np.arange(0.4, 1.4, 0.01)
    best_s, best_n = None, np.inf
    for s in grid:
        losses = row_nll(y, cens, mu, np.full(len(y), float(s)))
        by = defaultdict(list)
        for j, loss in zip(jid, losses):
            by[str(j)].append(float(loss))
        nll = float(np.mean([np.mean(v) for v in by.values()]))
        if nll < best_n:
            best_n, best_s = nll, s
    return float(best_s), float(best_n)


def _per_scaf_sigmas(preds, folds, min_rows=20):
    """Return {rid: sigma} per-scaffold LOO for a set of predictions."""
    by_fold = defaultdict(dict)
    for rid, p in preds.items():
        by_fold[p["fold"]][rid] = p
    grid = np.arange(0.4, 1.4, 0.01)
    out = {}
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
            if len(rows_sc) >= min_rows:
                s, _ = _scan_sigma(rows_sc, grid=grid)
                scaf_sigma[sc] = s
            else:
                scaf_sigma[sc] = s_global
        for rid, p in by_fold[f].items():
            sc = int(p["scaf"])
            out[rid] = scaf_sigma.get(sc, s_global)
    return out


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

    # reference rows
    def rows_for(rids):
        return {rid: {"jid": ref[rid]["jid"], "fold": ref[rid]["fold"],
                      "scaf": ref[rid]["scaf"], "y": ref[rid]["y"],
                      "cens": ref[rid]["cens"]} for rid in rids}

    rids_all = rows_for(common)
    rids_mlp = rows_for(common)

    # per-scaf sigma maps for each member + nuisance
    sig07 = {rid: 0.7 for rid in common}
    member_sig = {}
    for m in ALL_MEMBERS:
        folds_m = sorted(set(members[m][r]["fold"] for r in members[m]))
        member_sig[m] = _per_scaf_sigmas(members[m], folds_m)
    nuis_sig = _per_scaf_sigmas(nuis, sorted(set(nuis[r]["fold"] for r in nuis)))

    # mixture score at frozen sigma=0.7
    mix07 = {
        "nuisance": _single_mixture(nuis, {rid: 0.7 for rid in nuis}),
        "xgb_lr03": _single_mixture(members[XGB_LR03], sig07),
        "t7_s99": _single_mixture(members[T7_S99], sig07),
        "3x_t7": _pooled_mixture(rids_mlp,
                                 {rid: [members[m][rid]["mu"] for m in MLP]
                                  for rid in common},
                                 {rid: [0.7] * len(MLP) for rid in common}),
        "7mem": _pooled_mixture(rids_all,
                                {rid: [members[m][rid]["mu"] for m in ALL_MEMBERS]
                                 for rid in common},
                                {rid: [0.7] * len(ALL_MEMBERS) for rid in common}),
    }

    # mixture score with per-scaffold sigma LOO
    mix_scaf = {
        "nuisance": _single_mixture(nuis, nuis_sig),
        "xgb_lr03": _single_mixture(members[XGB_LR03], member_sig[XGB_LR03]),
        "t7_s99": _single_mixture(members[T7_S99], member_sig[T7_S99]),
        "3x_t7": _pooled_mixture(rids_mlp,
                                 {rid: [members[m][rid]["mu"] for m in MLP]
                                  for rid in common},
                                 {rid: [member_sig[m][rid] for m in MLP]
                                  for rid in common}),
        "7mem": _pooled_mixture(rids_all,
                                {rid: [members[m][rid]["mu"] for m in ALL_MEMBERS]
                                 for rid in common},
                                {rid: [member_sig[m][rid] for m in ALL_MEMBERS]
                                 for rid in common}),
    }

    out = {
        "mixture_frozen07": {k: round(v, 4) for k, v in mix07.items()},
        "mixture_per_scaf_sigma": {k: round(v, 4) for k, v in mix_scaf.items()},
        "mixture_frozen07_rel_gain_pct": {
            k: round(100.0 * (mix07["nuisance"] - v) / mix07["nuisance"], 2)
            for k, v in mix07.items() if k != "nuisance"},
        "mixture_per_scaf_rel_gain_pct": {
            k: round(100.0 * (mix_scaf["nuisance"] - v) / mix_scaf["nuisance"], 2)
            for k, v in mix_scaf.items() if k != "nuisance"},
        "n_rows": len(common),
        "note": (
            "Ranking robustness under the mixture-of-predictives proper scoring "
            "rule (no free params, all OOF).  Frozen primary (single-Gaussian "
            "NLL of muavg) stays 0.8527/0.8166; this table uses a DIFFERENT "
            "scoring rule to confirm the ranking is robust to aggregation choice."),
    }
    Path(f"{R}/mixture_predictives_ranking.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
