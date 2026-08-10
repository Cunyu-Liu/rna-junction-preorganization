"""RNAMake / physical-ensemble prior baseline (contract §9.3 / §4.2).

Yesselman et al. (PNAS 2019) and Denny et al. (Cell 2018) motivate a physical
conformational-ensemble prior for junction assembly energetics.  This baseline
computes TRAIN-ONLY frozen physical features of each unique junction sequence
with ViennaRNA (the widely-available, deterministic RNA folding engine), then
fits a single low-capacity linear head on top under the SAME right-censored
objective and the SAME optimizer budget as every other parametric baseline.

Features (per contiguous junction sequence, all sequence-derived, NO label
exposure; computed once per unique junction and cached):
  - MFE free energy (kcal/mol),            [static-structure branch]
  - partition-function / ensemble energy,  [ensemble branch]
  - ensemble defect (expected # of wrongly predicted nts), [ensemble branch]
  - GC/AT content,                         [composition]
  - length,                                [length]
  - mean base-pair probability over all positions, [structure density]

The linear head reuses CensoredObjective + fit_lbfgs (identical to frozen_lm).
This answers "does a physical-ensemble prior help beyond the statistical
baselines", NOT a claim that folding is novel.  It is a comparator family along
with static_only / topology_only documented in the auditing.
"""
from __future__ import annotations

import numpy as np

TAU = 0.7
CAP = -7.1
RIDGE = 1.0

_FEATURE_NAMES = ["mfe", "ensemble_energy", "ensemble_defect",
                  "gc", "length", "mean_bpp"]


def _contig(seq):
    return str(seq or "").replace("_", "").replace("&", "").upper()


def _vienna_features(seq):
    import ViennaRNA
    s = _contig(seq)
    fc = ViennaRNA.fold_compound(s)
    mfe_struct, mfe = fc.mfe()
    mfe = float(mfe)
    ens_struct, ens = fc.pf()          # ensemble free energy (kcal/mol)
    ens = float(ens)
    defect = float(fc.ensemble_defect(mfe_struct))
    n = len(s)
    gc = (s.count("G") + s.count("C")) / max(n, 1)
    # mean base-pair probability
    fc.pf()
    bpp = fc.bpp()
    total = 0.0
    cnt = 0.0
    for i in range(n):
        row = bpp[i]
        for j in range(n):
            if j == i:
                continue
            try:
                total += float(row[j])
                cnt += 1.0
            except Exception:
                pass
    mean_bpp = (total / cnt) if cnt > 0 else 0.0
    return np.asarray([mfe, ens, defect, gc, float(n), mean_bpp], dtype=float)


def build_physical_cache(sequences):
    """Return {contig_seq: feature vector (6,)} computed once per unique seq."""
    seqs = sorted({_contig(s) for s in sequences if _contig(s)})
    cache = {}
    for s in seqs:
        cache[s] = _vienna_features(s)
    return cache


def _add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_physical_head(train, feat_cache, ridge=RIDGE):
    """Fit low-capacity linear head on frozen physical features (train-only)."""
    from audit.core.censored_objective import CensoredObjective, fit_lbfgs
    X = np.asarray([feat_cache[_contig(r["junction_seq"])] for r in train])
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    X = (X - mean) / sd
    Xb = _add_intercept(X)
    y = np.asarray([r["y"] for r in train], dtype=float)
    cens = np.asarray([r["cens"] for r in train], dtype=bool)
    obj = CensoredObjective(Xb, y, cens)
    rec = fit_lbfgs(obj, np.zeros(Xb.shape[1]), ridge=ridge,
                    maxiter=2000, gtol=1e-8)
    return {"kind": "physical_prior", "beta": rec["beta"],
            "mean": mean, "sd": sd, "gate": rec}


def predict_physical_head(model, test, feat_cache):
    from scipy.special import log_ndtr
    X = np.asarray([feat_cache[_contig(r["junction_seq"])] for r in test])
    X = (X - model["mean"]) / model["sd"]
    Xb = _add_intercept(X)
    mu = Xb @ model["beta"]
    sigma = np.full(len(mu), TAU)
    support = np.ones(len(mu), dtype=bool)
    abstain = np.zeros(len(mu), dtype=bool)
    a = (mu - CAP) / sigma
    cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
    return mu, sigma, cp, support, abstain


def physical_feature_names():
    return list(_FEATURE_NAMES)
