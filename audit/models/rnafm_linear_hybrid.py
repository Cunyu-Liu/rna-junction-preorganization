"""RNA-FM frozen-embedding plain-linear hybrid (winning-head representation).

Provenance: the folding-proxy representations (ViennaRNA 11-D, then enriched
21-D) are saturated: the 21-D extended hybrid was WORSE than the 11-D base
(-1.12%), so hand-engineered thermodynamic scalars cannot push the decisive
+2.71% sequence increment further.  This model replaces the sequence block
with the frozen, pooled RNA-FM embedding (1920-D: mean + max + [CLS] of the
layer-12 hidden states), keeping the SAME winning plain-linear right-censored
head and the SAME motif + scaffold + topology nuisance basis.

  x = [1, motif_onehot, scaffold_onehot, topology(3), RNA-FM(1920, train-scaled)]
  mu = x @ beta    (right-censored Gaussian, tau=0.7, CAP=-7.1)
  ridge on all coefficients except the intercept.

The 1920-D block is large relative to 1714 unique junctions, so a modestly
larger ridge is used to control overfitting.  Standardization is fit on TRAIN
junction embeddings only; unseen scaffolds/motifs are left at 0 (abstain on
unseen scaffold), mirroring the ViennaRNA hybrid so the increment is
attributable to the sequence block alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.rnafm_features import build_raw_by_jid, fit_scaler, transform
from audit.core.censored_objective import CensoredObjective, fit_lbfgs
from audit.data.audit_dataset import parse_parts
from audit.models.vienna_linear_hybrid import _nuisance_basis

RIDGE = 5.0
MAXITER = 2000
GTOL = 1e-8


def make_rnafm_linear_hybrid(cache: dict):
    """Factory taking the precomputed {seq: embedding} cache."""
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = build_raw_by_jid(train_rows, cache)
        missing = [j for j in tr_jids if j not in by_jid]
        if missing:
            raise RuntimeError(f"{len(missing)} train junctions missing RNA-FM embedding")
        mean, sd = fit_scaler(tr_jids, by_jid)
        Xv = np.zeros((len(train_rows), len(mean)))
        for i, r in enumerate(train_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        obj = CensoredObjective(X, y, cens)
        rec = fit_lbfgs(obj, np.zeros(X.shape[1]), ridge=RIDGE,
                        mask_intercept=True, maxiter=MAXITER, gtol=GTOL)
        return {"kind": "rnafm_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_nuisance": Xn.shape[1], "n_rnafm": Xv.shape[1]}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = build_raw_by_jid(test_rows, cache)
        Xv = np.zeros((len(test_rows), model["n_rnafm"]))
        for i, r in enumerate(test_rows):
            if str(r["jid"]) in by_jid:
                Xv[i] = transform([str(r["jid"])], by_jid, model["mean"], model["sd"])[0]
        X = np.hstack([Xn, Xv])
        mu = X @ model["beta"]
        sigma = np.full(len(mu), 0.7)
        from scipy.special import log_ndtr
        a = (mu + 7.1) / 0.7
        cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        abstain = ~seen_scaf
        return mu, sigma, cp, seen_scaf, abstain

    return fit, predict