"""RNA-FM + ViennaRNA combined plain-linear hybrid (winning-head representation).

The two sequence representations are orthogonal families:
  - ViennaRNA (11-D folding proxy): hand-engineered thermodynamic/secondary-
    structure scalars (MFE, ensemble dG, base-pair probabilities, positional
    entropy, GC).  Alone it gives the decisive +2.71% sequence increment.
  - RNA-FM (1920-D frozen pooled embedding): unsupervised learned sequence
    representation from a 12-layer RNA language model.

This model stacks BOTH on top of the SAME winning plain-linear right-censored
head and the SAME motif + scaffold + topology nuisance basis, testing whether
the learned representation adds increment on top of the folding proxy (i.e.
they are complementary rather than redundant representations of the junction).

  x = [1, motif_onehot, scaffold_onehot, topology(3),
       ViennaRNA(11, train-scaled), RNA-FM(1920, train-scaled)]
  mu = x @ beta    (right-censored Gaussian, tau=0.7, CAP=-7.1)
  ridge on all coefficients except the intercept.

Standardization of each sequence block is fit on TRAIN junction features only;
unseen scaffolds/motifs are left at 0 (abstain on unseen scaffold), mirroring
vienna_linear_hybrid so any increment is attributable to the sequence blocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.rnafm_features import build_raw_by_jid as rnafm_raw_by_jid
from audit.benchmark.rnafm_features import fit_scaler as rnafm_fit_scaler
from audit.benchmark.rnafm_features import transform as rnafm_transform
from audit.benchmark.vienna_features import build_raw_by_jid as vienna_build_raw
from audit.benchmark.vienna_features import fit_scaler as vienna_fit_scaler
from audit.benchmark.vienna_features import transform as vienna_transform
from audit.core.censored_objective import CensoredObjective, fit_lbfgs
from audit.models.vienna_linear_hybrid import _nuisance_basis

RIDGE = 5.0
MAXITER = 2000
GTOL = 1e-8


def make_rnafm_vienna_linear_hybrid(cache: dict):
    """Factory taking the precomputed {seq: embedding} RNA-FM cache."""
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})

        # ViennaRNA block (11-D, train-scaled) and RNA-FM block (1920-D,
        # train-scaled), both built per TRAIN ROW (not per unique jid).
        v_by_jid = vienna_build_raw(train_rows)
        v_mean, v_sd = vienna_fit_scaler(tr_jids, v_by_jid)
        r_by_jid = rnafm_raw_by_jid(train_rows, cache)
        missing = [j for j in tr_jids if j not in r_by_jid]
        if missing:
            raise RuntimeError(f"{len(missing)} train junctions missing RNA-FM embedding")
        r_mean, r_sd = rnafm_fit_scaler(tr_jids, r_by_jid)
        Xv = np.zeros((len(train_rows), len(v_mean)))
        Xr = np.zeros((len(train_rows), len(r_mean)))
        for i, r in enumerate(train_rows):
            j = str(r["jid"])
            Xv[i] = vienna_transform([j], v_by_jid, v_mean, v_sd)[0]
            Xr[i] = rnafm_transform([j], r_by_jid, r_mean, r_sd)[0]

        X = np.hstack([Xn, Xv, Xr])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        obj = CensoredObjective(X, y, cens)
        rec = fit_lbfgs(obj, np.zeros(X.shape[1]), ridge=RIDGE,
                        mask_intercept=True, maxiter=MAXITER, gtol=GTOL)
        return {"kind": "rnafm_vienna_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "v_mean": v_mean, "v_sd": v_sd, "v_by_jid": v_by_jid,
                "r_mean": r_mean, "r_sd": r_sd, "r_by_jid": r_by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_rnafm": Xr.shape[1]}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        r_by_jid = rnafm_raw_by_jid(test_rows, cache)
        Xr = np.zeros((len(test_rows), model["n_rnafm"]))
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            if j in model["v_by_jid"]:
                Xv[i] = vienna_transform([j], model["v_by_jid"], model["v_mean"], model["v_sd"])[0]
            if j in r_by_jid:
                Xr[i] = rnafm_transform([j], r_by_jid, model["r_mean"], model["r_sd"])[0]
        X = np.hstack([Xn, Xv, Xr])
        mu = X @ model["beta"]
        sigma = np.full(len(mu), 0.7)
        from scipy.special import log_ndtr
        a = (mu + 7.1) / 0.7
        cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        return mu, sigma, cp, seen_scaf, ~seen_scaf

    return fit, predict