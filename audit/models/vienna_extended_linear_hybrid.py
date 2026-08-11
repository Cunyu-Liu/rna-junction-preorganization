"""ViennaRNA-extended plain-linear hybrid (winning-head representation expansion).

Provenance: vienna_linear_hybrid (11-D ViennaRNA + motif/scaffold/topology nuisance
on the winning plain-linear right-censored head) first produced a DECISIVE positive
sequence increment over the strongest simple baseline (+2.71%, edit-cluster CI
[0.0129, 0.0763] > 0, leave-one-largest robust).  This model is the SAME winning
head and nuisance basis, but with the RICHER ViennaRNA-extended representation
(21 features: base 11 + bpp-profile + cross-part interaction + MFE topology +
entropy profile) in place of the 11-D block.  It tests whether a richer folding
representation pushes the sequence increment further.

  x = [1, motif_onehot, scaffold_onehot, topology(3), vienna_extended(21)]
  mu = x @ beta    (right-censored Gaussian, tau=0.7, CAP=-7.1)
  ridge on all coefficients except the intercept.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_extended_features import build_raw_by_jid, fit_scaler, transform
from audit.core.censored_objective import CensoredObjective, fit_lbfgs
from audit.data.audit_dataset import parse_parts
from audit.models.vienna_linear_hybrid import _nuisance_basis

RIDGE = 1.0
MAXITER = 2000
GTOL = 1e-8


def make_vienna_extended_linear_hybrid():
    """Return (fit, predict) for the winning-head + ViennaRNA-extended hybrid."""
    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = build_raw_by_jid(train_rows)
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
        return {"kind": "vienna_extended_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1]}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = build_raw_by_jid(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
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


VIENNA_EXTENDED_LINEAR_HYBRID = {
    "vienna_extended_linear_hybrid": make_vienna_extended_linear_hybrid(),
}