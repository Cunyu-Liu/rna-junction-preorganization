"""ViennaRNA-representation plain-linear hybrid (winning-head increment test).

Head-diagnosis + representation shootout established two facts on the decisive
joint axis:
  1. the plain right-censored linear head systematically beats the latent-
     operator head (motif_topology_hierarchy 1.0917 vs vienna_latent_operator
     1.1420);
  2. ViennaRNA folding features are the best sequence representation so far
     (they largely restore the 63-D penalty) but under the latent head they
     still lose to the plain-linear nuisance model.

This model couples the WINNING head class (plain linear ridge) with the BEST
sequence representation (ViennaRNA), appended as additional linear features on
top of the motif + scaffold + topology nuisance basis.  It answers the single
most decision-relevant question for representation rank: does a sequence
representation add any increment over the strongest simple nuisance model, when
both share the same (best) head?

  x = [1, motif_onehot, scaffold_onehot, topology(3), viennaRNA(11, train-scaled)]
  mu = x @ beta    (right-censored Gaussian, tau=0.7, CAP=-7.1)
  ridge on all coefficients except the intercept.

Standardization of the ViennaRNA block is fit on TRAIN junction raw features
only, and unseen scaffolds/motifs are left at 0 (model abstains on unseen
scaffold), mirroring motif_topology_hierarchy so the increment is attributable
to the sequence block alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_features import build_raw_by_jid, fit_scaler, transform
from audit.core.censored_objective import CensoredObjective, fit_lbfgs
from audit.data.audit_dataset import parse_parts

RIDGE = 1.0
MAXITER = 2000
GTOL = 1e-8


def _nuisance_basis(rows, motifs, scafs, topo_ok=True):
    """Build [1, motif_onehot, scaffold_onehot, topology(3)] columns."""
    mi = {m: i + 1 for i, m in enumerate(motifs)}
    si = {s: i + 1 + len(motifs) for i, s in enumerate(scafs)}
    nf = 1 + len(motifs) + len(scafs) + 3
    X = np.zeros((len(rows), nf))
    X[:, 0] = 1.0
    off = 1 + len(motifs) + len(scafs)
    for i, r in enumerate(rows):
        m = str(r["motif"])
        s = int(r["scaf"])
        if m in mi:
            X[i, mi[m]] = 1.0
        if s in si:
            X[i, si[s]] = 1.0
        parts = parse_parts(str(r["junction_seq"]))
        full = "".join(parts)
        X[i, off] = len(full)
        X[i, off + 1] = len(parts[0]) if parts else 0
        X[i, off + 2] = len(parts[1]) if len(parts) > 1 else 0
    return X


def make_vienna_linear_hybrid():
    """Return (fit, predict) for the winning-head + ViennaRNA hybrid."""
    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = build_raw_by_jid(train_rows)
        mean, sd = fit_scaler(tr_jids, by_jid)
        ji = {j: i for i, j in enumerate(tr_jids)}
        Xv = np.zeros((len(train_rows), len(mean)))
        for i, r in enumerate(train_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        obj = CensoredObjective(X, y, cens)
        rec = fit_lbfgs(obj, np.zeros(X.shape[1]), ridge=RIDGE,
                        mask_intercept=True, maxiter=MAXITER, gtol=GTOL)
        return {"kind": "vienna_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1]}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        te_jids = sorted({str(r["jid"]) for r in test_rows})
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
        support = np.ones(len(mu), dtype=bool)
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        abstain = ~seen_scaf
        support = seen_scaf
        return mu, sigma, cp, support, abstain

    return fit, predict


VIENNA_LINEAR_HYBRID = {
    "vienna_linear_hybrid": make_vienna_linear_hybrid(),
}