"""Interaction-augmented ViennaRNA plain-linear hybrid (nonlinear/interaction step).

The decisive shootout (r09) established that the base 11-D ViennaRNA block is
the ONLY representation with a confirmed sequence increment (+2.7% vs the
nuisance-only model, CI excluding zero), and that every richer sequence block
(21-D extended Vienna, RNA-FM 1920-D, RNA-FM PCA-64-D) fails to add increment.
A plausible reading is that the sequence effect is real but *interactive*:
the linear head (mu = x @ beta) cannot capture a sequence effect whose sign or
magnitude depends on the scaffold/motif context, even though the marginal
effect is small.

This model keeps the winning plain right-censored linear head (ridge, L-BFGS,
tau=0.7, CAP=-7.1) and appends two interaction blocks to the base nuisance +
ViennaRNA basis:

  x = [nuisance, vienna11, vienna11 x scaffold (per-scaffold blocks),
       vienna11 x motif (per-motif blocks)]
  mu = x @ beta

For a row in scaffold s, only the scaffold-s Vienna interaction block is
active (others are zeroed); likewise for motif.  Unseen scaffolds/motifs get a
zero interaction block (and the model abstains on unseen scaffold, mirroring
vienna_linear_hybrid).  Ridge is applied to every coefficient except the
intercept, so the larger interaction basis is regularized and cannot freely
overfit.

Standardization of the ViennaRNA block is fit on TRAIN junction raw features
only (no test leakage), identical to vienna_linear_hybrid.
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


def _nuisance_basis(rows, motifs, scafs):
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


def _interaction_blocks(rows, key_fn, index_map, Xv, n_vienna):
    """For each group (scaffold/motif) in index_map, a block holding Xv where
    the row belongs to that group, else zeros.  Returns (X_blocks, n_cols)."""
    groups = sorted(index_map)
    if not groups:
        return np.zeros((len(rows), 0)), 0
    Xb = np.zeros((len(rows), len(groups) * n_vienna))
    for i, r in enumerate(rows):
        g = key_fn(r)
        if g in index_map:
            gi = groups.index(g)
            Xb[i, gi * n_vienna:(gi + 1) * n_vienna] = Xv[i]
    return Xb, len(groups) * n_vienna


def make_vienna_interaction_linear_hybrid(scaffold_interact=True, motif_interact=True):
    """Return (fit, predict) for the interaction-augmented ViennaRNA hybrid."""
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

        blocks = [Xn, Xv]
        si = {s: i for i, s in enumerate(scafs)}
        mi = {m: i for i, m in enumerate(motifs)}
        n_scaf_int = 0
        n_motif_int = 0
        if scaffold_interact:
            Xb, n_scaf_int = _interaction_blocks(
                train_rows, lambda r: int(r["scaf"]), si, Xv, Xv.shape[1])
            blocks.append(Xb)
        if motif_interact:
            Xb, n_motif_int = _interaction_blocks(
                train_rows, lambda r: str(r["motif"]), mi, Xv, Xv.shape[1])
            blocks.append(Xb)
        X = np.hstack(blocks)
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        obj = CensoredObjective(X, y, cens)
        rec = fit_lbfgs(obj, np.zeros(X.shape[1]), ridge=RIDGE,
                        mask_intercept=True, maxiter=MAXITER, gtol=GTOL)
        return {"kind": "vienna_interaction_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_scaf_int": n_scaf_int, "n_motif_int": n_motif_int}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = build_raw_by_jid(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, model["mean"], model["sd"])[0]
        si = {s: i for i, s in enumerate(model["scafs"])}
        mi = {m: i for i, m in enumerate(model["motifs"])}
        blocks = [Xn, Xv]
        if model["n_scaf_int"] > 0:
            Xb, _ = _interaction_blocks(test_rows, lambda r: int(r["scaf"]), si, Xv, Xv.shape[1])
            blocks.append(Xb)
        if model["n_motif_int"] > 0:
            Xb, _ = _interaction_blocks(test_rows, lambda r: str(r["motif"]), mi, Xv, Xv.shape[1])
            blocks.append(Xb)
        X = np.hstack(blocks)
        mu = X @ model["beta"]
        sigma = np.full(len(mu), 0.7)
        from scipy.special import log_ndtr
        a = (mu + 7.1) / 0.7
        cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        support = seen_scaf
        abstain = ~seen_scaf
        return mu, sigma, cp, support, abstain

    return fit, predict


VIENNA_INTERACTION_LINEAR_HYBRID = {
    "vienna_interaction_linear_hybrid": make_vienna_interaction_linear_hybrid(),
}
