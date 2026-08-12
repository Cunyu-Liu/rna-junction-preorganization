"""Nonlinear (shallow-MLP) hybrid on RICHER representations (nonlinear step 2).

The decisive r10b shootout showed the shallow MLP head is the bottleneck-breaker:
on the SAME base feature set (nuisance + 11-D ViennaRNA) it beat the plain-linear
head by +8.87% and the nuisance-only model by +11.33% (passing the 10% gate),
with clean convergence on all 37 folds.  The linear head had saturated on every
richer sequence representation (RNA-FM PCA, extended ViennaRNA, scaffold/motif
interactions); the hypothesis driving THIS module is that a nonlinear head can
finally extract signal from those richer representations that a linear head
structurally cannot.

This module reuses the exact MLP core (architecture, right-censored Gaussian
NLL, Adam + weight decay, plateau-based convergence, eligibility gate) from
nonlinear_mlp_hybrid, but swaps the feature block:

  - nonlinear_mlp_rnafm_pca_hybrid
      x = [nuisance, ViennaRNA(11, train-scaled), RNA-FM-PCA(K, train-fit)]
      RNA-FM PCA (mean + components) is fit on TRAIN embeddings only.

  - nonlinear_mlp_extended_hybrid
      x = [nuisance, ViennaRNA-extended(21, train-scaled)]

Every standardization/PCA is fit on TRAIN only (no test leakage); unseen
scaffolds/motifs are left at 0 (abstain on unseen scaffold).  Trains on GPU when
CUDA is available, CPU otherwise (unit tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.rnafm_features import build_raw_by_jid as rnafm_build_raw
from audit.benchmark.vienna_features import build_raw_by_jid as vienna_build_raw
from audit.benchmark.vienna_features import fit_scaler as vienna_fit_scaler
from audit.benchmark.vienna_features import transform as vienna_transform
from audit.benchmark.vienna_extended_features import build_raw_by_jid as vienna_ext_build_raw
from audit.benchmark.vienna_extended_features import fit_scaler as vienna_ext_fit_scaler
from audit.benchmark.vienna_extended_features import transform as vienna_ext_transform
from audit.models.nonlinear_mlp_hybrid import (
    _train_mlp,
    _nuisance_basis as _nuisance_basis_11,
)
from audit.models.rnafm_pca_linear_hybrid import _fit_pca, _apply_pca

DEFAULT_K = 64
_DEFAULT_WD = 1e-3


def _nuisance_basis(rows, motifs, scafs):
    return _nuisance_basis_11(rows, motifs, scafs)


def make_nonlinear_mlp_extended_hybrid(hidden=(64, 32), dropout=0.0,
                                       weight_decay=None):
    """Return (fit, predict) for MLP on nuisance + ViennaRNA-extended(21).

    dropout/weight_decay are passed through to the MLP core so a caller can
    stabilize the higher-dimensional 21-D representation against per-fold
    overfitting (the base 11-D extended-MLP loses on mean NLL due to a few
    catastrophic folds even though it wins on median).
    """
    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = vienna_ext_build_raw(train_rows)
        mean, sd = vienna_ext_fit_scaler(tr_jids, by_jid)
        Xv = np.zeros((len(train_rows), len(mean)))
        for i, r in enumerate(train_rows):
            Xv[i] = vienna_ext_transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net, gate = _train_mlp(X, y, cens, device, X.shape[1], hidden=hidden,
                               dropout=dropout,
                               weight_decay=(weight_decay if weight_decay is not None
                                             else _DEFAULT_WD))
        return {"kind": "nonlinear_mlp_extended_hybrid", "net": net, "gate": gate,
                "motifs": motifs, "scafs": scafs, "mean": mean, "sd": sd,
                "by_jid": by_jid, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "device": device, "hidden": list(hidden),
                "dropout": dropout, "weight_decay": (weight_decay if weight_decay is not None
                                                     else _DEFAULT_WD)}

    def predict(model, test_rows):
        import torch
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = vienna_ext_build_raw(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = vienna_ext_transform([str(r["jid"])], by_jid, model["mean"], model["sd"])[0]
        X = np.hstack([Xn, Xv])
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=model["device"])
            mu = model["net"](Xt).squeeze(-1).cpu().numpy()
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


def make_nonlinear_mlp_extended_hybrid_reg(hidden=(64, 32), dropout=0.1,
                                           weight_decay=1e-2):
    """Regularized 21-D extended-Vienna MLP.

    The un-regularized 21-D extended MLP wins on median NLL over the base 11-D
    MLP but loses on mean NLL because a few folds overfit catastrophically.
    This variant adds dropout + a 10x larger weight-decay to stabilize those
    folds, testing whether the richer folding representation's real (median)
    signal can be converted into a robust mean gain.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=hidden, dropout=dropout,
                                              weight_decay=weight_decay)


def make_nonlinear_mlp_extended_hybrid_reg_strong():
    """Extended-MLP with stronger regularization (dropout=0.2, wd=3e-2).

    Probe whether more aggressive regularization on the 21-D folding features
    pushes the robust mean NLL even lower without erasing signal.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(64, 32), dropout=0.2,
                                              weight_decay=3e-2)


def make_nonlinear_mlp_extended_hybrid_reg_light():
    """Extended-MLP with lighter dropout (0.05) at the same wd=1e-2.

    Probe whether a milder regularizer retains more signal while still taming
    the catastrophic folds seen in the un-regularized 21-D model.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(64, 32), dropout=0.05,
                                              weight_decay=1e-2)


def make_nonlinear_mlp_extended_hybrid_reg_wider():
    """Extended-MLP with a wider hidden layer (128,64) at the reference reg.

    Probe whether more capacity helps the nonlinear head fit the 21-D folding
    features once regularization already prevents catastrophic overfitting.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(128, 64), dropout=0.1,
                                              weight_decay=1e-2)


def make_nonlinear_mlp_extended_hybrid_reg_deep():
    """Extended-MLP with a third hidden layer (96,64,32) at the reference reg.

    Probe whether depth adds representational power on the richer folding
    representation, again under the reference regularization budget.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(96, 64, 32), dropout=0.1,
                                              weight_decay=1e-2)


def make_nonlinear_mlp_extended_hybrid_reg_deep4():
    """Extended-MLP with a fourth hidden layer (96,64,32,16) at reference reg.

    The r14 scan found the 3-layer (96,64,32) reg_deep decisive (+13.17% over
    nuisance, CI excludes 0).  This probe adds one more layer at the same
    (dropout=0.1, wd=1e-2) budget to test whether depth gains continue.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(96, 64, 32, 16),
                                              dropout=0.1, weight_decay=1e-2)


def make_nonlinear_mlp_extended_hybrid_reg_deep4w():
    """Extended-MLP with a wider four-layer stack (128,96,64,32) at ref reg.

    Probes whether a 4-layer stack with more capacity at every level (vs the
    (96,64,32,16) taper) extracts more from the 21-D folding features without
    re-introducing catastrophic overfitting.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(128, 96, 64, 32),
                                              dropout=0.1, weight_decay=1e-2)


def make_nonlinear_mlp_extended_hybrid_reg_deep5():
    """Extended-MLP with a five-layer stack (128,96,64,32,16) at ref reg.

    Aggressive-depth probe: checks whether a 5-layer nonlinear head on the 21-D
    folding features keeps improving mean NLL or begins to overfit under the
    reference regularization budget.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(128, 96, 64, 32, 16),
                                              dropout=0.1, weight_decay=1e-2)


def make_nonlinear_mlp_rnafm_extended_reg_deep(cache: dict, k: int = DEFAULT_K,
                                               hidden=(96, 64, 32), dropout=0.1,
                                               weight_decay=1e-2):
    """MLP on nuisance + extended-ViennaRNA(21) + RNA-FM-PCA(K), reg_deep arch.

    Combines the winning r14 architecture (reg_deep: 3 hidden layers under the
    reference regularization budget) with BOTH the folding proxy (21-D extended
    ViennaRNA) and the learned RNA-FM representation.  Tests whether the learned
    embedding is complementary to the folding proxy once the deeper nonlinear
    head is in place -- i.e. whether reg_deep's +13% gain over nuisance can be
    extended further.
    """
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})

        v_by_jid = vienna_ext_build_raw(train_rows)
        v_mean, v_sd = vienna_ext_fit_scaler(tr_jids, v_by_jid)

        r_by_jid = rnafm_build_raw(train_rows, cache)
        missing = [j for j in tr_jids if j not in r_by_jid]
        if missing:
            raise RuntimeError(f"{len(missing)} train junctions missing RNA-FM embedding")
        Xr_raw = np.asarray([r_by_jid[j] for j in tr_jids], dtype=float)
        k_eff = min(k, Xr_raw.shape[0], Xr_raw.shape[1])
        pca_mean, comps, scale = _fit_pca(Xr_raw, k_eff)

        Xv = np.zeros((len(train_rows), len(v_mean)))
        Xr = np.zeros((len(train_rows), k_eff))
        for i, r in enumerate(train_rows):
            j = str(r["jid"])
            Xv[i] = vienna_ext_transform([j], v_by_jid, v_mean, v_sd)[0]
            Xr[i] = _apply_pca(r_by_jid[j][None, :], pca_mean, comps, scale)[0]

        X = np.hstack([Xn, Xv, Xr])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net, gate = _train_mlp(X, y, cens, device, X.shape[1], hidden=hidden,
                               dropout=dropout, weight_decay=weight_decay)
        return {"kind": "nonlinear_mlp_rnafm_extended_reg_deep", "net": net,
                "gate": gate, "motifs": motifs, "scafs": scafs,
                "v_mean": v_mean, "v_sd": v_sd, "v_by_jid": v_by_jid,
                "pca_mean": pca_mean, "comps": comps, "scale": scale,
                "k": k_eff, "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_rnafm_pca": k_eff, "device": device, "hidden": list(hidden),
                "dropout": dropout, "weight_decay": weight_decay}

    def predict(model, test_rows):
        import torch
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        Xr = np.zeros((len(test_rows), model["n_rnafm_pca"]))
        r_by_jid = rnafm_build_raw(test_rows, cache)
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            if j in model["v_by_jid"]:
                Xv[i] = vienna_ext_transform([j], model["v_by_jid"], model["v_mean"],
                                             model["v_sd"])[0]
            if j in r_by_jid:
                Xr[i] = _apply_pca(r_by_jid[j][None, :], model["pca_mean"],
                                   model["comps"], model["scale"])[0]
        X = np.hstack([Xn, Xv, Xr])
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=model["device"])
            mu = model["net"](Xt).squeeze(-1).cpu().numpy()
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


def make_nonlinear_mlp_rnafm_pca_hybrid(cache: dict, k: int = DEFAULT_K,
                                        hidden=(64, 32)):
    """Return (fit, predict) for MLP on nuisance + ViennaRNA(11) + RNA-FM-PCA(K)."""
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})

        v_by_jid = vienna_build_raw(train_rows)
        v_mean, v_sd = vienna_fit_scaler(tr_jids, v_by_jid)

        r_by_jid = rnafm_build_raw(train_rows, cache)
        missing = [j for j in tr_jids if j not in r_by_jid]
        if missing:
            raise RuntimeError(f"{len(missing)} train junctions missing RNA-FM embedding")
        Xr_raw = np.asarray([r_by_jid[j] for j in tr_jids], dtype=float)
        k_eff = min(k, Xr_raw.shape[0], Xr_raw.shape[1])
        pca_mean, comps, scale = _fit_pca(Xr_raw, k_eff)

        Xv = np.zeros((len(train_rows), len(v_mean)))
        Xr = np.zeros((len(train_rows), k_eff))
        for i, r in enumerate(train_rows):
            j = str(r["jid"])
            Xv[i] = vienna_transform([j], v_by_jid, v_mean, v_sd)[0]
            Xr[i] = _apply_pca(r_by_jid[j][None, :], pca_mean, comps, scale)[0]

        X = np.hstack([Xn, Xv, Xr])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net, gate = _train_mlp(X, y, cens, device, X.shape[1])
        return {"kind": "nonlinear_mlp_rnafm_pca_hybrid", "net": net, "gate": gate,
                "motifs": motifs, "scafs": scafs,
                "v_mean": v_mean, "v_sd": v_sd, "v_by_jid": v_by_jid,
                "pca_mean": pca_mean, "comps": comps, "scale": scale,
                "k": k_eff, "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_rnafm_pca": k_eff, "device": device, "hidden": list(hidden)}

    def predict(model, test_rows):
        import torch
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        Xr = np.zeros((len(test_rows), model["n_rnafm_pca"]))
        r_by_jid = rnafm_build_raw(test_rows, cache)
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            if j in model["v_by_jid"]:
                Xv[i] = vienna_transform([j], model["v_by_jid"], model["v_mean"], model["v_sd"])[0]
            if j in r_by_jid:
                Xr[i] = _apply_pca(r_by_jid[j][None, :], model["pca_mean"],
                                   model["comps"], model["scale"])[0]
        X = np.hstack([Xn, Xv, Xr])
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=model["device"])
            mu = model["net"](Xt).squeeze(-1).cpu().numpy()
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


def make_nonlinear_mlp_rnafm_only_pca_hybrid(cache: dict, k: int = DEFAULT_K,
                                             hidden=(64, 32)):
    """MLP on nuisance + RNA-FM-PCA only (isolate learned rep without folding proxy)."""
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        r_by_jid = rnafm_build_raw(train_rows, cache)
        missing = [j for j in tr_jids if j not in r_by_jid]
        if missing:
            raise RuntimeError(f"{len(missing)} train junctions missing RNA-FM embedding")
        Xr_raw = np.asarray([r_by_jid[j] for j in tr_jids], dtype=float)
        k_eff = min(k, Xr_raw.shape[0], Xr_raw.shape[1])
        pca_mean, comps, scale = _fit_pca(Xr_raw, k_eff)
        Xr = np.zeros((len(train_rows), k_eff))
        for i, r in enumerate(train_rows):
            Xr[i] = _apply_pca(r_by_jid[str(r["jid"])][None, :], pca_mean, comps, scale)[0]
        X = np.hstack([Xn, Xr])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net, gate = _train_mlp(X, y, cens, device, X.shape[1])
        return {"kind": "nonlinear_mlp_rnafm_only_pca_hybrid", "net": net, "gate": gate,
                "motifs": motifs, "scafs": scafs,
                "pca_mean": pca_mean, "comps": comps, "scale": scale,
                "k": k_eff, "n_nuisance": Xn.shape[1], "n_rnafm_pca": k_eff,
                "device": device, "hidden": list(hidden)}

    def predict(model, test_rows):
        import torch
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        Xr = np.zeros((len(test_rows), model["n_rnafm_pca"]))
        r_by_jid = rnafm_build_raw(test_rows, cache)
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            if j in r_by_jid:
                Xr[i] = _apply_pca(r_by_jid[j][None, :], model["pca_mean"],
                                   model["comps"], model["scale"])[0]
        X = np.hstack([Xn, Xr])
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=model["device"])
            mu = model["net"](Xt).squeeze(-1).cpu().numpy()
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
