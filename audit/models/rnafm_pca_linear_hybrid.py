"""PCA-reduced RNA-FM plain-linear hybrid (winning-head representation).

The frozen pooled RNA-FM embedding is 1920-D (mean+max+CLS of the layer-12
hidden states).  A raw 1920-D linear head overfits on the ~1400-train-row
per-fold regime (first RNA-FM shootout gave pooled-NLL ~2.4 vs ~1.5 for the
folding-proxy hybrid, with extreme out-of-range mu).  This model applies a
train-only PCA reduction to K principal components before the linear head,
so the learned representation is tested without dimensional overfitting.

  x = [1, motif_onehot, scaffold_onehot, topology(3),
       ViennaRNA(11, train-scaled), RNA-FM-PCA(K, train-fit)]
  mu = x @ beta    (right-censored Gaussian, tau=0.7, CAP=-7.1)
  ridge on all coefficients except the intercept.

Never-leakage guarantees:
  - ViennaRNA scaler is fit on TRAIN only.
  - RNA-FM PCA (mean + components) is fit on TRAIN embeddings only and applied
    to both train and test exactly as learned from train.
  - unseen scaffolds/motifs are left at 0 (abstain on unseen scaffold).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.rnafm_features import build_raw_by_jid, RENDER_DIM
from audit.benchmark.vienna_features import build_raw_by_jid as vienna_build_raw
from audit.benchmark.vienna_features import fit_scaler as vienna_fit_scaler
from audit.benchmark.vienna_features import transform as vienna_transform
from audit.core.censored_objective import CensoredObjective, fit_lbfgs
from audit.models.vienna_linear_hybrid import _nuisance_basis

RIDGE = 5.0
MAXITER = 2000
GTOL = 1e-8
DEFAULT_K = 64


def _fit_pca(X, k):
    """Fit a K-component PCA on the given (n, d) matrix; return (mean, comps, scale).

    Applied to TRAIN embeddings only.  `scale` standardizes each principal
    component to unit variance using the train singular values.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    pca.fit(X)
    comps = pca.components_  # (k, d)
    scale = np.sqrt(pca.explained_variance_)  # std of each component on train
    scale = np.where((scale > 1e-8) & np.isfinite(scale), scale, 1.0)
    return pca.mean_, comps, scale


def _apply_pca(X, mean, comps, scale):
    return (X - mean) @ comps.T / scale


def make_rnafm_pca_linear_hybrid(cache: dict, k: int = DEFAULT_K):
    """Factory taking the precomputed {seq: embedding} RNA-FM cache."""
    assert cache, "RNA-FM embedding cache is empty; run rnafm_extract.py first"
    assert k <= RENDER_DIM

    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})

        # ViennaRNA block (11-D, train-scaled), per row.
        v_by_jid = vienna_build_raw(train_rows)
        v_mean, v_sd = vienna_fit_scaler(tr_jids, v_by_jid)

        # RNA-FM embeddings -> PCA (train-only), per row.
        r_by_jid = build_raw_by_jid(train_rows, cache)
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
        obj = CensoredObjective(X, y, cens)
        rec = fit_lbfgs(obj, np.zeros(X.shape[1]), ridge=RIDGE,
                        mask_intercept=True, maxiter=MAXITER, gtol=GTOL)
        return {"kind": "rnafm_pca_linear_hybrid", "beta": rec["beta"],
                "gate": rec, "motifs": motifs, "scafs": scafs,
                "v_mean": v_mean, "v_sd": v_sd, "v_by_jid": v_by_jid,
                "pca_mean": pca_mean, "comps": comps, "scale": scale,
                "k": k_eff, "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_rnafm_pca": k_eff}

    def predict(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        Xr = np.zeros((len(test_rows), model["n_rnafm_pca"]))
        r_by_jid = build_raw_by_jid(test_rows, cache)
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            if j in model["v_by_jid"]:
                Xv[i] = vienna_transform([j], model["v_by_jid"], model["v_mean"], model["v_sd"])[0]
            if j in r_by_jid:
                Xr[i] = _apply_pca(r_by_jid[j][None, :], model["pca_mean"],
                                   model["comps"], model["scale"])[0]
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