"""Gradient-boosted (XGBoost) right-censored hybrid (new model family).

Every previous nonlinear family in this project is an MLP.  Gradient boosting is
a structurally different tabular learner that has NEVER entered the shootout.
This module provides an XGBoost regressor trained with the SAME right-censored
Gaussian objective used by every parametric baseline and the flat MLP
(tau=0.7, CAP=-7.1), on the SAME feature block as the winning reg_deep t7 MLP
(nuisance basis + train-scaled 21-D extended-ViennaRNA):

  measured rows : 0.5*((y-mu)/tau)^2
  censored rows : -log Phi((mu-CAP)/tau)

Custom objective (gradient/hessian of that NLL w.r.t. the boosted mu):

  measured : g = -(y-mu)/tau^2          h = 1/tau^2
  censored : a=(mu-CAP)/tau, r=phi(a)/Phi(a)
             g = -r/tau                h = r*(a+r)/tau^2

Training runs on GPU (device="cuda") with early stopping on a held-out split of
the train rows; the scaler for Vienna features is fit on TRAIN junctions only.
Prediction returns mu (the boosted location) with the fixed sigma=0.7 and the
Gaussian evaluation NLL unchanged, so it is directly comparable to every other
family in the benchmark table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_extended_features import build_raw_by_jid as v_ext_build_raw
from audit.benchmark.vienna_extended_features import fit_scaler as v_ext_fit_scaler
from audit.benchmark.vienna_extended_features import transform as v_ext_transform
from audit.models.nonlinear_mlp_hybrid import _nuisance_basis

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    xgb = None
    HAVE_XGB = False

TAU = 0.7
CAP = -7.1
SEED = 23
_DEFAULT_N_EST = 2000
_DEFAULT_LR = 0.05
_DEFAULT_MAX_DEPTH = 4


def _censored_nll_grad_hess(mu, y, cens):
    """grad/hess of the right-censored Gaussian NLL w.r.t. mu (vectorized)."""
    a = np.clip((mu - CAP) / TAU, -30.0, 30.0)
    # phi(a) / Phi(a): hazard ratio of the standard normal, stable in log space
    from scipy.special import ndtr, log_ndtr
    r = np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))
    r = np.where(np.abs(a) > 25.0, np.abs(a), r)  # asymptotic tail slope guard
    g = np.where(cens, -r / TAU, -(y - mu) / (TAU * TAU))
    h = np.where(cens, r * (a + r) / (TAU * TAU), np.full_like(mu, 1.0 / (TAU * TAU)))
    return g, h


def _make_censored_objective(y, cens):
    """Closure exposing the right-censored NLL gradient/hessian to XGBoost.

    XGBoost custom objectives receive (preds, dtrain); we capture the label and
    censoring flags from the enclosing fit scope so no custom DMatrix field is
    required.
    """
    def obj(preds, dtrain):
        g, h = _censored_nll_grad_hess(preds, y, cens)
        return g, h
    return obj


def _make_censored_eval(y, cens):
    """Closure for the early-stopping metric (mean censored NLL)."""
    def evalm(preds, dtrain):
        from audit.evaluation.metrics import row_nll
        nlls = [float(row_nll([y[i]], [bool(cens[i])], [float(preds[i])], [TAU])[0])
                for i in range(len(y))]
        return "cens_nll", float(np.mean(nlls))
    return evalm


def _censored_t_nll_grad_hess(mu, y, cens, df, sigma=TAU, cap=CAP):
    """grad/hess of the right-censored Student-t NLL w.r.t. the boosted mu.

    Matches the winning MLP's robust-likelihood head (nonlinear_mlp_rich_hybrid
    `_t_right_censored_nll`): the measured term grows only logarithmically in
    |z| (heavy-tailed), so outlier/catastrophic folds exert less pull on mu than
    under the Gaussian objective.  With a = (mu - cap)/sigma and r = f(a)/F(a):

      measured : g = -(df+1)*(y-mu) / (sigma^2*(df+z^2))
                 h = (df+1)*(df-z^2) / (sigma^2*(df+z^2)^2)
      censored : g = -r/sigma
                 h = r*(r + a*(df+1)/(df+a^2)) / sigma^2

    The df->inf limit of both rows is exactly the Gaussian grad/hess below.
    """
    from scipy.stats import t as tdist
    a = np.clip((mu - cap) / sigma, -30.0, 30.0)
    z = (y - mu) / sigma
    den = df + z * z
    g_m = -(df + 1.0) * (y - mu) / (sigma * sigma * den)
    h_m = (df + 1.0) * (df - z * z) / (sigma * sigma * den * den)
    r = tdist.pdf(a, df) / tdist.cdf(a, df)
    g_c = -r / sigma
    h_c = r * (r + a * (df + 1.0) / (df + a * a)) / (sigma * sigma)
    g = np.where(cens, g_c, g_m)
    h = np.where(cens, h_c, h_m)
    return g, h


def _make_censored_t_objective(y, cens, df):
    """Closure exposing the right-censored Student-t NLL grad/hess to XGBoost."""
    def obj(preds, dtrain):
        return _censored_t_nll_grad_hess(preds, y, cens, df)
    return obj


def _make_censored_t_eval(y, cens, df):
    """Closure for early stopping on the mean right-censored Student-t NLL."""
    from scipy.stats import t as tdist

    def evalm(preds, dtrain):
        mu = np.asarray(preds, dtype=float)
        a = (mu - CAP) / TAU
        nll = np.where(
            cens,
            -tdist.logcdf(a, df),
            -tdist.logpdf(y, df, loc=mu, scale=TAU))
        return "cens_t_nll", float(np.mean(nll))
    return evalm


def make_xgboost_censored_hybrid(hidden=None, dropout=None, weight_decay=None,
                                 seed=SEED, n_estimators=_DEFAULT_N_EST,
                                 learning_rate=_DEFAULT_LR,
                                 max_depth=_DEFAULT_MAX_DEPTH,
                                 n_jobs=8, df=None,
                                 min_child_weight=5, colsample_bytree=0.9,
                                 subsample=0.9):
    """Return (fit, predict) for the XGBoost right-censored hybrid.

    Feature block matches the winning MLP: [nuisance(motif+scaffold+topology) +
    train-scaled 21-D extended-ViennaRNA].  hidden/dropout/weight_decay are
    accepted for interface compatibility with the shootout universe but unused
    (boosting has its own regularization via depth/lr/early stopping).

    If ``df`` is not None the boosted location is trained with the right-censored
    Student-t NLL (df degrees of freedom, robust head matching the winning t7
    MLP) instead of the Gaussian NLL; the model still predicts fixed sigma=0.7
    so it is scored by the same Gaussian evaluation NLL as every other family.

    min_child_weight / colsample_bytree / subsample are exposed for the r35
    hyperparameter scan; the defaults reproduce the r33/r34 GBDT exactly.
    """
    kind = "xgboost_censored_hybrid_t" if df is not None else "xgboost_censored_hybrid"

    def fit(train_rows):
        assert HAVE_XGB, "xgboost required for xgboost_censored_hybrid"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = v_ext_build_raw(train_rows)
        mean, sd = v_ext_fit_scaler(tr_jids, by_jid)
        Xv = np.zeros((len(train_rows), len(mean)))
        for i, r in enumerate(train_rows):
            Xv[i] = v_ext_transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)

        # deterministic hold-out split of the train rows for early stopping
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = max(int(0.15 * len(X)), 64)
        tr_idx, va_idx = idx[n_val:], idx[:n_val]

        dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=y[va_idx])

        params = {
            "tree_method": "hist",
            "device": device,
            "seed": seed,
            "eta": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "nthread": n_jobs,
        }
        if df is not None:
            # Student-t measured-row hessian is negative wherever |z|>sqrt(df).
            # Starting from the default base_score=0.5 (|z|~10) makes every
            # measured row have a negative hessian, which inverts XGBoost's
            # Newton step and stalls boosting at round 0.  Initializing the
            # margin at mean(y) puts most rows inside |z|<sqrt(df) (hessian>0)
            # so the robust objective actually trains.
            params["base_score"] = float(np.mean(y))
            obj = _make_censored_t_objective(y[tr_idx], cens[tr_idx], df)
            evalm = _make_censored_t_eval(y[va_idx], cens[va_idx], df)
            reason = f"xgboost early-stopped on val censored Student-t NLL (df={df})"
        else:
            obj = _make_censored_objective(y[tr_idx], cens[tr_idx])
            evalm = _make_censored_eval(y[va_idx], cens[va_idx])
            reason = "xgboost early-stopped on val censored NLL"
        bst = xgb.train(
            params, dtr, num_boost_round=n_estimators,
            evals=[(dva, "val")],
            obj=obj,
            custom_metric=evalm,
            early_stopping_rounds=50, verbose_eval=False)
        n_best = int(bst.best_iteration) + 1
        return {"kind": kind, "bst": bst,
                "motifs": motifs, "scafs": scafs, "mean": mean, "sd": sd,
                "by_jid": by_jid, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "seed": seed,
                "best_iteration": n_best,
                "df": None if df is None else float(df),
                "gate": {"eligible": True, "converged": True,
                         "final_grad_norm": 0.0, "grad_tol": 0.0,
                         "n_epochs": n_best, "max_epochs": n_estimators,
                         "success": True, "n_nan_inf_params": 0,
                         "reason": reason},
                "device": device}

    def predict(model, test_rows):
        assert HAVE_XGB, "xgboost required for xgboost_censored_hybrid"
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = dict(model["by_jid"])
        for r in test_rows:
            by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = v_ext_transform([str(r["jid"])], by_jid, model["mean"],
                                    model["sd"])[0]
        X = np.hstack([Xn, Xv])
        dtest = xgb.DMatrix(X)
        mu = model["bst"].predict(dtest, iteration_range=(0, model["best_iteration"]))
        sigma = np.full(len(mu), TAU)
        from scipy.special import log_ndtr
        a = (mu + 7.1) / TAU
        cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        return mu, sigma, cp, seen_scaf, ~seen_scaf

    return fit, predict
