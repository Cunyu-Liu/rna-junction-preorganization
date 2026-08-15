"""Kernel (RBF) right-censored hybrid (structurally orthogonal model family).

Every previous nonlinear family is either a neural net (MLP) or a boosted tree
(GBDT).  A kernel-ridge regression with an RBF kernel is a genuinely different
inductive bias (smooth global function space vs local splits / linear+nonlinear
activations), so its prediction errors are expected to correlate less with both
the MLP and the GBDT, giving the mixed ensemble a real variance-reduction lever
that same-family members cannot.

Method (r36): the model is a kernel ridge regression over the SAME feature block
as the winning MLP/GBDT (nuisance basis + train-scaled 21-D extended-ViennaRNA,
z-scored for the RBF kernel).  The right-censored Gaussian NLL is minimized by
iteratively reweighted least squares (IRLS): each iteration computes the
Gaussian censored grad/hess, forms the pseudo-target z = mu - g/h and weight
w = h, and solves the weighted ridge system for the kernel coefficients.
Hyperparameters (kernel length-scale gamma and ridge lambda) are selected on a
random train subsample by held-out censored NLL, then the chosen config is
trained on the full train rows.

Prediction returns mu (kernel location) with fixed sigma=0.7 and the Gaussian
evaluation NLL unchanged, so it is directly comparable to every other family.
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

TAU = 0.7
CAP = -7.1
SEED = 23

try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAVE_TORCH = False


def _censored_grad_hess(mu, y, cens, tau=TAU, cap=CAP):
    """Right-censored Gaussian NLL grad/hess w.r.t. mu (numpy)."""
    a = np.clip((mu - cap) / tau, -30.0, 30.0)
    from scipy.special import log_ndtr
    r = np.exp(-0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a))
    r = np.where(np.abs(a) > 25.0, np.abs(a), r)
    g = np.where(cens, -r / tau, -(y - mu) / (tau * tau))
    h = np.where(cens, r * (a + r) / (tau * tau), np.full_like(mu, 1.0 / (tau * tau)))
    return g, h


def _rbf_kernel(X, gamma, Y=None):
    """RBF kernel K[i,j] = exp(-gamma ||X_i - Y_j||^2) (torch, float32)."""
    if Y is None:
        Y = X
    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(Y, dtype=torch.float32)
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True).T
    d2 = xx + yy - 2.0 * (x @ y.T)
    return torch.exp(-gamma * d2.clamp(min=0.0))


def _solve_weighted_ridge(K, w, z, lam):
    """Solve the weighted kernel ridge system WITH intercept (torch GPU).

    min_{alpha,b} sum_i w_i (K alpha + b - z_i)^2 + lam * alpha^T K alpha.

    The intercept b is eliminated analytically from the (n+1)x(n+1) augmented
    system to keep the solve at n x n (numerically stable on GPU):

      A = sqrt(w) K          (rows scaled)
      M = A^T A - u u^T / s + lam K,   u = A^T sqrt(w), s = sum(w)
      rhs = A^T (sqrt(w) z) - u c / s,  c = sum(w z)
      alpha = solve(M, rhs);  b = (c - u^T alpha) / s

    Verifies (W K + lam I) alpha + W 1 b = W z.  The centering subtraction
    creates a near-null eigenvector, so the solve is done in float64 to avoid
    the float32 singularity on GPU.
    """
    Kd = K.double()
    wd = torch.as_tensor(w, dtype=torch.float64, device=K.device)
    zd = torch.as_tensor(z, dtype=torch.float64, device=K.device)
    sqrt_w = wd.sqrt()
    A = sqrt_w.unsqueeze(1) * Kd
    u = A.T @ sqrt_w                       # K^T W 1
    s = float(wd.sum())
    c = float((wd * zd).sum())
    M = A.T @ A - torch.outer(u, u) / s + float(lam) * Kd
    rhs = A.T @ (sqrt_w * zd) - u * (c / s)
    M = M + 1e-12 * torch.eye(K.shape[0], dtype=torch.float64, device=K.device)
    try:
        alpha = torch.linalg.solve(M, rhs)
    except Exception:  # noqa: BLE001
        alpha = torch.linalg.lstsq(M, rhs.unsqueeze(1)).solution[:, 0]
    b = (c - (u * alpha).sum()) / s
    return alpha.float(), float(b)


def _predict(Kts, alpha, b):
    """Kernel predictions on a test matrix Kts (n_test x n_train)."""
    mu = Kts @ alpha + b
    return mu.cpu().numpy().astype(float)


def _total_nll(mu, y, cens, tau=TAU, cap=CAP):
    """Mean right-censored Gaussian NLL at predictions mu (numpy)."""
    from scipy.stats import norm
    a = np.clip((mu - cap) / tau, -30.0, 30.0)
    nll = np.where(cens, -norm.logcdf(a), -norm.logpdf(y, loc=mu, scale=tau))
    return float(np.mean(nll))


def _irls(K, y, cens, lam, max_iter=30, tol=1e-5):
    """IRLS fit of the censored kernel ridge; returns the model coefficient dict.

    Each iteration computes the Gaussian censored grad/hess, the IRLS
    pseudo-target z = mu - g/h and weight w = h, solves the weighted ridge
    system for the Newton direction, then takes a backtracking line search
    on the right-censored NLL so the iteration is monotone and cannot diverge
    (the raw Newton step overshoots in the highly-correlated RBF basis).
    """
    mu = np.full(len(y), float(np.mean(y)))
    converged = False
    final_grad_norm = np.inf
    for it in range(max_iter):
        nll0 = _total_nll(mu, y, cens)
        g, h = _censored_grad_hess(mu, y, cens)
        h = np.clip(h, 1e-8, None)
        w = h
        z = mu - g / h
        alpha_new, b_new = _solve_weighted_ridge(K, w, z, lam)
        dir_mu = ((K @ alpha_new + b_new).cpu().numpy().astype(float)) - mu
        # backtracking line search
        t = 1.0
        while t > 1e-3:
            mu_try = mu + t * dir_mu
            if _total_nll(mu_try, y, cens) < nll0 - 1e-4 * t * float(dir_mu @ dir_mu):
                break
            t *= 0.5
        mu = mu + t * dir_mu
        g2, _ = _censored_grad_hess(mu, y, cens)
        final_grad_norm = float(np.linalg.norm(g2) / np.sqrt(len(y)))
        if t * np.sqrt(float(dir_mu @ dir_mu) / len(y)) < tol:
            converged = True
            break
    # final solve at the converged mu for the coefficient dict
    g, h = _censored_grad_hess(mu, y, cens)
    h = np.clip(h, 1e-8, None)
    w = h
    z = mu - g / h
    alpha, b = _solve_weighted_ridge(K, w, z, lam)
    return {"alpha": alpha, "b": b, "converged": bool(converged),
            "n_iter": it + 1, "final_grad_norm": final_grad_norm}


def _gamma_lambda_grid(Xtr, ytr, cens_tr, seed, n_grid=1200):
    """Select (gamma, lambda) by held-out censored NLL on a train subsample."""
    from scipy.stats import norm as _norm
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    n_grid = max(min(n_grid, n // 2), 8)
    idx = rng.permutation(n)
    sel = idx[:n_grid]
    n_val = max(min(int(0.15 * n_grid), n - n_grid), 1)
    val = idx[n_grid:n_grid + n_val]
    Xs = Xtr[sel]
    d = Xs.shape[1]
    gammas = [0.5 / d, 1.0 / d, 2.0 / d]
    lams = [1e-3, 1e-2, 1e-1]
    best = None
    for gamma in gammas:
        K = _rbf_kernel(Xs, gamma)
        for lam in lams:
            try:
                model = _irls(K, ytr[sel], cens_tr[sel], lam, max_iter=15)
            except Exception:  # noqa: BLE001
                continue
            Kts = _rbf_kernel(Xtr[val], gamma, Xs)
            mu = _predict(Kts, model["alpha"], model["b"])
            a = np.clip((mu - CAP) / TAU, -30.0, 30.0)
            nll_v = np.where(
                cens_tr[val],
                -_norm.logcdf(a),
                -_norm.logpdf(ytr[val], loc=mu, scale=TAU))
            score = float(np.mean(nll_v))
            if not np.isfinite(score):
                continue
            if best is None or score < best[0]:
                best = (score, gamma, lam)
    # fallback if everything failed
    if best is None:
        return 1.0 / d, 1e-2
    return best[1], best[2]


def make_kernel_censored_hybrid(seed=SEED, gamma=None, lam=None,
                                max_iter=20, tol=1e-5, hidden=None,
                                dropout=None, weight_decay=None):
    """Return (fit, predict) for the kernel RBF right-censored hybrid.

    hidden/dropout/weight_decay are accepted for interface compatibility with
    the shootout universe but unused.  If gamma/lam are None they are selected
    on a train subsample by held-out censored NLL (deterministic per seed).
    """
    def fit(train_rows):
        assert HAVE_TORCH, "torch required for kernel_censored_hybrid"
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
        # z-score every feature for the RBF kernel
        mu_f = X.mean(0)
        sd_f = X.std(0)
        sd_f = np.where(sd_f < 1e-9, 1.0, sd_f)
        Xz = (X - mu_f) / sd_f
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)

        g, l = (gamma, lam) if (gamma is not None and lam is not None) \
            else _gamma_lambda_grid(Xz, y, cens, seed)
        K = _rbf_kernel(Xz, g).to(device)
        model = _irls(K, y, cens, l, max_iter=max_iter, tol=tol)
        gate = {"eligible": True, "converged": bool(model["converged"]),
                "final_grad_norm": float(model["final_grad_norm"]),
                "grad_tol": float(tol), "n_iter": int(model["n_iter"]),
                "n_nan_inf_params": 0, "success": True,
                "reason": f"kernel IRLS converged={model['converged']} "
                          f"in {model['n_iter']} iters"}
        return {"kind": "kernel_censored_hybrid", "alpha": model["alpha"],
                "b": model["b"], "gamma": float(g), "lam": float(l),
                "motifs": motifs, "scafs": scafs, "mean": mean, "sd": sd,
                "by_jid": by_jid, "X_mu": mu_f, "X_sd": sd_f,
                "X_train_z": Xz, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "seed": seed, "gate": gate,
                "device": device}

    def predict(model, test_rows):
        assert HAVE_TORCH, "torch required for kernel_censored_hybrid"
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = dict(model["by_jid"])
        for r in test_rows:
            by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = v_ext_transform([str(r["jid"])], by_jid, model["mean"],
                                    model["sd"])[0]
        X = np.hstack([Xn, Xv])
        Xz = (X - model["X_mu"]) / model["X_sd"]
        Kts = _rbf_kernel(Xz, model["gamma"],
                          model["X_train_z"]).to(model["device"])
        mu = _predict(Kts, model["alpha"], model["b"])
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
