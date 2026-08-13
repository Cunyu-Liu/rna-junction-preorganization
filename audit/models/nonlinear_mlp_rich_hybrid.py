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
from audit.benchmark.vienna_local_context_features import build_raw_by_jid as lctx_build_raw
from audit.benchmark.vienna_local_context_features import fit_scaler as lctx_fit_scaler
from audit.benchmark.vienna_local_context_features import transform as lctx_transform
from audit.models.nonlinear_mlp_hybrid import (
    _train_mlp,
    _nuisance_basis as _nuisance_basis_11,
    LR, MAX_EPOCHS, PATIENCE, LOSS_TOL, PLATEAU_WINDOW, PLATEAU_REL_TOL,
    BATCH, GRAD_TOL,
)
from audit.models.rnafm_pca_linear_hybrid import _fit_pca, _apply_pca

try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAVE_TORCH = False

DEFAULT_K = 64
_DEFAULT_WD = 1e-3


def _nuisance_basis(rows, motifs, scafs):
    return _nuisance_basis_11(rows, motifs, scafs)


def make_nonlinear_mlp_extended_hybrid(hidden=(64, 32), dropout=0.0,
                                       weight_decay=None, seed=None):
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
                                             else _DEFAULT_WD),
                               seed=seed)
        return {"kind": "nonlinear_mlp_extended_hybrid", "net": net, "gate": gate,
                "motifs": motifs, "scafs": scafs, "mean": mean, "sd": sd,
                "by_jid": by_jid, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "device": device, "hidden": list(hidden),
                "dropout": dropout,
                "weight_decay": (weight_decay if weight_decay is not None
                                 else _DEFAULT_WD),
                "seed": seed}

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


def make_nonlinear_mlp_extended_hybrid_reg_deep(seed=None):
    """Extended-MLP with a third hidden layer (96,64,32) at the reference reg.

    Probe whether depth adds representational power on the richer folding
    representation, again under the reference regularization budget.
    """
    return make_nonlinear_mlp_extended_hybrid(hidden=(96, 64, 32), dropout=0.1,
                                              weight_decay=1e-2, seed=seed)


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


_SIGMA_FLOOR = 0.05          # matches the row_nll sigma clamp
_CAP = -7.1
_HET_SEED = 17


def _train_mlp_het(Xtr, ytr, cens_tr, device, in_dim, hidden=(96, 64, 32),
                   dropout=0.1, weight_decay=1e-2):
    """Train a two-output MLP that learns BOTH mu and a heteroscedastic sigma.

    The base reg_deep models fix sigma=0.7 in the right-censored Gaussian NLL,
    so the head only ever optimizes mu.  This variant gives the model a second
    output that predicts per-input sigma = softplus(raw) + floor, trained with
    the EXACT right-censored Gaussian NLL used by row_nll:

      measured : 0.5*log(2pi) + log(sigma) + 0.5*((y-mu)/sigma)^2
      censored : -log Phi((mu-CAP)/sigma)

    Because sigma enters both the measured and censored terms, the model cannot
    cheat by inflating sigma (that blows up the censored loss).  Early stopping
    and the eligibility gate mirror _train_mlp so the fold remains auditable.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from audit.models.nonlinear_mlp_hybrid import (
        MAX_EPOCHS, PATIENCE, LOSS_TOL, PLATEAU_WINDOW, PLATEAU_REL_TOL,
        BATCH, GRAD_TOL,
    )

    class _MLP2(nn.Module):
        def __init__(self, in_dim, hidden):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                d = h
            self.shared = nn.Sequential(*layers)
            self.mu_head = nn.Linear(d, 1)
            self.sigma_head = nn.Linear(d, 1)

        def forward(self, x):
            h = self.shared(x)
            mu = self.mu_head(h).squeeze(-1)
            raw = self.sigma_head(h).squeeze(-1)
            sigma = F.softplus(raw) + _SIGMA_FLOOR
            return mu, sigma

    def het_nll(mu, sigma, y, cens):
        z = (y - mu) / sigma
        nll_m = 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793,
                                             device=mu.device)) + torch.log(sigma) + 0.5 * z * z
        a = (mu - _CAP) / sigma
        log_phi = torch.special.log_ndtr(a.clamp(min=-30.0, max=30.0))
        nll_c = -log_phi
        return torch.where(cens, nll_c, nll_m).mean()

    torch.manual_seed(_HET_SEED)
    net = _MLP2(in_dim, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    ct = torch.tensor(cens_tr, dtype=torch.bool, device=device)
    n = Xt.shape[0]

    best_loss = float("inf")
    best_state = None
    epochs_since_best = 0
    n_epochs = 0
    final_loss = float("inf")
    loss_history = []
    plateau_reached = False
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            mu, sigma = net(Xt[idx])
            loss = het_nll(mu, sigma, yt[idx], ct[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_losses))
        loss_history.append(final_loss)
        n_epochs = epoch + 1
        if final_loss < best_loss - LOSS_TOL:
            best_loss = final_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                plateau_reached = True
                break
        if len(loss_history) >= PLATEAU_WINDOW:
            base = abs(loss_history[-PLATEAU_WINDOW]) or 1e-8
            rel = abs(final_loss - loss_history[-PLATEAU_WINDOW]) / base
            if rel < PLATEAU_REL_TOL:
                plateau_reached = True
                break

    net.load_state_dict(best_state)
    net.eval()

    net.zero_grad()
    mu_all, sigma_all = net(Xt)
    lossg = het_nll(mu_all, sigma_all, yt, ct)
    lossg.backward()
    gn = 0.0
    for p in net.parameters():
        if p.grad is not None:
            gn += float(torch.norm(p.grad).item() ** 2)
    total_grad_norm = float(np.sqrt(gn))
    net.eval()

    params_finite = all(bool(torch.isfinite(p).all().item()) for p in net.parameters())
    converged = bool(plateau_reached and params_finite and np.isfinite(final_loss))
    eligible = bool(converged and np.isfinite(total_grad_norm))
    gate = {
        "eligible": eligible, "converged": converged,
        "final_grad_norm": total_grad_norm, "grad_tol": GRAD_TOL,
        "n_epochs": n_epochs, "max_epochs": MAX_EPOCHS,
        "final_train_nll": final_loss, "best_train_nll": best_loss,
        "plateau_reached": plateau_reached, "success": converged,
        "n_nan_inf_params": int(not params_finite),
    }
    return net, gate


_T_SEED = 23
_DEFAULT_DF = 5.0


def _betacf(a, b, x, itmax=200, eps=3e-7, fpmin=1e-30):
    """Lentz's continued fraction for the incomplete beta (differentiable).

    Numerical-Recipes betacf, vectorized over the batch and run in float32 (or
    the input precision).  Every op is differentiable so gradients flow through
    the censored Student-t survival to mu.
    """
    import torch
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(torch.abs(d) < fpmin, torch.full_like(d, fpmin), d)
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        mf = float(m)
        m2 = 2.0 * mf
        aa = (mf * (b - mf) * x) / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < fpmin, torch.full_like(d, fpmin), d)
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < fpmin, torch.full_like(c, fpmin), c)
        d = 1.0 / d
        h = h * d * c
        aa = -((a + mf) * (qab + mf) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < fpmin, torch.full_like(d, fpmin), d)
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < fpmin, torch.full_like(c, fpmin), c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if bool((torch.abs(delta - 1.0) < eps).all()):
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a,b) (differentiable), NR betai."""
    import torch
    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    x = x.clamp(1e-12, 1.0 - 1e-12)
    switch = x > (a + 1.0) / (a + b + 2.0)
    xs = torch.where(switch, 1.0 - x, x)
    aa = torch.where(switch, b, a)
    bb = torch.where(switch, a, b)
    bts = torch.exp(
        torch.special.gammaln(aa + bb) - torch.special.gammaln(aa)
        - torch.special.gammaln(bb) + aa * torch.log(xs) + bb * torch.log1p(-xs))
    cf = _betacf(aa, bb, xs)
    i = bts * cf / aa
    return torch.where(switch, 1.0 - i, i)


if HAVE_TORCH:

    class _StudentTSurvival(torch.autograd.Function):
        """S(a)=P(T_nu>a) with correct value (incomplete beta) and gradient dS/da=-f(a).

        The naive ``S(a)=1-F(a)`` via ``x=nu/(nu+a^2)`` is even in ``a``, so its
        autograd gradient vanishes at a=0 even though the true dS/da=-f(a) is
        nonzero at the median.  We therefore keep the beta-computed VALUE but supply
        the exact analytic gradient in backward (f = standard Student-t pdf).
        """

        @staticmethod
        def forward(ctx, a, nu):
            ctx.nu = float(nu)
            ctx.save_for_backward(a)
            with torch.no_grad():
                x = (nu / (nu + a * a)).clamp(1e-12, 1.0 - 1e-12)
                ib = _betai(nu / 2.0, 0.5, x)
                F = torch.where(a >= 0, 1.0 - 0.5 * ib, 0.5 * ib)
                S = (1.0 - F).clamp(min=1e-12, max=1.0)
            return S

        @staticmethod
        def backward(ctx, grad_output):
            a = ctx.saved_tensors[0]
            nu = ctx.nu
            dist = torch.distributions.StudentT(
                df=torch.full_like(a, nu), loc=torch.zeros_like(a),
                scale=torch.ones_like(a))
            f = torch.exp(dist.log_prob(a))
            return -f * grad_output, None


def _student_t_survival(a, nu):
    """P(T_nu > a) for a standard Student-t (differentiable, exact gradient)."""
    return _StudentTSurvival.apply(
        a, torch.as_tensor(float(nu), device=a.device, dtype=a.dtype))


def _t_right_censored_nll(mu, y, cens, df, sigma=0.7, cap=-7.1):
    """Right-censored Student-t NLL (mean over batch), heavier-tailed than Gaussian.

    measured rows : nll_m = -log f_t(y; mu, sigma, df)
        = log(sigma) + 0.5*(df+1)*log(1 + ((y-mu)/sigma)^2 / df) + const(df)
    censored rows : nll_c = -log P(Y >= cap) = -log S_t((cap-mu)/sigma)

    The measured term grows only logarithmically in |z| (vs quadratically for the
    Gaussian), so a few extreme / outlier rows (the catastrophic folds that drag
    mean evaluation NLL) exert far less pull on mu.
    """
    import torch
    dist = torch.distributions.StudentT(
        df=torch.full_like(mu, float(df)), loc=mu, scale=sigma)
    nll_m = -dist.log_prob(y)
    a = (cap - mu) / sigma
    nll_c = -torch.log(_student_t_survival(a, float(df)))
    return torch.where(cens, nll_c, nll_m).mean()


def _train_mlp_t(Xtr, ytr, cens_tr, device, in_dim, hidden=(96, 64, 32),
                 dropout=0.1, weight_decay=1e-2, df=_DEFAULT_DF,
                 seed=_T_SEED, swa_n=0,
                 lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 loss_tol=LOSS_TOL, plateau_window=PLATEAU_WINDOW,
                 plateau_rel_tol=PLATEAU_REL_TOL):
    """Train the reg_deep MLP with the robust right-censored Student-t objective.

    Mirrors _train_mlp (same architecture, Adam + weight decay, plateau-based
    early stopping, eligibility gate) but swaps the Gaussian NLL for the
    Student-t NLL with `df` degrees of freedom.  Fixed sigma=0.7 and cap=-7.1
    match the evaluation metric; only the training objective differs.
    """
    import torch
    from audit.models.nonlinear_mlp_hybrid import (
        _MLP, MAX_EPOCHS, PATIENCE, LOSS_TOL, PLATEAU_WINDOW, PLATEAU_REL_TOL,
        BATCH, GRAD_TOL,
    )
    torch.manual_seed(seed)
    net = _MLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    ct = torch.tensor(cens_tr, dtype=torch.bool, device=device)
    n = Xt.shape[0]

    best_loss = float("inf")
    best_state = None
    epochs_since_best = 0
    n_epochs = 0
    final_loss = float("inf")
    loss_history = []
    plateau_reached = False
    # SWA: rolling average of the last `swa_n` epochs' weights.
    swa_sum = None        # running sum (float32) of recent weights
    swa_epochs = []       # recent state dicts (kept only if swa_n>0)
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            mu = net(Xt[idx]).squeeze(-1)
            loss = _t_right_censored_nll(mu, yt[idx], ct[idx], df=df)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_losses))
        loss_history.append(final_loss)
        n_epochs = epoch + 1
        if swa_n > 0:
            sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            swa_epochs.append(sd)
            if swa_sum is None:
                swa_sum = {k: v.detach().clone() for k, v in sd.items()}
            else:
                for k in sd:
                    swa_sum[k].add_(sd[k])
            if len(swa_epochs) > swa_n:
                dropped = swa_epochs.pop(0)
                for k in sd:
                    swa_sum[k].sub_(dropped[k])
        if final_loss < best_loss - loss_tol:
            best_loss = final_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                plateau_reached = True
                break
        if len(loss_history) >= plateau_window:
            base = abs(loss_history[-plateau_window]) or 1e-8
            rel = abs(final_loss - loss_history[-plateau_window]) / base
            if rel < plateau_rel_tol:
                plateau_reached = True
                break

    if swa_n > 0 and swa_sum is not None and len(swa_epochs) > 0:
        nk = len(swa_epochs)
        swa_state = {k: (swa_sum[k] / float(nk)).clone() for k in swa_sum}
        if all(bool(torch.isfinite(swa_state[k]).all().item()) for k in swa_state):
            net.load_state_dict(swa_state)
        else:
            net.load_state_dict(best_state)
    else:
        net.load_state_dict(best_state)
    net.eval()

    net.zero_grad()
    mu_all = net(Xt).squeeze(-1)
    lossg = _t_right_censored_nll(mu_all, yt, ct, df=df)
    lossg.backward()
    gn = 0.0
    for p in net.parameters():
        if p.grad is not None:
            gn += float(torch.norm(p.grad).item() ** 2)
    total_grad_norm = float(np.sqrt(gn))
    net.eval()

    params_finite = all(bool(torch.isfinite(p).all().item()) for p in net.parameters())
    converged = bool(plateau_reached and params_finite and np.isfinite(final_loss))
    eligible = bool(converged and np.isfinite(total_grad_norm))
    gate = {
        "eligible": eligible, "converged": converged,
        "final_grad_norm": total_grad_norm, "grad_tol": GRAD_TOL,
        "n_epochs": n_epochs, "max_epochs": MAX_EPOCHS,
        "final_train_nll": final_loss, "best_train_nll": best_loss,
        "plateau_reached": plateau_reached, "success": converged,
        "n_nan_inf_params": int(not params_finite), "df": float(df),
    }
    return net, gate


def make_nonlinear_mlp_extended_hybrid_het(hidden=(96, 64, 32), dropout=0.1,
                                           weight_decay=1e-2):
    """Extended-Vienna(21) MLP with a learned heteroscedastic sigma head.

    Same reg_deep feature block as the winning ensemble member, but the head
    outputs both mu and sigma (softplus-parameterized, floored at 0.05),
    trained with the exact right-censored Gaussian NLL.  Tests whether letting
    the model learn per-input uncertainty (instead of the fixed 0.7) lowers the
    evaluation NLL further.
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
        net, gate = _train_mlp_het(X, y, cens, device, X.shape[1], hidden=hidden,
                                   dropout=dropout, weight_decay=weight_decay)
        return {"kind": "nonlinear_mlp_extended_hybrid_het", "net": net,
                "gate": gate, "motifs": motifs, "scafs": scafs, "mean": mean,
                "sd": sd, "by_jid": by_jid, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "device": device, "hidden": list(hidden),
                "dropout": dropout, "weight_decay": weight_decay}

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
            mu, sigma = model["net"](Xt)
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
        from scipy.special import log_ndtr
        a = (mu + 7.1) / sigma
        cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
        seen_scaf = np.zeros(len(mu), dtype=bool)
        for i, r in enumerate(test_rows):
            if int(r["scaf"]) in model["scafs"]:
                seen_scaf[i] = True
        return mu, sigma, cp, seen_scaf, ~seen_scaf

    return fit, predict


def make_nonlinear_mlp_extended_hybrid_localctx(hidden=(96, 64, 32), dropout=0.1,
                                                weight_decay=1e-2):
    """reg_deep MLP on nuisance + extended-Vienna(21) + join-local-context(24).

    Adds the position-anchored join-local-context one-hot block (see
    vienna_local_context_features) to the winning reg_deep feature set.  Tests
    whether explicit edit-junction region sequence identity adds signal beyond
    the folding aggregates that smooth it over.
    """
    def fit(train_rows):
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})

        v_by_jid = vienna_ext_build_raw(train_rows)
        v_mean, v_sd = vienna_ext_fit_scaler(tr_jids, v_by_jid)
        l_by_jid = lctx_build_raw(train_rows)
        l_mean, l_sd = lctx_fit_scaler(tr_jids, l_by_jid)

        Xv = np.zeros((len(train_rows), len(v_mean)))
        Xl = np.zeros((len(train_rows), len(l_mean)))
        for i, r in enumerate(train_rows):
            j = str(r["jid"])
            Xv[i] = vienna_ext_transform([j], v_by_jid, v_mean, v_sd)[0]
            Xl[i] = lctx_transform([j], l_by_jid, l_mean, l_sd)[0]

        X = np.hstack([Xn, Xv, Xl])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net, gate = _train_mlp(X, y, cens, device, X.shape[1], hidden=hidden,
                               dropout=dropout, weight_decay=weight_decay)
        return {"kind": "nonlinear_mlp_extended_hybrid_localctx", "net": net,
                "gate": gate, "motifs": motifs, "scafs": scafs,
                "v_mean": v_mean, "v_sd": v_sd, "v_by_jid": v_by_jid,
                "l_mean": l_mean, "l_sd": l_sd, "l_by_jid": l_by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "n_localctx": Xl.shape[1], "device": device,
                "hidden": list(hidden), "dropout": dropout,
                "weight_decay": weight_decay}

    def predict(model, test_rows):
        import torch
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        v_by_jid = vienna_ext_build_raw(test_rows)
        l_by_jid = lctx_build_raw(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        Xl = np.zeros((len(test_rows), model["n_localctx"]))
        for i, r in enumerate(test_rows):
            j = str(r["jid"])
            Xv[i] = vienna_ext_transform([j], v_by_jid, model["v_mean"], model["v_sd"])[0]
            Xl[i] = lctx_transform([j], l_by_jid, model["l_mean"], model["l_sd"])[0]
        X = np.hstack([Xn, Xv, Xl])
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


def make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=_DEFAULT_DF,
                                                  hidden=(96, 64, 32),
                                                  dropout=0.1,
                                                  weight_decay=1e-2,
                                                  seed=_T_SEED,
                                                  swa_n=0):
    """reg_deep MLP trained with a robust right-censored Student-t objective.

    Same winning feature block (nuisance + 21-D extended-Vienna, reg_deep arch)
    as the best r14 single model, but the head minimizes a heavy-tailed
    Student-t NLL with `df` degrees of freedom instead of the Gaussian NLL.
    Heavy tails down-weight extreme / outlier rows, so a few catastrophic folds
    (which drag the mean evaluation NLL) should exert far less pull on mu.
    Prediction still returns mu with the fixed sigma=0.7 and the Gaussian
    evaluation NLL is unchanged -- only the training objective differs.
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
        net, gate = _train_mlp_t(X, y, cens, device, X.shape[1], hidden=hidden,
                                 dropout=dropout, weight_decay=weight_decay,
                                 df=df, seed=seed, swa_n=swa_n)
        return {"kind": "nonlinear_mlp_extended_hybrid_reg_deep_t", "net": net,
                "gate": gate, "motifs": motifs, "scafs": scafs, "mean": mean,
                "sd": sd, "by_jid": by_jid, "n_nuisance": Xn.shape[1],
                "n_vienna": Xv.shape[1], "device": device, "hidden": list(hidden),
                "dropout": dropout, "weight_decay": weight_decay, "df": float(df),
                "seed": int(seed), "swa_n": int(swa_n)}

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
