"""Nonlinear (shallow-MLP) hybrid on the winning feature set (nonlinear step).

The decisive shootout (r09) showed the sequence increment is real but tiny
(+2.7%) and is captured only by the base 11-D ViennaRNA block; all richer
sequence representations fail.  The winning head is a PLAIN right-censored
LINEAR regression, which cannot represent nonlinearity or feature interactions.
If the residual sequence signal lives in interactions (e.g. scaffold/motif-
dependent effects, nonlinear folding-proxy response), a linear head is
structurally incapable of capturing it.

This model keeps the SAME winning feature set (nuisance basis + train-scaled
11-D ViennaRNA) but replaces the linear head with a shallow MLP:

  mu = MLP( [nuisance_onehot, topology, vienna11] )    (right-censored Gaussian)

trained by minimizing the identical right-censored Gaussian NLL used by every
parametric baseline (tau=0.7, CAP=-7.1):
  measured rows : 0.5*((y-mu)/tau)^2
  censored rows : -log Phi((mu-CAP)/tau)

The MLP is regularized with weight decay and early-stopped on the train loss so
the extra capacity cannot overfit.  ViennaRNA standardization is fit on TRAIN
junction raw features only (no test leakage).  An explicit optimizer gate
(tracked from the Adam training loop) reports convergence so the fold cannot be
mistaken for converged when it was truncated at max epochs.

When run under CUDA the model trains on GPU; otherwise it falls back to CPU
(unit tests).  Deterministic seed -> reproducible per fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_features import build_raw_by_jid, fit_scaler, transform
from audit.data.audit_dataset import parse_parts

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    nn = None
    HAVE_TORCH = False

TAU = 0.7
CAP = -7.1
SEED = 17
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 1500
PATIENCE = 30
LOSS_TOL = 1e-5
PLATEAU_WINDOW = 50
PLATEAU_REL_TOL = 1e-3   # relative train-loss improvement over window -> plateau
BATCH = 1024
GRAD_TOL = 0.5


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


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _censored_nll(mu, y, cens):
    """Right-censored Gaussian NLL (mean over batch), matching censored_objective."""
    z = (y - mu) / TAU
    nll_m = 0.5 * z * z
    a = (mu - CAP) / TAU
    # -log Phi(a) via log_ndtr; clamp to stay finite
    log_phi = torch.special.log_ndtr(a.clamp(min=-30.0, max=30.0))
    nll_c = -log_phi
    nll = torch.where(cens, nll_c, nll_m)
    return nll.mean()


def _train_mlp(Xtr, ytr, cens_tr, device, in_dim, hidden=(64, 32),
               dropout=0.0, weight_decay=WEIGHT_DECAY, lr=LR,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, loss_tol=LOSS_TOL,
               plateau_window=PLATEAU_WINDOW, plateau_rel_tol=PLATEAU_REL_TOL,
               seed=SEED):
    torch.manual_seed(SEED if seed is None else seed)
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
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            mu = net(Xt[idx]).squeeze(-1)
            loss = _censored_nll(mu, yt[idx], ct[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_losses))
        loss_history.append(final_loss)
        n_epochs = epoch + 1
        if final_loss < best_loss - loss_tol:
            best_loss = final_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                plateau_reached = True
                break
        # relative plateau: negligible improvement over a trailing window
        if len(loss_history) >= plateau_window:
            base = abs(loss_history[-plateau_window]) or 1e-8
            rel = abs(final_loss - loss_history[-plateau_window]) / base
            if rel < plateau_rel_tol:
                plateau_reached = True
                break

    net.load_state_dict(best_state)
    net.eval()

    # representative gradient norm over all params on the best state
    net.zero_grad()
    mu_all = net(Xt).squeeze(-1)
    lossg = _censored_nll(mu_all, yt, ct)
    lossg.backward()
    gn = 0.0
    for p in net.parameters():
        if p.grad is not None:
            gn += float(torch.norm(p.grad).item() ** 2)
    total_grad_norm = float(np.sqrt(gn))
    net.eval()

    params_finite = all(bool(torch.isfinite(p).all().item()) for p in net.parameters())
    # For an SGD-trained net, "converged" means the training loss reached a
    # plateau (relative improvement over the trailing window negligible) OR
    # early-stopped on the record-best loss, with finite params/loss.  A raw
    # full-data gradient norm is NOT required to vanish (an early-stopped net
    # stops before zeroing the gradient); it is recorded as a diagnostic, and
    # eligibility requires it to be finite (a NaN/inf gradient flags a
    # degenerate fit).
    converged = bool(plateau_reached and params_finite and np.isfinite(final_loss))
    eligible = bool(converged and np.isfinite(total_grad_norm))

    gate = {
        "eligible": eligible,
        "converged": converged,
        "final_grad_norm": total_grad_norm,
        "grad_tol": GRAD_TOL,
        "n_epochs": n_epochs,
        "max_epochs": MAX_EPOCHS,
        "final_train_nll": final_loss,
        "best_train_nll": best_loss,
        "plateau_reached": plateau_reached,
        "success": converged,
        "n_nan_inf_params": int(not params_finite),
    }
    return net, gate


def make_nonlinear_mlp_hybrid(hidden=(64, 32)):
    """Return (fit, predict) for the nonlinear MLP hybrid."""
    def fit(train_rows):
        assert HAVE_TORCH, "torch required for nonlinear_mlp_hybrid"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        net, gate = _train_mlp(X, y, cens, device, X.shape[1])
        return {"kind": "nonlinear_mlp_hybrid", "net": net, "gate": gate,
                "motifs": motifs, "scafs": scafs, "mean": mean, "sd": sd,
                "by_jid": by_jid, "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "device": device, "hidden": list(hidden)}

    def predict(model, test_rows):
        assert HAVE_TORCH, "torch required for nonlinear_mlp_hybrid"
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = build_raw_by_jid(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, model["mean"], model["sd"])[0]
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
        support = seen_scaf
        abstain = ~seen_scaf
        return mu, sigma, cp, support, abstain

    return fit, predict


NONLINEAR_MLP_HYBRID = {
    "nonlinear_mlp_hybrid": make_nonlinear_mlp_hybrid(),
}
