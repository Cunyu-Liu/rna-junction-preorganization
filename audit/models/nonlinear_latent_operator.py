"""Nonlinear latent-operator head: MLP junction map + operator-aware calibration.

This closes the contract §9.1 gap: the "unified operator-aware head" was only
ever evaluated with a LINEAR sequence map (v1.31 / vienna_latent_operator /
no_sequence_latent_operator all use  q_j ~ N(X_j @ theta, sigma_q^2)), while the
winning nonlinear MLP family uses a FLAT head (mu = MLP(x_row)) with no explicit
junction latent and no scaffold-specific slope.  Neither the linear latent-operator
family nor the flat MLP family combines the two:

    q_j ~ N(f_theta(x_j), sigma_q^2)      f_theta = nonlinear MLP (junction-level)
    Y_js | q_j ~ N(a_s + b_s q_j, tau^2)  operator-aware head (scaffold-specific)
    right-censored marginal via Gauss-Hermite (GH48), trained jointly.

Features x_j are JUNCTION-level only (motif one-hot + topology + train-scaled
21-D extended-ViennaRNA): scaffold identity is NOT in the MLP input because it
enters through the operator head a_s/b_s, exactly as in v1.31.  This keeps the
comparison to the flat MLP clean: same representation budget, different head
structure (explicit junction latent + scaffold slope vs direct row-level mu).

Training: full-batch Adam on the GH marginal right-censored (Student-t or
Gaussian) NLL, plateau-based early stopping, and the same eligibility gate as
the flat MLP family.  Prediction mirrors v1.31: mu = a_s + b_s * f(x_j) with
sigma = sqrt(tau^2 + b_s^2 * sigma_q^2); unseen scaffold -> abstain.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.vienna_extended_features import build_raw_by_jid as v_ext_build_raw
from audit.benchmark.vienna_extended_features import fit_scaler as v_ext_fit_scaler
from audit.benchmark.vienna_extended_features import transform as v_ext_transform
from audit.data.audit_dataset import parse_parts
from audit.models.nonlinear_mlp_hybrid import (
    LR, MAX_EPOCHS, PATIENCE, LOSS_TOL, PLATEAU_WINDOW, PLATEAU_REL_TOL,
    GRAD_TOL,
)
from audit.models.nonlinear_mlp_rich_hybrid import _student_t_survival
from audit.numerics.v131_corrected_objective import hermite

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
SIGMA_Q = 1.0
GH = 48
_DEFAULT_DF = 7.0          # match the winning t7 single member
_SEED = 23
_REF_SLOPE = 1.0           # ref scaffold slope fixed at 1 (matches v1.31)
_SLOPE_RIDGE = 5.0
_LOG_B_CLAMP = 1.5


def _junction_panel(train_rows):
    """Build junction-level grouping: (flat_j, flat_s, flat_y, flat_c) per row."""
    jids = sorted({str(r["jid"]) for r in train_rows})
    ji = {j: i for i, j in enumerate(jids)}
    scaffolds = sorted({int(r["scaf"]) for r in train_rows})
    si = {s: i for i, s in enumerate(scaffolds)}
    flat_j = [ji[str(r["jid"])] for r in train_rows]
    flat_s = [si[int(r["scaf"])] for r in train_rows]
    flat_y = [float(r["y"]) for r in train_rows]
    flat_c = [bool(r["cens"]) for r in train_rows]
    return {"jids": jids, "scaffolds": scaffolds,
            "flat_j": np.asarray(flat_j, dtype=int),
            "flat_s": np.asarray(flat_s, dtype=int),
            "flat_y": np.asarray(flat_y, dtype=float),
            "flat_c": np.asarray(flat_c, dtype=bool)}


def _junction_feats(rows, jids, motifs, mean, sd, by_jid):
    """Junction-level feature matrix: [motif one-hot, topology(3), vienna21].

    `by_jid` must contain a sequence entry for EVERY jid in `jids` (callers pass
    a builder that covers both train and test rows); the scaler (mean, sd) is
    always the train-only scaler to avoid leakage.
    """
    mi = {m: i for i, m in enumerate(motifs)}
    X = np.zeros((len(jids), 1 + len(motifs) + 3 + len(mean)))
    for row, j in enumerate(jids):
        r = next(x for x in rows if str(x["jid"]) == j)
        m = str(r["motif"])
        if m in mi:
            X[row, 1 + mi[m]] = 1.0
        parts = parse_parts(str(r["junction_seq"]))
        full = "".join(parts)
        off = 1 + len(motifs)
        X[row, off] = len(full)
        X[row, off + 1] = len(parts[0]) if parts else 0
        X[row, off + 2] = len(parts[1]) if len(parts) > 1 else 0
        v = v_ext_transform([j], by_jid, mean, sd)[0]
        X[row, off + 3:] = v
    return X


def _latent_marginal_nll(f, panel, a, b, nodes_t, log_w_t, df=None,
                         tau=TAU, cap=CAP, sigma_q=SIGMA_Q):
    """GH-marginal right-censored NLL over junctions (mean over junctions).

    f : (n_j,) junction latent locations.
    q_jk = f_j + sqrt(2)*sigma_q*nodes_k ; mu_ik = a_s + b_s * q_{j_i,k}.
    Returns the scalar mean junction-macro marginal NLL (data term only).
    """
    q = f[:, None] + math.sqrt(2.0 * sigma_q) * nodes_t[None, :]        # (n_j, G)
    mu = a[panel["flat_s"]][:, None] + b[panel["flat_s"]][:, None] * q[panel["flat_j"]]  # (n_rows, G)
    y = torch.as_tensor(panel["flat_y"], dtype=mu.dtype, device=mu.device)[:, None]
    c = torch.as_tensor(panel["flat_c"], dtype=torch.bool, device=mu.device)[:, None]
    if df is None:
        z = (y - mu) / tau
        nll_m = 0.5 * z * z
        a_t = (mu - cap) / tau
        log_phi = torch.special.log_ndtr(a_t.clamp(min=-30.0, max=30.0))
        nll_c = -log_phi
    else:
        # Student-t training objective (evaluation stays Gaussian at fixed tau)
        dist = torch.distributions.StudentT(
            df=torch.full_like(mu, float(df)), loc=mu, scale=tau)
        nll_m = -dist.log_prob(y)
        a_t = (cap - mu) / tau
        nll_c = -torch.log(_student_t_survival(a_t, float(df)))
    ll_row = torch.where(c, -nll_c, -nll_m)                            # (n_rows, G)
    grouped = torch.zeros((q.shape[0], q.shape[1]), dtype=mu.dtype, device=mu.device)
    grouped.index_add_(0, torch.as_tensor(panel["flat_j"], device=mu.device, dtype=torch.long), ll_row)
    log_marginal = torch.logsumexp(grouped + log_w_t[None, :], dim=1)  # (n_j,)
    return -log_marginal.mean()


def _train_latent_mlp(Xj, panel, device, in_dim, hidden=(96, 64, 32),
                      dropout=0.1, weight_decay=1e-2, df=_DEFAULT_DF,
                      seed=_SEED, lr=LR, max_epochs=MAX_EPOCHS,
                      patience=PATIENCE, loss_tol=LOSS_TOL,
                      plateau_window=PLATEAU_WINDOW,
                      plateau_rel_tol=PLATEAU_REL_TOL):
    """Train the nonlinear latent-operator head (full-batch Adam, GH marginal)."""
    import torch
    from audit.models.nonlinear_mlp_hybrid import _MLP
    torch.manual_seed(seed)
    net = _MLP(in_dim, hidden=hidden, dropout=dropout).to(device)

    n_j = Xj.shape[0]
    n_scaf = len(panel["scaffolds"])
    ref = panel["scaffolds"].index(2) if 2 in panel["scaffolds"] else 0
    # a_s: per-scaffold intercept (init to per-scaffold mean y); b_s = exp(log_b)
    a_init = np.zeros(n_scaf)
    for s in range(n_scaf):
        rows_s = np.where(panel["flat_s"] == s)[0]
        if len(rows_s):
            a_init[s] = float(np.mean(panel["flat_y"][rows_s]))
    a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32, device=device))
    log_b_free = nn.Parameter(torch.zeros(n_scaf - 1, dtype=torch.float32, device=device))

    def full_b():
        b = torch.full((n_scaf,), _REF_SLOPE, dtype=torch.float32, device=device)
        free = [s for s in range(n_scaf) if s != ref]
        vals = torch.exp(log_b_free.clamp(-_LOG_B_CLAMP, _LOG_B_CLAMP))
        for k, s in enumerate(free):
            b[s] = vals[k]
        return b

    opt = torch.optim.Adam(
        [{"params": net.parameters(), "weight_decay": weight_decay},
         {"params": [a], "weight_decay": 0.0},
         {"params": [log_b_free], "weight_decay": 0.0}],
        lr=lr)
    # slope regularization on free log_b (matches v1.31 slope_ridge)
    def slope_reg():
        return _SLOPE_RIDGE * 0.5 * (log_b_free * log_b_free).sum() / max(n_scaf, 1)

    Xt = torch.tensor(Xj, dtype=torch.float32, device=device)
    nodes, lw = hermite(GH)
    nodes_t = torch.tensor(nodes, dtype=torch.float32, device=device)
    log_w_t = torch.tensor(lw, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    epochs_since_best = 0
    n_epochs = 0
    final_loss = float("inf")
    loss_history = []
    plateau_reached = False
    for epoch in range(max_epochs):
        net.train()
        f = net(Xt).squeeze(-1)
        b = full_b()
        loss = _latent_marginal_nll(f, panel, a, b, nodes_t, log_w_t, df=df)
        loss = loss + slope_reg()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
        loss_history.append(final_loss)
        n_epochs = epoch + 1
        if final_loss < best_loss - loss_tol:
            best_loss = final_loss
            best_state = {
                "net": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                "a": a.detach().cpu().clone(), "log_b_free": log_b_free.detach().cpu().clone(),
            }
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

    net.load_state_dict(best_state["net"])
    a.data.copy_(best_state["a"])
    log_b_free.data.copy_(best_state["log_b_free"])
    net.eval()

    net.zero_grad()
    f = net(Xt).squeeze(-1)
    b = full_b()
    lossg = _latent_marginal_nll(f, panel, a, b, nodes_t, log_w_t, df=df) + slope_reg()
    lossg.backward()
    gn = 0.0
    for p in net.parameters():
        if p.grad is not None:
            gn += float(torch.norm(p.grad).item() ** 2)
    total_grad_norm = float(np.sqrt(gn))
    net.eval()

    params_finite = all(bool(torch.isfinite(p).all().item())
                        for p in list(net.parameters()) + [a, log_b_free])
    converged = bool(plateau_reached and params_finite and np.isfinite(final_loss))
    eligible = bool(converged and np.isfinite(total_grad_norm))
    gate = {
        "eligible": eligible, "converged": converged,
        "final_grad_norm": total_grad_norm, "grad_tol": GRAD_TOL,
        "n_epochs": n_epochs, "max_epochs": max_epochs,
        "final_train_nll": final_loss, "best_train_nll": best_loss,
        "plateau_reached": plateau_reached, "success": converged,
        "n_nan_inf_params": int(not params_finite), "df": df,
    }
    b_full = full_b().detach().cpu().numpy()
    return {"net": net, "a": a.detach().cpu().numpy(), "b": b_full, "ref": ref,
            "gate": gate, "n_junctions": n_j}


def make_nonlinear_latent_operator(hidden=(96, 64, 32), dropout=0.1,
                                   weight_decay=1e-2, df=_DEFAULT_DF,
                                   seed=_SEED):
    """Return (fit, predict) for the nonlinear latent-operator head.

    The MLP maps junction-level features (motif one-hot + topology +
    train-scaled 21-D extended-ViennaRNA) to a junction latent location f_j;
    the operator-aware head then predicts mu = a_s + b_s * q_j with q_j the
    latent (sigma_q=1) and scaffolds sharing a fixed ref slope.  Train objective
    is the GH-marginal right-censored NLL (Student-t if df is not None).
    """
    def fit(train_rows):
        assert HAVE_TORCH, "torch required for nonlinear_latent_operator"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        motifs = sorted({str(r["motif"]) for r in train_rows})
        panel = _junction_panel(train_rows)
        tr_jids = panel["jids"]
        by_jid = v_ext_build_raw(train_rows)
        mean, sd = v_ext_fit_scaler(tr_jids, by_jid)
        Xj = _junction_feats(train_rows, tr_jids, motifs, mean, sd, by_jid)
        m = _train_latent_mlp(Xj, panel, device, Xj.shape[1], hidden=hidden,
                              dropout=dropout, weight_decay=weight_decay,
                              df=df, seed=seed)
        return {"kind": "nonlinear_latent_operator", **m,
                "motifs": motifs, "scaffolds": panel["scaffolds"],
                "jids": panel["jids"],
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_junction_feats": Xj.shape[1], "device": device,
                "hidden": list(hidden), "dropout": dropout,
                "weight_decay": weight_decay, "df": df, "seed": seed}

    def predict(model, test_rows):
        assert HAVE_TORCH, "torch required for nonlinear_latent_operator"
        import torch
        te_jids = sorted({str(r["jid"]) for r in test_rows})
        # sequence map must cover every test jid (train jids too, for the scale)
        by_jid = dict(model["by_jid"])
        for r in test_rows:
            by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
        Xj = _junction_feats(test_rows, te_jids, model["motifs"],
                             model["mean"], model["sd"], by_jid)
        je = {j: i for i, j in enumerate(te_jids)}
        si = {s: i for i, s in enumerate(model["scaffolds"])}
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(Xj, dtype=torch.float32, device=model["device"])
            f = model["net"](Xt).squeeze(-1).cpu().numpy()
        n = len(test_rows)
        mu = np.zeros(n)
        sigma = np.full(n, TAU)
        cp = np.zeros(n)
        support = np.ones(n, dtype=bool)
        abstain = np.zeros(n, dtype=bool)
        from scipy.special import log_ndtr
        a, b = model["a"], model["b"]
        for i, r in enumerate(test_rows):
            j = je.get(str(r["jid"]))
            if int(r["scaf"]) not in si or j is None:
                abstain[i] = True
                support[i] = False
                mu[i] = 0.0
                sigma[i] = TAU
            else:
                s = si[int(r["scaf"])]
                mu[i] = a[s] + b[s] * f[j]
                sigma[i] = float(np.sqrt(TAU * TAU + (b[s] * SIGMA_Q) ** 2))
            cp[i] = float(np.exp(np.clip(log_ndtr((mu[i] - CAP) / sigma[i]), -50.0, 0.0)))
        return mu, sigma, cp, support, abstain

    return fit, predict
