"""Phase 1 strong-simple / publicly-reproducible baselines (contract Phase 1).

Contract Phase 1 requires the strongest simple and publicly reproducible
baselines to be established under identical rows, folds, metric, budget and
external-information boundaries.  P0.5 already provides the nuisance-only
minimum set (global / scaffold / hierarchy).  Phase 1 ADDS the sequence-aware
and topology-aware baselines that must be beaten before any model can claim
incremental information:

  motif_topology_hierarchy : mu = b0 + motif + topology(part lengths) + scaffold
                             partial pooling (nuisance + weak structure only)
  onehot_kmer_ridge        : linear ridge on junction one-hot/k-mer composition
                             (train-only standardization)
  position_aware_additive  : additive position-aware one-hot linear model
  edit_knn                 : right-censored KNN smoother on junction Levenshtein
                             distance (train-only neighbor graph; no test info)
  mutation_graph           : graph Laplacian smoother over edit-mutation graph
  small_mlp                : small MLP (>=2 hidden layers, <=64 units) on the
                             shared 63-dim position-aware features, right-censored
                             loss, GPU if available (train-only scaling)

External / non-reproducible baselines (Denny-native oracle, Denny-train-only,
RNAMake physical ensemble, frozen RNA-LM) are registered separately as
UNAVAILABLE_NOT_COMPARED with evidence in TaskEquivalenceTable — they are NOT
fabricated into this numeric leaderboard (contract Phase 1 failure handling:
"不可复现者标 UNAVAILABLE_NOT_COMPARED，不得抄论文数字入榜").

Every baseline follows the P0.5 model interface:
  fit(train_rows) -> model
  predict(model, test_rows) -> (mu, sigma, censor_prob, support, abstain)
and uses the same right-censored NLL metric (tau=0.7, CAP=-7.1).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.features import raw_features, build_raw_by_jid, fit_scaler, transform
from audit.data.audit_dataset import parse_parts

CAP = -7.1
TAU = 0.7
RIDGE = 1.0
MLP_RIDGE = 1e-3
MLP_HIDDEN = (64, 32)
MLP_EPOCHS = 40
MLP_LR = 1e-2
KNN_K = 11


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _rows_to_arrays(rows):
    y = np.asarray([r["y"] for r in rows], dtype=float)
    cens = np.asarray([r["cens"] for r in rows], dtype=bool)
    return y, cens


def _gauss_tobit_loss(y, cens, mu, sigma=TAU):
    """Vectorized right-censored Gaussian negative log-likelihood (per-row)."""
    from scipy.special import log_ndtr
    out = np.zeros(len(y))
    m = ~cens
    if m.any():
        z = (y[m] - mu[m]) / sigma
        out[m] = 0.5 * z * z
    c = cens
    if c.any():
        a = (mu[c] - CAP) / sigma
        out[c] = -np.clip(log_ndtr(a), -50.0, 50.0)
    return out


def _predict_from_mu(mu, sigma=None):
    from scipy.special import log_ndtr
    if sigma is None:
        sigma = np.full(len(mu), TAU)
    sigma = np.asarray(sigma, dtype=float)
    a = (mu - CAP) / sigma
    cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
    support = np.ones(len(mu), dtype=bool)
    abstain = np.zeros(len(mu), dtype=bool)
    return mu, sigma, cp, support, abstain


def _sequence_kmer_features(seq, k=3):
    """Count-based k-mer composition of the junction sequence (train-only scale later)."""
    parts = parse_parts(seq)
    full = "".join(parts)
    alphabet = "ACGU"
    kmer_count = defaultdict(int)
    for i in range(len(full) - k + 1):
        kmer_count[full[i:i + k]] += 1
    out = np.zeros(4 ** k)
    for kmer, cnt in kmer_count.items():
        if len(kmer) == k and all(b in alphabet for b in kmer):
            idx = 0
            for b in kmer:
                idx = idx * 4 + alphabet.index(b)
            out[idx] = cnt
    return out


# ---------------------------------------------------------------------------
# 1. motif/topology partial pooling
# ---------------------------------------------------------------------------

def fit_motif_topology(train, ridge=RIDGE):
    from scipy.optimize import minimize
    motifs = sorted({str(r["motif"]) for r in train})
    scafs = sorted({int(r["scaf"]) for r in train})
    mi = {m: i + 1 for i, m in enumerate(motifs)}
    si = {s: i + 1 + len(motifs) for i, s in enumerate(scafs)}
    # topology: total length + two part lengths (already in 63-dim but build explicitly)
    nf = 1 + len(motifs) + len(scafs) + 3
    X = np.zeros((len(train), nf))
    X[:, 0] = 1.0
    for i, r in enumerate(train):
        X[i, mi[str(r["motif"])]] = 1.0
        X[i, si[int(r["scaf"])]] = 1.0
        parts = parse_parts(r["junction_seq"])
        full = "".join(parts)
        off = 1 + len(motifs) + len(scafs)
        X[i, off] = len(full)
        X[i, off + 1] = len(parts[0]) if parts else 0
        X[i, off + 2] = len(parts[1]) if len(parts) > 1 else 0
    y, cens = _rows_to_arrays(train)

    def f(beta):
        mu = X @ beta
        g = np.zeros_like(beta)
        loss = 0.0
        m = ~cens
        if m.any():
            z = (y[m] - mu[m]) / TAU
            loss += 0.5 * np.sum(z * z)
            g += -(X[m].T @ z) / TAU
        c = cens
        if c.any():
            from scipy.special import log_ndtr
            a = (mu[c] - CAP) / TAU
            phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
            sa = np.clip(-log_ndtr(a), -50.0, 50.0)
            loss += float(np.sum(sa))
            d = -phi / np.exp(-sa)
            g += -(X[c].T @ d) / TAU
        reg = 0.5 * ridge * float(beta[1:] @ beta[1:])
        g_reg = np.zeros(nf); g_reg[1:] = ridge * beta[1:]
        return loss + reg, g + g_reg

    res = minimize(f, np.zeros(nf), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-8})
    return {"kind": "motif_topology", "beta": res.x, "motifs": motifs, "scafs": scafs,
            "mi": mi, "si": si}


def predict_motif_topology(model, test):
    nf = model["beta"].shape[0]
    X = np.zeros((len(test), nf))
    X[:, 0] = 1.0
    off = 1 + len(model["motifs"]) + len(model["scafs"])
    seen = np.zeros(len(test), dtype=bool)
    for i, r in enumerate(test):
        if str(r["motif"]) in model["mi"]:
            X[i, model["mi"][str(r["motif"])]] = 1.0
        if int(r["scaf"]) in model["si"]:
            X[i, model["si"][int(r["scaf"])]] = 1.0
            seen[i] = True
        parts = parse_parts(r["junction_seq"])
        full = "".join(parts)
        X[i, off] = len(full)
        X[i, off + 1] = len(parts[0]) if parts else 0
        X[i, off + 2] = len(parts[1]) if len(parts) > 1 else 0
    mu = X @ model["beta"]
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    abstain = ~seen
    support = seen
    return mu, sigma, cp, support, abstain


# ---------------------------------------------------------------------------
# 2. one-hot / k-mer ridge
# ---------------------------------------------------------------------------

def _build_kmer_matrix(rows, by_jid, k=3):
    jids = sorted({str(r["jid"]) for r in rows})
    ji = {j: i for i, j in enumerate(jids)}
    X = np.zeros((len(rows), 4 ** k))
    for i, r in enumerate(rows):
        X[i] = _sequence_kmer_features(by_jid[str(r["jid"])], k=k)
    return X, ji


def _add_intercept(X):
    """Prepend a unit column so the linear predictor can capture the base
    location (mu0 ~ -7.5) instead of being forced toward 0 on standardized
    features.  The intercept is left unpenalized by the ridge."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_kmer_ridge(train, k=3, ridge=RIDGE):
    from scipy.optimize import minimize
    by_jid = build_raw_by_jid(train)
    X, _ = _build_kmer_matrix(train, by_jid, k=k)
    mean = X.mean(axis=0); sd = X.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    X = (X - mean) / sd
    Xb = _add_intercept(X)
    y, cens = _rows_to_arrays(train)
    nf = Xb.shape[1]

    def f(beta):
        mu = Xb @ beta
        g = np.zeros_like(beta)
        loss = 0.0
        m = ~cens
        if m.any():
            z = (y[m] - mu[m]) / TAU
            loss += 0.5 * np.sum(z * z)
            g += -(Xb[m].T @ z) / TAU
        c = cens
        if c.any():
            from scipy.special import log_ndtr
            a = (mu[c] - CAP) / TAU
            phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
            sa = np.clip(-log_ndtr(a), -50.0, 50.0)
            loss += float(np.sum(sa))
            d = -phi / np.exp(-sa)
            g += -(Xb[c].T @ d) / TAU
        # penalize all coefficients except the intercept (beta[0])
        reg = 0.5 * ridge * float(beta[1:] @ beta[1:])
        gr = np.zeros(nf); gr[1:] = ridge * beta[1:]
        return loss + reg, g + gr

    res = minimize(f, np.zeros(nf), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-8})
    return {"kind": "kmer_ridge", "beta": res.x, "mean": mean, "sd": sd, "k": k}


def predict_kmer_ridge(model, test):
    by_jid = build_raw_by_jid(test)
    X, _ = _build_kmer_matrix(test, by_jid, k=model["k"])
    X = (X - model["mean"]) / model["sd"]
    Xb = _add_intercept(X)
    mu = Xb @ model["beta"]
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    return mu, sigma, cp, support, abstain


# ---------------------------------------------------------------------------
# 3. position-aware additive (shared 63-dim features, linear ridge)
# ---------------------------------------------------------------------------

def fit_position_additive(train, ridge=RIDGE):
    from scipy.optimize import minimize
    tr_jids = sorted({str(r["jid"]) for r in train})
    by_jid = build_raw_by_jid(train)
    mean, sd = fit_scaler(tr_jids, by_jid)
    X = transform(tr_jids, by_jid, mean, sd)
    ji = {j: i for i, j in enumerate(tr_jids)}
    Xr = np.zeros((len(train), X.shape[1]))
    for i, r in enumerate(train):
        Xr[i] = X[ji[str(r["jid"])]]
    y, cens = _rows_to_arrays(train)
    Xb = _add_intercept(Xr)
    nf = Xb.shape[1]

    def f(beta):
        mu = Xb @ beta
        g = np.zeros_like(beta)
        loss = 0.0
        m = ~cens
        if m.any():
            z = (y[m] - mu[m]) / TAU
            loss += 0.5 * np.sum(z * z)
            g += -(Xb[m].T @ z) / TAU
        c = cens
        if c.any():
            from scipy.special import log_ndtr
            a = (mu[c] - CAP) / TAU
            phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
            sa = np.clip(-log_ndtr(a), -50.0, 50.0)
            loss += float(np.sum(sa))
            d = -phi / np.exp(-sa)
            g += -(Xb[c].T @ d) / TAU
        reg = 0.5 * ridge * float(beta[1:] @ beta[1:])
        gr = np.zeros(nf); gr[1:] = ridge * beta[1:]
        return loss + reg, g + gr

    res = minimize(f, np.zeros(nf), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-8})
    return {"kind": "position_additive", "beta": res.x, "mean": mean, "sd": sd}


def predict_position_additive(model, test):
    te_jids = sorted({str(r["jid"]) for r in test})
    by_jid = build_raw_by_jid(test)
    X = transform(te_jids, by_jid, model["mean"], model["sd"])
    ji = {j: i for i, j in enumerate(te_jids)}
    Xr = np.zeros((len(test), X.shape[1]))
    for i, r in enumerate(test):
        Xr[i] = X[ji[str(r["jid"])]]
    Xb = _add_intercept(Xr)
    mu = Xb @ model["beta"]
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    return mu, sigma, cp, support, abstain


# ---------------------------------------------------------------------------
# 4. edit KNN (right-censored local smoother)
# ---------------------------------------------------------------------------

def _lev(a, b):
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fit_edit_knn(train, k=KNN_K):
    """Neighbor graph over DISTINCT junction identities (train-only).

    Each junction is a biological identity observed across multiple
    scaffold/context rows; a neighbor graph over raw rows would duplicate the
    same junction many times and bias the neighborhood.  We therefore build the
    graph on the distinct junction sequences present in train, with each node's
    value = mean observed (censored rows count at CAP) over that junction's
    train rows.  Test rows map to their own junction's node value if present,
    else to the k nearest distinct train junctions by Levenshtein distance.
    """
    seqs = sorted({str(r["junction_seq"]) for r in train})
    seq_idx = {s: i for i, s in enumerate(seqs)}
    node_val = np.zeros(len(seqs))
    node_n = np.zeros(len(seqs))
    for r in train:
        s = str(r["junction_seq"])
        i = seq_idx[s]
        node_val[i] += float(r["y"])
        node_n[i] += 1.0
    node_val = node_val / np.maximum(node_n, 1.0)
    return {"kind": "edit_knn", "seqs": seqs, "seq_idx": seq_idx, "node_val": node_val, "k": k}


def predict_edit_knn(model, test):
    mu = np.zeros(len(test))
    seqs = model["seqs"]
    node_val = model["node_val"]
    for i, r in enumerate(test):
        tseq = str(r["junction_seq"])
        if tseq in model.get("seq_idx", {}):
            j = model["seq_idx"][tseq]
            if j is not None:
                mu[i] = node_val[j]
                continue
        dists = [_lev(tseq, s) for s in seqs]
        idx = np.argsort(dists)[:model["k"]]
        mu[i] = float(np.mean(node_val[idx]))
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    return mu, sigma, cp, support, abstain


# ---------------------------------------------------------------------------
# 5. mutation graph smoother (Laplacian)
# ---------------------------------------------------------------------------

def fit_mutation_graph(train):
    keys = sorted({str(r["symmetry_key"]) for r in train})
    ki = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if _lev(keys[i], keys[j]) <= 1:
                adj[i].add(j)
                adj[j].add(i)
    # node value = mean y per key
    y_by = defaultdict(list)
    for r in train:
        y_by[str(r["symmetry_key"])].append(float(r["y"]))
    node_val = np.array([np.mean(y_by[k]) for k in keys])
    return {"kind": "mutation_graph", "keys": keys, "ki": ki, "adj": adj, "node_val": node_val}


def predict_mutation_graph(model, test):
    keys, ki, adj, node_val = model["keys"], model["ki"], model["adj"], model["node_val"]
    mu = np.zeros(len(test))
    for i, r in enumerate(test):
        k = str(r["symmetry_key"])
        if k in ki:
            v = ki[k]
            nb = adj[v]
            if nb:
                mu[i] = np.mean([node_val[x] for x in nb])
            else:
                mu[i] = node_val[v]
        else:
            mu[i] = float(np.mean(node_val))  # unseen key -> graph marginal
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    return mu, sigma, cp, support, abstain


# ---------------------------------------------------------------------------
# 6. small MLP (right-censored, GPU)
# ---------------------------------------------------------------------------

def fit_small_mlp(train, hidden=MLP_HIDDEN, ridge=MLP_RIDGE, epochs=MLP_EPOCHS, lr=MLP_LR):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_jids = sorted({str(r["jid"]) for r in train})
    by_jid = build_raw_by_jid(train)
    mean, sd = fit_scaler(tr_jids, by_jid)
    X = transform(tr_jids, by_jid, mean, sd)
    ji = {j: i for i, j in enumerate(tr_jids)}
    Xr = np.zeros((len(train), X.shape[1]))
    for i, r in enumerate(train):
        Xr[i] = X[ji[str(r["jid"])]]
    y, cens = _rows_to_arrays(train)
    Xt = torch.tensor(Xr, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    ct = torch.tensor(cens, dtype=torch.bool)
    nf = X.shape[1]
    layers = []
    prev = nf
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, 1))
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=ridge)
    Xt, yt, ct = Xt.to(dev), yt.to(dev), ct.to(dev)

    def loss(mu):
        mu = mu.squeeze(1)
        # measured
        m = ~ct
        l = torch.zeros(len(yt), dtype=torch.float32, device=dev)
        if m.any():
            z = (yt[m] - mu[m]) / TAU
            l[m] = 0.5 * z * z
        if ct.any():
            # log P(Y>=CAP) = log_Phi((mu-CAP)/tau), differentiable in mu
            a = (mu[ct] - CAP) / TAU
            l[ct] = -torch.clamp(torch.special.log_ndtr(a), -50.0, 50.0)
        return l.mean()

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(Xt)
        lo = loss(out)
        lo.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        mu_tr = net(Xt).squeeze(1).cpu().numpy()
    return {"kind": "small_mlp", "net": net, "dev": dev, "mean": mean, "sd": sd,
            "train_mean_mu": float(np.mean(mu_tr))}


def predict_small_mlp(model, test):
    import torch
    te_jids = sorted({str(r["jid"]) for r in test})
    by_jid = build_raw_by_jid(test)
    X = transform(te_jids, by_jid, model["mean"], model["sd"])
    ji = {j: i for i, j in enumerate(te_jids)}
    Xr = np.zeros((len(test), X.shape[1]))
    for i, r in enumerate(test):
        Xr[i] = X[ji[str(r["jid"])]]
    Xt = torch.tensor(Xr, dtype=torch.float32).to(model["dev"])
    model["net"].eval()
    with torch.no_grad():
        mu = model["net"](Xt).squeeze(1).cpu().numpy()
    mu, sigma, cp, support, abstain = _predict_from_mu(mu)
    return mu, sigma, cp, support, abstain


PHASE1_MODELS = {
    "motif_topology_hierarchy": (fit_motif_topology, predict_motif_topology),
    "onehot_kmer_ridge": (fit_kmer_ridge, predict_kmer_ridge),
    "position_aware_additive": (fit_position_additive, predict_position_additive),
    "edit_knn": (fit_edit_knn, predict_edit_knn),
    "mutation_graph_smoother": (fit_mutation_graph, predict_mutation_graph),
    "small_mlp": (fit_small_mlp, predict_small_mlp),
}
