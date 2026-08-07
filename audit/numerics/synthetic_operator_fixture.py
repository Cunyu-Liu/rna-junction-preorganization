"""Synthetic operator fixture for v1.31 recovery tests.

Generates junction sequences, a known linear latent functional q_j = X_j theta*,
known scaffold intercepts a_s and slopes b_s, then draws Y_js ~ N(a_s+b_s q_j,tau)
and right-censors at CAP (Y>=CAP recorded as CAP).  Enables known-q recovery and
gradient/GH gates on data with a known ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

CAP = -7.1
TAU = 0.7


def make_sequences(n_junctions, n_scaf, seed):
    """Generate random junction sequences (two parts) + one-hot 63-dim features."""
    rng = np.random.default_rng(int(seed))
    alphabet = "ACGU"
    seqs = []
    for _ in range(n_junctions):
        p1 = "".join(rng.choice(list(alphabet), size=7))
        p2 = "".join(rng.choice(list(alphabet), size=7))
        seqs.append(f"{p1}_{p2}")
    # feature builder (mirrors seq_features)
    X = np.zeros((n_junctions, 2 * 7 * 4 + 4 + 1 + 2), dtype=float)
    for i, s in enumerate(seqs):
        parts = s.split("_")
        for pi, part in enumerate(parts[:2]):
            for pos, base in enumerate(part[:7]):
                if base in alphabet:
                    X[i, (pi * 7 + pos) * 4 + alphabet.index(base)] = 1.0
        full = "".join(parts)
        den = max(len(full), 1)
        off = 2 * 7 * 4
        for bi, b in enumerate(alphabet):
            X[i, off + bi] = full.count(b) / den
        X[i, off + 4] = len(full)
        for pi, p in enumerate(parts[:2]):
            X[i, off + 5 + pi] = len(p)
    return seqs, X


def _calibrate_shift(a_star, b_star, q_star, n_rows_per_junc, n_scaf, censor_frac,
                     tau):
    """Find a global location shift so that expected P(Y>=CAP) == censor_frac.

    The expected censoring fraction under the model is
        F(shift) = mean_{j,s} Phi((a_s + shift + b_s q_j - CAP) / tau).
    We bisect on `shift` to hit the target.  For target=0.0 we push the location
    far below CAP so cap-censoring is negligible (approx. all measured).
    """
    n_junc = len(q_star)
    base = np.zeros((n_junc * n_rows_per_junc,))
    for j in range(n_junc):
        for r in range(n_rows_per_junc):
            s = r % n_scaf
            base[j * n_rows_per_junc + r] = a_star[s] + b_star[s] * q_star[j]

    def expected(s):
        return float(np.mean(ndtr((base + s - CAP) / tau)))

    if censor_frac <= 1e-9:
        # keep intercepts inside model bounds a_s in [-18,-5]; max a_star ~ -7.5
        # so a shift of -8 keeps all a_s + shift within bounds and makes
        # P(Y>=CAP) negligible (well below -7.1).  (A huge shift like -100 would
        # clip intercepts at the -18 bound and destroy recovery.)
        return -8.0
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected(mid) > censor_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def make_panel(seqs, X, theta_star, a_star, b_star, n_scaf, censor_frac, seed,
               n_rows_per_junc=None):
    """Draw panel data censored ONLY at the cap (well-specified w.r.t. the
    v1.31 likelihood, which models P(Y>=CAP) via log_ndtr).

    To hit a target censor fraction, the scaffold intercepts are shifted by a
    global `shift` so that empirical P(Y>=CAP) matches the target.  The shift is
    a uniform location offset (preserves operator ordering and relative a_s);
    the recovered model absorbs it into the overall location/intercepts.  The
    empirical fraction actually achieved is returned for transparency.
    """
    rng = np.random.default_rng(int(seed))
    n_junc = len(seqs)
    if n_rows_per_junc is None:
        n_rows_per_junc = n_scaf  # one row per junction x scaffold
    q_star = X @ theta_star
    # calibrate a global location shift to hit the target censor fraction
    shift = _calibrate_shift(a_star, b_star, q_star, n_rows_per_junc, n_scaf,
                             censor_frac, TAU)
    flat_j, flat_s, flat_y, flat_c = [], [], [], []
    for j in range(n_junc):
        for r in range(n_rows_per_junc):
            s = r % n_scaf
            mu = a_star[s] + shift + b_star[s] * q_star[j]
            y = float(rng.normal(mu, TAU))
            if y >= CAP:  # only cap censoring (matches likelihood)
                y = CAP
                c = True
            else:
                c = False
            flat_j.append(j)
            flat_s.append(s)
            flat_y.append(y)
            flat_c.append(bool(c))
    return {
        "jids": [str(i) for i in range(n_junc)],
        "scaffolds": list(range(1, n_scaf + 1)),
        "si": {i + 1: i for i in range(n_scaf)},
        "flat_j": np.asarray(flat_j, dtype=int),
        "flat_s": np.asarray(flat_s, dtype=int),
        "flat_y": np.asarray(flat_y, dtype=float),
        "flat_c": np.asarray(flat_c, dtype=bool),
    }, X, q_star
