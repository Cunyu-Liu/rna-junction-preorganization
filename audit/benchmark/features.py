"""P0.5 shared junction-sequence feature builder (63-dim, train-only scaling).

Replicates the legacy v1.31 feature construction exactly: two parts x up-to-7
positions x 4 bases one-hot (56) + base composition (4) + total length (1) +
two part lengths (2) = 63.  Standardization is fit on TRAIN junction raw
features only and applied to both train and test, so no test statistic leaks
into the model (contract P0.4/P0.5 leakage rule).
"""
from __future__ import annotations

import numpy as np

from audit.data.audit_dataset import parse_parts

ALPHABET = "ACGU"


def raw_features(seq: str) -> np.ndarray:
    out = np.zeros(2 * 7 * 4 + 4 + 1 + 2)
    parts = parse_parts(seq)
    for pi, part in enumerate(parts[:2]):
        for pos, base in enumerate(part[:7]):
            if base in ALPHABET:
                out[(pi * 7 + pos) * 4 + ALPHABET.index(base)] = 1.0
    full = "".join(parts)
    den = max(len(full), 1)
    off = 2 * 7 * 4
    for bi, b in enumerate(ALPHABET):
        out[off + bi] = full.count(b) / den
    out[off + 4] = len(full)
    for pi, p in enumerate(parts[:2]):
        out[off + 5 + pi] = len(p)
    return out


def build_raw_by_jid(rows):
    by_jid = {}
    for r in rows:
        by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
    return by_jid


def fit_scaler(jids, by_jid):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mean, sd


def transform(jids, by_jid, mean, sd):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    return (Xraw - mean) / sd
