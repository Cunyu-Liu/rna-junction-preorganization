"""k-mer composition feature builder for the latent-operator representation
shootout.

Replaces the 63-D position/composition map with normalized k-mer composition
frequencies over the FULL junction sequence.  k-mer composition captures local
sequence motif content (GC content, stacking partners, trinucleotide
preorganization) that one-hot position encoding loses, and is computed from
sequence alone (no external weights or network required).

Features:
  - counts of all k-mers (default k=3 -> 4**3 = 64), normalized to frequency
  - total length (kept for scale)
  - length of part 1, length of part 2

Standardization is fit on TRAIN junction raw features only and applied to both
train and test (no test leakage), consistent with audit.benchmark.features.
"""
from __future__ import annotations

import itertools

import numpy as np

from audit.data.audit_dataset import parse_parts

ALPHABET = "ACGU"


def _kmers(k: int):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]


_KMER_CACHE = {}


def _kmer_index(k: int):
    if k not in _KMER_CACHE:
        _KMER_CACHE[k] = {m: i for i, m in enumerate(_kmers(k))}
    return _KMER_CACHE[k]


def raw_features(seq: str, k: int = 3) -> np.ndarray:
    parts = parse_parts(seq)
    full = "".join(parts)
    idx = _kmer_index(k)
    nkm = len(idx)
    counts = np.zeros(nkm, dtype=float)
    n = len(full)
    for i in range(n - k + 1):
        sub = full[i:i + k]
        if sub in idx:
            counts[idx[sub]] += 1.0
    den = max(n - k + 1, 1)
    freq = counts / den
    out = np.concatenate([
        freq,
        [float(n)],
        [float(len(parts[0])) if parts else 0.0],
        [float(len(parts[1])) if len(parts) > 1 else 0.0],
    ])
    return out


def build_raw_by_jid(rows, k: int = 3):
    by_jid = {}
    for r in rows:
        by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
    return by_jid


def fit_scaler(jids, by_jid, k: int = 3):
    Xraw = np.asarray([raw_features(by_jid[j], k=k) for j in jids], dtype=float)
    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mean, sd


def transform(jids, by_jid, mean, sd, k: int = 3):
    Xraw = np.asarray([raw_features(by_jid[j], k=k) for j in jids], dtype=float)
    return (Xraw - mean) / sd