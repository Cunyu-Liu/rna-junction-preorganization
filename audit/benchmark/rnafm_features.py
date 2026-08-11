"""RNA-FM frozen-embedding feature builder for the winning-head hybrid.

Clinical contrast with the folding-proxy features: instead of hand-engineered
thermodynamic scalars (MFE, bpp, entropy), this representation is the pooled
frozen output of RNA-FM (Chen et al. 2022), a 12-layer BERT-style RNA language
model pretrained on 23.7M ncRNA sequences.  It is a fully unsupervised,
label-free representation: embeddings are extracted once (offline, on GPU)
for every unique junction sequence and cached to disk.  Because the encoder is
frozen and never sees fold splits or labels, the cache introduces no leakage;
only the per-fold standardization is fit on train (exactly as in the ViennaRNA
and k-mer hybrids).

Pooled representation (per junction sequence, layer 12, hidden 640):
  - mean-pooled per-position embedding (masked to non-padding tokens)
  - max-pooled per-position embedding
  - [CLS] token embedding
  concatenated -> 1920 dimensions.

This module is CPU/test-safe: it only reads the precomputed cache and does the
train-only scaling.  The GPU extraction lives in audit.benchmark.rnafm_extract.
"""
from __future__ import annotations

import numpy as np

from audit.data.audit_dataset import parse_parts

# 640 (hidden) * 3 (mean, max, cls)
RENDER_DIM = 1920
HIDDEN = 640


def load_cache(cache_path) -> dict:
    """Load {junction_seq: pooled_embedding[1920]} from an .npz cache."""
    data = np.load(cache_path, allow_pickle=True)
    seqs = data["seqs"]
    vecs = data["vecs"]
    return {str(s): np.asarray(v, dtype=float) for s, v in zip(seqs, vecs)}


def build_raw_by_jid(rows, cache: dict):
    """Map jid -> pooled RNA-FM embedding for a list of admitted rows."""
    by_jid = {}
    for r in rows:
        seq = str(r["junction_seq"])
        if seq in cache:
            by_jid[str(r["jid"])] = cache[seq]
    return by_jid


def fit_scaler(jids, by_jid):
    """Standardize on TRAIN junction embeddings only (no test leakage)."""
    Xraw = np.asarray([by_jid[j] for j in jids], dtype=float)
    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mean, sd


def transform(jids, by_jid, mean, sd):
    Xraw = np.asarray([by_jid[j] for j in jids], dtype=float)
    return (Xraw - mean) / sd


def pooled_embedding(seq: str, model, alphabet, device) -> np.ndarray:
    """Return the 1920-dim pooled RNA-FM embedding for one RNA sequence.

    Uses the frozen last-layer (hidden 640) representation, pooling mean/max
    over non-padding positions and the [CLS] token.  Requires torch + GPU;
    used only by the offline extraction script.
    """
    import torch

    conv = alphabet.get_batch_converter()
    # '_' is not in the RNA-FM alphabet -> encode junction boundary as gap '-'.
    _, _, tokens = conv([("seq", seq.replace("_", "-"))])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[model.args.layers])
        rep = out["representations"][model.args.layers]  # (1, L, hidden)
    pad = alphabet.padding_idx
    m = (tokens != pad).unsqueeze(-1).float()
    denom = m.sum(1).clamp(min=1)
    meanp = ((rep * m).sum(1) / denom)[0]
    maxp = (rep * m + (1 - m) * -1e9).max(1).values[0]
    cls = rep[0, 0, :]
    return np.concatenate([meanp.float().cpu().numpy(),
                           maxp.float().cpu().numpy(),
                           cls.float().cpu().numpy()]).astype(np.float64)