"""Frozen RNA foundation-model baseline (contract §9.4 / §12.2).

A frozen RNA-FM embedding is used as a fixed feature representation of the
junction sequence; only a single low-capacity linear head is fit on top, under
the SAME right-censored objective and the SAME inner-search budget (L-BFGS-B,
maxiter=2000, gtol=1e-8, ridge=1.0) as the other parametric baselines.  This
answers only "does a modern frozen representation baseline perform better",
NOT a claim that adding a foundation model is a novel contribution.

The foundation model weights are FROZEN (no fine-tuning, no label exposure).
Embeddings are computed once per unique junction sequence and cached, so the
head fit sees only train-fold rows (no leakage).  Pretraining exposure is
recorded in the config/status for auditability.

Sequence input: the contiguous junction sequence (``junction_seq`` with the
``_``/``&`` structural separators removed), consistent with the sequence
baselines (kmer / position / edit-knn) that consume ``junction_seq``.
"""
from __future__ import annotations

import numpy as np

TAU = 0.7
CAP = -7.1
RIDGE = 1.0


def _contig(seq):
    return str(seq or "").replace("_", "").replace("&", "").upper()


def embed_sequences(sequences, tokenizer, model, device, pool="mean"):
    """Frozen embeddings of unique contiguous junction sequences.

    sequences: iterable of unique strings. Returns {seq: np.ndarray (D,)}.
    The model/tokenizer are used in eval (frozen) mode.
    """
    import torch

    seqs = sorted({_contig(s) for s in sequences if _contig(s)})
    embs = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), 64):
            batch = seqs[i:i + 64]
            enc = tokenizer(batch, padding=True, return_tensors="pt")
            enc = {k: (v.to(device) if torch.cuda.is_available() else v)
                   for k, v in enc.items()}
            out = model(**enc)
            hid = out.last_hidden_state
            # mask padding tokens via attention mask
            attn = enc.get("attention_mask")
            if attn is not None:
                mask = attn.unsqueeze(-1).float()
                summed = (hid * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            else:
                pooled = hid.mean(dim=1)
            pooled = pooled.float().cpu().numpy()
            for s, v in zip(batch, pooled):
                embs[s] = v
    return embs


def _add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_frozen_head(train, embs, ridge=RIDGE):
    """Fit the low-capacity linear head on frozen embeddings.

    embs: {contig_junction_seq: (D,)}.  Returns model dict with beta + gate,
    plus the embedding standardizer (per-dim mean/sd on train rows only).
    """
    from audit.core.censored_objective import CensoredObjective, fit_lbfgs

    X = np.asarray([embs[_contig(r["junction_seq"])] for r in train])
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    X = (X - mean) / sd
    Xb = _add_intercept(X)
    y = np.asarray([r["y"] for r in train], dtype=float)
    cens = np.asarray([r["cens"] for r in train], dtype=bool)
    obj = CensoredObjective(Xb, y, cens)
    rec = fit_lbfgs(obj, np.zeros(Xb.shape[1]), ridge=ridge,
                    maxiter=2000, gtol=1e-8)
    return {"kind": "frozen_rnafm_head", "beta": rec["beta"],
            "mean": mean, "sd": sd, "gate": rec}


def predict_frozen_head(model, test, embs):
    """Predictions from the fitted head on frozen embeddings of test rows."""
    X = np.asarray([embs[_contig(r["junction_seq"])] for r in test])
    X = (X - model["mean"]) / model["sd"]
    Xb = _add_intercept(X)
    mu = Xb @ model["beta"]
    sigma = np.full(len(mu), TAU)
    support = np.ones(len(mu), dtype=bool)
    abstain = np.zeros(len(mu), dtype=bool)
    from scipy.special import log_ndtr
    a = (mu - CAP) / sigma
    cp = np.exp(np.clip(log_ndtr(a), -50.0, 0.0))
    return mu, sigma, cp, support, abstain