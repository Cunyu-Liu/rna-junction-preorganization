"""Edit-site / junction local-context feature builder.

The 21-D extended-ViennaRNA block encodes per-junction FOLDING aggregates, which
smooth over position-specific determinants.  The "edit site" in this dataset is
not a single canonical position (edit components are union-find chains of
Hamming-distance-1 junctions), so we encode the junction-local context that is
position-anchored and generalizes across held-out edit components:

  window = last W bases of the left arm ++ first W bases of the right arm
           (the join / junction region where preorganization happens)

Each of the 2W window positions is one-hot encoded over {A,C,G,U} so the
feature is a sparse binary vector that is constant per junction and needs no
standardization.  Because the edit-junction region differs across components,
the base->effect mapping learned on training folds transfers to held-out
components (no leakage), mirroring how the folding block already transfers.

Only per-junction sequence is used (junction_seq), never scaffold/context/y.
"""
from __future__ import annotations

import numpy as np

from audit.data.audit_dataset import parse_parts

ALPHABET = "ACGU"
W = 3                 # window half-width on each arm
N_POS = 2 * W
N_FEAT = N_POS * len(ALPHABET)


def raw_features(seq: str) -> np.ndarray:
    """Return a binary (N_FEAT,) one-hot of the join-local window.

    window = last W of left arm + first W of right arm.  Positions that do not
    exist (arm shorter than W) are left as all-zero one-hots.
    """
    parts = parse_parts(seq)
    left = parts[0] if parts else ""
    right = parts[1] if len(parts) > 1 else ""
    tail = left[-W:] if left else ""
    head = right[:W] if right else ""
    window = tail + head          # length <= 2W, anchored at the join
    feat = np.zeros(N_FEAT, dtype=float)
    for i, base in enumerate(window):
        if base in ALPHABET:
            feat[i * len(ALPHABET) + ALPHABET.index(base)] = 1.0
    return feat


def build_raw_by_jid(rows):
    by_jid = {}
    for r in rows:
        by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
    return by_jid


def fit_scaler(jids, by_jid):
    # one-hot binary features need no standardization -> identity transform.
    return np.zeros(N_FEAT, dtype=float), np.ones(N_FEAT, dtype=float)


def transform(jids, by_jid, mean, sd):
    return np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
