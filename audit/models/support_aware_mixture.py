"""Phase 3 Candidate C: support-aware gated mixture with abstention.

Motivation (from Phase 2): corrected_v1_31 produces catastrophic right-censored
NLL on low-support extrapolation folds (notably context_lomo), and edit-axis
gain is marginal/low-support. Candidate C treats "extrapolation vs interpolation"
explicitly: it computes per-junction support features from outer-train only, and
abstains (excludes from scoring) when a junction lacks local sequence-neighbour
support. Supported rows use a train-only local sequence predictor (edit-KNN
censored location); unsupported rows are abstained and reported separately.

This yields a coverage-risk curve (coverage vs supported NLL vs catastrophic
folds) and directly addresses the §9.3 success criteria: edit / blocked-context /
low-support strata improve under abstention, with abstention rules frozen before
any outer test selection.

Performance: the Levenshtein distance between every pair of distinct junction
sequences in the admitted universe is precomputed ONCE into a global cache, so
support_features / _local_mu for arbitrary train/test splits are fast lookups
instead of O(n_test x n_train) recomputation per inner fold.

Design notes (contract §9.3 / P0.4 abstention rule):
  - support features are computed from TRAIN rows only (no test leakage)
  - abstained rows are excluded from the macro NLL and reported as support strata
  - the gate threshold sweep is a pre-registered grid, selected via nested CV
"""
from __future__ import annotations

import numpy as np

from audit.evaluation.metrics import junction_macro_nll

CAP = -7.1
TAU = 0.7
SUPPORT_DIST = 3          # edit distance defining a "near" train neighbour
KNN_K = 11
# pre-registered gate grid on min-edit-distance (1000 ~ inf)
GATE_GRID = [1, 2, 3, 5, 8, 12, 1000]

# ---- global distance cache (built once over the admitted universe) ----
_DIST_CACHE = {}          # seq_index -> {seq: row_index}
_SEQ_LIST = []            # list of distinct junction seqs
_DIST_MAT = None          # (N, N) int matrix of Levenshtein distances


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


def build_distance_cache(all_rows):
    """Precompute the full pairwise Levenshtein matrix over all distinct
    junction sequences present in `all_rows`. Call ONCE per process."""
    global _SEQ_LIST, _DIST_MAT
    seqs = sorted({str(r["junction_seq"]) for r in all_rows})
    _SEQ_LIST = seqs
    n = len(seqs)
    mat = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            d = _lev(seqs[i], seqs[j])
            mat[i, j] = d
            mat[j, i] = d
    _DIST_MAT = mat
    return seqs, mat


def _seq_index(tseq):
    from bisect import bisect_left
    i = bisect_left(_SEQ_LIST, tseq)
    if i < len(_SEQ_LIST) and _SEQ_LIST[i] == tseq:
        return i
    return -1


def _dist_to_train_seq(train_seqs_idx, tseq):
    """min distance and neighbor counts from tseq to a set of train seq indices."""
    ti = _seq_index(tseq)
    if ti < 0:
        ti = 0  # fallback; should not happen within admitted universe
    row = _DIST_MAT[ti]
    dists = row[train_seqs_idx] if len(train_seqs_idx) else np.empty(0, dtype=np.int32)
    return dists


def support_features(train_rows, test_rows, dist=SUPPORT_DIST):
    """Per distinct test junction: min_edit_dist to nearest train junction seq,
    n_neighbors within `dist`, scaffold_seen, context_seen. Train-only."""
    if _DIST_MAT is None:
        # lazily build from the union of the provided rows (tests / small calls)
        build_distance_cache(list(train_rows) + list(test_rows))
    train_seqs_idx = np.array([_seq_index(str(r["junction_seq"])) for r in train_rows], dtype=np.int64)
    train_scafs = {str(r["scaf"]) for r in train_rows}
    train_ctxs = {str(r["helix_seq"]) for r in train_rows}
    jid_seq, jid_scaf, jid_ctx = {}, {}, {}
    for r in test_rows:
        j = str(r["jid"])
        jid_seq.setdefault(j, str(r["junction_seq"]))
        jid_scaf.setdefault(j, str(r["scaf"]))
        jid_ctx.setdefault(j, str(r["helix_seq"]))
    feats = {}
    for j in jid_seq:
        dists = _dist_to_train_seq(train_seqs_idx, jid_seq[j])
        min_d = float(dists.min()) if dists.size else float("inf")
        feats[j] = {
            "min_edit_dist": min_d,
            "n_neighbors": int((dists <= dist).sum()),
            "scaffold_seen": jid_scaf[j] in train_scafs,
            "context_seen": jid_ctx[j] in train_ctxs,
        }
    return feats


def fit_local(train_rows, k=KNN_K):
    """Train-only edit-KNN censored location over distinct junction identities."""
    seqs = sorted({str(r["junction_seq"]) for r in train_rows})
    seq_idx = {s: i for i, s in enumerate(seqs)}
    node_val = np.zeros(len(seqs))
    node_n = np.zeros(len(seqs))
    for r in train_rows:
        i = seq_idx[str(r["junction_seq"])]
        node_val[i] += float(r["y"])
        node_n[i] += 1.0
    node_val = node_val / np.maximum(node_n, 1.0)
    return {"seqs": seqs, "seq_idx": seq_idx, "node_val": node_val, "k": k}


def _local_mu(model, tseq):
    if tseq in model.get("seq_idx", {}):
        return float(model["node_val"][model["seq_idx"][tseq]])
    # vectorized: distances from tseq to all model seqs via global matrix
    train_seqs_idx = np.array([_seq_index(s) for s in model["seqs"]], dtype=np.int64)
    dists = _dist_to_train_seq(train_seqs_idx, tseq)
    k = min(model["k"], len(dists))
    idx = np.argsort(dists)[:k]
    return float(np.mean(model["node_val"][idx]))


def predict_gated(model, feats, test_rows, d_thresh=SUPPORT_DIST, k_thresh=1):
    """mu/sigma from local predictor on supported rows; abstain otherwise."""
    n = len(test_rows)
    mu = np.zeros(n)
    sigma = np.full(n, TAU)
    cp = np.zeros(n)
    abstain = np.zeros(n, dtype=bool)
    support = np.zeros(n, dtype=bool)
    for i, r in enumerate(test_rows):
        f = feats[str(r["jid"])]
        if f["min_edit_dist"] <= d_thresh and f["n_neighbors"] >= k_thresh:
            mu[i] = _local_mu(model, str(r["junction_seq"]))
            sigma[i] = TAU
            a = (mu[i] - CAP) / TAU
            cp[i] = float(np.exp(np.clip(_log_ndtr(a), -50.0, 0.0)))
            support[i] = True
        else:
            abstain[i] = True
    return mu, sigma, cp, support, abstain


def _log_ndtr(a):
    from scipy.special import log_ndtr
    return log_ndtr(a)


def supported_metrics(rows, mu, sigma, support_mask):
    """junction-macro NLL over supported rows only + coverage."""
    sup = [r for r, s in zip(rows, support_mask) if s]
    if not sup:
        return {"coverage": 0.0, "supported_nll": None, "n_supported": 0}
    nll = junction_macro_nll(sup, [mu[i] for i, s in enumerate(support_mask) if s],
                             [sigma[i] for i, s in enumerate(support_mask) if s])
    return {"coverage": float(len(sup)) / len(rows), "supported_nll": nll,
            "n_supported": len(sup)}
