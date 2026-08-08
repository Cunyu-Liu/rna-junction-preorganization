"""Shared Phase 2 helpers: data loading, gain, model fit, permutation generators.

Reuses the frozen Phase 1 protocol (rows, splits, metric, model adapters) so all
Phase 2 statistics are computed under the identical rows/folds/metric.
"""
from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

CAP = -7.1
EPS = 1e-8

REFERENCE_BASELINE = "train_only_scaffold"
CANDIDATE = "corrected_v1_31"


def load_rows(ledger_path):
    rows = {}
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("layer") != "admitted" or o.get("excluded"):
            continue
        rows[str(o["source_row_id"])] = o
    return rows


def load_splits(manifest_path):
    by_fold = defaultdict(set)
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    return axis, by_fold, len(by_fold)


def load_fold_metrics(csv_path):
    """Return {(axis, model_id, fold): macro_nll} from Phase 1 FoldMetrics.csv."""
    import csv
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("macro_nll") not in (None, ""):
                out[(r["axis"], r["model_id"], int(r["fold"]))] = float(r["macro_nll"])
    return out


def axis_gain(fold_metrics, axis, folds, ref=REFERENCE_BASELINE, cand=CANDIDATE):
    """Mean over folds of NLL(ref)-NLL(cand). Returns (gain, per_fold dict)."""
    per = {}
    for fd in folds:
        nb = fold_metrics.get((axis, ref, fd))
        nc = fold_metrics.get((axis, cand, fd))
        if nb is None or nc is None:
            continue
        per[fd] = nb - nc
    if not per:
        return None, {}
    return float(np.mean(list(per.values()))), per


def observed_fold_gains(fold_metrics, axis, n_folds, ref=REFERENCE_BASELINE, cand=CANDIDATE):
    per = {}
    for fd in range(n_folds):
        nb = fold_metrics.get((axis, ref, fd))
        nc = fold_metrics.get((axis, cand, fd))
        if nb is not None and nc is not None:
            per[fd] = nb - nc
    return per


# ---------------------------------------------------------------------------
# permutation generators (group-preserving, deterministic per seed)
# ---------------------------------------------------------------------------

def _positions_by_jid(rows):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[str(r["jid"])].append(i)
    return g


def permute_labels_within_junction(rows, rng):
    """Shuffle (y,cens) within each junction group. Sequence stays with its
    junction; only the scaffold/context-specific label assignment is broken."""
    out = [dict(r) for r in rows]
    g = _positions_by_jid(rows)
    for positions in g.values():
        if len(positions) < 2:
            continue
        ys = [out[p]["y"] for p in positions]
        cs = [out[p]["cens"] for p in positions]
        perm = rng.permutation(len(positions))
        for k, p in enumerate(positions):
            out[p]["y"] = ys[perm[k]]
            out[p]["cens"] = cs[perm[k]]
    return out


def permute_sequence_pairing(rows, rng):
    """Reassign each train row a randomly-drawn junction sequence (permuted
    among the distinct junctions present in train), keeping labels in place.
    Breaks the sequence -> label link (v1.30 sequence-null analog)."""
    jids = sorted({str(r["jid"]) for r in rows})
    seq_by_jid = {str(r["jid"]): str(r["junction_seq"]) for r in rows}
    perm_jids = rng.permutation(jids).tolist()
    new_jid_for = {j: p for j, p in zip(jids, perm_jids)}
    out = []
    for r in rows:
        o = dict(r)
        o["junction_seq"] = seq_by_jid[new_jid_for[str(r["jid"])]]
        out.append(o)
    return out


# ---------------------------------------------------------------------------
# model fit/predict (full-config candidate + scaffold baseline)
# ---------------------------------------------------------------------------

def _make_candidate():
    from audit.benchmark.legacy_adapters import make_v131_adapter
    return make_v131_adapter()


def _make_baseline():
    from audit.benchmark.baselines import fit_scaffold, predict_scaffold
    return fit_scaffold, predict_scaffold


def fit_predict(rows, test_rows, model_id):
    """Fit on `rows` (possibly permuted), predict on test_rows.
    Returns (mu, sigma). model_id in {corrected_v1_31, train_only_scaffold}."""
    if model_id == "corrected_v1_31":
        fit, pred = _make_candidate()
    else:
        fit, pred = _make_baseline()
    model = fit(rows)
    mu, sigma, _cp, _supp, _abst = pred(model, test_rows)
    return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)


def permutation_gain(train_rows, test_rows, model_id, perm_type, rng):
    """Fit model on permuted train, evaluate right-censored junction-macro NLL
    on real test. Returns junction-macro NLL (higher = worse)."""
    from audit.evaluation.metrics import junction_macro_nll
    if perm_type == "label":
        train_p = permute_labels_within_junction(train_rows, rng)
    elif perm_type == "sequence":
        train_p = permute_sequence_pairing(train_rows, rng)
    else:
        raise ValueError(perm_type)
    mu, sigma = fit_predict(train_p, test_rows, model_id)
    return junction_macro_nll(test_rows, mu, sigma)


def axis_permutation_nll(axis_folds, perm_type, seed, model_id):
    """Run one permutation across all folds of an axis; return axis-mean NLL."""
    rng = np.random.default_rng(seed)
    vals = []
    for _test_ids, train_rows, test_rows in axis_folds:
        vals.append(permutation_gain(train_rows, test_rows, model_id, perm_type, rng))
    return float(np.mean(vals)) if vals else None


def build_axis_folds(rows, by_fold, n_folds):
    """Pre-compute per-fold (test_ids, train_rows, test_rows) once."""
    out = []
    for fold in range(n_folds):
        test_ids = by_fold.get(fold, set())
        test_rows = [r for sid, r in rows.items() if sid in test_ids]
        train_rows = [r for sid, r in rows.items() if sid not in test_ids]
        out.append((test_ids, train_rows, test_rows))
    return out
