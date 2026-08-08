"""Phase 2 support ledger + stratified gain analysis (contract Phase 2).

Computes per-test-row support/stratification features and the stratified gain
(candidate vs reference baseline) by: neighbor distance, neighbor count,
context support, and censor fraction.  Emits SupportLedger.parquet (row level)
and writes stratified bootstrap intervals.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit.p2.common import load_rows, load_splits
from audit.p2.bootstrap import bootstrap_ci

NEIGHBOR_DIST_THRESHOLD = 2


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


def _junction_seq_by_jid(rows):
    return {str(r["jid"]): str(r["junction_seq"]) for r in rows}


def _neighbor_features(test_jids, train_jids, seq_by_jid, threshold=NEIGHBOR_DIST_THRESHOLD):
    """For each distinct test junction, count of train junctions within edit
    distance <= threshold, and min distance to a train junction."""
    train_seqs = [seq_by_jid[j] for j in train_jids]
    out = {}
    for tj in test_jids:
        s = seq_by_jid[tj]
        dists = [_lev(s, ts) for ts in train_seqs]
        dmin = min(dists)
        count = sum(1 for d in dists if d <= threshold)
        out[tj] = {"neighbor_min_dist": int(dmin), "neighbor_count": int(count)}
    return out


def build_support_ledger(rows, protocol_dir, axes):
    """Per-row support features across axes/folds. Returns list of records."""
    all_seq_by_jid = _junction_seq_by_jid(list(rows.values()))
    recs = []
    for axis in axes:
        _a, by_fold, n_folds = load_splits(Path(protocol_dir) / f"SplitManifest_{axis}.jsonl")
        for fold in range(n_folds):
            test_ids = by_fold.get(fold, set())
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            train_jids = sorted({str(r["jid"]) for r in train_rows})
            train_ctx = {str(r["helix_seq"]) for r in train_rows}
            test_jids = sorted({str(r["jid"]) for r in test_rows})
            nbr = _neighbor_features(test_jids, train_jids, all_seq_by_jid)
            # per-junction censor fraction in test
            jid_y = defaultdict(list)
            for r in test_rows:
                jid_y[str(r["jid"])].append(bool(r["cens"]))
            for r in test_rows:
                j = str(r["jid"])
                cens_list = jid_y[j]
                recs.append({
                    "axis": axis, "fold": fold, "source_row_id": str(r["source_row_id"]),
                    "jid": str(r["jid"]), "scaf": int(r["scaf"]),
                    "context": str(r["helix_seq"]), "y": float(r["y"]), "cens": bool(r["cens"]),
                    "censor_fraction_junction": float(np.mean(cens_list)),
                    "junction_test_n": len(cens_list),
                    "context_support": str(r["helix_seq"]) in train_ctx,
                    "neighbor_min_dist": nbr[j]["neighbor_min_dist"],
                    "neighbor_count": nbr[j]["neighbor_count"],
                })
    return recs


def stratified_gain(predictions, axes, ledger_recs):
    """Compute per-stratum gain (baseline_nll - candidate_nll) for each factor.
    predictions: list of dicts (Phase 1 Predictions.jsonl)."""
    ledger = {(lr["axis"], lr["fold"], lr["source_row_id"]): lr for lr in ledger_recs}
    cand = { (p["axis"], p["fold"], p["source_row_id"]): p for p in predictions
             if p["model_id"] == "corrected_v1_31"}
    base = { (p["axis"], p["fold"], p["source_row_id"]): p for p in predictions
             if p["model_id"] == "train_only_scaffold"}
    factors = {
        "neighbor_min_dist": lambda lr: "near(<=1)" if lr["neighbor_min_dist"] <= 1 else "far(>1)",
        "neighbor_count": lambda lr: "low(0)" if lr["neighbor_count"] <= 0 else "high(>0)",
        "context_support": lambda lr: "seen" if lr["context_support"] else "unseen",
        "censor_fraction_junction": lambda lr: "low_cens" if lr["censor_fraction_junction"] < 0.5 else "high_cens",
    }
    out = []
    for axis in axes:
        for factor, key_fn in factors.items():
            stratum_keys = defaultdict(list)  # stratum -> [(fold, sid)]
            for (a, fold, sid), lr in ledger.items():
                if a != axis:
                    continue
                if (a, fold, sid) not in cand or (a, fold, sid) not in base:
                    continue
                stratum_keys[key_fn(lr)].append((a, fold, sid))
            for stratum, keys in stratum_keys.items():
                if not keys:
                    continue
                c_nll = _per_stratum_macro_nll([cand[k] for k in keys], axis)
                b_nll = _per_stratum_macro_nll([base[k] for k in keys], axis)
                out.append({"axis": axis, "factor": factor, "stratum": stratum,
                            "n_rows": len(keys), "candidate_nll": c_nll, "baseline_nll": b_nll,
                            "gain": b_nll - c_nll})
    return out


def _per_stratum_macro_nll(rows, axis):
    by = defaultdict(list)
    for p in rows:
        by[str(p["jid"])].append(float(p["nll"]))
    if not by:
        return None
    return float(np.mean([np.mean(v) for v in by.values()]))


def write_support(out_dir, ledger_recs, strat_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ledger_recs)
    df.to_parquet(out_dir / "SupportLedger.parquet", index=False)
    sdf = pd.DataFrame(strat_rows)
    sdf.to_csv(out_dir / "StratifiedGain.csv", index=False)
    return out_dir / "SupportLedger.parquet", out_dir / "StratifiedGain.csv"
