"""Phase 2 permutation null runner (contract Phase 2).

Runs group-preserving label / sequence-pairing permutation nulls and computes
the null distribution of the axis-mean gain:

    gain = mean over folds [ NLL(train_only_scaffold) - NLL(corrected_v1_31) ]

Under each permutation the TRAIN association between sequence and outcome is
destroyed (label shuffle within junction groups, or sequence-pairing shuffle);
both the candidate and the reference baseline are refit on the SAME permuted
train and evaluated on the real frozen test rows, so the null gain is directly
comparable to the observed genuine gain.

Parallelised across CPU cores (candidate fits are single-threaded), using fork
so the frozen axis-fold structures are shared via copy-on-write.

Emits NullResults.parquet (per-permutation candidate/baseline NLL + gain) and
appends the run record into NullProtocol.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit.p2.common import (
    CANDIDATE, REFERENCE_BASELINE,
    load_rows, load_splits, build_axis_folds,
    permute_labels_within_junction, permute_sequence_pairing,
    fit_predict,
)
from audit.evaluation.metrics import junction_macro_nll

# Worker context shared via fork copy-on-write: {axis_name: list of folds}
_POOL_CTX = {"folds": {}, "axis": None}


def _permuted_train(train_rows, perm_type, rng):
    if perm_type == "label":
        return permute_labels_within_junction(train_rows, rng)
    if perm_type == "sequence":
        return permute_sequence_pairing(train_rows, rng)
    raise ValueError(perm_type)


def _perm_pair(seed, perm_type):
    """Return (candidate_nll, baseline_nll) for one permutation over all folds
    of the configured axis. A single permuted train is generated per fold and
    shared by both models."""
    folds = _POOL_CTX["folds"]
    rng = np.random.default_rng(seed)
    c_nll, b_nll = [], []
    for _test_ids, train_rows, test_rows in folds:
        train_p = _permuted_train(train_rows, perm_type, rng)
        # candidate
        mu_c, sig_c = fit_predict(train_p, test_rows, CANDIDATE)
        c_nll.append(junction_macro_nll(test_rows, mu_c, sig_c))
        # baseline (same permuted train)
        mu_b, sig_b = fit_predict(train_p, test_rows, REFERENCE_BASELINE)
        b_nll.append(junction_macro_nll(test_rows, mu_b, sig_b))
    return float(np.mean(c_nll)) if c_nll else None, float(np.mean(b_nll)) if b_nll else None


def _worker(args):
    seed, perm_type = args
    try:
        c, b = _perm_pair(seed, perm_type)
        return {"seed": seed, "perm_type": perm_type, "axis": _POOL_CTX["axis"],
                "nll_candidate": c, "nll_baseline": b, "gain": (b - c) if (c is not None and b is not None) else None}
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "perm_type": perm_type, "axis": _POOL_CTX["axis"],
                "nll_candidate": None, "nll_baseline": None, "gain": None, "error": str(e)}


def run_axis_permutations(rows, manifest_path, axis, perm_type, n_perms,
                          n_workers, out_dir, start_seed=0):
    _axis, by_fold, n_folds = load_splits(manifest_path)
    folds = build_axis_folds(rows, by_fold, n_folds)
    _POOL_CTX["folds"] = folds
    _POOL_CTX["axis"] = axis
    seeds = [start_seed + i for i in range(n_perms)]
    t0 = time.time()
    if n_workers <= 1:
        recs = [_worker((s, perm_type)) for s in seeds]
    else:
        with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
            recs = pool.map(_worker, [(s, perm_type) for s in seeds])
    elapsed = time.time() - t0
    df = pd.DataFrame(recs)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"NullResults_{axis}_{perm_type}.parquet"
    df.to_parquet(parquet, index=False)
    valid = df["gain"].notna()
    return {
        "axis": axis, "perm_type": perm_type, "n_perms": n_perms,
        "n_valid": int(valid.sum()), "elapsed_s": round(elapsed, 1),
        "mean_gain": float(np.nanmean(df["gain"])),
        "sd_gain": float(np.nanstd(df["gain"])) if valid.any() else None,
        "p975_gain": float(np.nanpercentile(df["gain"], 97.5)) if valid.any() else None,
        "parquet": str(parquet),
    }


def _init_worker():
    pass
