"""R2 group/multiway cluster uncertainty (contract §12.3 / FrozenGateSpec).

The genuine axis-level statistic is the *junction-macro* mean NLL difference
between the matched no-sequence model and the full (corrected v1.31) model:

    theta_j        = mean over rows of (nll_ns - nll_full)  within junction j
    theta          = mean over junctions of theta_j          (equal-weight macro)
    relative_gain  = theta / mean_j( mean-row-nll_ns over junction j )

Clustering follows the estimand: junction is the repeated predictive unit
(each junction appears in 4-9 contexts; contract §7.4 warns row-level
bootstrap overinflates support).  We therefore:
  * junction-cluster percentile bootstrap for every axis  -> 95% CI on theta,
  * junction x context two-way cluster bootstrap for the context axis
    (contract: "context 使用 junction x context multiway cluster"),
  * junction-level pairing null (same statistic as genuine, sign-flip within
    junction) -> 1000-null distribution, one-sided p and 97.5% upper bound.

Gate (fail-closed, FrozenGateSpec §8.6/§9.2):
  group-bootstrap 95% CI lower bound > 0  AND  null 97.5% upper bound < genuine
  (relative_gain >= 0.10 and 5/5 fold positivity are handled by the caller.)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.evaluation.metrics import row_nll

SEED = 17
N_NULL = 1000
N_BOOT = 1000
ALPHA = 0.05
JOINT_CLUSTER_AXES = {"context_lomo"}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _iter_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_axis_rows(run_root: Path, axis: str):
    """Eligible full-coverage rows for (axis) where BOTH full and no-sequence
    make a supported prediction.  Returns list of dicts with jid/context/scaf
    and per-row d = nll_ns - nll_full and nll_ns."""
    full_path = run_root / "r05_v131" / "Predictions_v1_31.jsonl"
    ns_path = run_root / "r1" / "Predictions_v2.jsonl"
    F = {}
    N = {}
    for p in _iter_jsonl(full_path):
        if p["axis"] == axis and p["model_id"] == "corrected_v1_31":
            F[p["source_row_id"]] = p
    for p in _iter_jsonl(ns_path):
        if p["axis"] == axis and p["model_id"] == "no_sequence_latent_operator":
            N[p["source_row_id"]] = p
    rows = []
    for rid, f in F.items():
        n = N.get(rid)
        if n is None:
            continue
        if not (f.get("support") and n.get("support")):
            continue
        if f.get("abstain") or n.get("abstain"):
            continue
        nll_full = float(row_nll([f["y"]], [f["cens"]], [f["mu"]], [f["sigma"]])[0])
        nll_ns = float(row_nll([n["y"]], [n["cens"]], [n["mu"]], [n["sigma"]])[0])
        rows.append({
            "source_row_id": rid, "jid": str(f["jid"]),
            "context": str(f["context"]), "scaf": int(f["scaf"]),
            "y": float(f["y"]), "cens": bool(f["cens"]),
            "d": nll_ns - nll_full, "nll_ns": nll_ns,
        })
    return rows


def _junction_deltas(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["jid"]].append(r["d"])
    return {j: float(np.mean(v)) for j, v in by.items()}


def _junction_nll_ns(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["jid"]].append(r["nll_ns"])
    return {j: float(np.mean(v)) for j, v in by.items()}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def axis_statistic(rows):
    """Return (theta, relative_gain, n_junctions) for the junction-macro mean
    NLL difference (no_sequence minus full).  Positive theta => full better."""
    deltas = _junction_deltas(rows)
    nll_ns = _junction_nll_ns(rows)
    if not deltas:
        return None, None, 0
    theta = float(np.mean(list(deltas.values())))
    base = float(np.mean(list(nll_ns.values())))
    rel = theta / base if base else None
    return theta, rel, len(deltas)


def junction_bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap resampling junctions (the repeated predictive
    unit).  Returns array of theta_b length n_boot."""
    deltas = _junction_deltas(rows)
    jids = list(deltas)
    if not jids:
        return np.array([np.nan])
    arr = np.array([deltas[j] for j in jids], dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([float(arr[rng.integers(0, len(arr), len(arr))].mean())
                     for _ in range(n_boot)])


def two_way_cluster_bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    """junction x context two-way cluster bootstrap for the context axis.

    For each bootstrap draw we resample junctions with replacement AND resample
    contexts with replacement; a resampled junction's value is the mean over the
    resampled contexts it actually occupies.  The axis value is the mean over
    the resampled junctions.  This captures correlation on both dimensions."""
    by_jid = defaultdict(list)
    for r in rows:
        by_jid[r["jid"]].append(r)
    jids = list(by_jid)
    cell = {}
    for j, rs in by_jid.items():
        ctxd = defaultdict(list)
        for r in rs:
            ctxd[r["context"]].append(r["d"])
        cell[j] = {c: float(np.mean(v)) for c, v in ctxd.items()}
    contexts = sorted({r["context"] for r in rows})
    if not jids or not contexts:
        return np.array([np.nan])
    rng = np.random.default_rng(seed)
    thetas = []
    for _ in range(n_boot):
        jb = rng.choice(jids, size=len(jids), replace=True)
        cb = rng.choice(contexts, size=len(contexts), replace=True)
        vals = []
        for j in jb:
            present = [c for c in cb if c in cell[j]]
            if present:
                vals.append(float(np.mean([cell[j][c] for c in present])))
        thetas.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(thetas)


def junction_pairing_null(rows, n_null=N_NULL, seed=SEED):
    """Junction-level pairing null: under exchangeability each per-junction
    delta is symmetric about 0; flip the sign of a random subset of junctions
    and recompute the SAME axis statistic.  Returns null theta array."""
    deltas = _junction_deltas(rows)
    jids = list(deltas)
    if not jids:
        return np.array([np.nan] * n_null)
    arr = np.array([deltas[j] for j in jids], dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_null)
    for k in range(n_null):
        signs = rng.choice([-1.0, 1.0], size=len(arr))
        out[k] = float((arr * signs).mean())
    return out


def percentile_ci(samples, alpha=ALPHA):
    lo = float(np.percentile(samples, 100 * alpha / 2.0))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2.0)))
    return lo, hi


# ---------------------------------------------------------------------------
# per-axis runner
# ---------------------------------------------------------------------------
def evaluate_axis(run_root: Path, axis: str):
    rows = load_axis_rows(run_root, axis)
    res = {"axis": axis, "n_rows": len(rows)}
    if not rows:
        res.update({"available": False, "reason": "no eligible full-coverage rows"})
        return res
    theta, rel, n_j = axis_statistic(rows)
    res.update({"available": True, "theta": theta, "relative_gain": rel,
                "n_junctions": n_j})
    res["n_contexts"] = len({r["context"] for r in rows})
    res["n_scaffolds"] = len({r["scaf"] for r in rows})

    boot = junction_bootstrap(rows)
    lo, hi = percentile_ci(boot)
    res["junction_boot_ci"] = [lo, hi]
    res["junction_boot_lower_gt_0"] = bool(lo > 0)
    res["junction_boot_95ci_upper"] = hi

    if axis in JOINT_CLUSTER_AXES:
        tw = two_way_cluster_bootstrap(rows)
        lo2, hi2 = percentile_ci(tw)
        res["two_way_ci"] = [lo2, hi2]
        res["two_way_lower_gt_0"] = bool(lo2 > 0)

    nulls = junction_pairing_null(rows)
    p = float((np.nansum(nulls >= theta) + 1) / (N_NULL + 1))
    res["null_p_value"] = p
    res["null_975_upper"] = float(np.percentile(nulls, 97.5))
    res["null_975_upper_lt_genuine"] = bool(np.percentile(nulls, 97.5) < theta)
    return res


def run(run_root: Path, axes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ax in axes:
        results.append(evaluate_axis(run_root, ax))
    report = {
        "run_root": str(run_root),
        "statistic": "junction-macro mean of (nll_no_sequence - nll_full)",
        "clustering": ("junction-cluster bootstrap (all axes); "
                       "junction x context two-way cluster (context_lomo)"),
        "null": f"junction-level pairing sign-flip, {N_NULL} per axis, seed {SEED}",
        "axes": results,
    }
    (out_dir / "MultiwayCluster.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report
