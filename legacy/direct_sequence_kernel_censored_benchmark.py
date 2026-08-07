#!/usr/bin/env python3
"""Direct, censor-aware sequence edit-kernel benchmark.

Unlike the historical EM-panel kernel, this probe does not construct a
panel-residual target first.  For each outer-test row it estimates a local
scaffold-specific location directly from train rows using a fixed sequence
edit kernel and the same right-censored likelihood used for scoring.  A
predeclared scaffold-baseline prior keeps unsupported or all-censored
neighborhoods conservative.  The pairing null permutes train targets before
both baseline and local fits, so it is a valid leakage control.

This is a selective local-smoothness probe, not a universal thermodynamic
model.  It remains exploratory until a parent-linked decision records its
split, support coverage, calibration and failure boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

import hetero_censored_benchmark as hetero
import sequence_fingerprint_factor_benchmark as seq


CAP = -7.1
TAU = 0.7
SEED = seq.SEED


def distance(a, b):
    pa, pb = seq.parse_parts(a), seq.parse_parts(b)
    if len(pa) != len(pb) or any(len(x) != len(y) for x, y in zip(pa, pb)):
        return None
    return int(sum(ca != cb for x, y in zip(pa, pb) for ca, cb in zip(x, y)))


def loglik_scalar(mu, row):
    if row["cens"]:
        return float(log_ndtr((float(mu) - CAP) / TAU))
    z = (float(row["y"]) - float(mu)) / TAU
    return float(-0.5 * math.log(2.0 * math.pi) - math.log(TAU) - 0.5 * z * z)


def local_location(neighbors, prior, prior_strength):
    """Penalized one-dimensional censored likelihood for one test row."""
    if not neighbors:
        return float(prior)
    # Normalize kernel weights so prior_strength has a stable effective-sample
    # interpretation across neighborhoods and motifs.
    weights = np.asarray([float(w) for _, w in neighbors], dtype=float)
    weights /= max(float(weights.sum()), 1e-12)
    mu = float(np.clip(prior, -18.0, 0.0))
    for _ in range(32):
        grad = float(prior_strength * (mu - prior))
        hess = float(prior_strength)
        for (row, _), weight in zip(neighbors, weights):
            if row["cens"]:
                a = (mu - CAP) / TAU
                log_cdf = float(log_ndtr(a))
                lam = math.exp(float(np.clip(-0.5 * a * a
                                             - 0.5 * math.log(2.0 * math.pi)
                                             - log_cdf, -50.0, 50.0)))
                grad += float(weight) * (-lam / TAU)
                hess += float(weight) * max(lam * (a + lam), 1e-10) / (TAU * TAU)
            else:
                grad += float(weight) * (mu - float(row["y"])) / (TAU * TAU)
                hess += float(weight) / (TAU * TAU)
        step = grad / max(hess, 1e-10)
        if not np.isfinite(step):
            return float(prior)
        candidate = float(np.clip(mu - step, -18.0, 0.0))
        if abs(candidate - mu) < 1e-7:
            mu = candidate
            break
        mu = candidate
    return float(mu)


def permute_targets(rows, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    out = []
    for row, idx in zip(rows, order):
        copy = dict(row)
        copy["y"] = float(rows[int(idx)]["y"])
        copy["cens"] = bool(rows[int(idx)]["cens"])
        out.append(copy)
    return out


def predict_local(train, test, baseline_mu, bandwidth, max_distance, prior_strength):
    # Unique sequences avoid repeatedly parsing the same construct.  The
    # target scaffold is retained in each neighbor lookup, so no operator is
    # silently transported across scaffolds.
    train_by_scaf = defaultdict(list)
    seq_cache = {}
    for row in train:
        raw = str(row["junction_seq"])
        seq_cache.setdefault(raw, seq.parse_parts(raw))
        train_by_scaf[int(row["scaf"])].append(row)
    mu = np.asarray(baseline_mu, dtype=float).copy()
    support = defaultdict(int)
    for i, row in enumerate(test):
        raw = str(row["junction_seq"])
        candidates = []
        for other in train_by_scaf.get(int(row["scaf"]), []):
            d = distance(raw, str(other["junction_seq"]))
            if d is None or d > int(max_distance):
                continue
            candidates.append((other, math.exp(-float(d) / float(bandwidth))))
        if not candidates:
            support["sequence_support_abstain"] += 1
            continue
        mu[i] = local_location(candidates, float(baseline_mu[i]), prior_strength)
        support["kernel_supported"] += 1
        support["neighbor_count_sum"] += int(len(candidates))
        support["distance_sum"] += int(sum(distance(raw, str(other["junction_seq"]))
                                             for other, _ in candidates))
    support["test_rows"] = int(len(test))
    return mu, dict(support)


def group_nll(rows, mu, measured_only=False, censored_only=False):
    keep = np.asarray([
        (not row["cens"] if measured_only else
         (row["cens"] if censored_only else True)) for row in rows
    ], dtype=bool)
    out = defaultdict(list)
    for row, value, ok in zip(rows, np.asarray(mu, dtype=float), keep):
        if ok:
            out[str(row["jid"])].append(-loglik_scalar(value, row))
    return {key: float(np.mean(values)) for key, values in out.items()}


def paired(base, cand, seed, n_boot=5000):
    keys = sorted(set(base) & set(cand))
    if not keys:
        return {"n_groups": 0, "mean_gain": None, "median_gain": None,
                "positive_fraction": None, "ci95": [None, None]}
    b = np.asarray([base[key] for key in keys], dtype=float)
    c = np.asarray([cand[key] for key in keys], dtype=float)
    gain = (b - c) / np.maximum(np.abs(b), 1e-12)
    rng = np.random.default_rng(seed)
    boot = gain[rng.integers(0, len(gain), size=(n_boot, len(gain)))].mean(axis=1)
    return {"n_groups": int(len(gain)), "mean_gain": float(np.mean(gain)),
            "median_gain": float(np.median(gain)),
            "positive_fraction": float(np.mean(gain > 0)),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def calibration(rows, mu):
    y = np.asarray([row["y"] for row in rows], dtype=float)
    cens = np.asarray([row["cens"] for row in rows], dtype=bool)
    mu = np.asarray(mu, dtype=float)
    measured = ~cens
    z = (y[measured] - mu[measured]) / TAU if measured.any() else np.asarray([])
    p_cens = np.exp(log_ndtr((mu - CAP) / TAU))
    return {"observed_censor_fraction": float(np.mean(cens)),
            "predicted_censor_probability_mean": float(np.mean(p_cens)),
            "censor_tail_difference": float(np.mean(p_cens) - np.mean(cens)),
            "measured_95_coverage": float(np.mean(np.abs(z) <= 1.959963984540054)) if len(z) else None,
            "measured_z_median": float(np.median(z)) if len(z) else None}


def evaluate(train, test, bandwidth, max_distance, prior_strength,
             shuffled_targets=False, seed=SEED):
    fit_rows = permute_targets(train, seed) if shuffled_targets else train
    baseline = hetero.fit_beta(fit_rows, mode="none", tau=TAU, ridge=1.0,
                               balanced=True, use_err=False)
    base_mu = np.asarray(hetero.predict(baseline, test), dtype=float)
    mu, support = predict_local(fit_rows, test, base_mu, bandwidth,
                                max_distance, prior_strength)
    base_j, cand_j = group_nll(test, base_mu), group_nll(test, mu)
    return {"baseline_macro_junction_nll": float(np.mean(list(base_j.values()))) if base_j else None,
            "candidate_macro_junction_nll": float(np.mean(list(cand_j.values()))) if cand_j else None,
            "gain": paired(base_j, cand_j, seed + 7000),
            "measured_only_gain": paired(group_nll(test, base_mu, measured_only=True),
                                          group_nll(test, mu, measured_only=True), seed + 8000),
            "censored_only_gain": paired(group_nll(test, base_mu, censored_only=True),
                                         group_nll(test, mu, censored_only=True), seed + 9000),
            "calibration": calibration(test, mu),
            "support": support,
            "n_test": int(len(test)),
            "n_test_junctions": int(len({str(row["jid"]) for row in test})),
            "n_test_censored": int(sum(bool(row["cens"]) for row in test)),
            "shuffled_targets": bool(shuffled_targets),
            "status": "EXPLORATORY_NOT_AUTHORITATIVE"}


def split_rows(rows, field, held):
    held = {str(value) for value in held}
    return ([row for row in rows if str(row[field]) not in held],
            [row for row in rows if str(row[field]) in held])


def blocks(rows, split, seed):
    if split == "motif_lomo":
        return [{x} for x in sorted({str(row["motif"]) for row in rows})], "motif"
    if split == "context_5fold":
        groups = sorted({str(row["helix_seq"]) for row in rows})
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        return [set(x.tolist()) for x in np.array_split(np.asarray(groups, dtype=object), 5)], "helix_seq"
    if split == "scaffold_lomo":
        return [{x} for x in sorted({str(row["scaf"]) for row in rows})], "scaf"
    if split == "edit_component_5fold":
        seq.edit_component_labels(rows, max_distance=1)
        groups = sorted({str(row["edit_component"]) for row in rows})
        rng = np.random.default_rng(seed + 29)
        rng.shuffle(groups)
        return [set(x.tolist()) for x in np.array_split(np.asarray(groups, dtype=object), 5)], "edit_component"
    if split == "sequence_5fold":
        groups = sorted({str(row["junction_seq"]) for row in rows})
        rng = np.random.default_rng(seed + 23)
        rng.shuffle(groups)
        return [set(x.tolist()) for x in np.array_split(np.asarray(groups, dtype=object), 5)], "junction_seq"
    raise ValueError(split)


def run(rows, splits, bandwidth, max_distance, prior_strength, controls=False, seed=SEED):
    result = {}
    for split in splits:
        group_blocks, field = blocks(rows, split, seed)
        folds = []
        for i, held in enumerate(group_blocks):
            train, test = split_rows(rows, field, held)
            fold = {"held_groups": sorted(held), "group_field": field,
                    "n_train": len(train), "n_test": len(test),
                    "genuine": evaluate(train, test, bandwidth, max_distance,
                                         prior_strength, seed=seed + 1000 * i)}
            if controls:
                fold["pairing_shuffle_null"] = evaluate(
                    train, test, bandwidth, max_distance, prior_strength,
                    shuffled_targets=True, seed=seed + 50000 + i)
            folds.append(fold)
        result[split] = {"split": split, "group_field": field, "folds": folds}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--splits", default="motif_lomo,edit_component_5fold,context_5fold,scaffold_lomo")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--max-distance", type=int, default=2)
    ap.add_argument("--prior-strength", type=float, default=4.0)
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    if a.bandwidth <= 0 or a.max_distance < 0 or a.prior_strength <= 0:
        raise SystemExit("bandwidth, prior-strength must be positive and max-distance non-negative")
    rows, fallback = seq.load_records(a.records)
    results = run(rows, [x for x in a.splits.split(",") if x], a.bandwidth,
                  a.max_distance, a.prior_strength, controls=a.controls)
    out = {"schema_version": "tecto-direct-sequence-kernel-censored-v1",
           "status": "EXPLORATORY_NOT_AUTHORITATIVE",
           "records_sha256": hashlib.sha256(a.records.read_bytes()).hexdigest(),
           "cap": CAP, "tau": TAU, "n_rows": len(rows),
           "n_measured": int(sum(not row["cens"] for row in rows)),
           "n_censored": int(sum(bool(row["cens"]) for row in rows)),
           "n_junctions": int(len({str(row["jid"]) for row in rows})),
           "n_motifs": int(len({str(row["motif"]) for row in rows})),
           "n_scaffolds": int(len({int(row["scaf"]) for row in rows})),
           "err10_fallback": fallback,
           "kernel": {"distance": "same-length ordered Hamming on junction_seq",
                       "bandwidth": float(a.bandwidth), "max_distance": int(a.max_distance),
                       "prior_strength": float(a.prior_strength),
                       "target_scaffold_only": True},
           "results": results,
           "rules": {"censoring": "right_censored_y_ge_minus_7_1",
                     "baseline": "train-only scaffold intercepts",
                     "forbidden": ["EM posterior panel targets", "DMS", "qMaPseq", "dg9", "dg11", "dg10_5mM", "outer labels", "foundation model"],
                     "claim_boundary": "local edit-graph selective prediction; not universal junction preorganization"}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(a.output), "splits": list(results)}, indent=2))


if __name__ == "__main__":
    main()
