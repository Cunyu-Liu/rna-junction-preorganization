#!/usr/bin/env python3
"""Sequence-only low-rank thermodynamic fingerprint benchmark.

This is an exploratory method-repair probe, not a contract implementation.
It asks whether a junction's *multi-scaffold residual fingerprint* is easier to
predict from sequence than individual row residuals.  The fingerprint is
constructed from the training side only, then compressed with a training-only
SVD and regressed from a fixed, low-capacity junction-sequence representation.

The primary outer split is motif-family LOMO.  A held-out motif has no labels,
no panel rows and no target-derived features in the sequence model.  All
feature scaling, missing-cell imputation, SVD rank and ridge selection happen
inside an inner motif-family split.  This is deliberately a bounded pilot:
it does not use dg9/dg11/dg10_5mM, does not use DMS, and does not modify the
v1.11 contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from sklearn.linear_model import Ridge

import hetero_censored_benchmark as hetero

CAP = -7.1
TAU = 0.7
SEED = hetero.SEED


def parse_parts(raw):
    return [x.upper() for x in str(raw or "").replace("&", "_").split("_")]


def symmetry_key(raw):
    """Canonicalize the reciprocal two-strand representation."""
    parts = parse_parts(raw)
    if len(parts) == 2:
        forward = "_".join(parts)
        swapped = "_".join(parts[::-1])
        return min(forward, swapped)
    return "_".join(parts)


def edit_component_labels(rows, max_distance=1):
    """Build an unsupervised one-edit graph over symmetry-canonical sequences.

    This is a source-level exposure grouping, not a label-derived feature. It
    is used only to ask whether a model survives a blocked sequence-family
    split. Edges require equal strand lengths and total Hamming distance at
    most ``max_distance``; reciprocal strand swaps are already canonicalized.
    """
    keys = sorted({str(row["symmetry_key"]) for row in rows})
    parsed = [tuple(key.split("_")) for key in keys]
    parent = list(range(len(keys)))
    size = [1] * len(keys)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a); b = find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for i in range(len(keys)):
        a_parts = parsed[i]
        for j in range(i):
            b_parts = parsed[j]
            if len(a_parts) != len(b_parts):
                continue
            if any(len(a) != len(b) for a, b in zip(a_parts, b_parts)):
                continue
            distance = sum(
                ca != cb for a, b in zip(a_parts, b_parts) for ca, cb in zip(a, b)
            )
            if distance <= int(max_distance):
                union(i, j)
    names = {}
    for i, key in enumerate(keys):
        root = find(i)
        names.setdefault(root, key)
    labels = {key: names[find(i)] for i, key in enumerate(keys)}
    for row in rows:
        row["edit_component"] = labels[str(row["symmetry_key"])]
    return labels


def seq_features(raw, max_parts=2, max_len=7):
    """Fixed position/composition features; no learned global vocabulary."""
    alphabet = "ACGU"
    out = np.zeros(max_parts * max_len * len(alphabet), dtype=float)
    parts = parse_parts(raw)
    for pi, part in enumerate(parts[:max_parts]):
        for pos, base in enumerate(part[:max_len]):
            if base in alphabet:
                out[(pi * max_len + pos) * 4 + alphabet.index(base)] = 1.0
    seq = "".join(parts)
    den = max(len(seq), 1)
    extras = [float(seq.count(b) / den) for b in alphabet]
    extras += [float(len(seq))]
    extras += [float(len(p)) for p in parts[:max_parts]]
    extras += [0.0] * max(0, max_parts - len(parts))
    return np.r_[out, np.asarray(extras, dtype=float)]


def num(x):
    try:
        if x is None or x == "":
            return None
        value = float(x)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def load_records(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("sublibrary") != "junction_conformations":
            continue
        y = num(raw.get("dg10"))
        scaf = num(raw.get("chip_scaffold"))
        if y is None or scaf is None:
            continue
        rows.append(
            {
                "jid": str(raw.get("junction_id")),
                "motif": str(raw.get("motif_type")),
                "scaf": int(scaf),
                "y": float(y),
                "cens": abs(float(y) - CAP) < 1e-8,
                "junction_seq": raw.get("junction_seq") or "",
                "helix_seq": raw.get("helix_seq") or "",
                "symmetry_key": symmetry_key(raw.get("junction_seq") or ""),
                "err10": num(raw.get("err10")),
                # The imported censored scaffold baseline builds its full
                # schema before selecting mode='none'; retain these source
                # fields even though this candidate never uses them as
                # sequence features.
                "dg_fold": num(raw.get("dg_fold")),
                "dg_fold_constrained": num(raw.get("dg_fold_constrained")),
                "dg9": num(raw.get("dg9")),
                "dg11": num(raw.get("dg11")),
                "dg10_5mM": num(raw.get("dg10_5mM")),
                "length": num(raw.get("length")),
                "helix_one_length": num(raw.get("helix_one_length")),
            }
        )
    finite = [r["err10"] for r in rows if r["err10"] is not None]
    fallback = float(np.median(finite)) if finite else 0.25
    for r in rows:
        if r["err10"] is None:
            r["err10"] = fallback
    return rows, fallback


def fit_m0(train):
    model = hetero.fit_beta(
        train, mode="none", tau=TAU, ridge=1.0, balanced=True, use_err=False
    )
    return model, predict_m0_fast(model, train)


def predict_m0_fast(model, rows):
    """Predict the mode='none' scaffold model without rebuilding rich schema.

    The shared benchmark's generic design function computes unused sequence
    descriptors even for mode='none'.  This direct reference-coded predictor
    is algebraically identical for the intercept+scaffold model and avoids
    repeating that expensive work for every inner candidate.
    """
    beta = np.asarray(model["beta"], dtype=float)
    scaffolds = list(model.get("scaffolds", []))
    out = np.full(len(rows), float(beta[0]) if len(beta) else 0.0, dtype=float)
    index = {int(s): i + 1 for i, s in enumerate(scaffolds[1:])}
    for i, row in enumerate(rows):
        j = index.get(int(row["scaf"]))
        if j is not None and j < len(beta):
            out[i] += float(beta[j])
    return out


def censored_residual(row, mu, mode):
    if not row["cens"]:
        return float(row["y"] - mu)
    if mode == "exclude":
        return np.nan
    # Sensitivity-only conditional mean on the common observation scale.  This
    # is not asserted to be a full multivariate censored likelihood.
    a = (CAP - float(mu)) / TAU
    log_lambda = -0.5 * a * a - 0.5 * math.log(2.0 * math.pi) - log_ndtr(-a)
    return float(TAU * math.exp(float(np.clip(log_lambda, -50.0, 50.0))))


def fit_feature_scaler(rows):
    unique = {}
    for row in rows:
        unique.setdefault(str(row["jid"]), row["junction_seq"])
    ids = sorted(unique)
    X = np.asarray([seq_features(unique[j]) for j in ids], dtype=float)
    mean = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < 1e-8] = 1.0
    return {"ids": ids, "raw_by_id": unique, "mean": mean, "sd": sd}


def transform_features(rows, scaler):
    X = np.asarray(
        [seq_features(row["junction_seq"]) for row in rows], dtype=float
    )
    return (X - scaler["mean"]) / scaler["sd"]


def build_fingerprint(train, mu_train, censor_mode):
    """Return a train-only junction x scaffold residual matrix.

    There is at most one canonical row per junction/scaffold.  If a future
    source revision contains duplicates, their finite residuals are averaged;
    this keeps the aggregation explicit rather than silently treating rows as
    independent observations.
    """
    by = defaultdict(lambda: defaultdict(list))
    for row, mu in zip(train, mu_train):
        value = censored_residual(row, mu, censor_mode)
        if np.isfinite(value):
            by[str(row["jid"])][int(row["scaf"])].append(float(value))
    jids = sorted(by)
    R = np.full((len(jids), 9), np.nan, dtype=float)
    for i, jid in enumerate(jids):
        for scaf, values in by[jid].items():
            R[i, scaf - 1] = float(np.mean(values))
    # Column means and imputation are fit on the training junctions only.
    col_mean = np.zeros(9, dtype=float)
    for s in range(9):
        vals = R[np.isfinite(R[:, s]), s]
        if len(vals):
            col_mean[s] = float(np.mean(vals))
    missing = ~np.isfinite(R)
    R_filled = np.where(missing, col_mean[None, :], R)
    centered = R_filled - col_mean[None, :]
    if len(jids):
        U, singular, Vt = np.linalg.svd(centered, full_matrices=False)
    else:
        U = np.zeros((0, 0)); singular = np.zeros(0); Vt = np.zeros((0, 9))
    return {
        "jids": jids,
        "col_mean": col_mean,
        "matrix": centered,
        "missing_count": int(np.sum(missing)),
        "U": U,
        "singular": singular,
        "Vt": Vt,
    }


def fit_model(
    train,
    test,
    rank,
    alpha,
    censor_mode,
    shuffled_targets=False,
    seed=SEED,
    base_model=None,
    mu_train=None,
    fingerprint=None,
    scaler=None,
):
    if base_model is None or mu_train is None:
        base_model, mu_train = fit_m0(train)
    m0 = base_model
    base_test = predict_m0_fast(m0, test)
    fp = build_fingerprint(train, mu_train, censor_mode) if fingerprint is None else fingerprint
    scaler = fit_feature_scaler(train) if scaler is None else scaler
    jids = fp["jids"]
    if not jids:
        return base_test, {"finite": False, "reason": "no_train_fingerprint"}
    # Each training junction has one multi-output target.  This removes the
    # unequal row/context density that affected row-level residual regressions.
    X = np.asarray(
        [(seq_features(scaler["raw_by_id"][jid]) - scaler["mean"]) / scaler["sd"] for jid in jids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    Y = np.asarray(fp["matrix"], dtype=float)
    if shuffled_targets:
        Y = Y[rng.permutation(len(Y))]
    kmax = min(int(rank), Y.shape[1], Y.shape[0])
    if kmax <= 0:
        pred_matrix = np.zeros((len(test), 9), dtype=float)
    else:
        # Low-rank target compression is fitted from training fingerprints;
        # the regressor itself is multi-output and predicts factor scores.
        if shuffled_targets:
            # Recompute the factor basis after the pairing permutation so the
            # null does not retain an aligned target-derived basis.
            _, _, Vt_null = np.linalg.svd(Y, full_matrices=False)
            V = Vt_null[:kmax].T
        else:
            V = fp["Vt"][:kmax].T
        scores = np.dot(Y, V)
        # ``solver='lsqr'`` avoids a platform-specific BLAS ``@`` warning in
        # the macOS sklearn cholesky path while solving the same ridge problem.
        reg = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-10)
        reg.fit(X, scores)
        Xte = transform_features(test, scaler)
        # sklearn's predict path uses the local BLAS ``@`` operator that has
        # emitted false overflow warnings in this environment; the explicit
        # dot is numerically equivalent and keeps the finite-value guard
        # meaningful.
        pred_scores = np.asarray(
            np.dot(Xte, np.asarray(reg.coef_).T) + np.asarray(reg.intercept_),
            dtype=float,
        )
        if pred_scores.ndim == 1:
            pred_scores = pred_scores[:, None]
        pred_matrix = np.dot(pred_scores, V.T) + fp["col_mean"][None, :]
    correction = np.asarray(
        [pred_matrix[i, int(row["scaf"]) - 1] for i, row in enumerate(test)],
        dtype=float,
    )
    mu = base_test + correction
    finite = bool(np.isfinite(mu).all())
    return mu, {
        "finite": finite,
        "rank": int(rank),
        "alpha": float(alpha),
        "censor_mode": censor_mode,
        "n_train_junctions": int(len(jids)),
        "n_train_fingerprint_missing_cells": int(fp["missing_count"]),
        "singular_values": [float(x) for x in fp["singular"][: min(9, len(fp["singular"]))]],
        "correction_mean": float(np.mean(correction)) if len(correction) else 0.0,
        "correction_sd": float(np.std(correction)) if len(correction) else 0.0,
        "max_abs_correction": float(np.max(np.abs(correction))) if len(correction) else 0.0,
        "shuffled_targets": bool(shuffled_targets),
    }


def grouped_nll(rows, mu, measured_only=False):
    keep = np.asarray([not r["cens"] for r in rows], dtype=bool) if measured_only else np.ones(len(rows), dtype=bool)
    selected = [row for row, flag in zip(rows, keep) if flag]
    selected_mu = np.asarray([value for value, flag in zip(mu, keep) if flag], dtype=float)
    losses = -hetero.loglik(
        np.asarray([r["y"] for r in selected], dtype=float),
        np.asarray([r["cens"] for r in selected], dtype=bool),
        selected_mu,
        np.full(len(selected), TAU, dtype=float),
    )
    out = defaultdict(list)
    for row, loss in zip(selected, losses):
        out[str(row["jid"])].append(float(loss))
    return {key: float(np.mean(values)) for key, values in out.items()}


def paired_gain(base, candidate, seed, n_boot=5000):
    keys = sorted(set(base) & set(candidate))
    if not keys:
        return {"n_groups": 0, "mean_gain": None, "median_gain": None, "positive_fraction": None, "ci95": [None, None]}
    b = np.asarray([base[k] for k in keys], dtype=float)
    c = np.asarray([candidate[k] for k in keys], dtype=float)
    gain = (b - c) / np.maximum(np.abs(b), 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gain), size=(n_boot, len(gain)))
    boot = np.mean(gain[idx], axis=1)
    return {
        "n_groups": int(len(gain)),
        "mean_gain": float(np.mean(gain)),
        "median_gain": float(np.median(gain)),
        "positive_fraction": float(np.mean(gain > 0)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def evaluate(
    train,
    test,
    rank,
    alpha,
    censor_mode,
    seed,
    shuffled_targets=False,
    base_model=None,
    mu_train=None,
    fingerprint=None,
    scaler=None,
):
    if base_model is None or mu_train is None:
        base_model, mu_train = fit_m0(train)
    base_m0 = base_model
    base_mu = predict_m0_fast(base_m0, test)
    base = grouped_nll(test, base_mu)
    mu, meta = fit_model(
        train,
        test,
        rank,
        alpha,
        censor_mode,
        shuffled_targets=shuffled_targets,
        seed=seed,
        base_model=base_m0,
        mu_train=mu_train,
        fingerprint=fingerprint,
        scaler=scaler,
    )
    cand = grouped_nll(test, mu) if meta.get("finite") else {}
    base_measured = grouped_nll(test, base_mu, measured_only=True)
    cand_measured = grouped_nll(test, mu, measured_only=True) if meta.get("finite") else {}
    return {
        "baseline_macro_junction_nll": float(np.mean(list(base.values()))) if base else None,
        "candidate_macro_junction_nll": float(np.mean(list(cand.values()))) if cand else None,
        "gain": paired_gain(base, cand, seed + 7000) if cand else None,
        "measured_only_gain": paired_gain(base_measured, cand_measured, seed + 8000) if cand_measured else None,
        "n_test_measured": int(sum(not row["cens"] for row in test)),
        "n_test_censored": int(sum(bool(row["cens"]) for row in test)),
        "meta": meta,
    }


def motif_split(rows, held):
    held = set(held)
    return (
        [r for r in rows if str(r["motif"]) not in held],
        [r for r in rows if str(r["motif"]) in held],
    )


def group_split(rows, group_field, held):
    held = set(held)
    return (
        [r for r in rows if str(r[group_field]) not in held],
        [r for r in rows if str(r[group_field]) in held],
    )


def inner_select(outer_train, seed, ranks, alphas, censor_modes, group_field="motif"):
    groups = sorted({str(r[group_field]) for r in outer_train})
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    held = set(shuffled[: max(1, int(round(0.2 * len(shuffled))))])
    tr, va = group_split(outer_train, group_field, held)
    base_model, mu_train = fit_m0(tr)
    scaler = fit_feature_scaler(tr)
    fingerprints = {mode: build_fingerprint(tr, mu_train, mode) for mode in censor_modes}
    candidates = []
    for mode in censor_modes:
        for rank in ranks:
            for alpha in alphas:
                score = evaluate(
                    tr,
                    va,
                    rank,
                    alpha,
                    mode,
                    seed + rank + int(alpha),
                    base_model=base_model,
                    mu_train=mu_train,
                    fingerprint=fingerprints[mode],
                    scaler=scaler,
                )
                candidates.append({"rank": rank, "alpha": alpha, "censor_mode": mode, **score})
    finite = [x for x in candidates if x["candidate_macro_junction_nll"] is not None]
    best = min(finite, key=lambda x: (x["candidate_macro_junction_nll"], x["rank"], x["alpha"], x["censor_mode"])) if finite else None
    return {
        "held_groups": sorted(held),
        "group_field": group_field,
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
        "best": best,
        "top_candidates": sorted(finite, key=lambda x: x["candidate_macro_junction_nll"])[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ranks", default="0,1,2,3,5,9")
    ap.add_argument("--alphas", default="1,10,30,100,300,1000")
    ap.add_argument("--censor-modes", default="exclude,truncated_mean")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument(
        "--split",
        choices=["motif_lomo", "symmetry_kfold", "edit_component_kfold", "edit_component_lomo"],
        default="motif_lomo",
    )
    a = ap.parse_args()
    rows, fallback = load_records(a.records)
    ranks = [int(x) for x in a.ranks.split(",") if x]
    alphas = [float(x) for x in a.alphas.split(",") if x]
    censor_modes = [x for x in a.censor_modes.split(",") if x]
    motifs = sorted({str(r["motif"]) for r in rows})
    if a.split == "motif_lomo":
        group_field = "motif"
        outer_blocks = [{motif} for motif in motifs]
    else:
        if a.split in {"edit_component_kfold", "edit_component_lomo"}:
            edit_component_labels(rows, max_distance=1)
            group_field = "edit_component"
        else:
            group_field = "symmetry_key"
        groups = sorted({str(r[group_field]) for r in rows})
        if a.split == "edit_component_lomo":
            outer_blocks = [{group} for group in groups]
        else:
            rng = np.random.default_rng(SEED + 17)
            shuffled_groups = list(groups)
            rng.shuffle(shuffled_groups)
            outer_blocks = [set(x.tolist()) for x in np.array_split(np.asarray(shuffled_groups, dtype=object), 5)]
    folds = []
    for i, held_groups in enumerate(outer_blocks):
        train, test = group_split(rows, group_field, held_groups)
        inner = inner_select(train, SEED + i, ranks, alphas, censor_modes, group_field=group_field)
        best = inner["best"]
        outer = None
        if best is not None:
            base_model, mu_train = fit_m0(train)
            scaler = fit_feature_scaler(train)
            fingerprint = build_fingerprint(train, mu_train, best["censor_mode"])
            outer = evaluate(
                train,
                test,
                best["rank"],
                best["alpha"],
                best["censor_mode"],
                SEED + 2000 + i,
                base_model=base_model,
                mu_train=mu_train,
                fingerprint=fingerprint,
                scaler=scaler,
            )
        folds.append(
            {
                "held_groups": sorted(held_groups),
                "held_motif": sorted(held_groups) if group_field == "motif" else None,
                "group_field": group_field,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_test_junctions": int(len({r["jid"] for r in test})),
                "inner": inner,
                "selected": None if best is None else {k: best[k] for k in ("rank", "alpha", "censor_mode")},
                "outer": outer,
            }
        )
    gains = [f["outer"]["gain"]["mean_gain"] for f in folds if f["outer"] and f["outer"]["gain"]]
    out = {
        "schema_version": "tecto-sequence-fingerprint-factor-v1",
        "status": "EXPLORATORY_NOT_AUTHORITATIVE",
        "records_sha256": hashlib.sha256(a.records.read_bytes()).hexdigest(),
        "n_rows": int(len(rows)),
        "n_junctions": int(len({r["jid"] for r in rows})),
        "n_motifs": int(len(motifs)),
        "split": a.split,
        "group_field": group_field,
        "ranks": ranks,
        "alphas": alphas,
        "censor_modes": censor_modes,
        "err10_fallback": fallback,
        "summary": {
            "n_folds": int(len(folds)),
            "n_finite_folds": int(len(gains)),
            "mean_fold_gain": float(np.mean(gains)) if gains else None,
            "median_fold_gain": float(np.median(gains)) if gains else None,
            "positive_fold_fraction": float(np.mean(np.asarray(gains) > 0)) if gains else None,
            "fold_gains": gains,
        },
        "controls": {},
        "rules": {
            "primary_split": "motif-family LOMO, symmetry-equivalent holdout, or one-edit sequence-component holdout according to --split; all rows and contexts for a held group stay in one outer fold",
            "fingerprint": "train-only junction-by-scaffold residual matrix after censored scaffold M0",
            "sequence_features": "fixed junction sequence position/composition features only",
            "forbidden_features": ["dg9", "dg11", "dg10_5mM", "DMS", "qMaPseq", "outer-test labels"],
            "claim_boundary": "even a positive result is sequence-only evidence on the admitted motif-family universe, not universal all-junction generalization",
        },
        "folds": folds,
    }
    if a.controls:
        # Negative controls use a fixed, predeclared rank/alpha/mode and the
        # same outer folds. They are deliberately not used to choose a model.
        control_spec = {"rank": 3, "alpha": 100.0, "censor_mode": "exclude"}
        control_rows = []
        for i, held_groups in enumerate(outer_blocks):
            train, test = group_split(rows, group_field, held_groups)
            base_model, mu_train = fit_m0(train)
            scaler = fit_feature_scaler(train)
            fingerprint = build_fingerprint(train, mu_train, control_spec["censor_mode"])
            genuine = evaluate(
                train,
                test,
                **control_spec,
                seed=SEED + 9000 + i,
                base_model=base_model,
                mu_train=mu_train,
                fingerprint=fingerprint,
                scaler=scaler,
            )
            shuffled = evaluate(
                train,
                test,
                **control_spec,
                seed=SEED + 9100 + i,
                shuffled_targets=True,
                base_model=base_model,
                mu_train=mu_train,
                fingerprint=fingerprint,
                scaler=scaler,
            )
            control_rows.append({"held_groups": sorted(held_groups), "genuine": genuine, "pairing_shuffle": shuffled})
        g = [x["genuine"]["gain"]["mean_gain"] for x in control_rows if x["genuine"]["gain"]]
        s = [x["pairing_shuffle"]["gain"]["mean_gain"] for x in control_rows if x["pairing_shuffle"]["gain"]]
        out["controls"] = {
            "fixed_spec": control_spec,
            "folds": control_rows,
            "genuine_mean_fold_gain": float(np.mean(g)) if g else None,
            "shuffle_mean_fold_gain": float(np.mean(s)) if s else None,
            "shuffle_positive_fraction": float(np.mean(np.asarray(s) > 0)) if s else None,
        }
        # Also replay the exact outer-selected hyperparameters under a fresh
        # pairing shuffle.  This is descriptive only: the shuffle never
        # participates in selecting the genuine candidate.
        selected_rows = []
        for i, fold in enumerate(folds):
            selected = fold.get("selected")
            if selected is None:
                continue
            train, test = group_split(rows, group_field, fold["held_groups"])
            base_model, mu_train = fit_m0(train)
            scaler = fit_feature_scaler(train)
            fingerprint = build_fingerprint(train, mu_train, selected["censor_mode"])
            null = evaluate(
                train,
                test,
                selected["rank"],
                selected["alpha"],
                selected["censor_mode"],
                SEED + 12000 + i,
                shuffled_targets=True,
                base_model=base_model,
                mu_train=mu_train,
                fingerprint=fingerprint,
                scaler=scaler,
            )
            selected_rows.append({"held_groups": fold["held_groups"], "selected": selected, "pairing_shuffle": null})
        selected_null = [
            x["pairing_shuffle"]["gain"]["mean_gain"]
            for x in selected_rows
            if x["pairing_shuffle"].get("gain")
        ]
        out["controls"]["selected_hyperparameter_pairing_shuffle"] = {
            "folds": selected_rows,
            "mean_fold_gain": float(np.mean(selected_null)) if selected_null else None,
            "positive_fraction": float(np.mean(np.asarray(selected_null) > 0)) if selected_null else None,
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(a.output), "summary": out["summary"], "controls": out["controls"]}, indent=2))


if __name__ == "__main__":
    main()
