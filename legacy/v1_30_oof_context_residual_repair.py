#!/usr/bin/env python3
"""v1.30 OOF context-residual repair for the frozen v1.28 mixture.

This probe tests one bounded hypothesis rather than searching a larger model:
the v1.28 ordinal+local-kernel predictor explains junction-sequence variation,
but a repeatable helix/boundary context can shift the whole conditional
functional.  A context random effect is therefore estimated as an additive
right-censored location correction.

The correction is deliberately fitted from *out-of-fold* v1.28 predictions on
the outer-training partition.  It is never fitted to in-sample predictions,
and no outer-test label enters the correction.  The primary context precision
is fixed at 1.0; the small sensitivity grid is reported but not used to choose
the primary result.  This keeps the repair falsifiable and prevents another
hyperparameter search from being mistaken for a scientific improvement.

The script is exploratory until a new parent-linked contract adjudicates the
results.  It does not modify v1.28 or v1.29 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

import sequence_fingerprint_factor_benchmark as seq
import v1_27_ordinal_calibrated_probe as probe27
import v1_28_nested_mixture_probe as mixture
import v1_28_strict_direct_kernel_probe as kernel_probe


CAP = -7.1
TAU = 0.7
PRIMARY_CONTEXT_PRECISION = 1.0
SENSITIVITY_CONTEXT_PRECISIONS = (0.5, 2.0, 4.0, 8.0)
CONTEXT_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# A correction must earn at least one percentage point of mean inner OOF
# relative NLL improvement before it is deployed.  This is a fixed selective
# rule, not a threshold tuned on an outer-test result; alpha=0 remains the
# fail-closed choice below this floor.
MIN_CONTEXT_SELECTION_GAIN = 0.01
OOF_FOLDS = 3
SEED = 20260820


def split_rows(rows, field: str, held):
    held = {str(x) for x in held}
    return (
        [r for r in rows if str(r[field]) not in held],
        [r for r in rows if str(r[field]) in held],
    )


def make_oof_blocks(rows, field: str, seed: int, n_folds: int = OOF_FOLDS):
    groups = sorted({str(r[field]) for r in rows})
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    return [set(x.tolist()) for x in np.array_split(
        np.asarray(groups, dtype=object), int(n_folds)
    )]


def right_censored_nll(rows, mu):
    """Return macro-junction NLL and measured/censored strata."""
    mu = np.asarray(mu, dtype=float)
    if len(mu) != len(rows) or not np.isfinite(mu).all():
        raise FloatingPointError("nonfinite_prediction")
    by_all = {}
    by_measured = {}
    by_censored = {}
    for row, value in zip(rows, mu):
        if bool(row["cens"]):
            loss = float(-log_ndtr((float(value) - CAP) / TAU))
            by_censored.setdefault(str(row["jid"]), []).append(loss)
        else:
            z = (float(row["y"]) - float(value)) / TAU
            loss = float(0.5 * math.log(2.0 * math.pi) + math.log(TAU)
                         + 0.5 * z * z)
            by_measured.setdefault(str(row["jid"]), []).append(loss)
        by_all.setdefault(str(row["jid"]), []).append(loss)

    def mean(groups):
        return float(np.mean([np.mean(x) for x in groups.values()])) if groups else None

    return {
        "macro_junction_nll": mean(by_all),
        "measured_macro_junction_nll": mean(by_measured),
        "censored_macro_junction_nll": mean(by_censored),
        "n_junctions": int(len(by_all)),
        "n_measured_junctions": int(len(by_measured)),
        "n_censored_junctions": int(len(by_censored)),
    }


def relative_gain(rows, base_mu, candidate_mu):
    base = right_censored_nll(rows, base_mu)
    cand = right_censored_nll(rows, candidate_mu)

    def gain(key):
        b = base.get(key)
        c = cand.get(key)
        return None if b is None or c is None else float((b - c) / max(abs(b), 1e-12))

    return {
        "all": gain("macro_junction_nll"),
        "measured": gain("measured_macro_junction_nll"),
        "right_censored": gain("censored_macro_junction_nll"),
    }


def calibration(rows, mu):
    measured = np.asarray([not bool(r["cens"]) for r in rows], dtype=bool)
    y = np.asarray([float(r["y"]) for r in rows], dtype=float)
    mu = np.asarray(mu, dtype=float)
    z = (y[measured] - mu[measured]) / TAU if measured.any() else np.asarray([])
    p_cens = np.exp(log_ndtr((mu - CAP) / TAU))
    return {
        "observed_censor_fraction": float(np.mean(~measured)) if len(rows) else None,
        "predicted_censor_probability_mean": float(np.mean(p_cens)) if len(rows) else None,
        "censor_tail_difference": float(np.mean(p_cens) - np.mean(~measured)) if len(rows) else None,
        "measured_95_coverage": float(np.mean(np.abs(z) <= 1.959963984540054)) if len(z) else None,
        "measured_z_median": float(np.median(z)) if len(z) else None,
    }


def fit_frozen_mixture(train_raw, test_raw, train_canonical, test_canonical,
                       selected, seed):
    """Fit exactly the v1.28 components with a parent-selected specification."""
    ordinal_spec = selected["ordinal"]
    kernel_spec = selected["kernel"]
    weight = float(selected["weight_ordinal"])
    ordinal_mu, ordinal_support = mixture.fit_ordinal(
        train_raw, test_raw, ordinal_spec, int(seed) + 101
    )
    kernel_mu, kernel_support = mixture.fit_kernel(
        train_canonical, test_canonical, kernel_spec, 3
    )
    mu = weight * np.asarray(ordinal_mu, dtype=float) + (1.0 - weight) * np.asarray(kernel_mu, dtype=float)
    return mu, {"ordinal": ordinal_support, "kernel": kernel_support}


def fit_context_delta(train, base_mu, precision):
    """Fit one train-only right-censored random effect per helix context.

    ``base_mu`` must be OOF predictions for ``train``.  A junction-balanced
    weight is used within the outer-training partition.  The prior is
    0.5*precision*delta^2, with delta measured in kcal/mol on the target scale.
    """
    by_context = {}
    for i, row in enumerate(train):
        by_context.setdefault(str(row["helix_seq"]), []).append(i)
    counts = {}
    for row in train:
        counts[str(row["jid"])] = counts.get(str(row["jid"]), 0) + 1
    weights = np.asarray([
        1.0 / max(counts[str(row["jid"])], 1) for row in train
    ], dtype=float)
    weights *= len(train) / max(float(weights.sum()), 1e-12)

    delta = {}
    diagnostics = {}
    for context, indices in sorted(by_context.items()):
        idx = np.asarray(indices, dtype=int)
        y = np.asarray([float(train[i]["y"]) for i in idx], dtype=float)
        cens = np.asarray([bool(train[i]["cens"]) for i in idx], dtype=bool)
        offset = np.asarray([float(base_mu[i]) for i in idx], dtype=float)
        w = weights[idx]
        d = 0.0
        for _ in range(100):
            mu = offset + d
            grad = -float(precision) * d
            hess = -float(precision)
            unc = ~cens
            if unc.any():
                grad += float(np.sum(w[unc] * (y[unc] - mu[unc]) / (TAU * TAU)))
                hess -= float(np.sum(w[unc]) / (TAU * TAU))
            if cens.any():
                a = (mu[cens] - CAP) / TAU
                log_phi_over_cdf = (
                    -0.5 * a * a - 0.5 * math.log(2.0 * math.pi)
                    - log_ndtr(a)
                )
                ratio = np.exp(np.clip(log_phi_over_cdf, -50.0, 50.0))
                grad += float(np.sum(w[cens] * ratio / TAU))
                hess -= float(np.sum(w[cens] * ratio * (a + ratio) / (TAU * TAU)))
            step = grad / max(abs(hess), 1e-12)
            new_d = float(np.clip(d + step, -4.0, 4.0))
            if abs(new_d - d) < 1e-8:
                d = new_d
                break
            d = new_d
        delta[context] = float(d)
        diagnostics[context] = {
            "n_rows": int(len(idx)),
            "n_junctions": int(len({str(train[i]["jid"]) for i in idx})),
            "n_censored": int(np.sum(cens)),
            "delta": float(d),
        }
    return delta, diagnostics


def apply_context_delta(mu, rows, delta):
    mu = np.asarray(mu, dtype=float).copy()
    support = np.asarray([str(r["helix_seq"]) in delta for r in rows], dtype=bool)
    for i, row in enumerate(rows):
        mu[i] += float(delta.get(str(row["helix_seq"]), 0.0))
    return mu, support


def make_oof_predictions(train_raw, train_canonical, field, selected, seed):
    """Cross-fit frozen v1.28 predictions for every outer-training row.

    The returned matrix has one row per OOF fold and one column per outer
    training record.  In row ``j`` *all* predictions are produced by a model
    trained without the groups held by fold ``j``.  This stronger form is
    needed for the correction-strength gate: it prevents a validation group's
    labels from influencing the offset used to score that group.
    """
    blocks = make_oof_blocks(train_raw, field, int(seed), OOF_FOLDS)
    matrix = np.full((len(blocks), len(train_raw)), np.nan, dtype=float)
    index_by_identity = {id(row): i for i, row in enumerate(train_raw)}
    records = []
    for fold, held in enumerate(blocks):
        fit_raw, val_raw = split_rows(train_raw, field, held)
        fit_canonical, val_canonical = split_rows(train_canonical, field, held)
        # Predict every outer-training row using the model that excludes the
        # held groups.  The held-group slice is used as the strict OOF value;
        # the other columns support the no-leakage alpha selection below.
        mu_all, supports = fit_frozen_mixture(
            fit_raw, train_raw, fit_canonical, train_canonical,
            selected, int(seed) + 10000 * (fold + 1)
        )
        for row, value in zip(train_raw, mu_all):
            matrix[fold, index_by_identity[id(row)]] = float(value)
        records.append({
            "fold": int(fold),
            "held_groups": sorted(held),
            "n_fit": int(len(fit_raw)),
            "n_validation": int(len(val_raw)),
            "kernel_support": supports["kernel"],
        })
    if not np.isfinite(matrix).all():
        raise FloatingPointError("incomplete_oof_prediction_matrix")
    diagonal = np.zeros(len(train_raw), dtype=float)
    for fold, held in enumerate(blocks):
        held_mask = np.asarray([
            str(row[field]) in held for row in train_raw
        ], dtype=bool)
        diagonal[held_mask] = matrix[fold, held_mask]
    return diagonal, matrix, blocks, records


def select_context_alpha(train, oof_matrix, blocks, field, precision, seed):
    """Select a correction multiplier using no-leakage group OOF scores.

    For validation block ``j``, both its parent prediction and the context
    offset are generated from data excluding block ``j``.  ``alpha=0`` is
    explicitly available and is preferred on ties, so an unsupported repair
    closes itself rather than being forced into the final predictor.
    """
    scores = {float(alpha): [] for alpha in CONTEXT_ALPHA_GRID}
    for fold, held in enumerate(blocks):
        held_mask = np.asarray([
            str(row[field]) in held for row in train
        ], dtype=bool)
        fit_rows = [row for row, keep in zip(train, ~held_mask) if keep]
        fit_base = np.asarray(oof_matrix[fold, ~held_mask], dtype=float)
        val_rows = [row for row, keep in zip(train, held_mask) if keep]
        val_base = np.asarray(oof_matrix[fold, held_mask], dtype=float)
        delta, _ = fit_context_delta(fit_rows, fit_base, precision)
        val_context, _ = apply_context_delta(val_base, val_rows, delta)
        base_nll = right_censored_nll(val_rows, val_base)["macro_junction_nll"]
        for alpha in CONTEXT_ALPHA_GRID:
            candidate = val_base + float(alpha) * (val_context - val_base)
            cand_nll = right_censored_nll(val_rows, candidate)["macro_junction_nll"]
            scores[float(alpha)].append(float((base_nll - cand_nll) / max(abs(base_nll), 1e-12)))
    means = {str(alpha): float(np.mean(values)) for alpha, values in scores.items()}
    best = min(
        CONTEXT_ALPHA_GRID,
        key=lambda alpha: (-means[str(alpha)], float(alpha)),
    )
    chosen = (
        float(best)
        if means[str(best)] >= MIN_CONTEXT_SELECTION_GAIN
        else 0.0
    )
    return {
        "chosen": float(chosen),
        "fold_scores": {str(alpha): values for alpha, values in scores.items()},
        "mean_gain_vs_parent": means,
        "selection": "3-fold group OOF; minimum inner gain floor; alpha=0 fail-closed; no outer-test labels",
        "minimum_inner_gain_floor": MIN_CONTEXT_SELECTION_GAIN,
        "best_alpha_before_floor": float(best),
        "seed": int(seed),
    }


def summarize_fold(rows, scaffold_mu, parent_mu, repaired_mu, support, delta_diag,
                   precision, oof_records):
    base_score = right_censored_nll(rows, scaffold_mu)
    parent_score = right_censored_nll(rows, parent_mu)
    repaired_score = right_censored_nll(rows, repaired_mu)
    return {
        "precision": float(precision),
        "baseline": base_score,
        "parent_v1_28": parent_score,
        "repaired": repaired_score,
        "parent_gain_vs_baseline": relative_gain(rows, scaffold_mu, parent_mu),
        "repair_gain_vs_parent": relative_gain(rows, parent_mu, repaired_mu),
        "repaired_gain_vs_baseline": relative_gain(rows, scaffold_mu, repaired_mu),
        "parent_calibration": calibration(rows, parent_mu),
        "repaired_calibration": calibration(rows, repaired_mu),
        "context_support_fraction": float(np.mean(support)) if len(support) else None,
        "context_fallback_fraction": float(1.0 - np.mean(support)) if len(support) else None,
        "context_delta_summary": {
            "n_contexts": int(len(delta_diag)),
            "max_abs_delta": float(max((abs(v["delta"]) for v in delta_diag.values()), default=0.0)),
            "median_abs_delta": float(np.median([abs(v["delta"]) for v in delta_diag.values()])) if delta_diag else 0.0,
        },
        "oof": oof_records,
        "n_test_rows": int(len(rows)),
        "n_test_junctions": int(len({str(r["jid"]) for r in rows})),
        "n_test_censored": int(sum(bool(r["cens"]) for r in rows)),
    }


def run(args):
    raw, fallback = seq.load_records(args.records)
    seq.edit_component_labels(raw, max_distance=1)
    canonical = kernel_probe.canonicalize_rows(raw)
    parent = json.loads(args.parent_result.read_text())
    if parent.get("records_sha256") != hashlib.sha256(args.records.read_bytes()).hexdigest():
        raise ValueError("parent/source hash mismatch")
    if parent.get("split") != args.split:
        raise ValueError(f"parent split {parent.get('split')} != requested {args.split}")
    field = str(parent["group_field"])
    folds = []
    for parent_fold in parent["folds"]:
        held = parent_fold["held_groups"]
        train_raw, test_raw = split_rows(raw, field, held)
        train_canonical, test_canonical = split_rows(canonical, field, held)
        selected = parent_fold["selected"]
        fold_seed = int(args.seed) + 100000 * int(parent_fold["fold"])
        oof_mu, oof_matrix, oof_blocks, oof_records = make_oof_predictions(
            train_raw, train_canonical, field, selected, fold_seed + 17
        )
        alpha_selection = select_context_alpha(
            train_raw, oof_matrix, oof_blocks, field,
            PRIMARY_CONTEXT_PRECISION, fold_seed + 31
        )
        alpha = float(alpha_selection["chosen"])
        delta, delta_diag = fit_context_delta(
            train_raw, oof_mu, PRIMARY_CONTEXT_PRECISION
        )
        parent_mu, parent_support = fit_frozen_mixture(
            train_raw, test_raw, train_canonical, test_canonical,
            selected, fold_seed + 101
        )
        baseline_mu = np.asarray(probe27.rc._m0_predict(train_raw, test_raw), dtype=float)
        context_mu, context_support = apply_context_delta(parent_mu, test_raw, delta)
        repaired_mu = parent_mu + alpha * (context_mu - parent_mu)
        sensitivities = {}
        for precision in SENSITIVITY_CONTEXT_PRECISIONS:
            d_sens, _ = fit_context_delta(train_raw, oof_mu, precision)
            context_sens, sup_sens = apply_context_delta(parent_mu, test_raw, d_sens)
            mu_sens = parent_mu + alpha * (context_sens - parent_mu)
            sensitivities[str(precision)] = {
                "repair_gain_vs_parent": relative_gain(test_raw, parent_mu, mu_sens),
                "repaired_gain_vs_baseline": relative_gain(test_raw, baseline_mu, mu_sens),
                "context_support_fraction": float(np.mean(sup_sens)) if len(sup_sens) else None,
            }
        fold = summarize_fold(
            test_raw, baseline_mu, parent_mu, repaired_mu, context_support,
            delta_diag, PRIMARY_CONTEXT_PRECISION, oof_records
        )
        fold.update({
            "fold": int(parent_fold["fold"]),
            "held_groups": sorted(held),
            "group_field": field,
            "n_train_rows": int(len(train_raw)),
            "n_train_junctions": int(len({str(r["jid"]) for r in train_raw})),
            "parent_selected": selected,
            "context_alpha_selection": alpha_selection,
            "selected_context_alpha": alpha,
            "parent_kernel_support": parent_support,
            "precision_sensitivity_not_selected": sensitivities,
        })
        folds.append(fold)

    parent_gain = [f["parent_gain_vs_baseline"]["all"] for f in folds]
    repair_gain = [f["repair_gain_vs_parent"]["all"] for f in folds]
    repaired_vs_base = [f["repaired_gain_vs_baseline"]["all"] for f in folds]
    out = {
        "schema_version": "tecto-v1.30-oof-context-residual-repair-v1",
        "status": "EXPLORATORY_NOT_AUTHORITATIVE",
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "parent_result_sha256": hashlib.sha256(args.parent_result.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": int(args.seed),
        "split": args.split,
        "group_field": field,
        "context_field": "helix_seq",
        "n_rows": int(len(raw)),
        "n_measured": int(sum(not r["cens"] for r in raw)),
        "n_censored": int(sum(bool(r["cens"]) for r in raw)),
        "n_junctions": int(len({str(r["jid"]) for r in raw})),
        "n_contexts": int(len({str(r["helix_seq"]) for r in raw})),
        "n_edit_components": int(len({str(r["edit_component"]) for r in raw})),
        "err10_fallback": fallback,
        "parent_method": "v1.28 frozen ordinal-calibrated plus direct same-scaffold censored kernel",
        "repair_method": "OOF parent prediction offset plus helix-context random effect under right-censored Gaussian likelihood",
        "primary_context_precision": PRIMARY_CONTEXT_PRECISION,
        "sensitivity_context_precisions": list(SENSITIVITY_CONTEXT_PRECISIONS),
        "context_alpha_grid": list(CONTEXT_ALPHA_GRID),
        "minimum_context_selection_gain": MIN_CONTEXT_SELECTION_GAIN,
        "oof_folds": OOF_FOLDS,
        "forbidden": ["DMS", "qMaPseq", "dg9", "dg11", "dg10_5mM", "interpolated labels", "outer-test labels", "foundation model"],
        "summary": {
            "n_folds": int(len(folds)),
            "parent_mean_gain_vs_scaffold": float(np.mean(parent_gain)),
            "repair_mean_gain_vs_parent": float(np.mean(repair_gain)),
            "repair_positive_fold_fraction": float(np.mean(np.asarray(repair_gain) > 0)),
            "repaired_mean_gain_vs_scaffold": float(np.mean(repaired_vs_base)),
            "parent_fold_gains_vs_scaffold": parent_gain,
            "repair_fold_gains_vs_parent": repair_gain,
            "repaired_fold_gains_vs_scaffold": repaired_vs_base,
            "context_support_fraction_by_fold": [f["context_support_fraction"] for f in folds],
        },
        "claim_boundary": "conditional context-adjusted repair only; no universal thermodynamic or cross-operator claim",
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summary": out["summary"]}, indent=2, sort_keys=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--parent-result", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["edit_component_5fold", "symmetry_5fold"])
    ap.add_argument("--seed", type=int, default=SEED)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
