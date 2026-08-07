#!/usr/bin/env python3
"""v1.28 nested mixture probe: ordinal utility + direct local kernel.

The two components have complementary inductive biases:

* the v1.27 ordinal component learns a scaffold-specific sequence utility from
  identified order pairs and calibrates it to the absolute censored tecto
  functional;
* the v1.28 kernel component fits a same-scaffold local location directly in
  the right-censored likelihood and abstains to the scaffold baseline when no
  edit-neighborhood support exists.

This probe tests a low-capacity convex mixture of the two predictions.  Each
component's hyperparameters are selected on the outer-training partition only;
the mixture weight is selected by grouped three-fold CV inside that partition.
The outer test is used once, and the selected full specification is reused for
all null controls.  The result is exploratory and does not unlock submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import direct_sequence_kernel_censored_benchmark as kernel
import sequence_fingerprint_factor_benchmark as seq
import v1_27_nested_ordinal_calibrated as nested27
import v1_27_ordinal_calibrated_probe as probe27
import v1_28_strict_direct_kernel_probe as kernel_probe

SEED = 20260806
WEIGHTS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def split_rows(rows, field, held):
    held = {str(x) for x in held}
    return ([r for r in rows if str(r[field]) not in held],
            [r for r in rows if str(r[field]) in held])


def fit_ordinal(train, test, spec, seed):
    ranker = probe27.rc._fit_ranker(
        train,
        operator_specific=True,
        C=float(spec["C"]),
        max_pairs=probe27.MAX_PAIRS_PER_SCAFFOLD,
        seed=int(seed),
    )
    z_train = probe27.safe_utility(ranker, train)
    z_test = probe27.safe_utility(ranker, test)
    calibrators = probe27.rc._fit_calibrators(
        train, z_train, mode="scaffold", ridge=float(spec["calibration_ridge"])
    )
    mu, support = probe27.rc._predict(calibrators, test, z_test)
    return np.asarray(mu, dtype=float), support


def fit_kernel(train_canonical, test_canonical, spec, max_distance):
    prepared = kernel_probe.prepare_cached(
        train_canonical, test_canonical, int(max_distance)
    )
    mu, support = kernel_probe.predict_cached(prepared, spec)
    return np.asarray(mu, dtype=float), support


def calibration(rows, mu):
    return kernel.calibration(rows, np.asarray(mu, dtype=float))


def score(rows, baseline_mu, mu_ordinal, mu_kernel, weight, seed,
          ordinal_support=None, kernel_support=None):
    mu = float(weight) * np.asarray(mu_ordinal, dtype=float) + (1.0 - float(weight)) * np.asarray(mu_kernel, dtype=float)
    base = kernel.group_nll(rows, baseline_mu)
    cand = kernel.group_nll(rows, mu)
    measured_base = kernel.group_nll(rows, baseline_mu, measured_only=True)
    measured_cand = kernel.group_nll(rows, mu, measured_only=True)
    cens_base = kernel.group_nll(rows, baseline_mu, censored_only=True)
    cens_cand = kernel.group_nll(rows, mu, censored_only=True)
    return {
        "weight_ordinal": float(weight),
        "baseline_macro_junction_nll": float(np.mean(list(base.values()))) if base else None,
        "candidate_macro_junction_nll": float(np.mean(list(cand.values()))) if cand else None,
        "gain": kernel.paired(base, cand, int(seed) + 7000),
        "measured_only_gain": kernel.paired(measured_base, measured_cand, int(seed) + 8000),
        "censored_only_gain": kernel.paired(cens_base, cens_cand, int(seed) + 9000),
        "calibration": calibration(rows, mu),
        "ordinal_support": ordinal_support,
        "kernel_support": kernel_support,
        "n_test": int(len(rows)),
        "n_test_junctions": int(len({str(r["jid"]) for r in rows})),
        "n_test_censored": int(sum(bool(r["cens"]) for r in rows)),
    }


def select_components(outer_train_raw, outer_train_canonical, field, seed, kernel_grid):
    ordinal = nested27.select_spec(outer_train_raw, field, int(seed) + 101)
    kernel_selection = kernel_probe.inner_select(
        outer_train_canonical, field, int(seed) + 101, kernel_grid
    )
    return ordinal, kernel_selection


def select_weight(outer_train_raw, outer_train_canonical, field, seed,
                  ordinal_spec, kernel_spec, max_distance):
    blocks = nested27.inner_blocks(outer_train_raw, field, int(seed) + 101)
    records = {float(w): [] for w in WEIGHTS}
    for j, held in enumerate(blocks):
        tr_raw, va_raw = split_rows(outer_train_raw, field, held)
        tr_c, va_c = split_rows(outer_train_canonical, field, held)
        baseline = np.asarray(probe27.rc._m0_predict(tr_raw, va_raw), dtype=float)
        ordinal_mu, _ = fit_ordinal(tr_raw, va_raw, ordinal_spec, int(seed) + 1000 + j)
        kernel_mu, _ = fit_kernel(tr_c, va_c, kernel_spec, max_distance)
        for weight in WEIGHTS:
            mu = float(weight) * ordinal_mu + (1.0 - float(weight)) * kernel_mu
            nll = kernel.group_nll(va_raw, mu)
            records[float(weight)].append(float(np.mean(list(nll.values()))) if nll else np.inf)
    means = {str(w): float(np.mean(v)) for w, v in records.items()}
    chosen = min(WEIGHTS, key=lambda w: (means[str(w)], float(w)))
    return {"chosen": float(chosen), "fold_scores": records, "mean_nll": means}


def fit_and_score(train_raw, test_raw, train_canonical, test_canonical,
                  ordinal_spec, kernel_spec, weight, seed, max_distance):
    baseline = np.asarray(probe27.rc._m0_predict(train_raw, test_raw), dtype=float)
    ordinal_mu, ordinal_support = fit_ordinal(train_raw, test_raw, ordinal_spec, int(seed) + 201)
    kernel_mu, kernel_support = fit_kernel(train_canonical, test_canonical, kernel_spec, max_distance)
    return score(
        test_raw, baseline, ordinal_mu, kernel_mu, weight, int(seed) + 300,
        ordinal_support=ordinal_support, kernel_support=kernel_support,
    )


def permute_sequence_rows(rows, seed):
    return probe27.sequence_pairing_permutation(rows, int(seed))


def permute_label_rows(rows, seed):
    return probe27.target_permutation_within_scaffold(rows, int(seed))


def run(args):
    records = args.records
    raw, fallback = seq.load_records(records)
    seq.edit_component_labels(raw, max_distance=1)
    canonical = kernel_probe.canonicalize_rows(raw)
    blocks, field = probe27.make_blocks(raw, args.split, int(args.seed))
    kernel_grid = [
        {"bandwidth": float(b), "max_distance": int(d), "prior_strength": float(p)}
        for b in args.bandwidths for d in args.max_distances for p in args.prior_strengths
    ]
    max_distance = max(int(x["max_distance"]) for x in kernel_grid)
    folds = []
    for i, held in enumerate(blocks):
        train_raw, test_raw = split_rows(raw, field, held)
        train_c, test_c = split_rows(canonical, field, held)
        fold_seed = int(args.seed) + 100000 * i
        ordinal_selection, kernel_selection = select_components(
            train_raw, train_c, field, fold_seed, kernel_grid
        )
        ordinal_spec = ordinal_selection["chosen"]
        kernel_spec = {k: kernel_selection["best"][k] for k in ("bandwidth", "max_distance", "prior_strength")}
        weight_selection = select_weight(
            train_raw, train_c, field, fold_seed, ordinal_spec,
            kernel_spec, max_distance,
        )
        weight = weight_selection["chosen"]
        genuine = fit_and_score(
            train_raw, test_raw, train_c, test_c, ordinal_spec,
            kernel_spec, weight, fold_seed, max_distance,
        )
        controls = {}
        for name, transform, offset in (
            ("sequence_pairing_null", permute_sequence_rows, 501),
            ("label_within_scaffold_null", permute_label_rows, 601),
            ("target_pairing_null", lambda rows, seed: kernel.permute_targets(rows, seed), 701),
        ):
            null_raw = transform(train_raw, fold_seed + offset)
            null_c = kernel_probe.canonicalize_rows(null_raw)
            controls[name] = fit_and_score(
                null_raw, test_raw, null_c, test_c, ordinal_spec,
                kernel_spec, weight, fold_seed + offset + 1000, max_distance,
            )
        folds.append({
            "fold": int(i),
            "held_groups": sorted(held),
            "group_field": field,
            "n_train": int(len(train_raw)),
            "n_test": int(len(test_raw)),
            "ordinal_selection": ordinal_selection,
            "kernel_selection": kernel_selection,
            "selected": {
                "ordinal": ordinal_spec,
                "kernel": kernel_spec,
                "weight_ordinal": float(weight),
                "weight_selection": weight_selection,
            },
            "genuine": genuine,
            "controls": controls,
        })
    gains = [f["genuine"]["gain"]["mean_gain"] for f in folds]
    null_summary = {}
    for name in ("sequence_pairing_null", "label_within_scaffold_null", "target_pairing_null"):
        vals = [f["controls"][name]["gain"]["mean_gain"] for f in folds]
        null_summary[name] = {
            "fold_gains": vals,
            "mean_fold_gain": float(np.mean(vals)),
            "positive_fold_fraction": float(np.mean(np.asarray(vals) > 0)),
        }
    out = {
        "schema_version": "tecto-v1.28-nested-ordinal-kernel-mixture-v1",
        "status": "EXPLORATORY_NESTED_NOT_AUTHORITATIVE",
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": int(args.seed),
        "split": args.split,
        "group_field": field,
        "n_rows": int(len(raw)),
        "n_measured": int(sum(not r["cens"] for r in raw)),
        "n_censored": int(sum(bool(r["cens"]) for r in raw)),
        "n_junctions": int(len({str(r["jid"]) for r in raw})),
        "n_edit_components": int(len({str(r["edit_component"]) for r in raw})),
        "err10_fallback": fallback,
        "kernel_grid": kernel_grid,
        "weight_grid": list(WEIGHTS),
        "summary": {
            "n_folds": len(folds),
            "fold_gains": gains,
            "mean_fold_gain": float(np.mean(gains)),
            "median_fold_gain": float(np.median(gains)),
            "positive_fold_fraction": float(np.mean(np.asarray(gains) > 0)),
            "nulls": null_summary,
            "selected_weights": [f["selected"]["weight_ordinal"] for f in folds],
        },
        "candidate": "outer-train-selected convex mixture of scaffold-specific identified-order utility/calibration and direct same-scaffold edit kernel",
        "baseline": "common train-only scaffold censored Gaussian baseline",
        "selection": "component hyperparameters selected outer-train-only; mixture weight selected in grouped three-fold inner CV; outer test used once",
        "likelihood": "right-censored Gaussian with CAP=-7.1 and tau=0.7",
        "forbidden": ["DMS", "qMaPseq", "dg9", "dg11", "dg10_5mM", "interpolated labels", "outer-test labels", "foundation model"],
        "claim_boundary": "a selective, sequence-only tecto method with local support and identified-order calibration; not universal two-way-junction thermodynamic generalization",
        "decision_rule": "candidate remains exploratory unless a parent-linked review accepts full nested selection, nulls, calibration, support and an independent split audit",
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", default="edit_component_5fold")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bandwidths", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--max-distances", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--prior-strengths", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    args = ap.parse_args()
    out = run(args)
    print(json.dumps({"output": str(args.output), "summary": out["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
