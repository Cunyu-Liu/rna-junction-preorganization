"""P0.3 frozen v3 specs: MetricSpec_v3, NullSpec_v3, PowerSpec_v3,
ClusterSimulationSpec (strict audit 2026-08-11).

The strict audit found four specification defects that must be fixed before any
new genuine result is admissible:

1. **Estimand drift** -- the R5 horizontal-comparison table averaged per-fold
   macro NLL with equal weight, but the frozen primary metric is the pooled-OOF
   junction-macro NLL.  MetricSpec_v3 freezes four distinct estimands and names
   the primary explicitly.
2. **Null is not a refit null** -- R2 used a junction-level pairing sign-flip on
   fixed prediction differences; that is not a sequence-pairing/refit null.
   NullSpec_v3 specifies an outer-train sequence permutation + refit null that
   reuses the SAME aggregation code path as genuine.
3. **Observed power** -- R4 reported observed power; PowerSpec_v3 forbids it and
   requires sensitivity power at pre-registered 2%/5%/10% bounds.
4. **Noise ceiling misnamed** -- R4's "NoiseCeiling" mixed operator spread with
   an in-sample junction-mean oracle.  The audit renames it to
   ObservedOperatorContextSpread + InSampleJunctionOracle.

All specs carry a SHA-256 and a frozen flag; they are generated BEFORE the new
genuine rerun (P0.5) so thresholds cannot be moved post-hoc.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ESTIMANDS = {
    "pooled_junction_macro": {
        "definition": "per-junction mean NLL, then equal-weight macro across junctions; PRIMARY",
        "role": "primary",
    },
    "fold_macro": {
        "definition": "equal-weight mean of per-fold pooled-junction NLL; explicitly NOT primary",
        "role": "secondary_diagnostic",
    },
    "context_macro": {
        "definition": "per-junction mean within a context, then equal-weight across contexts",
        "role": "secondary_diagnostic",
    },
    "operator_bundle_macro": {
        "definition": "per-junction mean within a scaffold/operator bundle, then equal-weight across bundles",
        "role": "secondary_diagnostic",
    },
}

NULL_SPEC = {
    "version": "v3",
    "frozen": True,
    "name": "outer_train_sequence_permutation_refit",
    "description": (
        "Shuffle junction-sequence labels on the OUTER TRAIN only, refit each "
        "model on the permuted train, score on the untouched test, and recompute "
        "the SAME pooled-OOF junction-macro statistic used for genuine.  This "
        "destroys the sequence->energy pairing while keeping scaffold/context "
        "structure and the matched no-sequence head.")
    ,
    "units": {
        "symmetry_5fold": "symmetry groups",
        "edit_5fold": "edit components",
        "context_lomo": "junction x context cells",
        "edit_x_nested_context": "edit components",
        "scaffold_lomo": "scaffold bundles",
    },
    "n_permutations_per_axis": 1000,
    "refit": True,
    "same_aggregation_as_genuine": True,
    "aggregation_fn": "pooled_junction_macro",
    "fail_closed_on_fit_failure": True,
    "statistic": "axis_level_pooled_junction_macro_nll_delta",
}

POWER_SPEC = {
    "version": "v3",
    "frozen": True,
    "name": "sensitivity_power",
    "forbid_observed_power": True,
    "require": ["2%", "5%", "10%"],
    "target_bounds_relative_gain_pct": [2.0, 5.0, 10.0],
    "error_rate_alpha": 0.05,
    "power_target": 0.80,
    "cluster_unit": {
        "symmetry_5fold": "symmetry group",
        "edit_5fold": "edit component",
        "edit_x_nested_context": "edit component",
        "context_lomo": "junction x context",
        "scaffold_lomo": "scaffold bundle",
    },
    "note": (
        "Report the relative gain that can be excluded at each pre-registered "
        "bound; never report observed power from the genuine result.")
}

CLUSTER_SIM_SPEC = {
    "version": "v3",
    "frozen": True,
    "name": "cluster_bootstrap_calibration_simulation",
    "purpose": (
        "Calibrate the two-way / wild-cluster / CR1 bootstrap CI coverage under "
        "the REAL incidence matrix and censoring fraction, so a negative result "
        "cannot be misread as a real null.")
    ,
    "simulate": ["null_effect", "small_effects"],
    "effect_sizes_relative_gain_pct": [0.0, 2.0, 5.0, 10.0],
    "report": {
        "ci_95_coverage": True,
        "type1_error": True,
        "leave_one_largest_component": True,
    },
    "incidence_source": "actual (edit_component, helix_seq, scaf) incidence from admitted rows",
    "censoring_fraction_source": "actual right-censor fraction from admitted rows",
}

GATE_SPEC_V3 = {
    "version": "v3",
    "frozen": True,
    "primary_metric": "pooled_junction_macro",
    "gates": {
        "coverage": {"full_coverage_requirement": 1.0,
                     "abstain_without_fallback": "INELIGIBLE"},
        "superiority": {"relative_gain_pct": 10.0},
        "uncertainty": {"ci_lower_gt": 0.0, "method": "edit_cluster_or_wild_cluster"},
        "fold_consistency": {"positive_folds_5fold": "5/5",
                             "catastrophic_fold_relative_gain_pct": -10.0},
        "null_separation": {"n_permutations": 1000, "null_percentile_975": 97.5,
                            "statistic": "axis_level_pooled_junction_macro_nll_delta",
                            "null_upper_lt_genuine": True},
        "interval": {"primary_contrast_width_kcal": 1.0},
        "synthetic_calibration": {"coverage_range": [0.9, 1.0]},
        "optimizer": {"method": "projected_gradient_or_strict_grad_norm",
                      "default_grad_tol": 1e-3},
    },
}


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_v3_specs(out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "MetricSpec_v3.json": {
            "version": "v3", "frozen": True,
            "primary_estimand": "pooled_junction_macro",
            "estimands": ESTIMANDS,
            "row_likelihood": {
                "measured": "-(0.5*log(2*pi) + log(sigma) + 0.5*((y-mu)/sigma)^2)",
                "censored": "log_Phi((mu-CAP)/sigma)",
            },
            "note": ("Four distinct estimands frozen; only pooled_junction_macro "
                     "is primary.  fold_macro must not be used for the primary "
                     "leaderboard or the R5 main table."),
        },
        "NullSpec_v3.json": NULL_SPEC,
        "PowerSpec_v3.json": POWER_SPEC,
        "ClusterSimulationSpec.json": CLUSTER_SIM_SPEC,
        "GateSpec_v3.json": GATE_SPEC_V3,
    }
    hashes = {}
    for name, obj in specs.items():
        (out_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[name] = _hash(obj)
    (out_dir / "SpecHashes_v3.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return hashes


if __name__ == "__main__":
    import sys
    print(json.dumps(write_v3_specs(Path(sys.argv[1])), indent=2))