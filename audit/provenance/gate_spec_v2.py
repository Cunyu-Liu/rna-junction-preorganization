"""R0.1 frozen GateSpec_v2 (contract §11.1 FrozenGateSpec).

Machine-readable, hash-bound success gates that must be frozen BEFORE any new
predictions are generated and must NOT be lowered after seeing results.  Every
claim class registers its estimand, eligible rows, support policy, comparator
set, minimum effect, CI/null/fold/calibration gates, and allowed wording.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GATE_SPEC = {
    "version": "v2",
    "frozen": True,
    "primary_metric": "pooled_OOF_junction_macro_right_censored_nll",
    "gates": {
        "coverage": {"full_coverage_requirement": 1.0,
                     "abstain_without_fallback": "INELIGIBLE"},
        "superiority": {"relative_gain_pct": 10.0},
        "uncertainty": {"ci_lower_gt": 0.0, "bootstrap_method": "group_aware"},
        "fold_consistency": {"positive_folds_5fold": "5/5",
                             "catastrophic_fold_relative_gain_pct": -10.0},
        "null_separation": {"n_permutations": 1000,
                            "null_percentile_975": 97.5,
                            "statistic": "axis_level_matching_genuine"},
        "interval": {"primary_contrast_width_kcal": 1.0},
        "synthetic_calibration": {"coverage_range": [0.9, 1.0]},
    },
    "claims": {
        "known_scaffold_conditional": {
            "estimand": "pooled_junction_macro", "eligible_rows": "full_coverage",
            "support_policy": "pre-registered fallback", "allowed_wording": "conditional predictor",
        },
        "joint_sequence_x_nested_context": {
            "estimand": "pooled_junction_macro + nested_context_macro",
            "eligible_rows": "full_coverage", "support_policy": "matched no-sequence",
            "allowed_wording": "sequence increment under seen-scaffold nested-context blocking",
        },
        "unseen_scaffold_bundle": {
            "estimand": "scaffold_bundle_macro", "eligible_rows": "full_coverage",
            "support_policy": "explicit fallback", "allowed_wording": "bundle transfer only",
        },
        "selective": {
            "estimand": "supported_junction_macro", "eligible_rows": "selective",
            "support_policy": "frozen coverage floor + coverage-matched comparator",
            "allowed_wording": "selective prediction, not primary",
        },
        "prospective_factorial_mechanism": {
            "estimand": "pre-registered primary", "eligible_rows": "prospective",
            "support_policy": "frozen model + pre-registered", "allowed_wording": "mechanism (requires new data)",
        },
        "sota_best_under_protocol": {
            "estimand": "pooled_junction_macro", "eligible_rows": "full_coverage",
            "support_policy": "complete comparator universe", "allowed_wording": "best under frozen public protocol only",
        },
    },
}


def gate_spec_hash(spec=None):
    return hashlib.sha256(json.dumps(
        spec or GATE_SPEC, sort_keys=True).encode()).hexdigest()


def write_gate_spec(out_dir: Path, spec=None):
    spec = spec or GATE_SPEC
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "GateSpec_v2.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return gate_spec_hash(spec)
