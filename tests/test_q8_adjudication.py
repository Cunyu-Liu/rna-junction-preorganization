#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent tests for Q8 corrected qMaP re-adjudication (v1.5)."""

import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
Q8_DIR = f"{RUN_ROOT}/qmap/q8"
EXPECTED_SUB_STATES = {
    "QMAP_GAIN_THRESHOLD": "MET",
    "QMAP_PERMUTATION_SIGNAL": "PRESENT",
    "QMAP_GAIN_BOOTSTRAP": "INCONCLUSIVE",
    "QMAP_REGISTERED_POINT_COVERAGE_RULE": "FAILED",
    "QMAP_CALIBRATION_DEFICIT_EVIDENCE": "INCONCLUSIVE",
    "QMAP_FULL_PREDECLARED_TRANSPORT_CRITERION": "NOT_MET",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_decision():
    return load_json(f"{Q8_DIR}/Q8_decision.json")


def test_decision_present_and_requires_no_model_rerun():
    d = load_decision()
    assert d["gate"] == "Q8"
    assert d["requires_no_model_rerun"] is True
    assert d["parent_run_id"] == "v1_4_boundary_audit_20260804T150707Z"


def test_six_sub_states_exact():
    d = load_decision()
    assert d["sub_states"] == EXPECTED_SUB_STATES


def test_deterministic_recomputation_reproduces_frozen_q7():
    """Q8 must deterministically reproduce frozen Q7 before any derived claim."""
    d = load_decision()
    assert d["deterministic_recomputation_reproduced_frozen_q7"] is True
    rc = d["recomputed"]
    assert abs(rc["gain"] - d["frozen"]["micro_gain"]) < 5e-4
    assert abs(rc["coverage"] - d["frozen"]["observed_coverage"]) < 5e-4
    assert abs(rc["width"] - d["frozen"]["observed_width"]) < 5e-4


def test_gain_threshold_is_met_not_below():
    d = load_decision()
    assert d["sub_states"]["QMAP_GAIN_THRESHOLD"] == "MET"
    assert d["frozen"]["micro_gain"] >= d["frozen"]["meaningful_threshold"]
    # forbidden language: must NOT report gain < 0.3
    assert d["frozen"]["micro_gain"] > 0.4


def test_coverage_rule_failed_but_deficit_inconclusive():
    d = load_decision()
    cu = d["calibration_uncertainty"]
    # registered point rule failed
    assert cu["registered_point_rule"]["passed"] is False
    # descriptive Wilson CI contains nominal 0.8 -> deficit evidence inconclusive
    wilson = cu["simple_wilson_95"]
    assert wilson["contains_nominal_0_8"] is True
    assert cu["registered_point_rule"]["observed"] == d["frozen"]["observed_coverage"]
    # simple Wilson must not be treated as primary (Bernoulli assumption invalid)
    assert "independent-Bernoulli" in wilson["note"]


def test_cluster_aware_bootstrap_present():
    d = load_decision()
    cb = d["calibration_uncertainty"]["cluster_aware_bootstrap"]
    assert cb["n_resamples"] >= 1000
    assert cb["gain_ci_95"][0] <= cb["gain_ci_95"][1]
    assert cb["coverage_ci_95"][0] <= cb["coverage_ci_95"][1]


def test_per_component_coverage_reported():
    d = load_decision()
    comps = d["calibration_uncertainty"]["per_component"]
    sizes = {c["component"]: c["n_test"] for c in comps}
    assert sizes == {0: 80, 1: 11, 2: 2, 3: 2}, sizes
    for c in comps:
        assert 0.0 <= c["coverage"] <= 1.0
        assert c["width"] > 0


def test_coverage_width_curve_multiple_levels():
    d = load_decision()
    curve = d["calibration_uncertainty"]["coverage_width_curve"]
    levels = {c["nominal_level"] for c in curve}
    assert levels == {0.60, 0.70, 0.80, 0.90}, levels
    # width must increase with nominal level (wider interval -> higher coverage)
    widths = [c["width"] for c in sorted(curve, key=lambda x: x["nominal_level"])]
    assert widths == sorted(widths), "width must be monotonic increasing in nominal level"


def test_11th_member_three_sensitivity_modes():
    d = load_decision()
    sens = d["membership_11th"]["sensitivity"]
    modes = {s["member_assignment"] for s in sens}
    assert modes == {"censored", "fitted", "excluded"}, modes  # all three reported
    assert d["membership_11th"]["source_status"] == "FIT_IDENTIFIED"
    assert d["membership_11th"]["member"] == "CCUGCC_ACUGG"


def test_membership_robustness_conclusion():
    """All three assignments must agree on the conjunctive full criterion (NOT_MET)."""
    d = load_decision()
    sens = {s["member_assignment"]: s for s in d["membership_11th"]["sensitivity"]}
    for mode in ("censored", "fitted", "excluded"):
        s = sens[mode]
        assert s["gain_threshold_met"] is True, f"{mode}: gain threshold must be met"
        assert s["coverage_ok"] is False, f"{mode}: coverage must fail (0.726 not in [0.75,0.85])"
    assert d["membership_11th"]["conclusion"] == "QMAP_SOURCE_MEMBERSHIP_ROBUST_NOT_MET"
    # forbidden: must not CLAIM exact source-authored membership (asserting the claim,
    # not merely mentioning the phrase in a negation)
    assert d["membership_11th"]["source_status"] == "FIT_IDENTIFIED"
    for key in ("comp_statement", "claim", "allowed_language"):
        v = str(d.get(key, ""))
        assert "exact source-authored membership is confirmed" not in v


def test_tsv_artifacts_present():
    assert os.path.exists(f"{Q8_DIR}/calibration.tsv")
    assert os.path.exists(f"{Q8_DIR}/membership_sensitivity.tsv")
    assert os.path.exists(f"{Q8_DIR}/q8_report.md")
    with open(f"{Q8_DIR}/membership_sensitivity.tsv") as f:
        lines = [l for l in f.read().strip().splitlines() if l]
    assert len(lines) == 4, "header + 3 membership modes"


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed} Q8 tests passed")


if __name__ == "__main__":
    run_all()