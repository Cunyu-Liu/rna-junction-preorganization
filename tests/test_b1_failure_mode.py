#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 independent tests — verify the synthetic failure-mode validation.

Checks:
  1. All 5 fixtures ran and produced results with expected outcomes.
  2. endpoint_reuse: E1 blocked, E2 preserved.
  3. censoring_misclassification: bias quantified.
  4. component_imbalance: estimand difference captured.
  5. baseline_failure: pseudo-gain revealed.
  6. coverage_width_tradeoff: useless uncertainty rejected.
  7. Decision state is PASS, false_pass=0, false_fail=0.
  8. Strategy doc exists for the user's对照实验 reference.
"""

import json
import os
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
B1_DIR = f"{RUN_ROOT}/benchmark/b1"

EXPECTED_OUTCOMES = {
    "endpoint_reuse": "BLOCK_TRANSPORT_PASS / PRESERVE_REAL_EXTERNAL",
    "censoring_misclassification": "BIAS_QUANTIFIED",
    "component_imbalance": "ESTIMAND_DIFFERENCE_CAPTURED",
    "baseline_failure": "PSEUDO_GAIN_REVEALED",
    "coverage_width_tradeoff": "USELESS_UNCERTAINTY_REJECTED",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def test_all_fixtures_results_exist_and_pass():
    for name, expected in EXPECTED_OUTCOMES.items():
        p = os.path.join(B1_DIR, "results", f"{name}.json")
        assert os.path.exists(p), f"missing result {name}"
        r = load_json(p)
        assert r["expected_outcome"] == expected, f"{name}: {r['expected_outcome']}"
        assert r["pass"] is True, f"{name} did not pass"


def test_endpoint_reuse():
    r = load_json(os.path.join(B1_DIR, "results", "endpoint_reuse.json"))
    assert r["observed"]["E1_block_transport_pass"] is True
    assert r["observed"]["E2_preserve_real_external"] is True


def test_censoring_bias_quantified():
    r = load_json(os.path.join(B1_DIR, "results", "censoring_misclassification.json"))
    assert r["observed"]["bias_quantified"] is True
    # correct likelihood error must be smallest
    scores = r["observed"]["estimates"]
    assert scores["correct_likelihood"] < scores["complete_case"]
    assert scores["correct_likelihood"] < scores["exactify"]
    assert scores["correct_likelihood"] < scores["wrong_direction"]


def test_component_difference_captured():
    r = load_json(os.path.join(B1_DIR, "results", "component_imbalance.json"))
    assert r["observed"]["difference_captured"] is True
    assert r["observed"]["imbalanced"]["micro_macro_spread"] > r["observed"]["balanced"]["micro_macro_spread"]
    assert "policy_estimand" in r["observed"]  # distinct target-policy estimand reported separately


def test_baseline_pseudo_gain():
    r = load_json(os.path.join(B1_DIR, "results", "baseline_failure.json"))
    assert r["observed"]["pseudo_gain_revealed"] is True
    assert r["observed"]["gain_vs_matched_baseline"] < 0.05


def test_coverage_rejected():
    r = load_json(os.path.join(B1_DIR, "results", "coverage_width_tradeoff.json"))
    assert r["observed"]["rejected"] is True
    assert r["observed"]["useful_uncertainty"] is False


def test_decision_pass_no_false():
    dec = load_json(os.path.join(B1_DIR, "B1_decision.json"))
    assert dec["state"] == "B1_FAILURE_MODE_VALIDATION_PASS", dec["state"]
    assert dec["false_pass"] == 0
    assert dec["false_fail"] == 0
    assert dec["all_fixtures_pass"] is True
    assert "biological models are correct" in dec["bare_claim"]


def test_strategy_doc_exists():
    # strategy doc lives in benchmark/b1/ (copied from the tracked deliverable)
    for p in (
        os.path.join(B1_DIR, "B1_strategy.md"),
        os.path.join(B1_DIR, "docs", "B1_strategy.md"),
    ):
        if os.path.exists(p):
            return
    raise AssertionError("B1 strategy doc not found")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)