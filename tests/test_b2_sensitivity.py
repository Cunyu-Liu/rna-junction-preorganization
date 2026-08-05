#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 independent tests — verify the POST_HOC_EXPLANATORY sensitivity ladder.

Checks:
  1. All 5 dimensions produced results carrying the POST_HOC_EXPLANATORY label.
  2. No result makes a confirmatory claim.
  3. selection: no sign flip between primary_95 and sensitivity_98.
  4. censoring: complete-case / wrong-direction recognized as failure illustration.
  5. weighting: micro/component/policy presented, not cherry-picked.
  6. operator: coverage compared against predeclared band.
  7. threshold: frozen 0.3 unchanged; no new gate.
  8. Decision state is B2_POST_HOC_SENSITIVITY_COMPLETE.
"""

import json
import os
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
B2_DIR = f"{RUN_ROOT}/analysis/b2"
LABEL = "POST_HOC_EXPLANATORY"

DIMS = ["selection", "censoring", "weighting", "operator", "threshold"]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def test_all_dimensions_labeled():
    for d in DIMS:
        r = load_json(os.path.join(B2_DIR, "results", f"{d}.json"))
        assert r["label"] == LABEL, f"{d} label: {r['label']}"
        assert r["no_confirmatory_claim"] is True, f"{d} made a confirmatory claim"


def test_selection_no_sign_flip():
    r = load_json(os.path.join(B2_DIR, "results", "selection.json"))
    assert r["gain_flip_sign"] is False


def test_censoring_failure_illustration():
    r = load_json(os.path.join(B2_DIR, "results", "censoring.json"))
    assert r["bias_illustration"] is True
    assert r["recognized_as_failure_illustration"] is True
    assert r["correct_likelihood"]["kept_survival_likelihood"] is True


def test_weighting_presented_not_cherry_picked():
    r = load_json(os.path.join(B2_DIR, "results", "weighting.json"))
    assert r["presented_not_cherry_picked"] is True
    assert "micro" in r and "component_weighted" in r and "target_policy_illustrative" in r


def test_operator_band():
    r = load_json(os.path.join(B2_DIR, "results", "operator.json"))
    assert r["coverage_band_predeclared"] == [0.75, 0.85]
    assert r["operator_source"] == "Q7_analysis_card.yaml (sealed spec)"


def test_threshold_frozen_unchanged():
    r = load_json(os.path.join(B2_DIR, "results", "threshold.json"))
    assert r["frozen_threshold"] == 0.3
    assert r["frozen_threshold_unchanged"] is True
    assert r["sensitivity_only_not_new_gate"] is True


def test_decision_state():
    dec = load_json(os.path.join(B2_DIR, "B2_decision.json"))
    assert dec["state"] == "B2_POST_HOC_SENSITIVITY_COMPLETE", dec["state"]
    assert dec["label"] == LABEL
    assert dec["no_confirmatory_claim"] is True
    assert dec["frozen_threshold_unchanged"] is True


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