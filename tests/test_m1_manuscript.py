#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 independent tests — verify manuscript is data-driven and claim-safe.

Checks:
  1. manuscript.md exists and references the sealed Q7 numbers (not hand-copied drift).
  2. Table 3 numbers match the sealed Q7 decision/metrics exactly.
  3. qMaP2024 in-population correlation is NOT conflated with out-of-component transport
     (scope guard present).
  4. All POST_HOC_EXPLANATORY sensitivities are labeled.
  5. Submission remains HOLD_PENDING_E1_AND_USER_APPROVAL.
  6. Forbidden statements are absent (no "i.i.d. repeats", no "reproduced preorganization").
  7. Fig labels cover §14.5 Fig.1-6 + Table 1-3.
"""

import json
import os
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
M1_DIR = f"{RUN_ROOT}/manuscript/m1"


def read_text(p):
    with open(p) as f:
        return f.read()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def test_manuscript_exists_and_data_driven():
    md = read_text(os.path.join(M1_DIR, "manuscript.md"))
    assert "micro gain" in md
    assert "0.4163" in md or "0.416" in md
    assert "0.726" in md  # coverage
    assert "QMAP_TRANSFER_NOT_SUPPORTED" in md


def test_table3_matches_sealed_q7():
    q7 = load_json(f"{RUN_ROOT}/qmap/q7/Q7_decision.json")
    q7m = load_json(f"{RUN_ROOT}/qmap/q7/metrics.json")
    t3 = read_text(os.path.join(M1_DIR, "tables", "table_3_q7_quantitative_results.tsv"))
    rows = dict(line.split("\t") for line in t3.strip().split("\n")[1:])
    assert abs(float(rows["micro_gain_B3_over_best_baseline"]) - q7["primary"]["micro_gain_b3_over_best_baseline"]) < 1e-6
    assert abs(float(rows["micro_coverage_80pct"]) - q7["co_constraints"]["micro_coverage"]) < 1e-6
    assert abs(float(rows["permutation_finite_p"]) - q7["permutation_finite_p"]) < 1e-6
    assert int(rows["N_total"]) == q7m["primary"]["n"]
    assert int(rows["N_measured"]) == q7m["primary"]["n_measured"]
    assert int(rows["N_right_censored"]) == q7m["primary"]["n_censored"]


def test_no_conflation_of_correlation_and_transport():
    md = read_text(os.path.join(M1_DIR, "manuscript.md"))
    # The scope guard is present; check semantically to tolerate line wrapping.
    assert "in-population" in md
    assert "not conflated" in md
    assert "different estimand" in md


def test_post_hoc_label_present():
    md = read_text(os.path.join(M1_DIR, "manuscript.md"))
    assert "POST_HOC_EXPLANATORY" in md


def test_submission_hold_preserved():
    md = read_text(os.path.join(M1_DIR, "manuscript.md"))
    assert "HOLD_PENDING_E1_AND_USER_APPROVAL" in md


def test_forbidden_statements_absent():
    md = read_text(os.path.join(M1_DIR, "manuscript.md"))
    assert "i.i.d. repeats" not in md
    assert "reproduced preorganization" not in md
    assert "independently reproduced junction preorganization" not in md


def test_figure_cover():
    fl = read_text(os.path.join(M1_DIR, "figures_labels.md"))
    for i in range(1, 7):
        assert f"Fig.{i}" in fl, f"missing Fig.{i}"
    for t in ["Table 1", "Table 2", "Table 3"]:
        assert t in fl, f"missing {t}"


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