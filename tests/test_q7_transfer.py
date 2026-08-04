#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent tests for Q7 corrected locked qMaP transfer rerun."""

import hashlib
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
Q7_DIR = f"{RUN_ROOT}/qmap/q7"
SPEC_DIR = f"{RUN_ROOT}/specs/qmap"
ALLOWED_STATES = {
    "QMAP_TRANSFER_SUPPORTED",
    "QMAP_TRANSFER_NOT_SUPPORTED",
    "QMAP_INCONCLUSIVE",
    "QMAP_NOT_ADMITTED",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_decision():
    return load_json(f"{Q7_DIR}/Q7_decision.json")


def load_metrics():
    return load_json(f"{Q7_DIR}/metrics.json")


def load_fold_metrics():
    rows = []
    with open(f"{Q7_DIR}/fold_metrics.tsv") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def test_decision_present_and_state():
    d = load_decision()
    assert d["gate"] == "Q7"
    assert d["state"] in ALLOWED_STATES, d["state"]
    assert d["run_id"] == "v1_4_boundary_audit_20260804T150707Z"


def test_metrics_consistency_with_decision():
    d = load_decision()
    m = load_metrics()
    assert abs(d["primary"]["micro_gain_b3_over_best_baseline"]
               - m["primary"]["micro_gain_b3_over_best_baseline"]) < 1e-9
    assert d["primary"]["meaningful_threshold"] == m["meaningful_gain_threshold"] == 0.3
    assert abs(d["co_constraints"]["micro_coverage"]
               - m["primary"]["micro_coverage_b3"]) < 1e-9
    assert d["co_constraints"]["coverage_ok"] == m["primary"]["coverage_ok"]


def test_coverage_calculation_is_weighted_sum():
    """micro coverage must be sum(coverage_i * n_i)/sum(n_i), NOT mean(...)/sum(n)."""
    rows = [r for r in load_fold_metrics() if r["population"] == "primary_95"]
    tot_n = sum(int(r["n_test"]) for r in rows)
    num = sum(float(r["coverage_b3"]) * int(r["n_test"]) for r in rows)
    micro_cov = num / tot_n
    d = load_decision()
    # TSV stores coverage rounded to 4 decimals, so allow 1e-3 tolerance.
    assert abs(micro_cov - d["co_constraints"]["micro_coverage"]) < 1e-3, \
        f"weighted coverage {micro_cov:.6f} != reported {d['co_constraints']['micro_coverage']:.6f}"
    # Reject the buggy aggregation: np.mean(weighted)/total_n (which yields ~0.18 here).
    buggy_cov = sum(float(r["coverage_b3"]) * int(r["n_test"]) for r in rows) / len(rows) / tot_n
    assert abs(d["co_constraints"]["micro_coverage"] - buggy_cov) > 0.3, \
        "reported coverage must NOT equal the buggy np.mean(...)/n aggregation"
    # sanity: coverage must be a plausible fraction in [0,1]
    assert 0.0 <= micro_cov <= 1.0


def test_nc2_negative_gain_is_a_pass():
    """A scrambled (non-informative) predictor showing a NEGATIVE gain is conservative,
    not a spurious positive, so NC2 must pass."""
    m = load_metrics()
    nc2_mean = m["negative_controls"]["NC2_non_informative_mean_gain"]
    assert m["negative_controls"]["NC2_pass"] is True, \
        f"NC2 should pass (no spurious positive), got mean gain {nc2_mean:.4f}"
    assert nc2_mean < 0.05, f"scrambled predictor must not show positive gain, got {nc2_mean:.4f}"


def test_censored_likelihood_counts():
    d = load_decision()
    m = load_metrics()
    assert m["primary"]["n_measured"] == 84
    assert m["primary"]["n_censored"] == 11
    assert m["primary"]["n"] == 95
    assert m["sensitivity"]["n"] == 98
    assert m["sensitivity"]["n_measured"] == 87
    assert m["sensitivity"]["n_censored"] == 11


def test_old_dg_not_in_primary_decision():
    m = load_metrics()
    d = load_decision()
    # B4 is the old_dg positive control; primary decision uses B3 vs best(B1,B2)
    assert d["primary"]["best_baseline"] in ("B1", "B2")
    assert d["old_dg_role"].startswith("same-platform positive control")
    assert d["primary"]["micro_gain_b3_over_best_baseline"] > 0  # B3 over B1


def test_spec_hashes_match_ondisk():
    d = load_decision()
    for fname, h in d["spec_hashes"].items():
        path = f"{SPEC_DIR}/{fname}"
        assert os.path.exists(path), f"missing spec {path}"
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == h, f"spec hash mismatch for {fname}: {actual} != {h}"


def test_component_sizes_and_integrity():
    cs = load_json(f"{Q7_DIR}/component_splits.json")
    assert cs["primary_component_sizes"] == [80, 11, 2, 2]
    assert cs["sensitivity_component_sizes"] == [83, 11, 2, 2]
    # each member appears in exactly one component
    all_members = []
    for cid, members in cs["component_membership"].items():
        all_members.extend(members)
    assert len(all_members) == 95, f"expected 95 primary members, got {len(all_members)}"
    assert len(set(all_members)) == 95, "members must be disjoint across components"


def test_permutation_finite_p():
    d = load_decision()
    m = load_metrics()
    assert 0 < d["permutation_finite_p"] < 1, "finite p must be strictly between 0 and 1"
    assert d["permutation_finite_p"] == m["permutation"]["finite_p"]
    assert m["permutation"]["n_resamples"] == 999


def test_decision_rule_consistent():
    """Decision must be NOT_SUPPORTED iff gain<threshold OR any co-constraint fails."""
    d = load_decision()
    threshold_met = d["primary"]["threshold_met"]
    cc = d["co_constraints"]
    co_ok = all([cc["coverage_ok"], cc["NC1_pass"], cc["NC2_pass"],
                 cc["NC3_pass"], cc["NC4_pass"], cc["permutation_significant"],
                 cc["per_component_consistency"]])
    if threshold_met and co_ok:
        assert d["state"] == "QMAP_TRANSFER_SUPPORTED", d["state"]
    else:
        assert d["state"] == "QMAP_TRANSFER_NOT_SUPPORTED", d["state"]


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed} Q7 tests passed")


if __name__ == "__main__":
    run_all()