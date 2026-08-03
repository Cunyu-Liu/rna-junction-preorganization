"""Unit tests for the M0 synthetic / operator-identification gate.

These tests verify the math/software behaves correctly on known synthetic
conditions (deterministic generator, censoring semantics, censored-likelihood
recovery, and negative-control spans). They are NOT a biological success claim.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "scripts"))

import m0_run  # noqa: E402


def test_generator_deterministic():
    a = m0_run.gen_synthetic(seed=42, censoring=True)
    b = m0_run.gen_synthetic(seed=42, censoring=True)
    assert a["rows"] == b["rows"]


def test_generator_censoring_semantics():
    d = m0_run.gen_synthetic(seed=7, censoring=True, base_mean=-6.3)
    for r in d["rows"]:
        # censored rows must be floored at CAP; uncensored must be above CAP
        if r["censored"]:
            assert r["dg10"] == m0_run.CAP
        else:
            assert r["dg10"] > m0_run.CAP


def test_censored_likelihood_recovers_mean():
    # Under substantial censoring the censored-likelihood estimate should be
    # close to the true family-conditional mean (no gross bias).
    d = m0_run.gen_synthetic(seed=2, censoring=True, n_constructs=2000, base_mean=-6.3)
    est = m0_run.estimate_interval(d["rows"], use_censored=True)
    truth = {}
    for fam in range(d["n_families"]):
        members = [j for j, i in d["truth"].items() if i["family"] == fam]
        truth[fam] = sum(d["truth"][j]["true_dg"] for j in members) / len(members)
    diffs = [abs(est[f]["point"] - truth[f]) for f in est if f in truth]
    assert diffs, "no family intervals produced"
    # tolerance: within ~2 kcal/mol given censoring + scaffold noise
    assert max(diffs) < 2.0


def test_negative_control_spans():
    # null signal: family-effect span should be far below min effect
    dnull = m0_run.gen_synthetic(seed=3, null=True, n_constructs=2000)
    means = [sum(r["dg10"] for r in dnull["rows"] if r["family"] == f)
             / sum(1 for r in dnull["rows"] if r["family"] == f)
             for f in range(dnull["n_families"])]
    assert max(means) - min(means) < m0_run.MIN_EFFECT


def test_min_effect_constant():
    assert m0_run.MIN_EFFECT == 1.0
    assert m0_run.CAP == -7.1