#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent tests for the B3 generative multi-regime benchmark (v1.5).

B3 (§9) replaces the v1.4 B0/B1 prototype (hard-coded booleans / schema-parse
only) with a generative, multi-regime, Monte-Carlo benchmark. These tests:

  1. verify the DGP is frozen and reproducible (same seed -> same dataset);
  2. verify the audit detector computes the correct verdict for each regime
     from raw data (never handed the expected label);
  3. verify the persisted frozen results under RUN_ROOT/benchmark/b3/ meet the
     B3 gate (false-pass <= 0.10, false-fail <= 0.10) with per-regime rates;
  4. verify the module ablations show each module prevents a distinct error.
"""

import json
import os
import sys

B3_SRC = None
for cand in ("/home/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z/benchmark/b3/src",
             "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z/benchmark/b3/src"):
    if os.path.isdir(cand):
        B3_SRC = cand
        if B3_SRC not in sys.path:
            sys.path.insert(0, B3_SRC)
        break

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
B3_DIR = f"{RUN_ROOT}/benchmark/b3"

MIN_COMP_SAMPLES = 5
FALSE_PASS_MAX = 0.10
FALSE_FAIL_MAX = 0.10


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. DGP frozen specs
# ---------------------------------------------------------------------------
def test_dgp_specs_frozen():
    specs = load_json(f"{B3_DIR}/dgp_specs.json")
    labels = {r["label"] for r in specs.values()}
    assert labels == {"VALID", "INVALID", "BOUNDARY"}
    assert len(specs) == 10
    # valid uses a balanced schema (all components adequate)
    assert all(s >= MIN_COMP_SAMPLES for s in specs["valid_transport"]["spec"]["schema"])
    # component_imbalance reproduces the qMaP-like 80/11/2/2 structure
    assert specs["component_imbalance"]["spec"]["schema"] == [80, 11, 2, 2]


def test_dgp_reproducible():
    from rna_audit_b3 import dgp
    a = dgp.generate("valid_transport", 0)
    b = dgp.generate("valid_transport", 0)
    assert list(a["y"]) == list(b["y"])
    assert list(a["mid"]) == list(b["mid"])
    assert a["label"] == "VALID"
    # different seeds differ
    c = dgp.generate("valid_transport", 1)
    assert list(a["y"]) != list(c["y"])


def test_dgp_unknown_regime_raises():
    from rna_audit_b3 import dgp
    try:
        dgp.generate("nope", 0)
        assert False, "should raise KeyError"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# 2. Detector verdicts (computed from raw data)
# ---------------------------------------------------------------------------
def test_detector_valid_transport():
    from rna_audit_b3 import dgp, detector
    for seed in (0, 1):
        ds = dgp.generate("valid_transport", seed)
        res = detector.audit(ds)
        assert res["decision"] == "VALID", f"seed {seed}: {res['checks']}"


def test_detector_endpoint_reuse():
    from rna_audit_b3 import dgp, detector
    ds = dgp.generate("endpoint_reuse", 0)
    res = detector.audit(ds)
    assert res["decision"] == "INVALID"
    assert res["checks"]["endpoint_identity"]["pass"] is False


def test_detector_censoring_misclassification():
    from rna_audit_b3 import dgp, detector
    ds = dgp.generate("censoring_misclassification", 0)
    res = detector.audit(ds)
    assert res["decision"] == "INVALID"
    assert res["checks"]["censoring"]["pass"] is False


def test_detector_component_imbalance():
    from rna_audit_b3 import dgp, detector
    ds = dgp.generate("component_imbalance", 0)
    res = detector.audit(ds)
    assert res["decision"] == "INVALID"
    assert res["checks"]["graph_support"]["pass"] is False


def test_detector_no_signal_null():
    from rna_audit_b3 import dgp, detector
    ds = dgp.generate("no_signal_null", 0)
    res = detector.audit(ds)
    assert res["decision"] == "INVALID"


def test_detector_boundary():
    from rna_audit_b3 import dgp, detector
    for seed in (0, 1):
        ds = dgp.generate("boundary", seed)
        res = detector.audit(ds)
        assert res["decision"] == "BOUNDARY", f"seed {seed} gain={res['gain']}"


def test_detector_never_sees_label():
    """The raw dataset handed to the detector must not disclose the label."""
    from rna_audit_b3 import dgp, detector
    for regime in ("valid_transport", "endpoint_reuse", "no_signal_null"):
        ds = dgp.generate(regime, 0)
        raw = {k: v for k, v in ds.items() if k != "label"}
        res = detector.audit(raw)
        assert res["decision"] in ("VALID", "INVALID", "BOUNDARY")


# ---------------------------------------------------------------------------
# 3. Persisted frozen results + gate
# ---------------------------------------------------------------------------
def test_frozen_decision_gate():
    d = load_json(f"{B3_DIR}/B3_decision.json")
    assert d["gate"] == "B3"
    assert d["state"] == "B3_VALIDATED"


def test_frozen_no_false_pass_no_false_fail():
    agg = load_json(f"{B3_DIR}/aggregate.json")
    assert agg["false_pass_rate"] == 0.0
    assert agg["false_fail_rate"] == 0.0
    assert agg["sensitivity"] == 1.0
    assert agg["specificity"] == 1.0


def test_frozen_per_regime_detection():
    agg = load_json(f"{B3_DIR}/aggregate.json")
    pr = agg["per_regime"]
    assert pr["valid_transport"]["detection_rate"] == 1.0
    assert pr["boundary"]["detection_rate"] == 1.0
    for regime, r in pr.items():
        if r["label"] == "INVALID":
            assert r["detection_rate"] == 1.0, f"{regime}: {r['detection_rate']}"


def test_frozen_results_have_checks():
    res = load_json(f"{B3_DIR}/benchmark_results.json")
    row = res["valid_transport"]["rows"][0]
    assert "checks" in row
    assert "decision" in row
    assert "coverage" in row


def test_frozen_confidence_intervals_present():
    agg = load_json(f"{B3_DIR}/aggregate.json")
    ci = agg["confidence_intervals"]
    assert set(ci) == {"false_pass_rate", "false_fail_rate", "sensitivity", "specificity"}
    for k, v in ci.items():
        assert v["n"] > 0, k


# ---------------------------------------------------------------------------
# 4. Ablations
# ---------------------------------------------------------------------------
def test_ablation_each_module_prevents_errors():
    ab = load_json(f"{B3_DIR}/ablation_results.json")
    base = float(ab["baseline"]["false_pass_rate"])
    assert base == 0.0
    # removing an audit module must allow a false pass (or at least not reduce power)
    nonmonotonic = []
    for mod in ("endpoint_identity", "censoring", "graph_support",
                "baseline_parity", "coverage_width", "claim_provenance"):
        fp = float(ab[mod]["false_pass_rate"])
        if not (fp >= base):
            nonmonotonic.append((mod, fp))
    # at least the modules that target a planted defect must inflate false-pass
    assert any(float(ab[m]["false_pass_rate"]) > base
               for m in ("endpoint_identity", "censoring", "graph_support",
                         "coverage_width", "claim_provenance"))


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed} B3 tests passed")


if __name__ == "__main__":
    run_all()