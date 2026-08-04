#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0 independent tests — verify the reusable benchmark + audit-schema freeze.

Checks:
  1. All 6 required schemas exist, parse as JSON, and are Draft 2020-12.
  2. All 5 required failure-mode fixtures exist with known ground truth.
  3. Both case studies (tecto, qmap) exist and are non-empty.
  4. The benchmark is NOT a dataset expansion: fixtures/case cards do not sum
     synthetic samples/reads/titrations into a biological N claim.
  5. Bonilla/Shin/Yesselman are marked as the same platform cluster (not counted
     as independent systems).
  6. B0_decision.json is valid, state is B0_BENCHMARK_FROZEN, inputs match the
     terminal states of C0/T6/Q6/Q7/N0.
  7. The CLI audit.py validates the schemas.
"""

import json
import os
import re
import subprocess
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
B0_DIR = f"{RUN_ROOT}/benchmark/b0"
PY = "/home/cunyuliu/miniconda3/envs/pc_cng/bin/python"

REQUIRED_SCHEMAS = [
    "EndpointRegistry.schema.json",
    "SourceMembershipRegistry.schema.json",
    "CensoringLedger.schema.json",
    "GraphSupportRegistry.schema.json",
    "ExposureRegistry.schema.json",
    "CanonicalStateManifest.schema.json",
]

REQUIRED_FIXTURES = [
    "endpoint_reuse",
    "censoring_misclassification",
    "component_imbalance",
    "baseline_failure",
    "coverage_width_tradeoff",
]

EXPECTED_INPUTS = {
    "C0": "C0_PASS",
    "T6": "TECTO_NEGATIVE_BOUND_AND_LOCKED",
    "Q6": "QMAP_SOURCE_RECONSTRUCTED",
    "Q7": "QMAP_TRANSFER_NOT_SUPPORTED",
    "N0": "METHODS_BOUNDARY_AUDIT",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def test_all_schemas_exist_and_parse():
    for name in REQUIRED_SCHEMAS:
        p = os.path.join(B0_DIR, "schemas", name)
        assert os.path.exists(p), f"missing schema {name}"
        s = load_json(p)
        assert s["$schema"].startswith("https://json-schema.org/draft/2020-12"), name
        assert s["type"] == "object", name


def test_all_fixtures_exist_with_truth():
    for name in REQUIRED_FIXTURES:
        p = os.path.join(B0_DIR, "fixtures", name, "fixture.json")
        assert os.path.exists(p), f"missing fixture {name}"
        fx = load_json(p)
        assert "purpose" in fx and "truth" in fx and "scenario" in fx, name


def test_case_studies_present():
    for cid in ("tecto", "qmap"):
        p = os.path.join(B0_DIR, "case_studies", cid, "case_card.json")
        assert os.path.exists(p), f"missing case study {cid}"
        c = load_json(p)
        assert c["case_id"] and c["decision"], cid


def test_not_dataset_expansion():
    # No fixture or case card may claim a summed biological N from synthetic data.
    for dpath, _, files in os.walk(B0_DIR):
        for fn in files:
            if fn == "case_card.json":
                c = load_json(os.path.join(dpath, fn))
                txt = json.dumps(c).lower()
                assert "biological n" not in txt, f"biological N claim in {dpath}"
                assert "independent measurement systems" not in txt or "not counted as independent" in txt.lower() or "share this cluster" in txt.lower(), f"platform cluster mis-stated in {dpath}"


def test_platform_cluster_rule():
    for cid in ("tecto", "qmap"):
        c = load_json(os.path.join(B0_DIR, "case_studies", cid, "case_card.json"))
        cluster = c["platform_cluster"].lower()
        assert "rna-map" in cluster or "tecto" in cluster, cluster
        assert "not counted as independent" in cluster, cluster


def test_decision_valid():
    dec = load_json(os.path.join(B0_DIR, "B0_decision.json"))
    assert dec["state"] == "B0_BENCHMARK_FROZEN", dec["state"]
    assert dec["not_dataset_expansion"] is True
    for k, v in EXPECTED_INPUTS.items():
        assert dec["inputs"][k] == v, f"input {k} mismatch: {dec['inputs'][k]} != {v}"


def test_cli_validates():
    res = subprocess.run([PY, os.path.join(B0_DIR, "cli", "audit.py"),
                          "validate", "--benchmark", B0_DIR],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["schemas_parsed"] == len(REQUIRED_SCHEMAS)


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