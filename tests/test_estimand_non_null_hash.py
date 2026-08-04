#!/usr/bin/env python3
"""test_estimand_non_null_hash — the bound EstimandSpec must be non-null and hash-bound."""
import json, os, hashlib, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def test_estimand_spec_exists_and_non_null():
    path = f"{RUN_ROOT}/specs/tecto/EstimandSpec.yaml"
    assert os.path.exists(path), "EstimandSpec.yaml missing"
    content = open(path).read()
    # target functional must be non-null, non-empty, with units
    assert "kcal/mol" in content
    assert "Delta G" in content
    assert "left-censored" in content or "left" in content
    assert "-7.1" in content

def test_estimand_hash_bound():
    binding = json.load(open(f"{RUN_ROOT}/tecto/t6/estimand_binding.json"))
    yaml_sha = binding["estimand_spec_yaml_sha256"]
    assert yaml_sha == sha256_file(f"{RUN_ROOT}/specs/tecto/EstimandSpec.yaml"), "YAML hash mismatch in binding"
    assert len(yaml_sha) == 64

def test_estimand_non_null_flag():
    binding = json.load(open(f"{RUN_ROOT}/tecto/t6/estimand_binding.json"))
    assert binding["estimand_non_null"] is True

def test_no_null_estimand():
    verification = json.load(open(f"{RUN_ROOT}/tecto/t6/t6_verification.json"))
    assert verification["estimand_non_null"] is True

if __name__ == "__main__":
    test_estimand_spec_exists_and_non_null()
    test_estimand_hash_bound()
    test_estimand_non_null_flag()
    test_no_null_estimand()
    print("test_estimand_non_null_hash: PASS")