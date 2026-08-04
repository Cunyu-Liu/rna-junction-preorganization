#!/usr/bin/env python3
"""test_manifest_seal — verify canonical payload + detached seal avoids self-hash paradox."""
import json, os, hashlib, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def test_schema_has_seal_fields():
    schema = json.load(open(f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"))
    for field in ["manifest_payload_sha256", "detached_seal_path", "detached_seal_sha256"]:
        assert field in schema["properties"], f"schema missing {field}"
    assert "additionalProperties" in schema

def test_no_self_hash_in_payload():
    # The schema must NOT require a self-checksum field (would create a cycle).
    schema = json.load(open(f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"))
    props = schema["properties"]
    forbidden = ["self_sha256", "manifest_sha256", "file_sha256"]
    for fld in forbidden:
        assert fld not in props, f"self-hash field {fld} must not be in payload schema"

def test_detached_seal_mechanism_documented():
    schema = json.load(open(f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"))
    assert "detached_seal_path" in schema["properties"]
    assert "detached_seal_sha256" in schema["properties"]

if __name__ == "__main__":
    test_schema_has_seal_fields()
    test_no_self_hash_in_payload()
    test_detached_seal_mechanism_documented()
    print("test_manifest_seal: PASS")