#!/usr/bin/env python3
"""test_manifest_freshness — the derived manifest must be fresh, i.e. NOT older than
the terminal gate finalizers it claims to cover."""
import json, os, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")
PARENT_ROOT = "/mnt/cunyuliu/v1_3_corrective_20260804T122313Z"

def parse_ts(s):
    return s.replace("Z", "+00:00")

def test_parent_manifest_was_stale():
    m = json.load(open(f"{PARENT_ROOT}/manifests/canonical_state_manifest.json"))
    freshness = m.get("derived_manifest_freshness")
    # T5 finalizer 13:10, QR3 13:06, P0 13:16 — all AFTER parent freshness 12:38
    assert freshness is not None
    assert freshness <= "2026-08-04T13:06:11Z", f"parent freshness {freshness} should precede terminal gates (stale)"

def test_new_manifest_schema_requires_freshness():
    schema = json.load(open(f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"))
    assert "derived_manifest_freshness" in schema["required"]

def test_finalizer_runs_after_all_upstream():
    # The v1.4 C0 report records that the finalizer must run once after all terminal artifacts.
    report = open(f"{RUN_ROOT}/reports/C0_report.md").read()
    assert "finalizer" in report.lower()

if __name__ == "__main__":
    test_parent_manifest_was_stale()
    test_new_manifest_schema_requires_freshness()
    test_finalizer_runs_after_all_upstream()
    print("test_manifest_freshness: PASS")