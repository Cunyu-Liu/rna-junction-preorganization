#!/usr/bin/env python3
"""test_manifest_completeness — parent manifest was incomplete (empty gate_decisions/sentinels,
output only manifest). v1.4 schema requires recursive coverage of all artifact classes."""
import json, os, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")
PARENT_ROOT = "/mnt/cunyuliu/v1_3_corrective_20260804T122313Z"

def test_parent_manifest_was_incomplete():
    m = json.load(open(f"{PARENT_ROOT}/manifests/canonical_state_manifest.json"))
    assert m.get("gate_decisions") == {}, "parent gate_decisions should have been empty (gap G05)"
    assert m.get("sentinels") == {}, "parent sentinels should have been empty (gap G05)"
    out = m.get("output_artifacts", {})
    assert set(out.keys()) == {"manifests/canonical_state_manifest.json"}, "parent output inventory incomplete (G04)"

def test_new_schema_covers_all_artifact_classes():
    schema = json.load(open(f"{RUN_ROOT}/specs/CanonicalStateManifest.schema.json"))
    for field in ["source_artifacts","source_checksums","licenses","input_artifacts","input_checksums",
                  "spec_artifacts","spec_checksums","output_artifacts","output_checksums",
                  "gate_decisions","sentinels","finalizers"]:
        assert field in schema["properties"], f"schema missing {field}"

def test_closure_audit_reproduced_10_gaps():
    audit = json.load(open(f"{RUN_ROOT}/provenance/parent_closure_audit.json"))
    assert audit["n_gaps"] == 10, f"expected 10 gaps, got {audit['n_gaps']}"
    assert audit["all_gaps_reproduced"] is True

if __name__ == "__main__":
    test_parent_manifest_was_incomplete()
    test_new_schema_covers_all_artifact_classes()
    test_closure_audit_reproduced_10_gaps()
    print("test_manifest_completeness: PASS")