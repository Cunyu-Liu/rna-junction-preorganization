#!/usr/bin/env python3
"""test_status_consistency — v1.4 must establish a single authoritative status and
freeze MANUSCRIPT_SUBMISSION=HOLD_PENDING_E1_AND_USER_APPROVAL everywhere."""
import json, os, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")
PARENT_ROOT = "/mnt/cunyuliu/v1_3_corrective_20260804T122313Z"

def test_authoritative_status_exists():
    path = f"{RUN_ROOT}/state/authoritative_status.json"
    assert os.path.exists(path), "authoritative_status.json missing"
    return json.load(open(path))

def test_submission_hold_frozen():
    s = test_authoritative_status_exists()
    assert s["authoritative_status"]["MANUSCRIPT_SUBMISSION"] == "HOLD_PENDING_E1_AND_USER_APPROVAL"
    assert s["interpretation"]["MANUSCRIPT_SUBMISSION"] == "HOLD_PENDING_E1_AND_USER_APPROVAL"

def test_operational_state_frozen():
    s = test_authoritative_status_exists()
    assert s["authoritative_status"]["CURRENT_OPERATIONAL_STATE"] == "BLOCKED_AT_V13_FINAL_CLOSURE_C0"

def test_conflict_ledger_exists():
    path = f"{RUN_ROOT}/state/status_conflict_ledger.tsv"
    assert os.path.exists(path)
    lines = open(path).read().strip().splitlines()
    assert len(lines) >= 3, "conflict ledger should have header + >=2 conflicts"

def test_v13_interpretation_of_p0_authorized():
    # P0's AUTHORIZED_UNDER_CLAIM_TIER must be interpreted as preparation-only.
    s = test_authoritative_status_exists()
    assert s["authoritative_status"]["MANUSCRIPT_PREPARATION"] == "AUTHORIZED_AFTER_C0_T6_Q6_Q7_N0"

if __name__ == "__main__":
    test_authoritative_status_exists()
    test_submission_hold_frozen()
    test_operational_state_frozen()
    test_conflict_ledger_exists()
    test_v13_interpretation_of_p0_authorized()
    print("test_status_consistency: PASS")