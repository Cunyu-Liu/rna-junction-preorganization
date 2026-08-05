#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1 independent tests — verify fresh-checkout reproduction, adversarial review and
submission adjudication.

Checks:
  1. fresh_checkout_report.md exists and reports reproduction pass.
  2. adversarial_review.md exists, covers all §15.3 checks, no open P1/P2.
  3. issue_registry.tsv exists with all 9 checks and no OPEN rows.
  4. submission_adjudication.json state is E1_REPRODUCED_CLAIMS_ADMISSIBLE_SUBMISSION_READY.
  5. submission_authorized is False and submission_status is HOLD_PENDING_USER_AUTHORIZATION.
  6. replay.sh still returns REPLAY_OK (fresh-checkout determinism).
"""

import json
import os
import subprocess
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
E1_DIR = f"{RUN_ROOT}/external_review/e1"

CHECKS = [
    "endpoint_leakage", "qmap_99_to_98", "qmap_84_11_2_1", "component_support",
    "censoring", "baseline_parity", "coverage_width", "post_hoc_labels", "claim_wording",
]


def read_text(p):
    with open(p) as f:
        return f.read()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def test_fresh_checkout_report():
    fc = read_text(os.path.join(E1_DIR, "fresh_checkout_report.md"))
    assert "All checks reproduced: **True**" in fc or "All checks reproduced: **True**" in fc
    assert "REPLAY_OK" in fc


def test_adversarial_review_covers_all_checks():
    adv = read_text(os.path.join(E1_DIR, "adversarial_review.md"))
    for c in CHECKS:
        assert c in adv, f"missing adversarial check {c}"
    assert "PASS (no open defects)" in adv


def test_issue_registry_no_open():
    reg = read_text(os.path.join(E1_DIR, "issue_registry.tsv"))
    lines = reg.strip().split("\n")[1:]
    assert len(lines) == 9, f"expected 9 findings, got {len(lines)}"
    for line in lines:
        fields = line.split("\t")
        assert fields[4] == "closed", f"open defect: {line}"


def test_adjudication_state():
    adj = load_json(os.path.join(E1_DIR, "submission_adjudication.json"))
    assert adj["state"] == "E1_REPRODUCED_CLAIMS_ADMISSIBLE_SUBMISSION_READY", adj["state"]
    assert adj["fresh_checkout_reproduced"] is True
    assert adj["adversarial_review_pass"] is True
    assert len(adj["open_defects"]) == 0


def test_submission_not_authorized():
    adj = load_json(os.path.join(E1_DIR, "submission_adjudication.json"))
    assert adj["submission_authorized"] is False
    assert adj["submission_status"] == "HOLD_PENDING_USER_AUTHORIZATION"


def test_replay_still_ok():
    res = subprocess.run(["bash", f"{RUN_ROOT}/release/r1/replay.sh"],
                         capture_output=True, text=True)
    assert res.returncode == 0 and "REPLAY_OK" in res.stdout


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