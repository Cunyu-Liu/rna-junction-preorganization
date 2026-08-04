#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N0 independent tests — verify the frozen novelty/claim/paper-spine gate.

Checks:
  1. Route determination follows the frozen §14.2 mapping given actual T6/Q7 states.
  2. N0 state is a valid allowed state (methods-boundary audit / transport-audit /
     audit-only / stop-manuscript).
  3. Novelty is NOT merely a new model / first chemical-probing / fair-split alone.
  4. Prohibited "runaway" claims are absent from the paper spine and primary claim.
  5. All N0 artifacts exist, are non-empty, and have fresh SHA-256 that match the
     decision's recorded artifact hashes.
  6. The forbidden-claims file contains the unconditional prohibitions from §14.4.
  7. Manuscript submission stays HOLD (no submission-authorization language).
"""

import csv
import hashlib
import json
import os
import re
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
N0_DIR = f"{RUN_ROOT}/novelty/n0"
T6_DEC = f"{RUN_ROOT}/tecto/t6/T6_decision.json"
Q7_DEC = f"{RUN_ROOT}/qmap/q7/Q7_decision.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Frozen §14.2 route mapping (reproduced independently, not from the script).
# ---------------------------------------------------------------------------
def expected_route(t6_state, q7_state):
    if t6_state == "TECTO_NEGATIVE_BOUND_AND_LOCKED":
        if q7_state == "QMAP_TRANSFER_SUPPORTED":
            return "TRANSPORT_AUDIT_METHODS"
        if q7_state == "QMAP_TRANSFER_NOT_SUPPORTED":
            return "METHODS_BOUNDARY_AUDIT"
        if q7_state == "QMAP_INCONCLUSIVE":
            return "BOUNDARY + BENCHMARK"
        return "TECTO_BOUNDARY/AUDIT"
    return "AUDIT_ONLY"


ALLOWED_N0_STATES = {
    "METHODS_BOUNDARY_AUDIT",
    "TRANSPORT_AUDIT_METHODS",
    "AUDIT_ONLY",
    "STOP_MANUSCRIPT",
    "BOUNDARY + BENCHMARK",
}

# Words that indicate a runaway claim the paper must not make. These must appear
# ONLY in negated form (the negative claim); the scan below flags positive
# affirmative usage, so the radical forms are checked as "not four i.i.d. repeats".
FORBIDDEN_TERMS = [
    "validated tectoRNA",
    "independently reproduced junction preorganization",
    "extrapolate to arbitrary",
    "large model is an innovation",
    "guarantees publishability",
    "preregistered",
    "submission is authorized",
    "submission authorized",
]

# affirmative-only forms (negated forms are permitted in the negative claim)
FORBIDDEN_AFFIRMATIVE = [
    "the four components are four i.i.d. repeats",
    "are four independent i.i.d. repeats",
]


def test_route_follows_contract():
    t6 = load_json(T6_DEC)["terminal_state"]
    q7 = load_json(Q7_DEC)["state"]
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    assert dec["route"] == expected_route(t6, q7), (
        f"route {dec['route']} != expected {expected_route(t6, q7)}"
    )
    assert dec["inputs"]["T6_terminal_state"] == t6
    assert dec["inputs"]["Q7_terminal_state"] == q7


def test_state_is_allowed():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    assert dec["state"] in ALLOWED_N0_STATES, dec["state"]


def test_route_matches_state():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    # A methods-boundary audit route must carry that state (not a stop).
    assert dec["state"] == "METHODS_BOUNDARY_AUDIT"
    assert dec["route"] == "METHODS_BOUNDARY_AUDIT"


def test_novelty_not_model_probing_split_alone():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    nov = dec["novelty"]
    assert nov["novelty_ok"] is True
    avoided = nov["avoided_claims"]
    for term in ("new-model-alone", "first-chemical-probing-alone", "fair-split-alone"):
        assert term in avoided, f"missing avoided claim {term}"
    # The contribution must be the unified audit, not a single-domain tool.
    assert "transport audit" in nov["contribution"].lower()


def test_no_runaway_claims_in_spine_or_primary():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    spine = open(f"{N0_DIR}/paper_spine.md").read()
    primary = dec["primary_claim"]
    # Scan only the primary-claim section of the spine (the negative claim section
    # legitimately restates banned statements in negated form).
    m = re.search(r"## Primary claim \(maximum allowed for this route\)\n(.*?)\n\n## Negative claim", spine, re.S)
    spine_primary = m.group(1) if m else ""
    haystack = (spine_primary + "\n" + primary).lower()
    for term in FORBIDDEN_TERMS:
        assert term not in haystack, f"runaway claim present: {term}"
    for term in FORBIDDEN_AFFIRMATIVE:
        assert term not in haystack, f"affirmative i.i.d. claim present: {term}"


def test_artifacts_exist_and_hashes_match():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    for rel, recorded in dec["artifacts"].items():
        path = os.path.join(N0_DIR, rel)
        assert os.path.exists(path), f"missing artifact {rel}"
        assert os.path.getsize(path) > 0, f"empty artifact {rel}"
        assert sha256(path) == recorded, f"hash mismatch for {rel}"


def test_prior_art_matrix_complete():
    rows = read_tsv(f"{N0_DIR}/prior_art_matrix.tsv")
    ids = {r["prior_art_id"] for r in rows}
    for expected in ("Denny2018", "Yesselman2019", "Bonilla2021", "Shin2023",
                     "qMaP2024", "PRIME2026", "DataSAIL"):
        assert expected in ids, f"missing prior art {expected}"
    for r in rows:
        assert r["key_claim"] and r["v14_differential"], f"empty field in {r['prior_art_id']}"


def test_forbidden_claims_contain_prohibitions():
    text = open(f"{N0_DIR}/forbidden_claims.txt").read()
    for phrase in ("7,500-construct DMS", "i.i.d. repeats", "guarantees publishability",
                   "preregistered", "more parameters compensate"):
        assert phrase in text, f"forbidden claim missing phrase: {phrase}"


def test_submission_held():
    dec = load_json(f"{N0_DIR}/N0_decision.json")
    assert "HOLD" in dec["scientific_disposition"].upper()
    spine = open(f"{N0_DIR}/paper_spine.md").read()
    assert "HOLD_PENDING_E1_AND_USER_APPROVAL" in spine


def test_citation_verification_marks_prime_preprint():
    rows = read_tsv(f"{N0_DIR}/citation_verification.tsv")
    prime = next(r for r in rows if r["ref_id"] == "PRIME2026")
    assert prime["peer_reviewed"] == "False", "PRIME2026 must be flagged as preprint"
    assert prime["status"] == "PREPRINT_NOTE"


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