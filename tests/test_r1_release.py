#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1 independent tests — verify the sealed release bundle + canonical manifest.

Checks:
  1. canonical_state_payload.json exists, is valid JSON, has a non-empty payload hash.
  2. detached_seal.sha256 exists and its recorded hash matches the payload FILE hash.
  3. No self-hash paradox: payload's manifest_payload_sha256 equals the hash of the
     canonical serialization (with seal fields empty), and the final file hash comes
     from the external seal.
  4. release_inventory.tsv lists all artifacts with correct hashes.
  5. replay.sh is executable and returns REPLAY_OK.
  6. R1_decision.json state is R1_RELEASE_SEALED, submission HOLD preserved.
  7. All 8 gate states are present in the sentinels map.
"""

import json
import os
import subprocess
import sys

RUN_ROOT = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
R1_DIR = f"{RUN_ROOT}/release/r1"
GATES = ["C0", "T6", "Q6", "Q7", "N0", "B0", "B1", "B2"]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def read_text(p):
    with open(p) as f:
        return f.read()


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_payload_exists_and_hash():
    p = os.path.join(R1_DIR, "canonical_state_payload.json")
    assert os.path.exists(p)
    payload = load_json(p)
    assert len(payload["manifest_payload_sha256"]) == 64
    assert payload["contract_sha256"] == "e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"


def test_seal_matches_payload_file():
    seal = read_text(os.path.join(R1_DIR, "detached_seal.sha256"))
    recorded = seal.split()[0]
    assert recorded == sha256_file(os.path.join(R1_DIR, "canonical_state_payload.json"))


def test_no_self_hash_paradox():
    payload = load_json(os.path.join(R1_DIR, "canonical_state_payload.json"))
    # The manifest_payload_sha256 must be the hash of the canonical serialization,
    # NOT the hash of the file that contains the hash (which would be a cycle).
    # Reconstruct: set seal fields to empty and re-serialize.
    clone = dict(payload)
    clone["manifest_payload_sha256"] = ""
    clone["detached_seal_path"] = ""
    clone["detached_seal_sha256"] = ""
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    import hashlib
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert payload["manifest_payload_sha256"] == recomputed, "payload hash must be the canonical-serialization hash"


def test_inventory_covers_artifacts():
    inv = read_text(os.path.join(R1_DIR, "release_inventory.tsv"))
    lines = inv.strip().split("\n")[1:]  # skip header
    assert len(lines) >= 4, f"inventory too short: {len(lines)}"
    for line in lines:
        rel, size, h = line.split("\t")
        assert len(h) == 64, f"bad hash for {rel}"
        p = os.path.join(R1_DIR, rel)
        assert os.path.exists(p), f"inventory lists missing file {rel}"
        assert sha256_file(p) == h, f"inventory hash mismatch for {rel}"


def test_replay_ok():
    rp = os.path.join(R1_DIR, "replay.sh")
    os.chmod(rp, 0o755)
    res = subprocess.run(["bash", rp], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "REPLAY_OK" in res.stdout


def test_decision_state_and_hold():
    dec = load_json(os.path.join(R1_DIR, "R1_decision.json"))
    assert dec["state"] == "R1_RELEASE_SEALED", dec["state"]
    assert dec["seal_matches_payload_file"] is True
    assert dec["no_self_hash_paradox"] is True
    assert dec["manuscript_submission"] == "HOLD_PENDING_E1_AND_USER_APPROVAL"


def test_all_gate_states_sealed():
    payload = load_json(os.path.join(R1_DIR, "canonical_state_payload.json"))
    for g in GATES:
        assert g in payload["sentinels"], f"missing gate {g} in sentinels"
    assert payload["sentinels"]["Q7"] == "QMAP_TRANSFER_NOT_SUPPORTED"
    assert payload["sentinels"]["N0"] == "METHODS_BOUNDARY_AUDIT"


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