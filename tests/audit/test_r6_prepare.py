"""Unit tests for the R6 engineering-prep module."""
import hashlib
from pathlib import Path

from audit.r6_prepare import (_sha256, license_ledger, PHASES)


def test_sha256_matches():
    p = Path("/tmp/r6_test_file.bin")
    p.write_bytes(b"hello world")
    assert _sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_phases_complete():
    for k in ("R0.5", "R0.6", "R1", "R2", "R3", "R4", "R5"):
        assert k in PHASES
    assert PHASES["R3"] == "D1_TRACK_A_LOCKED"


def test_license_ledger_marks_legal_pending():
    led = license_ledger()
    assert len(led) >= 3
    for r in led:
        assert r["owner"] == "PENDING_LEGAL"
    assert any(r["type"] == "data" for r in led)
    assert any(r["type"] == "code" for r in led)
