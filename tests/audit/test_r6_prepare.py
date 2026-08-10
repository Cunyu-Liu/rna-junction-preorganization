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


def test_reproduce_md_uses_run_root_relative_checksum():
    """Checksum paths are run_root-relative, so the verify command must run from run_root."""
    from audit.r6_prepare import reproduce_md
    cfg = {"run_root": "/tmp/fake_root"}
    git = {"branch": "b", "commit": "c", "remote": "r"}
    md = reproduce_md(cfg, git)
    assert "sha256sum -c r6/checksums.sha256" in md
    assert "cd r6 && sha256sum -c checksums.sha256" not in md
