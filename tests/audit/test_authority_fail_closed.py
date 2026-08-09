"""R0.1 fail-closed authority tests (contract R0.1 / §13.2).

Any of these conditions must force the authority gate FAIL:
  - strict contract missing or hash mismatch
  - commit is the literal string "HEAD"
  - run DAG has a dangling parent edge
  - run DAG has a cycle
  - canonical source missing or hash mismatch
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from audit.provenance import authority_v2 as A


# The frozen durable canonical source (from authority_v2.CANONICAL_SOURCE_SHA).
# The nominal-pass test reads this real file so the source-hash check can pass.
CANONICAL_SOURCE_PATH = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"


def _write_contract(tmp, content=b"strict contract"):
    p = tmp / "contract.md"
    p.write_bytes(content)
    return p


def _fake_git_commit(tmp):
    """A fake git-rev-parse that returns a real-looking 40-char SHA."""
    return "5f28320cf8262a2dd6c3f75fa06d0dc74719a2c3"


def _make(tmp, contract_content=b"strict contract", source_content="__REAL__",
          commit=None, node_ids=("A", "B"), edges=(("A", "B"),)):
    strict = _write_contract(tmp, contract_content)
    source = tmp / "canonical.jsonl"
    if source_content == "__REAL__":
        source.write_bytes(Path(CANONICAL_SOURCE_PATH).read_bytes())
    else:
        source.write_bytes(source_content)
    # override current_commit
    orig = A.current_commit
    A.current_commit = lambda wt: (commit if commit is not None else _fake_git_commit(tmp))
    try:
        status, ok = A.run_authority_gate(tmp, strict, source, list(node_ids), list(edges))
    finally:
        A.current_commit = orig
    return status, ok


def test_nominal_pass():
    tmp = Path(tempfile.mkdtemp())
    status, ok = _make(tmp, contract_content=_contract_bytes())
    assert ok is True
    assert status["state"] == "PASS"


def _contract_bytes():
    """Return the bytes of the actual authorized strict contract.

    The file must be tracked in the repo under contract/ so its SHA-256 equals
    A.AUTHORIZED_CONTRACT_SHA.  If the contract is ever replaced, this test
    intentionally fails (the authority is bound to a fixed hash).
    """
    import hashlib
    import audit.provenance.authority_v2 as A
    p = Path(__file__).resolve().parent.parent.parent / "contract" \
        / "rna_junction_post_execution_strict_audit_2026-08-09.md"
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == A.AUTHORIZED_CONTRACT_SHA, \
        "contract bytes do not match AUTHORIZED_CONTRACT_SHA"
    return data


def test_missing_contract_fails():
    tmp = Path(tempfile.mkdtemp())
    strict = tmp / "missing.md"  # not created
    source = tmp / "canonical.jsonl"; source.write_bytes(b"SOURCE")
    orig = A.current_commit; A.current_commit = lambda wt: _fake_git_commit(tmp)
    try:
        status, ok = A.run_authority_gate(tmp, strict, source, ["A", "B"], [("A", "B")])
    finally:
        A.current_commit = orig
    assert ok is False
    assert status["state"] == "FAIL"
    assert status["checks"]["contract_present"] is False


def test_commit_literal_HEAD_fails():
    tmp = Path(tempfile.mkdtemp())
    status, ok = _make(tmp, contract_content=_contract_bytes(), commit="HEAD")
    assert ok is False
    assert status["checks"]["commit_is_real_sha"] is False


def test_dangling_parent_fails():
    tmp = Path(tempfile.mkdtemp())
    status, ok = _make(tmp, contract_content=_contract_bytes(),
                       node_ids=["A", "B"], edges=[("X", "B")])  # X dangling
    assert ok is False
    assert status["checks"]["run_dag_no_dangling"] is False


def test_cycle_fails():
    tmp = Path(tempfile.mkdtemp())
    status, ok = _make(tmp, contract_content=_contract_bytes(),
                       node_ids=["A", "B"], edges=[("A", "B"), ("B", "A")])
    assert ok is False
    assert status["checks"]["run_dag_no_cycles"] is False


def test_source_hash_mismatch_fails():
    tmp = Path(tempfile.mkdtemp())
    status, ok = _make(tmp, contract_content=_contract_bytes(), source_content=b"DIFFERENT")
    assert ok is False
    assert status["checks"]["canonical_source_hash_matches"] is False
