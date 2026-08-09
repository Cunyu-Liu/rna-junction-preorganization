"""Unit tests for the R3 D1 decision-gate module."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit import r3_decision_gate as r3


def _write_r2(tmp_path, joint_verdict):
    r2 = {"axes": [
        {"axis": "symmetry_5fold", "verdict": "NOT_SUPPORTED_OR_INCONCLUSIVE"},
        {"axis": "edit_x_nested_context", "verdict": joint_verdict},
    ]}
    d = tmp_path / "r2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "CoreHypothesisDecision_v3.json").write_text(json.dumps(r2))


def test_r3_missing_r2_blocks(tmp_path):
    cfg = {"run_id": "x", "run_root": str(tmp_path)}
    status = r3.main(cfg)
    assert status["state"] == "BLOCKED_PENDING_R2"


def test_r3_no_joint_signal_locks_track_a(tmp_path):
    _write_r2(tmp_path, "NOT_SUPPORTED_OR_INCONCLUSIVE")
    cfg = {"run_id": "x", "run_root": str(tmp_path)}
    status = r3.main(cfg)
    assert status["state"] == "D1_DECIDED"
    assert status["decision"] == "TRACK_A_LOCKED"
    assert status["joint_supported"] is False
    reg = json.loads((tmp_path / "r3" / "CandidateRegistry_v2.json").read_text())
    by_id = {c["candidate_id"]: c for c in reg}
    assert by_id["support_aware_mixture"]["status"] == "REJECTED"
    assert by_id["corrected_v1_31_latent_operator"]["status"] == "TRACK_A_ONLY"


def test_r3_joint_signal_allows_dual_track(tmp_path):
    _write_r2(tmp_path, "SEQUENCE_INCREMENT_SUPPORTED")
    cfg = {"run_id": "x", "run_root": str(tmp_path)}
    status = r3.main(cfg)
    assert status["decision"] == "DUAL_TRACK"
    assert status["joint_supported"] is True
    reg = json.loads((tmp_path / "r3" / "CandidateRegistry_v2.json").read_text())
    by_id = {c["candidate_id"]: c for c in reg}
    assert by_id["corrected_v1_31_latent_operator"]["status"] == "ALLOWED"
    assert by_id["support_aware_mixture"]["status"] == "REJECTED"
