#!/usr/bin/env python3
"""Verify contract and phase-lock invariants without reading scientific data.

The verifier is deliberately conservative: a missing or mismatched contract
source is a blocking result, not an implicit pass. It checks only governance
metadata and never changes a manifest or phase state.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


CODE_ROOT = Path("/home/cunyuliu/rna_junction_preorganization_v1_1_20260801")
EXPECTED_CONTRACT_SHA256 = (
    "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    violations: list[str] = []
    contract_path = CODE_ROOT / "contract" / "1.1.docx"
    manifest_dir = CODE_ROOT / "manifests"

    project = load_json(manifest_dir / "project_manifest.json")
    phase = load_json(manifest_dir / "phase_status.json")
    acceptance = load_json(manifest_dir / "acceptance_phase0.json")
    registry = load_json(manifest_dir / "data_registry.json")
    audit = load_json(manifest_dir / "source_metadata_audit.json")

    observed_contract_sha256 = None
    if contract_path.is_file():
        observed_contract_sha256 = sha256(contract_path)
        if observed_contract_sha256 != EXPECTED_CONTRACT_SHA256:
            violations.append("CONTRACT_SOURCE_HASH_MISMATCH")
    else:
        violations.append("CONTRACT_SOURCE_ABSENT")

    expected_hash_locations = {
        "project_manifest": project.get("contract", {}).get("sha256"),
        "acceptance_phase0": acceptance.get("contract_sha256"),
        "data_registry": registry.get("contract_sha256"),
        "source_metadata_audit": audit.get("contract_sha256"),
    }
    for name, value in expected_hash_locations.items():
        if value != EXPECTED_CONTRACT_SHA256:
            violations.append(f"{name.upper()}_CONTRACT_HASH_INCONSISTENT")

    gates = phase.get("gates", {})
    required_locks = {
        "phase_0_5_four_specs": "LOCKED_UNTIL_PHASE_0_PASS",
        "phase_1_registry_qc_benchmark": "LOCKED",
        "phase_2_synthetic_robust_identification": "LOCKED",
        "phase_3_tecto_real_data": "LOCKED",
        "phase_4_dms_increment": "LOCKED",
        "phase_5_target_preorganization": "LOCKED",
        "phase_6_sequence_deployment": "FORBIDDEN_UNTIL_L3",
        "phase_7_external_validation": "LOCKED",
        "phase_8_reproducibility_freeze": "LOCKED",
    }
    for gate_name, required_state in required_locks.items():
        if gates.get(gate_name) != required_state:
            violations.append(f"PHASE_LOCK_INVARIANT_FAILED:{gate_name}")

    if acceptance.get("pass") is not False:
        violations.append("PHASE_0_ACCEPTANCE_MUST_REMAIN_FALSE_WHILE_BLOCKED")
    if registry.get("status") != "NOT_EXECUTED":
        violations.append("SOURCE_REGISTRY_MUST_REMAIN_NOT_EXECUTED")
    if audit.get("status") == "PASS":
        violations.append("METADATA_ONLY_AUDIT_CANNOT_BE_PHASE_0_PASS")

    status = "PASS_GOVERNANCE_INVARIANTS" if not violations else "BLOCKED_FAIL_CLOSED"
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "code_root": str(CODE_ROOT),
        "status": status,
        "violations": violations,
        "contract_path": str(contract_path),
        "expected_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "observed_contract_sha256": observed_contract_sha256,
        "project_status": project.get("status"),
        "phase_status": phase.get("overall_status"),
        "phase_0_acceptance": acceptance.get("status"),
        "source_registry_status": registry.get("status"),
        "metadata_audit_status": audit.get("status"),
        "scientific_gate_effect": "NO_UNLOCK",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    sys.exit(main())
