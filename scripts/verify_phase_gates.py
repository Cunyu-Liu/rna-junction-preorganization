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
MANUAL_MATCHING_COMPONENT_PATH = Path(
    "/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801"
) / "phase0" / "audits" / "manual_matching_acceptance.json"


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
    matching = load_json(manifest_dir / "matching_audit.json")
    payload_inventory = load_json(manifest_dir / "phase0_payload_inventory.json")
    manual_matching = None
    if MANUAL_MATCHING_COMPONENT_PATH.is_file():
        try:
            manual_matching = load_json(MANUAL_MATCHING_COMPONENT_PATH)
        except (OSError, json.JSONDecodeError, ValueError):
            violations.append("MANUAL_MATCHING_COMPONENT_JSON_INVALID")
        else:
            allowed_manual_statuses = {
                "BLOCKED_MANUAL_MATCHING_AUDIT",
                "PASS_MANUAL_MATCHING_COMPONENT",
            }
            if manual_matching.get("status") not in allowed_manual_statuses:
                violations.append("MANUAL_MATCHING_COMPONENT_STATUS_UNEXPECTED")
            if manual_matching.get("raw_sequence_content_emitted") is not False:
                violations.append("MANUAL_MATCHING_COMPONENT_MUST_NOT_EMIT_RAW_SEQUENCE")
            if manual_matching.get("primary_labels_admitted") is not False:
                violations.append("MANUAL_MATCHING_COMPONENT_MUST_NOT_ADMIT_PRIMARY_LABELS")
            if manual_matching.get("scientific_gate_effect") != "NO_PHASE_0_PASS":
                violations.append("MANUAL_MATCHING_COMPONENT_MUST_NOT_UNLOCK_PHASE_0")

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
        "matching_audit": matching.get("contract_sha256"),
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
    allowed_registry_statuses = {
        "NOT_EXECUTED",
        "PHASE_0_METADATA_PARTIAL_PAYLOAD_AUDIT_IN_PROGRESS",
        "PHASE_0_PUBLIC_FASTQ_PAYLOAD_AUDIT_IN_PROGRESS",
        "PHASE_0_OFFICIAL_PROCESSED_DMS_PAYLOAD_ACQUIRED_IDENTITY_RECONCILIATION_PENDING",
    }
    if registry.get("status") not in allowed_registry_statuses:
        violations.append("SOURCE_REGISTRY_STATUS_UNEXPECTED")
    if audit.get("status") == "PASS":
        violations.append("METADATA_ONLY_AUDIT_CANNOT_BE_PHASE_0_PASS")
    if matching.get("primary_labels_admitted") is not False:
        violations.append("MATCHING_AUDIT_MUST_NOT_ADMIT_PRIMARY_LABELS_WHILE_BLOCKED")
    if matching.get("status") != "BLOCKED_PENDING_PRIMARY_PAYLOADS":
        violations.append("MATCHING_AUDIT_STATUS_MUST_REMAIN_BLOCKED_WHILE_PHASE_0_INCOMPLETE")

    for artifact in payload_inventory.get("artifacts", []):
        if artifact.get("kind") != "FASTQ_batch_payload_audit":
            continue
        if artifact.get("raw_sequence_content_emitted") is not False:
            violations.append("FASTQ_BATCH_AUDIT_MUST_NOT_EMIT_RAW_SEQUENCE")
        if artifact.get("scientific_labels_admitted") is not False:
            violations.append("FASTQ_BATCH_AUDIT_MUST_NOT_ADMIT_SCIENTIFIC_LABELS")
        if artifact.get("scientific_gate_effect") != "NO_PHASE_0_PASS":
            violations.append("FASTQ_BATCH_AUDIT_MUST_NOT_UNLOCK_PHASE_0")
        if artifact.get("status") not in {"BATCH_COMPLETE", "BATCH_PARTIAL_PENDING_OR_BLOCKED", "ONE_SELECTED_RUN_PAIR_AUDIT_COMPLETE_BATCH_PENDING"}:
            violations.append("FASTQ_BATCH_AUDIT_STATUS_UNEXPECTED")

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
        "matching_audit_status": matching.get("status"),
        "manual_matching_component_path": str(MANUAL_MATCHING_COMPONENT_PATH),
        "manual_matching_component_status": (
            manual_matching.get("status") if manual_matching is not None else "NOT_PRESENT"
        ),
        "scientific_gate_effect": "NO_UNLOCK",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    sys.exit(main())
