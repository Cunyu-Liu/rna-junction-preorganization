#!/usr/bin/env python3
"""Register construct identity evidence while keeping raw/processed crosswalk locked."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def backup(path: Path, history: Path, run_id: str) -> None:
    target = history / f"{path.stem}_{run_id}{path.suffix}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite history backup: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())


def write_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary manifest: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    code_root = args.code_root.resolve()
    audit_path = args.audit.resolve()
    manifests = code_root / "manifests"
    history = manifests / "history"
    now = utc_now()
    contract_path = code_root / "contract/1.1.docx"
    if sha256(contract_path) != CONTRACT_SHA256:
        raise SystemExit("contract SHA256 does not match the frozen 1.1 contract")
    audit = load(audit_path)
    if audit.get("status") != "OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY_SET_EQUAL":
        raise SystemExit("identity audit is not a successful condition-set pass")
    if audit.get("raw_processed_run_crosswalk_status") != (
        "BLOCKED_NO_DIRECT_SRA_RUN_TOKEN_IN_OFFICIAL_ARCHIVE_MEMBER_NAMES"
    ):
        raise SystemExit("unexpected raw/processed crosswalk state")

    data_registry_path = manifests / "data_registry.json"
    payload_inventory_path = manifests / "phase0_payload_inventory.json"
    acceptance_path = manifests / "acceptance_phase0.json"
    phase_status_path = manifests / "phase_status.json"
    for path in (
        data_registry_path,
        payload_inventory_path,
        acceptance_path,
        phase_status_path,
    ):
        backup(path, history, args.run_id)

    relative_audit = str(audit_path.relative_to(artifact_root))
    audit_sha256 = sha256(audit_path)
    artifact = {
        "source_id": "deenalattha_2026_dms",
        "kind": "official_processed_construct_identity_audit",
        "relative_path": relative_audit,
        "size_bytes": audit_path.stat().st_size,
        "sha256": audit_sha256,
        "status": audit["status"],
        "construct_json_member_count": audit["construct_json_member_count"],
        "construct_condition_reference_count": audit["construct_condition_reference_count"],
        "construct_condition_sequence_count": audit["construct_condition_sequence_count"],
        "raw_processed_run_crosswalk_status": audit["raw_processed_run_crosswalk_status"],
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }

    inventory = load(payload_inventory_path)
    artifacts = inventory.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item.get("relative_path") != relative_audit]
    artifacts.append(artifact)
    inventory["status"] = (
        "OFFICIAL_PROCESSED_CONSTRUCT_IDENTITY_AUDITED_RAW_PROCESSED_CROSSWALK_PENDING"
    )
    inventory["last_refresh_utc"] = now
    inventory["primary_labels_admitted"] = False
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["phase0_gate_effect"] = "NO_PHASE_0_PASS"
    append_unique(
        inventory.setdefault("required_next_evidence", []),
        "raw_fastq_to_processed_construct_crosswalk_bound_to_official_run_manifest",
    )
    write_replace(payload_inventory_path, inventory)

    registry = load(data_registry_path)
    source = next(
        (
            item
            for item in registry.get("sources", [])
            if item.get("source_id") == "deenalattha_2026_dms"
        ),
        None,
    )
    if source is None:
        raise SystemExit("DMS source registry entry is missing")
    source.update(
        {
            "processed_construct_identity_status": audit["status"],
            "processed_construct_identity_audit": relative_audit,
            "processed_construct_identity_audit_sha256": audit_sha256,
            "construct_identity_binding_status": (
                "OFFICIAL_PROCESSED_CONDITION_IDENTITY_PASS_RAW_SRA_CROSSWALK_PENDING"
            ),
            "raw_processed_run_crosswalk_status": audit[
                "raw_processed_run_crosswalk_status"
            ],
            "primary_labels_admitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    updates = registry.setdefault("phase0_evidence_updates", [])
    updates[:] = [item for item in updates if item.get("run_id") != args.run_id]
    updates.append(
        {
            "run_id": args.run_id,
            "official_processed_construct_identity_audit": relative_audit,
            "official_processed_construct_identity_audit_sha256": audit_sha256,
            "processed_construct_identity_status": audit["status"],
            "construct_condition_count": audit["construct_json_member_count"],
            "construct_condition_reference_count": audit[
                "construct_condition_reference_count"
            ],
            "construct_condition_sequence_count": audit[
                "construct_condition_sequence_count"
            ],
            "raw_processed_run_crosswalk_status": audit[
                "raw_processed_run_crosswalk_status"
            ],
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    registry["status"] = (
        "PHASE_0_OFFICIAL_PROCESSED_DMS_PAYLOAD_ACQUIRED_IDENTITY_RECONCILIATION_PENDING"
    )
    registry["phase0_acceptance"]["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    write_replace(data_registry_path, registry)

    acceptance = load(acceptance_path)
    acceptance["pass"] = False
    acceptance["status"] = (
        "IN_PROGRESS_CONSTRUCT_IDENTITY_AUDITED_RAW_PROCESSED_CROSSWALK_PENDING"
    )
    acceptance["next_authorized_action"] = (
        "Complete raw FASTQ to processed construct crosswalk against the official "
        "run manifest and DMS processing provenance; do not unlock Phase 0."
    )
    append_unique(acceptance.setdefault("evidence_paths", []), relative_audit)
    acceptance["note"] = (
        acceptance.get("note", "").rstrip()
        + " Official processed-condition construct identity audit passed at "
        + args.run_id
        + " for all condition files, but no direct SRA run token appears in the "
        + "archive member namespace; raw FASTQ to processed-DMS crosswalk remains "
        + "open and the scientific gate stays fail-closed."
    )
    write_replace(acceptance_path, acceptance)

    phase_status = load(phase_status_path)
    phase_status["overall_status"] = (
        "PHASE_0_OFFICIAL_PROCESSED_CONSTRUCT_IDENTITY_AUDITED_RAW_PROCESSED_CROSSWALK_PENDING"
    )
    phase_status["scientific_gate_effect"] = "NO_UNLOCK"
    blocker = (
        "Official processed-condition construct identity sets are equal, but raw "
        "ENA FASTQ to processed-DMS construct crosswalk is not directly bound by "
        "an official SRA run token/provenance artifact."
    )
    if isinstance(phase_status.get("blocking_conditions"), list):
        append_unique(phase_status["blocking_conditions"], blocker)
    else:
        phase_status["blocking_conditions"] = [blocker]
    phase_status["transition_evidence"] = relative_audit
    write_replace(phase_status_path, phase_status)

    print(
        json.dumps(
            {
                "status": "REGISTERED_IDENTITY_PASS_CROSSWALK_PENDING",
                "audit": relative_audit,
                "audit_sha256": audit_sha256,
                "data_registry_sha256": sha256(data_registry_path),
                "payload_inventory_sha256": sha256(payload_inventory_path),
                "acceptance_sha256": sha256(acceptance_path),
                "phase_status_sha256": sha256(phase_status_path),
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
