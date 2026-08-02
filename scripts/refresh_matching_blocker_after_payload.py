#!/usr/bin/env python3
"""Refresh matching-audit blocker semantics after the official payload arrives."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
NEW_STATUS = "BLOCKED_RAW_PROCESSED_CROSSWALK_AND_MANUAL_REVIEW_PENDING"


def now() -> str:
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


def write_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary file: {temporary}")
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
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    artifact_root = args.artifact_root.resolve()
    identity_audit_path = args.identity_audit.resolve()
    contract = code_root / "contract/1.1.docx"
    if sha256(contract) != CONTRACT_SHA256:
        raise SystemExit("contract SHA256 does not match the frozen 1.1 contract")
    identity_audit = load(identity_audit_path)
    if identity_audit.get("status") != (
        "OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY_SET_EQUAL"
    ):
        raise SystemExit("identity audit is not a successful condition-set pass")

    manifests = code_root / "manifests"
    matching_path = manifests / "matching_audit.json"
    registry_path = manifests / "data_registry.json"
    history = manifests / "history"
    for path in (matching_path, registry_path):
        backup_path = history / f"{path.stem}_{args.run_id}{path.suffix}"
        if backup_path.exists():
            raise FileExistsError(f"refusing to overwrite history backup: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(path.read_bytes())

    matching = load(matching_path)
    reasons = []
    for reason in matching.get("blocking_reasons", []):
        if reason == "DMS JSON/supplementary payload has not been obtained or validated.":
            reason = (
                "Official processed-DMS archive and condition-level construct "
                "identity are verified, but raw FASTQ to processed-DMS crosswalk "
                "is not bound to the official ENA/SRA run manifest."
            )
        append_unique(reasons, reason)
    append_unique(
        reasons,
        "No manual matched/rejected_or_ambiguous review has been performed.",
    )
    matching["status"] = NEW_STATUS
    matching["blocking_reasons"] = reasons
    matching["primary_labels_admitted"] = False
    matching["gate_effect"] = "NO_PHASE_0_PASS"
    matching["evidence_class"] = "AUDIT_SCHEMA_ONLY_NO_MATCHING_ROWS"
    matching["failure_preservation"] = {
        **matching.get("failure_preservation", {}),
        "official_processed_payload": str(
            identity_audit_path.relative_to(artifact_root)
        ),
        "official_processed_payload_sha256": identity_audit.get("payload_sha256"),
        "official_construct_identity_audit": str(
            identity_audit_path.relative_to(artifact_root)
        ),
        "official_construct_identity_audit_sha256": sha256(identity_audit_path),
    }
    manual = dict(matching.get("manual_audit", {}))
    manual["status"] = "NOT_STARTED_CROSSWALK_AND_REVIEW_PENDING"
    manual["observed_matched_cases"] = None
    manual["observed_rejected_or_ambiguous_cases"] = None
    manual["matching_accuracy"] = None
    manual["fatal_ambiguity"] = None
    matching["manual_audit"] = manual
    matching["refreshed_at_utc"] = now()
    matching["raw_sequence_content_emitted"] = False
    write_replace(matching_path, matching)

    registry = load(registry_path)
    registry["matching_audit_status"] = NEW_STATUS
    updates = registry.setdefault("phase0_evidence_updates", [])
    updates[:] = [item for item in updates if item.get("run_id") != args.run_id]
    updates.append(
        {
            "run_id": args.run_id,
            "matching_audit": str(matching_path.relative_to(code_root)),
            "matching_audit_sha256": sha256(matching_path),
            "matching_audit_status": NEW_STATUS,
            "official_construct_identity_audit": str(
                identity_audit_path.relative_to(artifact_root)
            ),
            "raw_processed_run_crosswalk_status": (
                "BLOCKED_NO_DIRECT_SRA_RUN_TOKEN_IN_OFFICIAL_ARCHIVE_MEMBER_NAMES"
            ),
            "manual_review_status": "NOT_STARTED_CROSSWALK_AND_REVIEW_PENDING",
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    registry["phase0_acceptance"]["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    write_replace(registry_path, registry)

    print(
        json.dumps(
            {
                "status": NEW_STATUS,
                "matching_audit_sha256": sha256(matching_path),
                "data_registry_sha256": sha256(registry_path),
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
