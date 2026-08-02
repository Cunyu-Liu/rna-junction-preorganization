#!/usr/bin/env python3
"""Register a public FASTQ reconciliation while keeping scientific gates locked."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-registry", type=Path, required=True)
    parser.add_argument("--payload-inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--relative-audit-path", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    audit = load(args.audit)
    artifact_sha256 = sha256(args.audit)
    artifact = {
        "source_id": "deenalattha_2026_dms",
        "kind": "FASTQ_batch_payload_audit",
        "relative_path": args.relative_audit_path,
        "size_bytes": args.audit.stat().st_size,
        "sha256": artifact_sha256,
        "status": audit["status"],
        "selected_run_count": audit["selected_run_count"],
        "file_count_expected": audit["file_count_expected"],
        "file_count_reconciled": audit["file_count_reconciled"],
        "missing_file_count": len(audit["missing_files"]),
        "size_mismatch_count": len(audit["size_mismatches"]),
        "preserved_partial_count": len(audit["preserved_partial_files"]),
        "raw_sequence_content_emitted": audit["raw_sequence_content_emitted"],
        "scientific_labels_admitted": audit["scientific_labels_admitted"],
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "batch_gate_effect": audit["batch_gate_effect"],
    }

    inventory = load(args.payload_inventory)
    artifacts = inventory.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item.get("relative_path") != args.relative_audit_path]
    artifacts.append(artifact)
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["phase0_gate_effect"] = "NO_PHASE_0_PASS"
    write(args.payload_inventory, inventory)

    registry = load(args.data_registry)
    updates = registry.setdefault("phase0_evidence_updates", [])
    updates[:] = [item for item in updates if item.get("run_id") != args.run_id]
    updates.append(
        {
            "run_id": args.run_id,
            "ena_fastq_reconciliation": args.relative_audit_path,
            "ena_fastq_reconciliation_sha256": artifact_sha256,
            "ena_fastq_reconciliation_status": audit["status"],
            "missing_file_count": len(audit["missing_files"]),
            "size_mismatch_count": len(audit["size_mismatches"]),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    registry["status"] = "PHASE_0_PUBLIC_FASTQ_PAYLOAD_AUDIT_IN_PROGRESS"
    write(args.data_registry, registry)
    print(json.dumps({"artifact": artifact, "data_registry_sha256": sha256(args.data_registry), "payload_inventory_sha256": sha256(args.payload_inventory)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
