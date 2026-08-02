#!/usr/bin/env python3
"""Register a completed toolchain audit without changing scientific gates."""

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
    return json.loads(path.read_text(encoding="utf-8"))


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
    artifact = {
        "source_id": "toolchain",
        "kind": "isolated_environment_audit",
        "relative_path": args.relative_audit_path,
        "size_bytes": args.audit.stat().st_size,
        "sha256": sha256(args.audit),
        "status": audit["status"],
        "training_started": audit["training_started"],
        "scientific_gate_effect": audit["scientific_gate_effect"],
        "cuda_probe": audit["cuda_probe"],
    }

    inventory = load(args.payload_inventory)
    artifacts = inventory.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item.get("relative_path") != args.relative_audit_path]
    artifacts.append(artifact)
    # The payload/scientific gate remains fail-closed until official inputs pass.
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["phase0_gate_effect"] = "NO_PHASE_0_PASS"
    write(args.payload_inventory, inventory)

    registry = load(args.data_registry)
    updates = registry.setdefault("phase0_evidence_updates", [])
    updates[:] = [item for item in updates if item.get("run_id") != args.run_id]
    updates.append(
        {
            "run_id": args.run_id,
            "toolchain_audit": args.relative_audit_path,
            "toolchain_audit_sha256": artifact["sha256"],
            "toolchain_status": audit["status"],
            "training_started": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    registry["status"] = "PHASE_0_PUBLIC_FASTQ_PAYLOAD_AUDIT_IN_PROGRESS"
    write(args.data_registry, registry)
    print(json.dumps({"audit": artifact, "data_registry_sha256": sha256(args.data_registry), "payload_inventory_sha256": sha256(args.payload_inventory)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
