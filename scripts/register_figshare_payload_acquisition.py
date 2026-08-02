#!/usr/bin/env python3
"""Register an acquired official Figshare payload without unlocking science.

The payload archive is treated as an opaque scientific artifact here.  This
script reads only the ZIP central directory and the engineering merge report,
records provenance and archive metadata, preserves manifest history, and keeps
construct identity binding and all scientific gates fail-closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import zipfile
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
    *,
    replace_existing: bool = False,
) -> None:
    if path.exists() and not replace_existing:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def backup_manifest(path: Path, history_root: Path, run_id: str) -> Path:
    backup = history_root / f"{path.stem}_{run_id}{path.suffix}"
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite history backup: {backup}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(path.read_bytes())
    return backup


def relative(path: Path, artifact_root: Path) -> str:
    return str(path.resolve().relative_to(artifact_root.resolve()))


def member_metadata(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "name": info.filename,
        "file_size": info.file_size,
        "compressed_size": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "is_directory": info.is_dir(),
    }


def append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--member-audit", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_root = args.artifact_root.resolve()
    code_root = args.code_root.resolve()
    payload = args.payload.resolve()
    merge_audit_path = args.merge_audit.resolve()
    member_audit_path = args.member_audit.resolve()
    manifests = code_root / "manifests"
    history = manifests / "history"
    started_at = utc_now()

    data_registry_path = manifests / "data_registry.json"
    payload_inventory_path = manifests / "phase0_payload_inventory.json"
    acceptance_path = manifests / "acceptance_phase0.json"
    phase_status_path = manifests / "phase_status.json"

    if sha256_file(code_root / "contract/1.1.docx") != CONTRACT_SHA256:
        raise SystemExit("contract SHA256 does not match the frozen 1.1 contract")
    if not payload.is_file() or not merge_audit_path.is_file():
        raise SystemExit("payload or merge audit is missing")
    if member_audit_path.exists():
        raise SystemExit(f"refusing to overwrite member audit: {member_audit_path}")

    merge_audit = load_json(merge_audit_path)
    if merge_audit.get("status") != "MERGE_COMPLETE_ZIP_INTEGRITY_PASS":
        raise SystemExit("merge audit is not a successful ZIP-integrity pass")
    payload_sha256 = sha256_file(payload)
    if payload_sha256 != merge_audit.get("merged_sha256"):
        raise SystemExit("payload SHA256 does not match merge audit")

    with zipfile.ZipFile(payload) as archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        sequence_members = [
            member_metadata(info)
            for info in infos
            if info.filename in {"data/csvs/library_sequences.csv", "data/csvs/p5_sequences.csv"}
        ]
        readme_members = [
            member_metadata(info)
            for info in infos
            if info.filename == "data/README.md"
        ]
        xlsx_members = [
            member_metadata(info)
            for info in infos
            if info.filename.lower().endswith((".xlsx", ".xls"))
        ]
        construct_json_count = sum(
            1
            for info in infos
            if info.filename.startswith("data/raw-jsons/constructs/")
            and info.filename.lower().endswith(".json")
        )
        mutation_histogram_count = sum(
            1
            for info in infos
            if info.filename.startswith("data/mutation-histograms/")
            and not info.is_dir()
        )
        reactivity_member_count = sum(
            1
            for info in infos
            if any(token in info.filename.lower() for token in ("reactivity", "dms"))
            and not info.is_dir()
        )
        top_level_names = sorted(
            {name.split("/", 1)[0] for name in names if name}
        )

    if not sequence_members:
        raise SystemExit("official sequence-reference members are missing")

    member_audit = {
        "schema": "phase0-figshare-payload-member-metadata-v1",
        "status": "OFFICIAL_PAYLOAD_MEMBER_METADATA_AUDITED_IDENTITY_BINDING_PENDING",
        "created_at_utc": started_at,
        "payload": relative(payload, artifact_root),
        "payload_sha256": payload_sha256,
        "payload_bytes": payload.stat().st_size,
        "merge_audit": relative(merge_audit_path, artifact_root),
        "merge_audit_sha256": sha256_file(merge_audit_path),
        "source_file_id": merge_audit.get("source_file_id"),
        "source_url": merge_audit.get("source_url"),
        "zip_member_count": len(infos),
        "top_level_names": top_level_names,
        "official_readme_members": readme_members,
        "official_sequence_reference_members": sequence_members,
        "sequences_xlsx_members": xlsx_members,
        "construct_json_member_count": construct_json_count,
        "mutation_histogram_member_count": mutation_histogram_count,
        "reactivity_like_member_count": reactivity_member_count,
        "identity_binding_status": (
            "OPEN_EXACT_FROZEN_CONSTRUCT_BINDING_REQUIRED; "
            "SEQUENCES_XLSX_NAME_NOT_PRESENT; "
            "LIBRARY_SEQUENCES_CSV_IS_OFFICIAL_CANDIDATE_REFERENCE_ONLY"
        ),
        "processed_dms_payload_admitted": True,
        "primary_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "audit_boundary": (
            "Central-directory metadata and engineering merge evidence only; "
            "no sequence, label, or effect-value content was emitted."
        ),
    }
    write_json_atomic(member_audit_path, member_audit)
    member_audit_sha256 = sha256_file(member_audit_path)

    for path in (
        data_registry_path,
        payload_inventory_path,
        acceptance_path,
        phase_status_path,
    ):
        if not path.is_file():
            raise SystemExit(f"required manifest is missing: {path}")
        backup_manifest(path, history, args.run_id)

    merge_rel = relative(merge_audit_path, artifact_root)
    member_rel = relative(member_audit_path, artifact_root)
    payload_rel = relative(payload, artifact_root)
    merge_artifact = {
        "source_id": "deenalattha_2026_dms",
        "kind": "official_figshare_uploaded_chunk_merge_audit",
        "relative_path": merge_rel,
        "size_bytes": merge_audit_path.stat().st_size,
        "sha256": sha256_file(merge_audit_path),
        "status": merge_audit["status"],
        "payload_relative_path": payload_rel,
        "payload_sha256": payload_sha256,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "primary_labels_admitted": False,
    }
    member_artifact = {
        "source_id": "deenalattha_2026_dms",
        "kind": "official_figshare_payload_member_metadata_audit",
        "relative_path": member_rel,
        "size_bytes": member_audit_path.stat().st_size,
        "sha256": member_audit_sha256,
        "status": member_audit["status"],
        "payload_relative_path": payload_rel,
        "payload_sha256": payload_sha256,
        "identity_binding_status": member_audit["identity_binding_status"],
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "primary_labels_admitted": False,
    }
    payload_artifact = {
        "source_id": "deenalattha_2026_dms",
        "kind": "official_processed_dms_payload_archive",
        "relative_path": payload_rel,
        "size_bytes": payload.stat().st_size,
        "sha256": payload_sha256,
        "status": "VERIFIED_OFFICIAL_FIGSHARE_PAYLOAD_ACQUIRED",
        "source_file_id": merge_audit.get("source_file_id"),
        "zip_member_count": member_audit["zip_member_count"],
        "processed_dms_payload_admitted": True,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }

    inventory = load_json(payload_inventory_path)
    artifacts = inventory.setdefault("artifacts", [])
    for rel in (merge_rel, member_rel, payload_rel):
        artifacts[:] = [item for item in artifacts if item.get("relative_path") != rel]
    artifacts.extend([merge_artifact, member_artifact, payload_artifact])
    inventory["status"] = (
        "OFFICIAL_PROCESSED_DMS_PAYLOAD_ACQUIRED_IDENTITY_RECONCILIATION_PENDING"
    )
    inventory["last_refresh_utc"] = started_at
    inventory["primary_labels_admitted"] = False
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["phase0_gate_effect"] = "NO_PHASE_0_PASS"
    append_unique(
        inventory.setdefault("required_next_evidence", []),
        "exact_frozen_construct_identity_binding_for_official_library_reference",
    )
    write_json_atomic(
        payload_inventory_path,
        inventory,
        replace_existing=True,
    )

    registry = load_json(data_registry_path)
    source = next(
        (item for item in registry.get("sources", []) if item.get("source_id") == "deenalattha_2026_dms"),
        None,
    )
    if source is None:
        raise SystemExit("DMS source registry entry is missing")
    source.update(
        {
            "processed_dms_payload_status": "VERIFIED_OFFICIAL_FIGSHARE_PAYLOAD_ACQUIRED",
            "processed_dms_payload_admitted": True,
            "processed_dms_payload_archive": payload_rel,
            "processed_dms_payload_archive_sha256": payload_sha256,
            "processed_dms_payload_merge_audit": merge_rel,
            "processed_dms_payload_member_audit": member_rel,
            "official_sequence_reference_status": (
                "OFFICIAL_LIBRARY_SEQUENCES_CSV_PRESENT_SEQUENCES_XLSX_NOT_IN_ARCHIVE"
            ),
            "construct_identity_binding_status": (
                "PENDING_EXACT_FROZEN_IDENTITY_AUDIT"
            ),
            "primary_labels_admitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    updates = registry.setdefault("phase0_evidence_updates", [])
    updates[:] = [item for item in updates if item.get("run_id") != args.run_id]
    updates.append(
        {
            "run_id": args.run_id,
            "official_processed_dms_payload": payload_rel,
            "official_processed_dms_payload_sha256": payload_sha256,
            "official_processed_dms_payload_merge_audit": merge_rel,
            "official_processed_dms_payload_member_audit": member_rel,
            "official_processed_dms_payload_status": "ACQUIRED_AND_ARCHIVE_VERIFIED",
            "official_sequence_reference": "data/csvs/library_sequences.csv",
            "sequences_xlsx_present": False,
            "construct_identity_binding_status": "PENDING_EXACT_FROZEN_IDENTITY_AUDIT",
            "primary_labels_admitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
    )
    registry["status"] = (
        "PHASE_0_OFFICIAL_PROCESSED_DMS_PAYLOAD_ACQUIRED_IDENTITY_RECONCILIATION_PENDING"
    )
    registry["phase0_acceptance"]["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    write_json_atomic(data_registry_path, registry, replace_existing=True)

    acceptance = load_json(acceptance_path)
    acceptance["pass"] = False
    acceptance["status"] = (
        "IN_PROGRESS_OFFICIAL_PAYLOAD_ACQUIRED_IDENTITY_BINDING_PENDING"
    )
    acceptance["next_authorized_action"] = (
        "Complete exact frozen construct identity binding and source-run "
        "reconciliation; do not unlock Phase 0 scientific gate."
    )
    for evidence in (merge_rel, member_rel, payload_rel):
        append_unique(acceptance.setdefault("evidence_paths", []), evidence)
    acceptance["note"] = (
        acceptance.get("note", "").rstrip()
        + " Official Figshare data.zip was acquired and verified at "
        + args.run_id
        + " (payload SHA256 "
        + payload_sha256
        + "); the archive contains an official library_sequences.csv "
        + "candidate reference but no Sequences.xlsx member, so exact "
        + "frozen construct identity binding remains open and the scientific "
        + "gate stays fail-closed."
    )
    write_json_atomic(acceptance_path, acceptance, replace_existing=True)

    phase_status = load_json(phase_status_path)
    phase_status["overall_status"] = (
        "PHASE_0_OFFICIAL_PAYLOAD_ACQUIRED_IDENTITY_RECONCILIATION_PENDING"
    )
    phase_status["scientific_gate_effect"] = "NO_UNLOCK"
    blocker = (
        "Official processed-DMS archive is verified, but exact frozen construct "
        "identity binding remains pending; Sequences.xlsx is absent from the "
        "archive and library_sequences.csv is not yet admitted as an equivalent "
        "frozen reference."
    )
    if isinstance(phase_status.get("blocking_conditions"), list):
        append_unique(phase_status["blocking_conditions"], blocker)
    else:
        phase_status["blocking_conditions"] = [blocker]
    phase_status["transition_evidence"] = member_rel
    write_json_atomic(phase_status_path, phase_status, replace_existing=True)

    print(
        json.dumps(
            {
                "status": member_audit["status"],
                "payload_sha256": payload_sha256,
                "member_audit": member_rel,
                "member_audit_sha256": member_audit_sha256,
                "data_registry_sha256": sha256_file(data_registry_path),
                "payload_inventory_sha256": sha256_file(payload_inventory_path),
                "acceptance_sha256": sha256_file(acceptance_path),
                "phase_status_sha256": sha256_file(phase_status_path),
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
