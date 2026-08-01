#!/usr/bin/env python3
"""Create a fail-closed ledger for the Phase 0 DMS reconstruction dependencies."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


CODE_ROOT = Path("/home/cunyuliu/rna_junction_preorganization_v1_1_20260801")
ARTIFACT_ROOT = Path("/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801")
CONTRACT = CODE_ROOT / "contract/1.1.docx"
README = ARTIFACT_ROOT / "phase0/source_metadata/dms_official_readme_ed75e36_20260802.md"
FEASIBILITY = ARTIFACT_ROOT / "phase0/audits/dms_reconstruction_feasibility_audit_20260802.json"
ROUTE = ARTIFACT_ROOT / "phase0/audits/figshare_readme_reprobe_20260801T200800Z.json"
PREVIOUS_ROUTE = ARTIFACT_ROOT / "phase0/audits/figshare_readme_reprobe_20260801T195800Z.json"
OUTPUT = ARTIFACT_ROOT / "phase0/audits/dms_phase0_dependency_ledger_20260802.json"

CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024 else None,
    }


def atomic_write(path: Path, value: dict[str, object]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


assert not OUTPUT.exists()
assert CONTRACT.is_file()
assert README.is_file()
assert FEASIBILITY.is_file()
assert ROUTE.is_file()
assert PREVIOUS_ROUTE.is_file()
assert sha256(CONTRACT) == CONTRACT_SHA256

feasibility = load_json(FEASIBILITY)
route = load_json(ROUTE)
previous_route = load_json(PREVIOUS_ROUTE)
assert feasibility.get("status") == "BLOCKED_RECONSTRUCTION_INPUTS_MISSING"
assert route.get("status") == "ROUTE_REPROBE_BLOCKED_NO_2XX"
assert route.get("payload_downloaded") is False
assert route.get("request_referer") == "https://figshare.com/"

ledger = {
    "schema_version": "phase0-dms-dependency-ledger-v1",
    "status": "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "contract_path": str(CONTRACT),
    "contract_sha256": sha256(CONTRACT),
    "code_root": str(CODE_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "scientific_gate_effect": "NO_PHASE_0_PASS",
    "primary_labels_admitted": False,
    "raw_sequence_content_emitted": False,
    "modeling_authorized": False,
    "source_provenance": {
        "repository": "https://github.com/YesselmanLabPublications/2025_char_3d_struct_features",
        "pinned_source_commit": "ed75e36bb36dc2f32c866c436c437d88a4743cf8",
        "readme_snapshot": file_record(README),
        "documented_processed_payload": "data.zip",
        "documented_figshare_file_id": "66482813",
        "documented_route": "https://ndownloader.figshare.com/files/66482813",
        "documented_referer": "https://figshare.com/",
    },
    "route_evidence": {
        "latest": file_record(ROUTE),
        "latest_status": route.get("status"),
        "latest_route_count": len(route.get("routes", [])) if isinstance(route.get("routes"), list) else None,
        "latest_all_http_codes": sorted({item.get("http_code") for item in route.get("routes", []) if isinstance(item, dict)}),
        "latest_payload_downloaded": route.get("payload_downloaded"),
        "previous": file_record(PREVIOUS_ROUTE),
        "previous_status": previous_route.get("status"),
        "access_control_bypassed": False,
    },
    "available_but_non_substituting_evidence": [
        {
            "evidence": "public FASTQ file-level/chunk-level integrity audits",
            "status": "AVAILABLE_FILE_INTEGRITY_ONLY",
            "cannot_substitute_for": ["construct-level DMS counts", "treatment/background hierarchy", "read-depth QC", "primary labels"],
        },
        {
            "evidence": "pinned processing source-code semantics and README",
            "status": "AVAILABLE_SOURCE_SEMANTICS_ONLY",
            "cannot_substitute_for": ["processed mutation histograms", "processed construct JSON", "traceable primary labels"],
        },
    ],
    "required_evidence": [
        {
            "requirement": "official processed-DMS payload or verified public route",
            "status": "BLOCKED_HTTP_403_NO_2XX",
            "evidence": str(ROUTE),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "source-defined mutation histograms",
            "status": "MISSING",
            "evidence": str(FEASIBILITY),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "reproducible rna-map executable and pinned processing environment",
            "status": "MISSING",
            "evidence": str(FEASIBILITY),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "construct reference FASTA and sequence/structure mapping",
            "status": "MISSING",
            "evidence": str(FEASIBILITY),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "processed construct JSON with source-defined fields",
            "status": "MISSING",
            "evidence": str(FEASIBILITY),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "construct-level count/background/read-depth/QC reconciliation",
            "status": "UNRESOLVED",
            "evidence": str(FEASIBILITY),
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "evidence-linked opaque matching table with at least 50 matched and 30 rejected-or-ambiguous cases",
            "status": "BLOCKED_PENDING_PRIMARY_PAYLOADS",
            "evidence": "manifests/acceptance_phase0.json",
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
        {
            "requirement": "manual agreement at least 0.95",
            "status": "BLOCKED_PENDING_MANUAL_MATCHING_AUDIT",
            "evidence": "phase0/audits/manual_matching_acceptance.json",
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        },
    ],
    "stop_rules": [
        "Do not derive mutation histograms or primary labels from FASTQ integrity summaries alone.",
        "Do not treat a README, source code, route HEAD response, smoke test, proxy metric, or training-set result as Phase 0 acceptance.",
        "Do not bypass Figshare access controls or replace the official payload with an unverified mirror.",
        "Do not start modeling or GPU training until all required evidence and manual matching gates pass.",
    ],
}

atomic_write(OUTPUT, ledger)
print(f"status={ledger['status']}")
print(f"ledger_path={OUTPUT}")
print(f"ledger_sha256={sha256(OUTPUT)}")
