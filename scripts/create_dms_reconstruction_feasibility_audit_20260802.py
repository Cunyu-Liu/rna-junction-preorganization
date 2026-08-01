#!/usr/bin/env python3
"""Record whether the public raw-data reconstruction prerequisites are present."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path


CODE_ROOT = Path("/home/cunyuliu/rna_junction_preorganization_v1_1_20260801")
ARTIFACT_ROOT = Path("/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801")
SOURCE_DIR = ARTIFACT_ROOT / "phase0/source_metadata/dms_processing_source_20260801T190000Z"
OUTPUT = ARTIFACT_ROOT / "phase0/audits/dms_reconstruction_feasibility_audit_20260802.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None}
    if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
        result["sha256"] = sha256(path)
    return result


def atomic_json_write(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=".dms_reconstruction_feasibility_audit_",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
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


assert not OUTPUT.exists()
source_files = [file_record(path) for path in sorted(SOURCE_DIR.glob("*")) if path.is_file()]
assert len(source_files) == 4

checked_paths = [
    ARTIFACT_ROOT / "phase0/source_metadata/sra_runinfo_PRJNA1188187.csv",
    ARTIFACT_ROOT / "phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json",
    ARTIFACT_ROOT / "phase0/source_payloads/dms_sra/data",
    ARTIFACT_ROOT / "phase0/source_payloads/dms_sra/data/raw-jsons",
    ARTIFACT_ROOT / "phase0/source_payloads/dms_sra/data/mutation-histograms",
]
checked = [file_record(path) for path in checked_paths]

required_tools = {
    "rna-map_executable": shutil.which("rna-map"),
    "python_module:rna_map": bool(importlib.util.find_spec("rna_map")),
    "python_module:pandas": bool(importlib.util.find_spec("pandas")),
    "python_module:Bio": bool(importlib.util.find_spec("Bio")),
    "python_module:numpy": bool(importlib.util.find_spec("numpy")),
}

payload = {
    "schema_version": "phase0-dms-reconstruction-feasibility-audit-v1",
    "status": "BLOCKED_RECONSTRUCTION_INPUTS_MISSING",
    "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "contract_sha256": "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9",
    "code_root": str(CODE_ROOT),
    "artifact_root": str(ARTIFACT_ROOT),
    "source_code_semantics_files": source_files,
    "checked_paths": checked,
    "required_tools": required_tools,
    "public_fastq_pair_audits_available": True,
    "source_defined_mutation_histograms_available": False,
    "construct_reference_fasta_available": False,
    "construct_sequence_structure_csv_available": False,
    "processed_construct_json_available": False,
    "construct_level_reconstruction_started": False,
    "raw_sequence_content_emitted": False,
    "primary_labels_admitted": False,
    "scientific_gate_effect": "NO_PHASE_0_PASS",
    "blocking_reason": "Public FASTQ file-integrity evidence exists, but source-defined rna-map inputs and processed-DMS payloads required for construct-level hierarchy are absent.",
    "required_next_evidence": [
        "official processed-DMS payload or verified public route",
        "construct reference FASTA and sequence/structure mapping",
        "source-defined mutation histograms or an independently reproducible rna-map run with pinned inputs and environment",
        "construct-level count/background/read-depth/QC reconciliation before matching",
    ],
}
atomic_json_write(OUTPUT, payload)
print(f"status={payload['status']}")
print(f"audit_path={OUTPUT}")
print(f"audit_sha256={sha256(OUTPUT)}")
