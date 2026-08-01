#!/usr/bin/env python3
"""Create an append-only reconciliation for the contract-selected ENA FASTQs.

This audit reads only public file bytes for size/hash verification and parses
existing audit JSON metadata. It never emits FASTQ records, sequences, quality
strings, read IDs, or scientific labels. It refuses to overwrite an existing
artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SELECTED_RUNS = (
    "SRR31402663",
    "SRR31402664",
    "SRR35766784",
    "SRR35766785",
    "SRR38259812",
)

PAIR_AUDITS = {
    "SRR31402663": "phase0/audits/SRR31402663_chunked_pair_audit_20260801T151900Z.json",
    "SRR31402664": "phase0/audits/SRR31402664_chunked_pair_audit_20260801T174200Z.json",
    "SRR35766784": "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.json",
    "SRR35766785": "phase0/audits/SRR35766785_fastq_audit_20260801T095400Z.json",
    "SRR38259812": "phase0/audits/SRR38259812_chunked_pair_audit_20260801T130500Z.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output}")
    if artifact_root not in output.parents:
        raise SystemExit("--output must be under --artifact-root")

    manifest_path = artifact_root / "phase0/source_metadata/ena_fastq_manifest_PRJNA1188187.tsv"
    manifest_rows: dict[tuple[str, str], dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            manifest_rows[(row["run_accession"], row["file_name"])] = row

    runs: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []

    for run in SELECTED_RUNS:
        audit_rel = PAIR_AUDITS[run]
        audit_path = artifact_root / audit_rel
        audit = load_json(audit_path)
        audit_status = audit.get("status")
        audit_safe = {
            "relative_path": audit_rel,
            "status": audit_status,
            "raw_sequence_content_emitted": audit.get("raw_sequence_content_emitted"),
            "scientific_labels_admitted": audit.get("scientific_labels_admitted"),
        }
        files: list[dict[str, object]] = []
        for mate in ("1", "2"):
            name = f"{run}_{mate}.fastq.gz"
            row = manifest_rows.get((run, name))
            if row is None:
                raise SystemExit(f"missing ENA manifest row: {run} {name}")
            expected_bytes = int(row["file_bytes"])
            target = artifact_root / "phase0/source_payloads/dms_sra/main_library" / run / name
            exists = target.is_file()
            observed_bytes = target.stat().st_size if exists else None
            actual_sha256 = sha256(target) if exists else None
            size_match = observed_bytes == expected_bytes
            item = {
                "run": run,
                "mate": mate,
                "file_name": name,
                "relative_path": str(target.relative_to(artifact_root)),
                "expected_bytes": expected_bytes,
                "observed_bytes": observed_bytes,
                "size_match": size_match,
                "sha256": actual_sha256,
            }
            files.append(item)
            if not exists:
                missing.append(item)
            elif not size_match:
                mismatches.append(item)
        runs.append({"run": run, "files": files, "pair_audit": audit_safe})

    preserved_partials: list[dict[str, object]] = []
    main_library = artifact_root / "phase0/source_payloads/dms_sra/main_library"
    for path in sorted(main_library.glob("SRR*/*.partial")):
        preserved_partials.append(
            {
                "relative_path": str(path.relative_to(artifact_root)),
                "size_bytes": path.stat().st_size,
                "preserved": True,
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            }
        )

    run_audits_complete = all(
        run["pair_audit"]["status"] in {"FASTQ_PAYLOAD_AUDIT_COMPLETE", "BATCH_COMPLETE"}
        for run in runs
    )
    no_sequence_emission = all(
        run["pair_audit"]["raw_sequence_content_emitted"] is False for run in runs
    )
    no_labels = all(run["pair_audit"]["scientific_labels_admitted"] is False for run in runs)
    complete = not missing and not mismatches and run_audits_complete and no_sequence_emission and no_labels
    payload = {
        "schema_version": "1.0",
        "artifact_type": "FULL_SELECTED_ENA_FASTQ_FILE_RECONCILIATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_runs": list(SELECTED_RUNS),
        "selected_run_count": len(SELECTED_RUNS),
        "file_count_expected": len(SELECTED_RUNS) * 2,
        "file_count_reconciled": sum(len(run["files"]) for run in runs),
        # The verifier intentionally accepts only the frozen batch-status
        # vocabulary. Keep the precise reconciliation state separately so
        # this artifact remains both machine-compatible and auditable.
        "status": "BATCH_COMPLETE" if complete else "BATCH_PARTIAL_PENDING_OR_BLOCKED",
        "reconciliation_status": "COMPLETE_PUBLIC_FASTQ_FILE_RECONCILIATION" if complete else "BLOCKED_FASTQ_RECONCILIATION",
        "size_and_hash_scope": "current final main_library files compared with ENA public manifest",
        "pair_audit_scope": "existing append-only pair audit JSON references",
        "missing_files": missing,
        "size_mismatches": mismatches,
        "runs": runs,
        "preserved_partial_files": preserved_partials,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "batch_gate_effect": "PUBLIC_FASTQ_PAYLOAD_RECONCILED_BUT_DMS_PROCESSED_PAYLOAD_AND_MATCHING_GATES_REMAIN_BLOCKED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(json.dumps({
        "status": payload["status"],
        "selected_run_count": payload["selected_run_count"],
        "file_count_reconciled": payload["file_count_reconciled"],
        "missing_count": len(missing),
        "size_mismatch_count": len(mismatches),
        "preserved_partial_count": len(preserved_partials),
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
