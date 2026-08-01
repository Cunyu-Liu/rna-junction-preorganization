#!/usr/bin/env python3
"""Batch-audit completed ENA FASTQ pairs without exporting read content."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from audit_fastq_payload import audit_one, audit_pair, sha256


STOP_RULE = (
    "Do not unlock Phase 0 from file-integrity evidence alone; require complete "
    "public-source, semantic, construct-matching, and manual-audit gates."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="append", dest="runs", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--code-file", action="append", dest="code_files", required=True, type=Path)
    args = parser.parse_args()

    if not args.contract.is_file():
        parser.error(f"contract file does not exist: {args.contract}")
    for code_file in args.code_files:
        if not code_file.is_file():
            parser.error(f"code file does not exist: {code_file}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = set(args.runs)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if selected and row["run_accession"] not in selected:
            continue
        grouped.setdefault(row["run_accession"], []).append(row)

    run_results: list[dict[str, object]] = []
    pending = 0
    failed = 0
    for run, file_rows in sorted(grouped.items()):
        paths = {Path(row["file_name"]).name: row for row in file_rows}
        r1_row = next((row for name, row in paths.items() if name.endswith("_1.fastq.gz")), None)
        r2_row = next((row for name, row in paths.items() if name.endswith("_2.fastq.gz")), None)
        if not r1_row or not r2_row:
            run_results.append({"run_accession": run, "status": "BLOCKED_MANIFEST_NOT_PAIRED"})
            failed += 1
            continue
        r1 = args.input_root / run / r1_row["file_name"]
        r2 = args.input_root / run / r2_row["file_name"]
        partial_files = sorted((args.input_root / run).glob("*.partial"))
        if partial_files:
            run_results.append(
                {
                    "run_accession": run,
                    "status": "BLOCKED_PARTIAL_FILES_PRESENT",
                    "partial_files": [str(path) for path in partial_files],
                }
            )
            failed += 1
            continue
        if not r1.is_file() or not r2.is_file():
            run_results.append({"run_accession": run, "status": "PENDING_PAYLOAD_FILES", "r1_exists": r1.is_file(), "r2_exists": r2.is_file()})
            pending += 1
            continue
        try:
            r1_audit = audit_one(r1, None)
            r2_audit = audit_one(r2, None)
            pair = audit_pair(r1, r2)
        except Exception as exc:  # preserve the failure in the audit artifact
            run_results.append({"run_accession": run, "status": "BLOCKED_AUDIT_ERROR", "error_type": type(exc).__name__})
            failed += 1
            continue
        expected_sizes = {"r1": int(r1_row["file_bytes"]), "r2": int(r2_row["file_bytes"])}
        observed_sizes = {"r1": r1.stat().st_size, "r2": r2.stat().st_size}
        size_match = expected_sizes == observed_sizes
        result_status = "COMPLETE" if size_match and pair["paired_ids_consistent"] and r1_audit["malformed_record_count"] == 0 and r2_audit["malformed_record_count"] == 0 else "BLOCKED_PAYLOAD_INTEGRITY"
        if result_status != "COMPLETE":
            failed += 1
        run_results.append({"run_accession": run, "status": result_status, "expected_compressed_bytes": expected_sizes, "observed_compressed_bytes": observed_sizes, "r1": r1_audit, "r2": r2_audit, "pair_audit": pair})

    status = "BATCH_COMPLETE" if not pending and not failed else "BATCH_PARTIAL_PENDING_OR_BLOCKED"
    result = {
        "schema_version": "phase0-ena-fastq-batch-audit-v1",
        "status": status,
        "run_id": args.run_id,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "environment": {
            "cwd": os.getcwd(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "randomness": {"deterministic": True, "seed": None, "reason": "file-integrity audit has no stochastic step"},
        "input_provenance": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "contract_path": str(args.contract.resolve()),
            "contract_sha256": sha256(args.contract),
            "input_root": str(args.input_root.resolve()),
            "code_files": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in [Path(__file__).resolve(), *args.code_files]
            ],
        },
        "selected_runs": sorted(selected),
        "run_results": run_results,
        "pending_run_count": pending,
        "failed_run_count": failed,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "stop_rule": STOP_RULE,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "run_count": len(run_results), "pending_run_count": pending, "failed_run_count": failed}, sort_keys=True))
    return 0 if status == "BATCH_COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
