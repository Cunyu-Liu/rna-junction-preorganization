#!/usr/bin/env python3
"""Record a partial-file size regression without reading payload bytes."""

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
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "observed_compressed_bytes": info.st_size,
        "inode": str(info.st_ino),
        "mtime_ns": info.st_mtime_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--partial", required=True, type=Path)
    parser.add_argument("--previous-audit", required=True, type=Path)
    parser.add_argument("--control-audit", required=True, type=Path)
    parser.add_argument("--continue-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-accession", required=True)
    parser.add_argument("--mate", required=True, type=int)
    args = parser.parse_args()

    contract = args.contract.resolve()
    partial = args.partial.resolve()
    previous = args.previous_audit.resolve()
    control = args.control_audit.resolve()
    continuation = args.continue_audit.resolve()
    output = args.output.resolve()
    if not contract.is_file() or not previous.is_file() or not control.is_file() or not continuation.is_file():
        parser.error("contract, previous audit, control audit, and continue audit must exist")
    if not partial.is_file():
        parser.error(f"partial file does not exist: {partial}")
    if output.exists():
        parser.error(f"refusing to overwrite existing audit: {output}")
    contract_sha256 = sha256_file(contract)
    if contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {contract_sha256}")
    prior = json.loads(previous.read_text(encoding="utf-8"))
    prior_size = prior.get("observed_compressed_bytes")
    expected = prior.get("expected_compressed_bytes")
    if not isinstance(prior_size, int) or not isinstance(expected, int):
        parser.error("previous audit lacks integer observed/expected compressed byte fields")
    current = snapshot(partial)
    current_size = current["observed_compressed_bytes"]
    status = "PARTIAL_SIZE_REGRESSION_AFTER_SAFE_CONTINUE_FROZEN" if current_size < prior_size else "PARTIAL_SIZE_OBSERVATION_FROZEN"
    report = {
        "schema_version": "phase0-partial-size-regression-audit-v1",
        "status": status,
        "observed_at_utc": utc_now(),
        "run_id": args.run_id,
        "run_accession": args.run_accession,
        "mate": args.mate,
        "contract_path": str(contract),
        "contract_sha256": contract_sha256,
        "partial_path": str(partial),
        "expected_compressed_bytes": expected,
        "observed_compressed_bytes": current_size,
        "prior_observation": {
            "artifact": str(previous),
            "artifact_sha256": sha256_file(previous),
            "observed_compressed_bytes": prior_size,
            "observed_at_utc": prior.get("observed_at_utc"),
        },
        "current_file_metadata": current,
        "continue_control_audit": {"path": str(continuation), "sha256": sha256_file(continuation)},
        "pause_control_audit": {"path": str(control), "sha256": sha256_file(control)},
        "observation_scope": "metadata-only stat comparison; no partial bytes were read or hashed",
        "cause_inferred": False,
        "scientific_meaning_inferred": False,
        "partial_file_deleted": False,
        "final_file_overwritten": False,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "required_follow_up": [
            "preserve the partial and original download log",
            "diagnose resume/truncation behavior without overwriting the partial",
            "perform final expected-size, SHA256, gzip, FASTQ-structure, and paired-read audit after terminal state",
            "retain this anomaly even if a later final audit succeeds",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output, report)
    print(json.dumps({"status": status, "observed_compressed_bytes": current_size, "prior_observed_compressed_bytes": prior_size, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
