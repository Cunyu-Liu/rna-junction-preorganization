#!/usr/bin/env python3
"""Audit FASTQ payload integrity and paired-read shape without emitting reads.

This audit is intentionally limited to file integrity and run-level structure.
It does not infer treatment labels, construct identities, mutation counts, or
scientific reactivity values.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fastq(handle: IO[str]) -> Iterator[tuple[str, str, str, str]]:
    while True:
        name = handle.readline()
        if not name:
            return
        sequence = handle.readline()
        plus = handle.readline()
        quality = handle.readline()
        if not sequence or not plus or not quality:
            raise ValueError("truncated FASTQ record")
        yield name.rstrip("\r\n"), sequence.rstrip("\r\n"), plus.rstrip("\r\n"), quality.rstrip("\r\n")


def normalized_read_id(name: str) -> str:
    token = name.split(maxsplit=1)[0]
    if token.endswith("/1") or token.endswith("/2"):
        return token[:-2]
    return token


def audit_one(path: Path, expected_sha256: str | None) -> dict:
    observed_sha256 = sha256(path)
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "hash_match": expected_sha256 is None or observed_sha256 == expected_sha256,
        "record_count": 0,
        "sequence_length_counts": {},
        "quality_length_counts": {},
        "sequence_alphabet": [],
        "malformed_record_count": 0,
    }
    sequence_lengths: Counter[int] = Counter()
    quality_lengths: Counter[int] = Counter()
    alphabet: set[str] = set()
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        for name, sequence, plus, quality in read_fastq(handle):
            if not name.startswith("@") or not plus.startswith("+") or len(sequence) != len(quality):
                result["malformed_record_count"] = int(result["malformed_record_count"]) + 1
            result["record_count"] = int(result["record_count"]) + 1
            sequence_lengths[len(sequence)] += 1
            quality_lengths[len(quality)] += 1
            alphabet.update(sequence)
    result["sequence_length_counts"] = {str(k): v for k, v in sorted(sequence_lengths.items())}
    result["quality_length_counts"] = {str(k): v for k, v in sorted(quality_lengths.items())}
    result["sequence_alphabet"] = sorted(alphabet)
    return result


def audit_pair(r1: Path, r2: Path, max_mismatches: int = 1000) -> dict:
    mismatches = 0
    compared = 0
    with gzip.open(r1, "rt", encoding="ascii", newline="") as left, gzip.open(
        r2, "rt", encoding="ascii", newline=""
    ) as right:
        left_iter = read_fastq(left)
        right_iter = read_fastq(right)
        while True:
            try:
                left_record = next(left_iter)
            except StopIteration:
                left_record = None
            try:
                right_record = next(right_iter)
            except StopIteration:
                right_record = None
            if left_record is None or right_record is None:
                if left_record is not None or right_record is not None:
                    mismatches += 1
                break
            compared += 1
            if normalized_read_id(left_record[0]) != normalized_read_id(right_record[0]):
                mismatches += 1
                if mismatches >= max_mismatches:
                    break
    return {
        "r1": str(r1),
        "r2": str(r2),
        "compared_record_pairs": compared,
        "read_id_mismatch_count_capped": mismatches,
        "paired_ids_consistent": mismatches == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mate", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-mate-sha256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files = [args.input] + ([args.mate] if args.mate else [])
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    payloads = [audit_one(args.input, args.expected_sha256)]
    if args.mate:
        payloads.append(audit_one(args.mate, args.expected_mate_sha256))

    result: dict[str, object] = {
        "schema_version": "phase0-fastq-payload-audit-v1",
        "status": "FASTQ_PAYLOAD_AUDIT_COMPLETE",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "payloads": payloads,
        "pair_audit": audit_pair(args.input, args.mate) if args.mate else None,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    if any(not bool(payload["hash_match"]) for payload in payloads):
        result["status"] = "BLOCKED_HASH_MISMATCH"
    if any(int(payload["malformed_record_count"]) for payload in payloads):
        result["status"] = "BLOCKED_MALFORMED_FASTQ"
    if result["pair_audit"] and not bool(result["pair_audit"]["paired_ids_consistent"]):
        result["status"] = "BLOCKED_PAIRED_ID_MISMATCH"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "payload_count": len(payloads),
                "record_counts": [payload["record_count"] for payload in payloads],
                "pair_audit": result["pair_audit"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "FASTQ_PAYLOAD_AUDIT_COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
