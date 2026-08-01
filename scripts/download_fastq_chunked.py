#!/usr/bin/env python3
"""Download one public FASTQ object with independently verified byte ranges.

The original downloader is intentionally left stopped and its partial file is
never used as input.  Each 128 MiB chunk is fetched with an explicit Range
request, checked for HTTP 206, exact Content-Range, exact byte count, and a
SHA256 digest before it is retained.  Only after every chunk passes are the
chunks concatenated into a separate recovery artifact and checked with
``gzip -t``.  This script never admits scientific labels or emits reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
MAX_ATTEMPTS = 8
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_last_response_headers(path: Path) -> tuple[int | None, tuple[int, int, int] | None]:
    status: int | None = None
    content_range: tuple[int, int, int] | None = None
    for line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            content_range = None
        elif line.lower().startswith("content-range:"):
            match = CONTENT_RANGE_RE.match(line.split(":", 1)[1].strip())
            if match:
                content_range = tuple(int(value) for value in match.groups())
    return status, content_range


def run_curl_chunk(url: str, start: int, end: int, output: Path, headers: Path) -> tuple[int, str]:
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "0",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--user-agent",
        "Mozilla/5.0",
        "--referer",
        "https://www.ebi.ac.uk/",
        "--range",
        f"{start}-{end}",
        "--dump-header",
        str(headers),
        "--output",
        str(output),
        url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stderr = (completed.stderr or "").strip()[-4000:]
    return completed.returncode, stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.chunk_bytes <= 0:
        parser.error("--chunk-bytes must be positive")
    if args.expected_bytes <= 0:
        parser.error("--expected-bytes must be positive")
    if not args.contract.is_file():
        parser.error(f"contract does not exist: {args.contract}")
    if not args.source_manifest.is_file():
        parser.error(f"source manifest does not exist: {args.source_manifest}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunk_root = args.output.parent / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    code_path = Path(__file__).resolve()
    total_chunks = (args.expected_bytes + args.chunk_bytes - 1) // args.chunk_bytes
    result: dict[str, object] = {
        "schema_version": "phase0-fastq-range-chunk-download-v1",
        "status": "CHUNKED_DOWNLOAD_IN_PROGRESS",
        "run_id": args.run_id,
        "started_at_utc": utc_now(),
        "url": args.url,
        "expected_bytes": args.expected_bytes,
        "chunk_bytes": args.chunk_bytes,
        "chunk_count": total_chunks,
        "chunk_root": str(chunk_root),
        "output": str(args.output),
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "source_manifest_path": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "code_path": str(code_path),
        "code_sha256": sha256(code_path),
        "command": list(sys.argv),
        "environment": {
            "cwd": os.getcwd(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "randomness": {"deterministic": True, "seed": None},
        "chunks": [],
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "stop_rule": "Do not install or accept the merged payload unless every range, size, hash, and gzip check passes.",
    }
    atomic_write_json(args.report, result)

    for index in range(total_chunks):
        start = index * args.chunk_bytes
        end = min(args.expected_bytes - 1, start + args.chunk_bytes - 1)
        expected_chunk_bytes = end - start + 1
        chunk_path = chunk_root / f"chunk_{index:05d}_{start}_{end}.bin"
        chunk_record: dict[str, object] = {
            "index": index,
            "start": start,
            "end": end,
            "expected_bytes": expected_chunk_bytes,
            "path": str(chunk_path),
            "attempts": [],
        }
        if chunk_path.is_file():
            if chunk_path.stat().st_size != expected_chunk_bytes:
                result["status"] = "CHUNKED_DOWNLOAD_BLOCKED_EXISTING_CHUNK_SIZE"
                result["failure"] = chunk_record
                result["finished_at_utc"] = utc_now()
                atomic_write_json(args.report, result)
                return 2
            chunk_record["status"] = "REUSED_EXISTING_EXACT_SIZE_CHUNK_HASH_PENDING"
            chunk_record["sha256"] = sha256(chunk_path)
            chunk_record["status"] = "REUSED_EXISTING_EXACT_SIZE_CHUNK"
            result["chunks"].append(chunk_record)
            atomic_write_json(args.report, result)
            continue

        passed = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            temporary = chunk_root / f".{chunk_path.name}.attempt{attempt}.{os.getpid()}.part"
            headers = chunk_root / f".{chunk_path.name}.attempt{attempt}.{os.getpid()}.headers"
            if temporary.exists():
                temporary.unlink()
            if headers.exists():
                headers.unlink()
            returncode, stderr = run_curl_chunk(args.url, start, end, temporary, headers)
            observed_bytes = temporary.stat().st_size if temporary.exists() else None
            status_code: int | None = None
            content_range: tuple[int, int, int] | None = None
            if headers.is_file():
                status_code, content_range = parse_last_response_headers(headers)
            attempt_record = {
                "attempt": attempt,
                "returncode": returncode,
                "observed_bytes": observed_bytes,
                "http_status": status_code,
                "content_range": list(content_range) if content_range else None,
                "stderr": stderr,
            }
            chunk_record["attempts"].append(attempt_record)
            valid = (
                returncode == 0
                and observed_bytes == expected_chunk_bytes
                and status_code == 206
                and content_range == (start, end, args.expected_bytes)
            )
            if valid:
                os.replace(temporary, chunk_path)
                chunk_record["sha256"] = sha256(chunk_path)
                chunk_record["status"] = "CHUNK_COMPLETE"
                passed = True
                break
            if temporary.exists():
                temporary.unlink()
            if headers.exists():
                headers.unlink()
            time.sleep(min(5 * attempt, 30))
        if not passed:
            result["status"] = "CHUNKED_DOWNLOAD_BLOCKED_CHUNK"
            result["failure"] = chunk_record
            result["chunks"].append(chunk_record)
            result["finished_at_utc"] = utc_now()
            atomic_write_json(args.report, result)
            return 2
        result["chunks"].append(chunk_record)
        atomic_write_json(args.report, result)

    assemble_tmp = args.output.parent / f".{args.output.name}.{args.run_id}.assemble.partial"
    if assemble_tmp.exists():
        parser.error(f"refusing to overwrite existing assembly temporary: {assemble_tmp}")
    digest = hashlib.sha256()
    total_written = 0
    with assemble_tmp.open("wb") as destination:
        for chunk_record in result["chunks"]:
            chunk_path = Path(str(chunk_record["path"]))
            with chunk_path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
                    total_written += len(block)
    if total_written != args.expected_bytes:
        result["status"] = "CHUNKED_MERGE_BLOCKED_SIZE"
        result["merged_bytes"] = total_written
        result["merged_sha256"] = digest.hexdigest()
        result["finished_at_utc"] = utc_now()
        atomic_write_json(args.report, result)
        return 2
    os.replace(assemble_tmp, args.output)
    result["merged_bytes"] = total_written
    result["merged_sha256"] = digest.hexdigest()
    gzip_check = subprocess.run(["gzip", "-t", str(args.output)], capture_output=True, text=True, check=False)
    result["gzip_test_returncode"] = gzip_check.returncode
    result["gzip_test_stderr"] = (gzip_check.stderr or "").strip()[-4000:]
    result["status"] = "CHUNKED_MERGE_COMPLETE_GZIP_OK" if gzip_check.returncode == 0 else "CHUNKED_MERGE_BLOCKED_GZIP"
    result["finished_at_utc"] = utc_now()
    atomic_write_json(args.report, result)
    print(json.dumps({"status": result["status"], "chunk_count": total_chunks, "merged_bytes": total_written, "merged_sha256": digest.hexdigest(), "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "CHUNKED_MERGE_COMPLETE_GZIP_OK" else 2


if __name__ == "__main__":
    sys.exit(main())
