#!/usr/bin/env python3
"""Probe one exact HTTP byte range and preserve auditable raw evidence.

This is a route-verification step, not a complete payload acquisition.  A
successful 206 response stores at most the requested range under the supplied
artifact root, records its hash, and still marks the processed payload as not
admitted.  A non-exact response is retained as fail-closed evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-range-route-probe-v1"
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_headers(path: Path) -> tuple[int | None, dict[str, str]]:
    blocks: list[tuple[int | None, dict[str, str]]] = []
    status: int | None = None
    headers: dict[str, str] = {}
    for raw_line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("HTTP/"):
            if status is not None or headers:
                blocks.append((status, headers))
            parts = line.split()
            status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            headers = {}
            continue
        if not line.strip():
            if status is not None or headers:
                blocks.append((status, headers))
            status = None
            headers = {}
            continue
        if status is not None and ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if status is not None or headers:
        blocks.append((status, headers))
    return blocks[-1] if blocks else (None, {})


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    match = CONTENT_RANGE_RE.fullmatch(value or "")
    if not match:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=DEFAULT_CHUNK_BYTES - 1)
    parser.add_argument("--referer", default="https://figshare.com/")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args()
    args.report = args.report.resolve()
    args.raw_root = args.raw_root.resolve()
    args.contract = args.contract.resolve()
    args.source_manifest = args.source_manifest.resolve()
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("run id contains unsupported characters")
    if args.start < 0 or args.end < args.start:
        parser.error("invalid byte range")
    if not args.contract.is_file() or not args.source_manifest.is_file():
        parser.error("contract and source manifest must exist")
    contract_sha256 = sha256_file(args.contract)
    if contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {contract_sha256}")
    if args.report.exists() or args.raw_root.exists():
        parser.error("refusing to overwrite an existing report or raw-root")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    requested_bytes = args.end - args.start + 1
    headers_path = args.raw_root / "headers.txt"
    body_path = args.raw_root / "range.partial"
    stderr_path = args.raw_root / "stderr.txt"
    status_path = args.raw_root / "status.txt"
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
        "--max-filesize",
        str(requested_bytes),
        "--user-agent",
        args.user_agent,
        "--referer",
        args.referer,
        "--range",
        f"{args.start}-{args.end}",
        "--dump-header",
        str(headers_path),
        "--output",
        str(body_path),
        args.url,
    ]
    started_at = utc_now()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    finished_at = utc_now()
    response_status, headers = parse_headers(headers_path) if headers_path.is_file() else (None, {})
    observed_range = parse_content_range(headers.get("content-range"))
    observed_bytes = body_path.stat().st_size if body_path.is_file() else 0
    body_sha256 = sha256_file(body_path) if body_path.is_file() and observed_bytes else None
    status_path.write_text(f"http_status={response_status}\ncurl_exit={completed.returncode}\n", encoding="utf-8")
    exact = (
        completed.returncode == 0
        and response_status == 206
        and observed_range is not None
        and observed_range[0] == args.start
        and observed_range[1] == args.end
        and observed_bytes == requested_bytes
    )
    if exact:
        status = "RANGE_PROBE_206_EXACT"
    elif response_status == 403:
        status = "BLOCKED_HTTP_403_RANGE_PROBE"
    else:
        status = "BLOCKED_NONEXACT_RANGE_RESPONSE"
    report = {
        "schema_version": "phase0-range-route-probe-v1",
        "status": status,
        "run_id": args.run_id,
        "created_at_utc": finished_at,
        "started_at_utc": started_at,
        "url": args.url,
        "request_referer": args.referer,
        "contract_path": str(args.contract),
        "contract_sha256": contract_sha256,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "requested_range": {"start": args.start, "end": args.end, "requested_bytes": requested_bytes},
        "observed_http_status": response_status,
        "observed_content_range": headers.get("content-range"),
        "observed_content_range_tuple": list(observed_range) if observed_range else None,
        "observed_content_length_header": headers.get("content-length"),
        "observed_total_bytes": observed_range[2] if observed_range else None,
        "observed_body_bytes": observed_bytes,
        "curl_exit_code": completed.returncode,
        "body_sha256": body_sha256,
        "status_probe": str(status_path),
        "headers_probe": str(headers_path),
        "body_probe": str(body_path),
        "stderr_probe": str(stderr_path),
        "status_probe_sha256": sha256_file(status_path),
        "headers_probe_sha256": sha256_file(headers_path) if headers_path.is_file() else None,
        "body_probe_sha256": sha256_file(body_path) if body_path.is_file() else None,
        "stderr_probe_sha256": sha256_file(stderr_path),
        "range_probe_body_downloaded": observed_bytes > 0,
        "payload_complete": False,
        "payload_downloaded": observed_bytes > 0,
        "processed_payload_admitted": False,
        "access_control_bypassed": False,
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "next_action": "Use a full 128 MiB chunked acquisition only after exact 206 Content-Range and total size are verified; never treat this partial range as a complete payload or scientific evidence.",
    }
    atomic_json_write(args.report, report)
    print(json.dumps({"status": status, "http_status": response_status, "observed_total_bytes": report["observed_total_bytes"], "observed_body_bytes": observed_bytes, "output": str(args.report)}, sort_keys=True))
    return 0 if exact or response_status == 403 else 2


if __name__ == "__main__":
    raise SystemExit(main())
