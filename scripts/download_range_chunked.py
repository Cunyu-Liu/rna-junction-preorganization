#!/usr/bin/env python3
"""Fail-closed 128 MiB HTTP range acquisition with auditable evidence.

This runner is deliberately generic: it does not infer file semantics and does
not admit any scientific labels.  It only accepts exact HTTP 206 responses,
exact Content-Range values, exact chunk sizes, and an optional post-merge gzip
check.  A failed run leaves verified chunks and metadata in place but never
installs a final output.
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
import sys
import time
from typing import Any
from urllib.parse import urlparse


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
MAX_ATTEMPTS = 8
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-range-acquisition-v1"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_headers(path: Path) -> tuple[int | None, dict[str, str]]:
    """Parse the final response block from curl's dump-header output."""

    try:
        lines = path.read_text(encoding="iso-8859-1").splitlines()
    except FileNotFoundError:
        return None, {}
    blocks: list[tuple[int | None, dict[str, str]]] = []
    status: int | None = None
    headers: dict[str, str] = {}
    for line in lines:
        if line.startswith("HTTP/"):
            if status is not None or headers:
                blocks.append((status, headers))
            status = None
            headers = {}
            parts = line.split(maxsplit=2)
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
            continue
        if not line.strip():
            if status is not None or headers:
                blocks.append((status, headers))
            status = None
            headers = {}
            continue
        if ":" in line and status is not None:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if status is not None or headers:
        blocks.append((status, headers))
    return blocks[-1] if blocks else (None, {})


def content_range_value(headers: dict[str, str]) -> tuple[int, int, int] | None:
    match = CONTENT_RANGE_RE.fullmatch(headers.get("content-range", ""))
    if not match:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def curl_command(args: argparse.Namespace, start: int, end: int, headers: Path, body: Path) -> list[str]:
    return [
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
        args.user_agent,
        "--referer",
        args.referer,
        "--range",
        f"{start}-{end}",
        "--dump-header",
        str(headers),
        "--output",
        str(body),
        args.url,
    ]


def failure_report(
    args: argparse.Namespace,
    *,
    status: str,
    started_at: str,
    finished_at: str,
    chunk_root: Path,
    chunks: list[dict[str, Any]],
    failure: dict[str, Any],
    output_sha256: str | None = None,
    output_installed: bool = False,
    postcheck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract_sha = sha256_file(args.contract)
    source_manifest_sha = sha256_file(args.source_manifest)
    return {
        "schema": "phase0-range-object-download-v1",
        "status": status,
        "run_id": args.run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "url": args.url,
        "referer": args.referer,
        "user_agent": args.user_agent,
        "output": str(args.output),
        "report": str(args.report),
        "chunk_root": str(chunk_root),
        "expected_bytes": args.expected_bytes,
        "chunk_bytes": args.chunk_bytes,
        "chunk_count": (args.expected_bytes + args.chunk_bytes - 1) // args.chunk_bytes,
        "contract_sha256": contract_sha,
        "source_manifest_sha256": source_manifest_sha,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "chunks": chunks,
        "failure": failure,
        "merged_sha256": output_sha256,
        "output_installed": output_installed,
        "postcheck": postcheck or {"requested": args.postcheck, "status": "not_run"},
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "stop_rule": "Stop on any non-exact range response, missing primary payload, failed integrity check, or unavailable official route; do not unlock matching or modeling.",
        "audit_boundary": "Engineering acquisition evidence only; this artifact is not a payload identity audit, matching result, model result, or scientific conclusion.",
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.expected_bytes <= 0:
        raise ValueError("--expected-bytes must be positive")
    if args.chunk_bytes <= 0:
        raise ValueError("--chunk-bytes must be positive")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("--run-id contains unsupported characters")
    if args.output == args.report:
        raise ValueError("--output and --report must be different paths")
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--url must be an absolute http(s) URL")
    if sha256_file(args.contract) != CONTRACT_SHA256:
        raise ValueError("contract SHA256 does not match the frozen 1.1 contract")
    if not args.source_manifest.is_file():
        raise ValueError(f"source manifest does not exist: {args.source_manifest}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.report}")


def run_chunk(args: argparse.Namespace, chunk_root: Path, index: int, start: int, end: int) -> dict[str, Any]:
    chunk_path = chunk_root / f"chunk-{index:06d}.bin"
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_root = chunk_root / f"chunk-{index:06d}.attempt-{attempt:02d}"
        attempt_root.mkdir()
        headers_path = attempt_root / "headers.txt"
        body_path = attempt_root / "body.partial"
        stderr_path = attempt_root / "stderr.txt"
        command = curl_command(args, start, end, headers_path, body_path)
        started_at = utc_now()
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
        finished_at = utc_now()
        response_status, headers = parse_headers(headers_path)
        observed_range = content_range_value(headers)
        observed_bytes = body_path.stat().st_size if body_path.exists() else 0
        error_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        attempt_record = {
            "attempt": attempt,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "returncode": completed.returncode,
            "http_status": response_status,
            "content_range": headers.get("content-range"),
            "observed_range": list(observed_range) if observed_range else None,
            "observed_bytes": observed_bytes,
            "expected_bytes": end - start + 1,
            "stderr": error_text[:4000],
            "headers_path": str(headers_path),
            "stderr_path": str(stderr_path),
        }
        attempts.append(attempt_record)
        exact = (
            completed.returncode == 0
            and response_status == 206
            and observed_range == (start, end, args.expected_bytes)
            and observed_bytes == end - start + 1
        )
        if exact:
            os.replace(body_path, chunk_path)
            return {
                "index": index,
                "start": start,
                "end": end,
                "bytes": observed_bytes,
                "sha256": sha256_file(chunk_path),
                "status": "VERIFIED",
                "chunk_path": str(chunk_path),
                "attempts": attempts,
            }
        # 4xx, an HTTP 206 contract mismatch, or any response with a body that
        # is not the requested range is a deterministic stop, not a retry loop.
        retryable = response_status is None or response_status == 408 or response_status == 429 or response_status >= 500
        if not retryable:
            break
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(2 * attempt, 10))
    return {
        "index": index,
        "start": start,
        "end": end,
        "bytes": 0,
        "sha256": None,
        "status": "BLOCKED",
        "attempts": attempts,
    }


def postcheck_gzip(path: Path) -> dict[str, Any]:
    command = ["gzip", "-t", str(path)]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    return {
        "requested": "gzip",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip()[:4000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--referer", default="https://figshare.com/")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--postcheck", choices=("none", "gzip"), default="none")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    args.contract = args.contract.resolve()
    args.source_manifest = args.source_manifest.resolve()
    started_at = utc_now()
    chunk_root = args.output.parent / f"{args.output.name}.chunks.{args.run_id}"
    chunks: list[dict[str, Any]] = []
    try:
        validate_args(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if chunk_root.exists():
            raise FileExistsError(f"refusing to reuse existing chunk root: {chunk_root}")
        chunk_root.mkdir()
        for index, start in enumerate(range(0, args.expected_bytes, args.chunk_bytes)):
            end = min(start + args.chunk_bytes, args.expected_bytes) - 1
            result = run_chunk(args, chunk_root, index, start, end)
            chunks.append(result)
            if result["status"] != "VERIFIED":
                report = failure_report(
                    args,
                    status="CHUNKED_DOWNLOAD_BLOCKED_CHUNK",
                    started_at=started_at,
                    finished_at=utc_now(),
                    chunk_root=chunk_root,
                    chunks=chunks,
                    failure={"reason": "non_exact_range_response", "chunk_index": index},
                )
                atomic_json_write(args.report, report)
                return 2

        assembly = args.output.parent / f".{args.output.name}.assemble.{args.run_id}.partial"
        if assembly.exists():
            raise FileExistsError(f"refusing to overwrite assembly temporary: {assembly}")
        digest = hashlib.sha256()
        total = 0
        with assembly.open("wb") as output_handle:
            for result in chunks:
                chunk_path = Path(result["chunk_path"])
                with chunk_path.open("rb") as chunk_handle:
                    while block := chunk_handle.read(8 * 1024 * 1024):
                        output_handle.write(block)
                        digest.update(block)
                        total += len(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if total != args.expected_bytes:
            report = failure_report(
                args,
                status="CHUNKED_MERGE_BLOCKED_SIZE_MISMATCH",
                started_at=started_at,
                finished_at=utc_now(),
                chunk_root=chunk_root,
                chunks=chunks,
                failure={"reason": "merged_size_mismatch", "observed_bytes": total},
                output_sha256=digest.hexdigest(),
            )
            atomic_json_write(args.report, report)
            return 2
        postcheck = {"requested": "none", "status": "NOT_REQUESTED"}
        if args.postcheck == "gzip":
            postcheck = postcheck_gzip(assembly)
            if postcheck["status"] != "PASS":
                report = failure_report(
                    args,
                    status="CHUNKED_MERGE_BLOCKED_POSTCHECK",
                    started_at=started_at,
                    finished_at=utc_now(),
                    chunk_root=chunk_root,
                    chunks=chunks,
                    failure={"reason": "postcheck_failed"},
                    output_sha256=digest.hexdigest(),
                    postcheck=postcheck,
                )
                atomic_json_write(args.report, report)
                return 2
        os.replace(assembly, args.output)
        report = failure_report(
            args,
            status="CHUNKED_MERGE_COMPLETE",
            started_at=started_at,
            finished_at=utc_now(),
            chunk_root=chunk_root,
            chunks=chunks,
            failure={},
            output_sha256=digest.hexdigest(),
            output_installed=True,
            postcheck=postcheck,
        )
        atomic_json_write(args.report, report)
        return 0
    except Exception as exc:  # preserve a machine-readable fail-closed record
        failure = {"reason": "runner_exception", "exception_type": type(exc).__name__, "message": str(exc)[:4000]}
        try:
            report = failure_report(
                args,
                status="CHUNKED_DOWNLOAD_BLOCKED_CONFIGURATION_OR_RUNTIME",
                started_at=started_at,
                finished_at=utc_now(),
                chunk_root=chunk_root,
                chunks=chunks,
                failure=failure,
            )
            if not args.report.exists():
                atomic_json_write(args.report, report)
        except Exception:
            pass
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
