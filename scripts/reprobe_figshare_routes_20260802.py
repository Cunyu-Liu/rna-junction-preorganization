#!/usr/bin/env python3
"""Re-probe the recorded public processed-DMS routes without downloading payloads."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_last_headers(path: Path) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("HTTP/"):
            parts = line.split()
            current = {"status_line": line, "status_code": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None}
            blocks.append(current)
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.lower()
        if normalized in {"content-type", "content-length", "location"}:
            current[normalized.replace("-", "_")] = value.strip()
    return blocks[-1] if blocks else {"status_line": None, "status_code": None}


def read_urls(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t")[-1] != "url":
        raise ValueError(f"unexpected route TSV header: {path}")
    routes: list[tuple[str, str]] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 6 or not fields[5].strip():
            raise ValueError(f"unexpected route TSV row: {line!r}")
        routes.append((fields[0], fields[5].strip()))
    if not routes:
        raise ValueError(f"no routes found: {path}")
    return routes


def probe(url: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="figshare_route_reprobe_") as temporary_dir:
        headers = Path(temporary_dir) / "headers.txt"
        completed = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--location",
                "--head",
                "--retry",
                "0",
                "--connect-timeout",
                "30",
                "--max-time",
                "120",
                "--user-agent",
                "Mozilla/5.0",
                "--referer",
                "https://www.ebi.ac.uk/",
                "--dump-header",
                str(headers),
                "--output",
                "/dev/null",
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        observed = parse_last_headers(headers) if headers.is_file() else {"status_line": None, "status_code": None}
        return {
            "url": url,
            "curl_exit": completed.returncode,
            "status_line": observed.get("status_line"),
            "http_code": observed.get("status_code"),
            "content_type": observed.get("content_type"),
            "content_length": observed.get("content_length"),
            "location": observed.get("location"),
            "stderr": (completed.stderr or "").strip()[-4000:],
            "payload_downloaded": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-route-tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-tsv", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if shutil.which("curl") is None:
        parser.error("curl is required")
    if not args.contract.is_file():
        parser.error(f"missing contract: {args.contract}")
    if not args.source_route_tsv.is_file():
        parser.error(f"missing route TSV: {args.source_route_tsv}")
    if args.output.exists() or args.report_tsv.exists():
        parser.error("refusing to overwrite an existing audit artifact")
    observed_contract_sha256 = sha256(args.contract)
    if observed_contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {observed_contract_sha256}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_tsv.parent.mkdir(parents=True, exist_ok=True)
    routes = read_urls(args.source_route_tsv)
    results = [{"route": route, **probe(url)} for route, url in routes]
    successful_head = [item for item in results if isinstance(item.get("http_code"), int) and 200 <= item["http_code"] < 300]
    status = "ROUTE_REPROBE_PUBLIC_HEAD_OK_PAYLOAD_NOT_DOWNLOADED" if successful_head else "ROUTE_REPROBE_BLOCKED_NO_2XX"
    payload = {
        "schema_version": "phase0-figshare-route-reprobe-v1",
        "status": status,
        "run_id": args.run_id,
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": observed_contract_sha256,
        "source_route_tsv": str(args.source_route_tsv.resolve()),
        "source_route_tsv_sha256": sha256(args.source_route_tsv),
        "routes": results,
        "payload_downloaded": False,
        "access_control_bypassed": False,
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "next_action": "Use the existing 128 MiB range downloader only after a verified 2xx route and expected payload size are established." if successful_head else "Preserve the route failure and continue searching only through official public routes; do not bypass access controls.",
    }
    atomic_write(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tsv_lines = ["run_id\troute\thttp_code\tcurl_exit\tcontent_type\tcontent_length\tpayload_downloaded\turl"]
    for item in results:
        tsv_lines.append("\t".join([args.run_id] + [str(item.get(key, "")) for key in ("route", "http_code", "curl_exit", "content_type", "content_length", "payload_downloaded", "url")]))
    atomic_write(args.report_tsv, "\n".join(tsv_lines) + "\n")
    print(json.dumps({"status": status, "route_count": len(results), "successful_head_count": len(successful_head), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
