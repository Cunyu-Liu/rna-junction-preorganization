#!/usr/bin/env python3
"""Create an append-only audit for a failed Figshare file-metadata probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_headers(path: Path) -> dict[str, object]:
    status_code = None
    content_type = None
    content_length = None
    for line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            status_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.lower().strip()
            if key == "content-type":
                content_type = value.strip()
            elif key == "content-length":
                content_length = value.strip()
    return {"status_code": status_code, "content_type": content_type, "content_length": content_length}


def atomic_dump(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--status-probe", required=True, type=Path)
    parser.add_argument("--headers-probe", required=True, type=Path)
    parser.add_argument("--body-probe", required=True, type=Path)
    parser.add_argument("--stderr-probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    status_path = args.status_probe.resolve()
    headers_path = args.headers_probe.resolve()
    body_path = args.body_probe.resolve()
    stderr_path = args.stderr_probe.resolve()
    output = args.output.resolve()
    for path in (status_path, headers_path, body_path, stderr_path):
        if not path.is_file():
            raise SystemExit(f"missing probe: {path}")
    status_lines = status_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not status_lines or status_lines[0].strip() != "403":
        raise SystemExit(f"expected HTTP 403 in status probe: {status_lines[:2]}")
    if body_path.stat().st_size != 0:
        raise SystemExit("metadata probe body is non-empty")
    header_data = parse_headers(headers_path)
    if header_data.get("status_code") != 403:
        raise SystemExit(f"expected final HTTP 403, observed {header_data}")
    curl_exit = next((line.split("=", 1)[1] for line in status_lines if line.startswith("curl_exit=")), None)
    if curl_exit != "22":
        raise SystemExit(f"expected curl exit 22, observed {curl_exit!r}")
    payload = {
        "schema_version": "phase0-figshare-file-metadata-probe-audit-v1",
        "status": "BLOCKED_HTTP_403_FILE_METADATA_PROBE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "observed_http_status": header_data.get("status_code"),
        "content_type": header_data.get("content_type"),
        "content_length_header": header_data.get("content_length"),
        "observed_body_bytes": body_path.stat().st_size,
        "curl_exit_code": int(curl_exit),
        "status_probe": str(status_path),
        "status_probe_sha256": sha256(status_path),
        "headers_probe": str(headers_path),
        "headers_probe_sha256": sha256(headers_path),
        "body_probe": str(body_path),
        "body_probe_sha256": sha256(body_path),
        "stderr_probe": str(stderr_path),
        "stderr_probe_sha256": sha256(stderr_path),
        "payload_downloaded": False,
        "access_control_bypassed": False,
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_dump(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output), "output_sha256": sha256(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
