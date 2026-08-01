#!/usr/bin/env python3
"""Create an append-only audit for a failed official Zenodo API probe."""

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
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--status-probe", required=True, type=Path)
    parser.add_argument("--headers-probe", required=True, type=Path)
    parser.add_argument("--stderr-probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    status_path = args.status_probe.resolve()
    headers_path = args.headers_probe.resolve()
    stderr_path = args.stderr_probe.resolve()
    output = args.output.resolve()
    for path in (status_path, headers_path, stderr_path):
        if not path.is_file():
            raise SystemExit(f"missing probe: {path}")
    status = status_path.read_text(encoding="utf-8", errors="replace").strip()
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
    if status != "7":
        raise SystemExit(f"expected curl exit 7, observed {status!r}")
    if "connection refused" not in stderr and "failed to connect" not in stderr:
        raise SystemExit("probe stderr does not establish a connection failure")

    payload = {
        "schema_version": "phase0-zenodo-api-route-audit-v1",
        "status": "BLOCKED_ZENODO_API_CONNECTION_REFUSED_HTTP_000",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "deenalattha_2026_dms",
        "zenodo_record_id": args.record_id,
        "route_role": "public_code_record_probe_not_processed_dms_payload",
        "status_probe": str(status_path),
        "status_probe_sha256": sha256(status_path),
        "headers_probe": str(headers_path),
        "headers_probe_sha256": sha256(headers_path),
        "stderr_probe": str(stderr_path),
        "stderr_probe_sha256": sha256(stderr_path),
        "http_code": 0,
        "curl_exit_code": 7,
        "payload_downloaded": False,
        "access_control_bypassed": False,
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "required_next_evidence": [
            "official processed-DMS payload route with verified content hash",
            "construct-level count/background/read-depth hierarchy",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_dump(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output), "output_sha256": sha256(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
