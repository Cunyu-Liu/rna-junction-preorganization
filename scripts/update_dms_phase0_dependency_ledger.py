#!/usr/bin/env python3
"""Derive a new fail-closed DMS dependency ledger from a route re-probe.

The previous ledger and route audit remain immutable.  This utility only
creates a new ledger when the new official route probe still has no 2xx
response and no payload was downloaded.
"""

from __future__ import annotations

import argparse
import copy
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


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def atomic_dump(path: Path, value: dict) -> None:
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
    parser.add_argument("--previous-ledger", required=True, type=Path)
    parser.add_argument("--route-audit", required=True, type=Path)
    parser.add_argument("--range-audit", type=Path)
    parser.add_argument("--metadata-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    previous = args.previous_ledger.resolve()
    route_path = args.route_audit.resolve()
    range_path = args.range_audit.resolve() if args.range_audit else None
    metadata_path = args.metadata_audit.resolve() if args.metadata_audit else None
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing ledger: {output}")
    if not previous.is_file() or not route_path.is_file():
        raise SystemExit("previous ledger and route audit must exist")

    old = load_json(previous)
    route = load_json(route_path)
    if old.get("status") != "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE":
        raise SystemExit("previous ledger is not fail-closed")
    if route.get("status") != "ROUTE_REPROBE_BLOCKED_NO_2XX":
        raise SystemExit("route audit is not a blocked no-2xx result")
    if route.get("payload_downloaded") is not False:
        raise SystemExit("route audit does not prove payload_downloaded=false")
    if route.get("access_control_bypassed") is not False:
        raise SystemExit("route audit does not prove access_control_bypassed=false")
    range_audit = None
    if range_path is not None:
        range_audit = load_json(range_path)
        if range_audit.get("status") != "BLOCKED_HTTP_403_RANGE_PROBE" or range_audit.get("payload_downloaded") is not False:
            raise SystemExit("range audit is not a blocked no-payload result")
        if range_audit.get("observed_body_bytes") != 0:
            raise SystemExit("range audit does not prove zero observed body bytes")
    metadata_audit = None
    if metadata_path is not None:
        metadata_audit = load_json(metadata_path)
        if metadata_audit.get("status") != "BLOCKED_HTTP_403_FILE_METADATA_PROBE" or metadata_audit.get("payload_downloaded") is not False:
            raise SystemExit("metadata audit is not a blocked no-payload result")
        if metadata_audit.get("observed_body_bytes") != 0:
            raise SystemExit("metadata audit does not prove zero observed body bytes")

    ledger = copy.deepcopy(old)
    ledger["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["derived_from_ledger"] = str(previous)
    ledger["derived_from_ledger_sha256"] = sha256(previous)
    ledger["latest_route_reprobe"] = file_record(route_path)
    ledger["latest_route_reprobe_status"] = route.get("status")
    ledger["latest_route_reprobe_payload_downloaded"] = False
    ledger["latest_route_reprobe_access_control_bypassed"] = False

    route_evidence = ledger.setdefault("route_evidence", {})
    prior_latest = route_evidence.get("latest")
    prior_status = route_evidence.get("latest_status")
    if prior_latest is not None:
        route_evidence["previous"] = prior_latest
        route_evidence["previous_status"] = prior_status
    route_evidence["latest"] = file_record(route_path)
    route_evidence["latest_status"] = route.get("status")
    route_evidence["latest_route_count"] = len(route.get("routes", [])) if isinstance(route.get("routes"), list) else None
    route_evidence["latest_all_http_codes"] = sorted(
        {item.get("http_code") for item in route.get("routes", []) if isinstance(item, dict)}
    )
    route_evidence["latest_payload_downloaded"] = False
    route_evidence["access_control_bypassed"] = False
    if range_path is not None and range_audit is not None:
        route_evidence["latest_range_probe"] = file_record(range_path)
        route_evidence["latest_range_probe_status"] = range_audit.get("status")
        route_evidence["latest_range_probe_observed_bytes"] = range_audit.get("observed_body_bytes")
        ledger["latest_range_probe"] = file_record(range_path)
        ledger["latest_range_probe_status"] = range_audit.get("status")
        ledger["latest_range_probe_payload_downloaded"] = False
    if metadata_path is not None and metadata_audit is not None:
        route_evidence["latest_metadata_probe"] = file_record(metadata_path)
        route_evidence["latest_metadata_probe_status"] = metadata_audit.get("status")
        route_evidence["latest_metadata_probe_observed_bytes"] = metadata_audit.get("observed_body_bytes")
        ledger["latest_metadata_probe"] = file_record(metadata_path)
        ledger["latest_metadata_probe_status"] = metadata_audit.get("status")
        ledger["latest_metadata_probe_payload_downloaded"] = False

    for requirement in ledger.get("required_evidence", []):
        if isinstance(requirement, dict) and requirement.get("requirement") == "official processed-DMS payload or verified public route":
            requirement["status"] = "BLOCKED_HTTP_403_NO_2XX"
            requirement["evidence"] = str(route_path)
            requirement["scientific_gate_effect"] = "NO_PHASE_0_PASS"
            if range_path is not None and range_audit is not None:
                requirement["status"] = "BLOCKED_HTTP_403_RANGE_PROBE"
                requirement["evidence"] = str(range_path)

    ledger["status"] = "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE"
    ledger["primary_labels_admitted"] = False
    ledger["modeling_authorized"] = False
    ledger["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_dump(output, ledger)
    checked = load_json(output)
    print(json.dumps({
        "status": checked["status"],
        "latest_route_status": checked["latest_route_reprobe_status"],
        "output": str(output),
        "output_sha256": sha256(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
