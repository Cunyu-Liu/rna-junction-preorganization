#!/usr/bin/env python3
"""Audit additional official Figshare OAI-PMH metadata formats only."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
MAX_METADATA_BYTES = 10 * 1024 * 1024
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-oai-format-audit-v1"
OAI_BASE = "https://api.figshare.com/v2/oai"
OAI_FORMATS = ("mets", "qdc", "rdf", "cerif")


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


def parse_headers(path: Path) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="iso-8859-1", errors="replace").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("HTTP/"):
            parts = line.split()
            current = {
                "status_line": line,
                "status_code": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
            }
            blocks.append(current)
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace("-", "_")
        if normalized in {"content_type", "content_length", "location", "etag", "last_modified"}:
            current[normalized] = value.strip()
    return blocks[-1] if blocks else {"status_line": None, "status_code": None}


def xml_summary(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"format_status": "EMPTY", "selected_elements": []}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return {"format_status": "NOT_XML", "error": str(exc)[:1000], "selected_elements": []}
    selected: list[dict[str, Any]] = []
    interesting_tags = {
        "identifier",
        "relatedIdentifier",
        "alternateIdentifier",
        "title",
        "size",
        "format",
        "file",
        "FLocat",
        "fileName",
        "resource",
        "url",
        "location",
    }
    interesting_attributes = {
        "ID",
        "SIZE",
        "MIMETYPE",
        "USE",
        "CHECKSUM",
        "CHECKSUMTYPE",
        "LOCTYPE",
        "href",
        "xlink:href",
        "relatedIdentifierType",
        "relationType",
        "alternateIdentifierType",
        "identifierType",
    }
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        text = (element.text or "").strip()
        attrs: dict[str, str] = {}
        for key, value in element.attrib.items():
            short_key = key.rsplit("}", 1)[-1]
            if short_key in interesting_attributes or key in interesting_attributes:
                attrs[short_key] = value[:1000]
        if tag in interesting_tags and (text or attrs):
            selected.append({"tag": tag, "text": text[:1000], "attributes": attrs})
        if len(selected) >= 200:
            break
    return {"format_status": "PARSED_XML", "selected_elements": selected}


def probe(format_name: str, raw_root: Path, referer: str, user_agent: str) -> dict[str, Any]:
    route_id = f"figshare_oai_{format_name}_getrecord"
    route_root = raw_root / route_id
    route_root.mkdir()
    query = urlencode({"verb": "GetRecord", "metadataPrefix": format_name, "identifier": "oai:figshare.com:article/27880434"})
    url = f"{OAI_BASE}?{query}"
    headers_path = route_root / "headers.txt"
    body_path = route_root / "metadata.body"
    stderr_path = route_root / "stderr.txt"
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
        "120",
        "--max-filesize",
        str(MAX_METADATA_BYTES),
        "--user-agent",
        user_agent,
        "--referer",
        referer,
        "--dump-header",
        str(headers_path),
        "--output",
        str(body_path),
        url,
    ]
    started_at = utc_now()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    finished_at = utc_now()
    headers = parse_headers(headers_path) if headers_path.is_file() else {"status_line": None, "status_code": None}
    body_bytes = body_path.stat().st_size if body_path.is_file() else 0
    return {
        "route_id": route_id,
        "metadata_prefix": format_name,
        "url": url,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "curl_exit": completed.returncode,
        "http_status": headers.get("status_code"),
        "status_line": headers.get("status_line"),
        "content_type": headers.get("content_type"),
        "content_length": headers.get("content_length"),
        "location": headers.get("location"),
        "body_bytes": body_bytes,
        "body_sha256": sha256_file(body_path) if body_bytes else None,
        "xml_summary": xml_summary(body_path),
        "headers_path": str(headers_path),
        "body_path": str(body_path),
        "stderr_path": str(stderr_path),
        "stderr": stderr_path.read_text(encoding="utf-8", errors="replace").strip()[:4000],
        "payload_downloaded": False,
        "processed_payload_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--referer", default="https://figshare.com/")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    args.output = args.output.resolve()
    args.raw_root = args.raw_root.resolve()
    if not args.contract.is_file():
        parser.error(f"missing contract: {args.contract}")
    contract_sha256 = sha256_file(args.contract)
    if contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {contract_sha256}")
    if args.output.exists() or args.raw_root.exists():
        parser.error("refusing to overwrite an existing audit artifact or raw-root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    results = [probe(format_name, args.raw_root, args.referer, args.user_agent) for format_name in OAI_FORMATS]
    successful = [item for item in results if isinstance(item.get("http_status"), int) and 200 <= item["http_status"] < 300]
    status = "FIGSHARE_OAI_FORMAT_METADATA_AVAILABLE" if successful else "BLOCKED_NO_2XX_FIGSHARE_OAI_FORMAT_ROUTE"
    report = {
        "schema_version": "phase0-figshare-oai-format-audit-v1",
        "status": status,
        "run_id": args.run_id,
        "checked_at_utc": utc_now(),
        "contract_path": str(args.contract),
        "contract_sha256": contract_sha256,
        "article_id": 27880434,
        "oai_formats": list(OAI_FORMATS),
        "routes": results,
        "successful_route_count": len(successful),
        "payload_downloaded": False,
        "processed_payload_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "next_action": "Treat any extracted file URL, size, or checksum as provenance metadata only; verify a separate exact 128 MiB range route before acquisition.",
        "audit_boundary": "Additional OAI metadata formats only; no processed payload, raw reads, labels, matching, model, or scientific conclusion is admitted.",
    }
    atomic_json_write(args.output, report)
    print(json.dumps({"status": status, "route_count": len(results), "successful_route_count": len(successful), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
