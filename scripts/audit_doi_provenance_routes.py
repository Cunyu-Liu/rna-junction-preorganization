#!/usr/bin/env python3
"""Audit DOI and OAI metadata routes without downloading scientific payloads."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
MAX_METADATA_BYTES = 10 * 1024 * 1024
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-doi-oai-provenance-v1"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")

ROUTES: tuple[dict[str, str], ...] = (
    {
        "route_id": "datacite_doi_metadata",
        "url": "https://api.datacite.org/dois/10.6084/m9.figshare.27880434",
        "format": "json",
    },
    {
        "route_id": "figshare_oai_datacite_getrecord",
        "url": "https://api.figshare.com/v2/oai?verb=GetRecord&metadataPrefix=oai_datacite&identifier=oai:figshare.com:article/27880434",
        "format": "xml",
    },
    {
        "route_id": "figshare_oai_dc_getrecord",
        "url": "https://api.figshare.com/v2/oai?verb=GetRecord&metadataPrefix=oai_dc&identifier=oai:figshare.com:article/27880434",
        "format": "xml",
    },
)


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
            pieces = line.split()
            current = {
                "status_line": line,
                "status_code": int(pieces[1]) if len(pieces) > 1 and pieces[1].isdigit() else None,
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


def json_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"format_status": "NOT_JSON", "error": str(exc)[:1000]}
    attributes: dict[str, Any] = {}
    if isinstance(value, dict) and isinstance(value.get("data"), dict):
        data = value["data"]
        if isinstance(data.get("attributes"), dict):
            attributes = data["attributes"]
    summary: dict[str, Any] = {"format_status": "PARSED_JSON"}
    for key in (
        "doi",
        "url",
        "publisher",
        "publisherIdentifier",
        "publicationYear",
        "resourceType",
        "types",
        "sizes",
        "alternateIdentifiers",
        "relatedIdentifiers",
        "contentUrl",
    ):
        if key in attributes:
            summary[key] = attributes[key]
    return summary


def xml_summary(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return {"format_status": "NOT_XML", "error": str(exc)[:1000]}
    values: dict[str, list[str]] = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        text = (element.text or "").strip()
        if text and tag in {"identifier", "resource", "title", "size", "format", "relatedIdentifier", "alternateIdentifier"}:
            values.setdefault(tag, []).append(text[:1000])
    return {"format_status": "PARSED_XML", "fields": values}


def probe(route: dict[str, str], raw_root: Path, referer: str, user_agent: str) -> dict[str, Any]:
    route_root = raw_root / route["route_id"]
    route_root.mkdir()
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
        route["url"],
    ]
    started_at = utc_now()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    finished_at = utc_now()
    headers = parse_headers(headers_path) if headers_path.is_file() else {"status_line": None, "status_code": None}
    body_bytes = body_path.stat().st_size if body_path.is_file() else 0
    summary = None
    if body_bytes:
        summary = json_summary(body_path) if route["format"] == "json" else xml_summary(body_path)
    return {
        "route_id": route["route_id"],
        "url": route["url"],
        "expected_format": route["format"],
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "curl_exit": completed.returncode,
        "http_status": headers.get("status_code"),
        "status_line": headers.get("status_line"),
        "content_type": headers.get("content_type"),
        "content_length": headers.get("content_length"),
        "location": headers.get("location"),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last_modified"),
        "body_bytes": body_bytes,
        "body_sha256": sha256_file(body_path) if body_bytes else None,
        "metadata_summary": summary,
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
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("run id contains unsupported characters")
    if not args.contract.is_file():
        parser.error(f"missing contract: {args.contract}")
    observed_contract_sha256 = sha256_file(args.contract)
    if observed_contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {observed_contract_sha256}")
    if args.output.exists() or args.raw_root.exists():
        parser.error("refusing to overwrite an existing audit artifact or raw-root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    results = [probe(route, args.raw_root, args.referer, args.user_agent) for route in ROUTES]
    successful = [item for item in results if isinstance(item.get("http_status"), int) and 200 <= item["http_status"] < 300]
    status = "DOI_OAI_PROVENANCE_METADATA_AVAILABLE" if successful else "BLOCKED_NO_2XX_DOI_OAI_PROVENANCE_ROUTE"
    report = {
        "schema_version": "phase0-doi-oai-provenance-route-audit-v1",
        "status": status,
        "run_id": args.run_id,
        "checked_at_utc": utc_now(),
        "contract_path": str(args.contract),
        "contract_sha256": observed_contract_sha256,
        "figshare_article_id": 27880434,
        "doi": "10.6084/m9.figshare.27880434",
        "routes": results,
        "successful_route_count": len(successful),
        "payload_downloaded": False,
        "processed_payload_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "next_action": "Review only metadata summaries; a 2xx metadata response is not a payload admission and cannot unlock matching or modeling.",
        "audit_boundary": "DOI/OAI provenance metadata evidence only; no processed payload, raw reads, labels, matching, model, or scientific conclusion is admitted.",
    }
    atomic_json_write(args.output, report)
    print(json.dumps({"status": status, "route_count": len(results), "successful_route_count": len(successful), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
