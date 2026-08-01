#!/usr/bin/env python3
"""Audit official Figshare provenance routes without downloading payload data.

The fixed routes below are metadata/landing-page routes only.  This script does
not request the processed payload file, does not bypass access controls, and
does not admit labels or scientific evidence.  Any file identifiers returned by
metadata are recorded as candidates for a separate, explicitly verified range
acquisition step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
MAX_METADATA_BYTES = 10 * 1024 * 1024
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-provenance-audit-v1"

ROUTES: tuple[dict[str, str], ...] = (
    {
        "route_id": "figshare_doi_resolver_head",
        "url": "https://doi.org/10.6084/m9.figshare.27880434",
        "mode": "head",
    },
    {
        "route_id": "figshare_article_landing_head",
        "url": "https://figshare.com/articles/dataset/27880434",
        "mode": "head",
    },
    {
        "route_id": "figshare_article_api_metadata",
        "url": "https://api.figshare.com/v2/articles/27880434",
        "mode": "metadata",
    },
    {
        "route_id": "figshare_article_files_metadata",
        "url": "https://api.figshare.com/v2/articles/27880434/files",
        "mode": "metadata",
    },
    {
        "route_id": "figshare_article_versions_metadata",
        "url": "https://api.figshare.com/v2/articles/27880434/versions",
        "mode": "metadata",
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


def parse_last_headers(path: Path) -> dict[str, Any]:
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
        key = key.strip().lower().replace("-", "_")
        if key in {"content_type", "content_length", "location", "etag", "last_modified"}:
            current[key] = value.strip()
    return blocks[-1] if blocks else {"status_line": None, "status_code": None}


def metadata_summary(body_path: Path, route_id: str) -> dict[str, Any]:
    if not body_path.is_file() or body_path.stat().st_size == 0:
        return {"json_status": "EMPTY", "candidate_files": []}
    try:
        value = json.loads(body_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"json_status": "NOT_JSON", "error": str(exc)[:1000], "candidate_files": []}

    candidates: list[dict[str, Any]] = []
    values: list[Any] = []
    if isinstance(value, dict) and isinstance(value.get("files"), list):
        values.extend(value["files"])
    elif isinstance(value, list):
        values.extend(value)
    for item in values:
        if not isinstance(item, dict):
            continue
        candidate = {
            "id": item.get("id"),
            "name": item.get("name"),
            "size": item.get("size"),
            "md5": item.get("md5"),
            "download_url": item.get("download_url"),
        }
        if any(value is not None for value in candidate.values()):
            candidates.append(candidate)
    summary: dict[str, Any] = {
        "json_status": "PARSED",
        "candidate_files": candidates,
        "route_id": route_id,
    }
    if isinstance(value, dict):
        for key in ("id", "title", "version", "doi", "url", "published_date", "modified_date"):
            if key in value:
                summary[key] = value[key]
    return summary


def probe(route: dict[str, str], raw_root: Path, referer: str, user_agent: str) -> dict[str, Any]:
    route_id = route["route_id"]
    route_root = raw_root / route_id
    route_root.mkdir()
    headers_path = route_root / "headers.txt"
    body_path = route_root / "metadata.json"
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
    ]
    if route["mode"] == "head":
        command.append("--head")
    command.append(route["url"])
    started_at = utc_now()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    finished_at = utc_now()
    headers = parse_last_headers(headers_path) if headers_path.is_file() else {"status_line": None, "status_code": None}
    body_bytes = body_path.stat().st_size if body_path.is_file() else 0
    return {
        "route_id": route_id,
        "mode": route["mode"],
        "url": route["url"],
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
        "body_sha256": sha256_file(body_path) if body_path.is_file() and body_bytes else None,
        "metadata_summary": metadata_summary(body_path, route_id) if route["mode"] == "metadata" else None,
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
    observed_contract_sha256 = sha256_file(args.contract)
    if observed_contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {observed_contract_sha256}")
    if args.output.exists() or args.raw_root.exists():
        parser.error("refusing to overwrite an existing audit artifact or raw-root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    results = [probe(route, args.raw_root, args.referer, args.user_agent) for route in ROUTES]
    successful = [item for item in results if isinstance(item.get("http_status"), int) and 200 <= item["http_status"] < 300]
    successful_metadata = [item for item in successful if item["mode"] == "metadata"]
    if successful_metadata:
        status = "PROVENANCE_METADATA_ROUTES_OK_PAYLOAD_NOT_DOWNLOADED"
        next_action = "Review candidate file IDs, expected sizes, and checksums; only then start a separately audited 128 MiB range acquisition."
    elif successful:
        status = "PROVENANCE_HEAD_ONLY_METADATA_BLOCKED"
        next_action = "Preserve landing-page evidence; do not infer a payload route from a HEAD-only response."
    else:
        status = "BLOCKED_NO_2XX_PROVENANCE_ROUTE"
        next_action = "Preserve the provenance block and continue only through newly identified official public routes; do not bypass access controls."
    report = {
        "schema_version": "phase0-figshare-provenance-route-audit-v1",
        "status": status,
        "run_id": args.run_id,
        "checked_at_utc": utc_now(),
        "contract_path": str(args.contract),
        "contract_sha256": observed_contract_sha256,
        "article_id": 27880434,
        "doi": "10.6084/m9.figshare.27880434",
        "routes": results,
        "successful_route_count": len(successful),
        "successful_metadata_route_count": len(successful_metadata),
        "payload_downloaded": False,
        "processed_payload_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "next_action": next_action,
        "audit_boundary": "Provenance and metadata routing evidence only; no processed payload, raw reads, labels, matching, model, or scientific conclusion is admitted.",
    }
    atomic_json_write(args.output, report)
    print(json.dumps({"status": status, "route_count": len(results), "successful_metadata_route_count": len(successful_metadata), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
