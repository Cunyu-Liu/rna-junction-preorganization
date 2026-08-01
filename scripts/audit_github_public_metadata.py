#!/usr/bin/env python3
"""Audit official GitHub release/tree metadata without downloading repository payloads."""

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
DEFAULT_USER_AGENT = "rna-junction-preorganization/phase0-github-metadata-audit-v1"
REPO = "YesselmanLabPublications/2025_char_3d_struct_features"
ROUTES = (
    ("releases", f"https://api.github.com/repos/{REPO}/releases?per_page=100"),
    ("tree_v1_1_0", f"https://api.github.com/repos/{REPO}/git/trees/v1.1.0?recursive=1"),
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
        if normalized in {"content_type", "content_length", "etag", "last_modified", "x_ratelimit_remaining"}:
            current[normalized] = value.strip()
    return blocks[-1] if blocks else {"status_line": None, "status_code": None}


def summarize_json(route_id: str, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {"format_status": "EMPTY"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"format_status": "NOT_JSON", "error": str(exc)[:1000]}
    if route_id == "releases":
        releases = value if isinstance(value, list) else []
        assets = [asset for release in releases if isinstance(release, dict) for asset in release.get("assets", []) if isinstance(asset, dict)]
        return {
            "format_status": "PARSED_JSON",
            "release_count": len(releases),
            "release_tags": [release.get("tag_name") for release in releases if isinstance(release, dict)],
            "release_assets": [
                {
                    "release_tag": release.get("tag_name"),
                    "name": asset.get("name"),
                    "size": asset.get("size"),
                    "browser_download_url": asset.get("browser_download_url"),
                }
                for release in releases
                if isinstance(release, dict)
                for asset in release.get("assets", [])
                if isinstance(asset, dict)
            ],
            "release_asset_count": len(assets),
        }
    tree = value if isinstance(value, dict) else {}
    entries = tree.get("tree", []) if isinstance(tree.get("tree"), list) else []
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
    interesting = [
        path
        for path in paths
        if any(token in path.lower() for token in ("data", "zip", "json", "csv", "dms", "fastq", "supp"))
    ]
    payload_like = [
        path
        for path in paths
        if path.lower().endswith((".fastq", ".fastq.gz", ".zip", ".tar", ".tar.gz", ".h5", ".hdf5", ".parquet"))
    ]
    return {
        "format_status": "PARSED_JSON",
        "tag": "v1.1.0",
        "truncated": tree.get("truncated"),
        "tree_entry_count": len(entries),
        "interesting_paths": interesting[:300],
        "payload_like_paths": payload_like[:100],
        "payload_like_path_count": len(payload_like),
    }


def probe(route_id: str, url: str, raw_root: Path, referer: str, user_agent: str) -> dict[str, Any]:
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
        "--header",
        "Accept: application/vnd.github+json",
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
        "url": url,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "curl_exit": completed.returncode,
        "http_status": headers.get("status_code"),
        "status_line": headers.get("status_line"),
        "content_type": headers.get("content_type"),
        "content_length": headers.get("content_length"),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last_modified"),
        "x_ratelimit_remaining": headers.get("x_ratelimit_remaining"),
        "body_bytes": body_bytes,
        "body_sha256": sha256_file(body_path) if body_bytes else None,
        "json_summary": summarize_json(route_id, body_path),
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
    parser.add_argument("--referer", default="https://github.com/")
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
    results = [probe(route_id, url, args.raw_root, args.referer, args.user_agent) for route_id, url in ROUTES]
    successful = [item for item in results if isinstance(item.get("http_status"), int) and 200 <= item["http_status"] < 300]
    tree = next((item for item in results if item["route_id"] == "tree_v1_1_0"), {})
    tree_summary = tree.get("json_summary", {}) if isinstance(tree, dict) else {}
    status = "GITHUB_PUBLIC_METADATA_AVAILABLE" if len(successful) == len(ROUTES) and tree_summary.get("truncated") is False else "BLOCKED_GITHUB_PUBLIC_METADATA_ROUTE"
    report = {
        "schema_version": "phase0-github-public-metadata-audit-v1",
        "status": status,
        "run_id": args.run_id,
        "checked_at_utc": utc_now(),
        "contract_path": str(args.contract),
        "contract_sha256": contract_sha256,
        "repository": REPO,
        "routes": results,
        "successful_route_count": len(successful),
        "release_asset_count": next((item.get("json_summary", {}).get("release_asset_count") for item in results if item["route_id"] == "releases"), None),
        "tree_payload_like_path_count": tree_summary.get("payload_like_path_count"),
        "tree_payload_like_paths": tree_summary.get("payload_like_paths", []),
        "metadata_only": True,
        "payload_downloaded": False,
        "processed_payload_admitted": False,
        "raw_sequence_content_emitted": False,
        "interpretation_boundary": "GitHub release/tree metadata inventories public code-repository metadata only; absence of a payload-like path is not proof that the external official payload is absent.",
    }
    atomic_json_write(args.output, report)
    print(json.dumps({"status": status, "route_count": len(results), "successful_route_count": len(successful), "release_asset_count": report["release_asset_count"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
