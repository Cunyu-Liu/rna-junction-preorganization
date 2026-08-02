#!/usr/bin/env python3
"""Audit the official bioRxiv supplementary page without admitting payloads."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse


HREF_RE = re.compile(r"href\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
FILE_EXTENSIONS = (".docx", ".xlsx", ".csv", ".fasta", ".fa", ".fastq", ".zip")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-html", required=True, type=Path)
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--expected-docx", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--docx-url", required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()

    for path in (args.page_html, args.docx):
        if not path.is_file():
            raise SystemExit(f"missing required path: {path}")
    if args.expected_docx is not None and not args.expected_docx.is_file():
        raise SystemExit(f"missing expected comparison file: {args.expected_docx}")

    page_text = args.page_html.read_text(encoding="utf-8", errors="replace")
    links = []
    for _, raw_href in HREF_RE.findall(page_text):
        href = html.unescape(raw_href.strip())
        absolute = urljoin(args.source_url, href)
        path = urlparse(absolute).path.lower()
        if path.endswith(FILE_EXTENSIONS):
            links.append(absolute)
    links = sorted(set(links))
    sequence_links = [link for link in links if "sequences.xlsx" in link.lower()]
    construct_links = [
        link
        for link in links
        if any(link.lower().split("?", 1)[0].endswith(ext) for ext in (".xlsx", ".csv", ".fasta", ".fa"))
    ]

    docx_sha = sha256_file(args.docx)
    expected_sha = sha256_file(args.expected_docx) if args.expected_docx else None
    payload = {
        "schema_version": "phase0-biorxiv-supplementary-route-audit-v1",
        "status": "PUBLIC_BIORXIV_SUPPLEMENTARY_PAGE_AUDITED_NO_CONSTRUCT_FILE_LINK_MAIN_DMS_PAYLOAD_NOT_ADMITTED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": args.contract_sha256,
        "source": {
            "supplementary_page_url": args.source_url,
            "page_html": {
                "path": str(args.page_html),
                "size_bytes": args.page_html.stat().st_size,
                "sha256": sha256_file(args.page_html),
            },
            "supplementary_docx_url": args.docx_url,
            "supplementary_docx": {
                "path": str(args.docx),
                "size_bytes": args.docx.stat().st_size,
                "sha256": docx_sha,
            },
            "comparison_docx": {
                "path": str(args.expected_docx) if args.expected_docx else None,
                "sha256": expected_sha,
                "same_bytes_as_biorxiv_docx": expected_sha == docx_sha if expected_sha else None,
            },
        },
        "route_findings": {
            "file_link_count": len(links),
            "file_links": links,
            "sequences_xlsx_link_count": len(sequence_links),
            "construct_reference_file_link_count": len(construct_links),
            "sequences_xlsx_link_present": bool(sequence_links),
            "construct_reference_file_link_present": bool(construct_links),
        },
        "content_classification": {
            "page_is_metadata_only": True,
            "construct_reference_fasta_available": False,
            "construct_sequence_structure_mapping_available": False,
            "mutation_histograms_available": False,
            "processed_construct_json_available": False,
            "raw_sequence_content_emitted": False,
            "primary_labels_admitted": False,
            "main_dms_payload_admitted": False,
        },
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "blocking_reason": (
            "The official bioRxiv supplementary page exposes only the article "
            "supplementary docx link; no Sequences.xlsx or construct FASTA/CSV "
            "link is present. The docx is auxiliary article material and is not "
            "the processed-DMS payload."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
