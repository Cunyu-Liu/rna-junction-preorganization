#!/usr/bin/env python3
"""Audit an article supplementary package without admitting it as assay payload.

The contract distinguishes article supplements from the processed-DMS payload
used for construct-level reconstruction.  This audit records package
provenance and schema-level facts while deliberately not emitting sequences,
labels, or reactivity values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docx_schema(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    paragraph_count = 0
    for paragraph in root.findall(".//w:p", WORD_NS):
        text = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)
        ).strip()
        paragraph_count += bool(text)

    tables = []
    for table_index, table in enumerate(root.findall(".//w:tbl", WORD_NS), 1):
        rows = []
        for row in table.findall("./w:tr", WORD_NS):
            cells = [
                "".join(node.text or "" for node in cell.findall(".//w:t", WORD_NS))
                .strip()
                for cell in row.findall("./w:tc", WORD_NS)
            ]
            rows.append(cells)
        header = rows[0] if rows else []
        tables.append(
            {
                "table_index": table_index,
                "row_count": len(rows),
                "max_column_count": max((len(row) for row in rows), default=0),
                "header": header,
            }
        )

    return {
        "nonempty_paragraph_count": paragraph_count,
        "table_count": len(tables),
        "tables": tables,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--pmcid", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-tree", required=True, type=Path)
    parser.add_argument("--source-tree-commit", required=True)
    args = parser.parse_args()

    for path in (args.package, args.docx, args.source_tree):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    if not zipfile.is_zipfile(args.package):
        raise SystemExit(f"supplementary package is not a ZIP archive: {args.package}")

    with zipfile.ZipFile(args.package) as archive:
        members = [
            {"name": info.filename, "size_bytes": info.file_size}
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]

    schema = docx_schema(args.docx)
    payload = {
        "schema_version": "phase0-pmc-supplementary-audit-v1",
        "status": "PUBLIC_SUPPLEMENTARY_SCHEMA_AUDITED_MAIN_DMS_PAYLOAD_NOT_ADMITTED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": args.contract_sha256,
        "source": {
            "pmcid": args.pmcid,
            "supplementary_url": args.source_url,
            "package": {
                "path": str(args.package),
                "size_bytes": args.package.stat().st_size,
                "sha256": sha256_file(args.package),
                "members": members,
            },
            "docx": {
                "path": str(args.docx),
                "size_bytes": args.docx.stat().st_size,
                "sha256": sha256_file(args.docx),
                "schema": schema,
            },
            "official_processing_source_tree": {
                "path": str(args.source_tree),
                "commit": args.source_tree_commit,
            },
        },
        "content_classification": {
            "scope": "article_figures_and_summary_tables",
            "construct_reference_fasta_available": False,
            "construct_sequence_structure_mapping_available": False,
            "mutation_histograms_available": False,
            "processed_construct_json_available": False,
            "background_read_depth_hierarchy_available": False,
            "raw_sequence_content_emitted": False,
            "primary_labels_admitted": False,
            "is_substitute_for_figshare_processed_payload": False,
        },
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "blocking_reason": (
            "The public supplementary package contains article figures and "
            "summary/example tables, but not the construct-level FASTA, "
            "sequence/structure mapping, mutation histograms, background/read "
            "depth hierarchy, or processed construct JSON required by the "
            "contract."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
