#!/usr/bin/env python3
"""Audit Denny XLSX OOXML structure without decoding cell values.

This dependency-free fallback is intentionally narrower than the semantic
workbook audit. It reads workbook relationships, sheet dimensions, row
elements, and cell-element counts only. It never decodes shared strings,
inline strings, numeric values, formulas, sequences, labels, or effects.
Therefore it is structural evidence only and cannot unlock Phase 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sheet_targets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationships = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rel_root
        if local_name(rel.tag) == "Relationship"
    }
    result: list[tuple[str, str]] = []
    for sheet in workbook_root.iter():
        if local_name(sheet.tag) != "sheet":
            continue
        name = sheet.attrib.get("name", "<unnamed>")
        relationship_id = sheet.attrib.get(f"{{{NS_REL}}}id")
        target = relationships.get(relationship_id or "")
        if not target:
            continue
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        result.append((name, path))
    return result


def audit_sheet(archive: ZipFile, path: str) -> dict[str, object]:
    row_count = 0
    nonempty_row_count = 0
    cell_element_count = 0
    first_nonempty_row = None
    dimension_ref = None
    with archive.open(path) as handle:
        for event, element in ET.iterparse(handle, events=("start", "end")):
            name = local_name(element.tag)
            if event == "start" and name == "dimension" and dimension_ref is None:
                dimension_ref = element.attrib.get("ref")
            if event != "end" or name != "row":
                continue
            row_count += 1
            cells = [child for child in element if local_name(child.tag) == "c"]
            cell_count = len(cells)
            cell_element_count += cell_count
            if cell_count:
                nonempty_row_count += 1
                if first_nonempty_row is None:
                    first_nonempty_row = element.attrib.get("r")
            element.clear()
    return {
        "worksheet_path": path,
        "dimension_ref": dimension_ref,
        "row_element_count": row_count,
        "nonempty_row_count": nonempty_row_count,
        "cell_element_count": cell_element_count,
        "first_nonempty_row": first_nonempty_row,
        "cell_values_decoded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        result = {
            "status": "BLOCKED_INPUT_MISSING",
            "input": str(args.input),
            "raw_cell_values_read": False,
            "raw_values_emitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 2

    observed = sha256(args.input)
    if observed != args.expected_sha256:
        result = {
            "status": "BLOCKED_HASH_MISMATCH",
            "input": str(args.input),
            "expected_sha256": args.expected_sha256,
            "observed_sha256": observed,
            "raw_cell_values_read": False,
            "raw_values_emitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 2

    try:
        with ZipFile(args.input) as archive:
            sheets = [
                {"name": name, **audit_sheet(archive, path)}
                for name, path in sheet_targets(archive)
            ]
    except (OSError, KeyError, ET.ParseError, ValueError) as exc:
        result = {
            "status": "BLOCKED_XLSX_OOXML_READ_ERROR",
            "input": str(args.input),
            "sha256": observed,
            "error_type": type(exc).__name__,
            "raw_cell_values_read": False,
            "raw_values_emitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "error_type": result["error_type"]}, sort_keys=True))
        return 2

    result = {
        "schema_version": "phase0-denny-xlsx-ooxml-structure-v1",
        "status": "STRUCTURAL_AUDIT_COMPLETE_SEMANTICS_UNRESOLVED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "sha256": observed,
        "size_bytes": args.input.stat().st_size,
        "sheet_count": len(sheets),
        "sheets": sheets,
        "cell_values_decoded": False,
        "raw_cell_values_read": False,
        "raw_values_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "semantic_gate_status": "UNRESOLVED_REQUIRES_DOCUMENTED_SCHEMA_AND_MANUAL_REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sha256": observed, "sheet_count": len(sheets)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
