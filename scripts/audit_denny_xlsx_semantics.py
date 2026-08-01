#!/usr/bin/env python3
"""Decode Denny workbook schema and aggregate semantic evidence safely.

This audit reads cell values internally but emits only headers, aggregate
statistics, keyword evidence, and masked text. It never emits raw sequences,
construct labels, or row-level effects. It is evidence for Phase 0 review,
not an accepted count mapping or primary-label table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
TARGET_COUNTS = {1687, 1713, 1636}
KEYWORDS = (
    "censor",
    "interpol",
    "raw",
    "replic",
    "bootstrap",
    "covar",
    "sublib",
    "sub-library",
    "variant",
    "count",
    "energy",
    "kcal",
    "mutation",
    "library",
    "9 bp",
    "11 bp",
    "9bp",
    "11bp",
    "limit",
    "saturat",
    "floor",
    "upper bound",
    "lower bound",
    "below",
    "above",
)
SEQUENCE_LIKE = re.compile(r"^[ACGUTNRYKMSWBDHVXacgutnrykmswbdhvx]{8,}$")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sheet_targets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationships = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels
        if local_name(rel.tag) == "Relationship"
    }
    result: list[tuple[str, str]] = []
    for sheet in workbook.iter():
        if local_name(sheet.tag) != "sheet":
            continue
        rel_id = sheet.attrib.get(f"{{{NS_REL}}}id", "")
        target = relationships.get(rel_id)
        if not target:
            continue
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        result.append((sheet.attrib.get("name", "<unnamed>"), path))
    return result


def shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root:
        if local_name(item.tag) != "si":
            continue
        values.append("".join(node.text or "" for node in item.iter() if local_name(node.tag) == "t"))
    return values


def cell_value(cell: ET.Element, strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        value = "".join(node.text or "" for node in cell.iter() if local_name(node.tag) == "t")
        return value or None
    value_node = next((child for child in cell if local_name(child.tag) == "v"), None)
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return strings[int(raw)]
        except (ValueError, IndexError):
            return None
    return raw


def column_name(ref: str) -> str:
    match = re.match(r"([A-Z]+)", ref)
    return match.group(1) if match else ref


def masked_text(value: str) -> str:
    clean = " ".join(value.split())
    if SEQUENCE_LIKE.fullmatch(clean):
        return "<MASKED_SEQUENCE_LIKE>"
    if len(clean) > 240:
        return clean[:237] + "..."
    return clean


def safe_semantic_text(value: str) -> str:
    clean = " ".join(value.split())
    lowered = clean.casefold()
    if any(keyword in lowered for keyword in KEYWORDS):
        return masked_text(clean)
    return "<REDACTED_NONSEMANTIC_TEXT>"


def number(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def audit_sheet(archive: ZipFile, path: str, strings: list[str]) -> dict[str, object]:
    columns: dict[str, dict[str, object]] = {}
    header_rows: list[dict[str, object]] = []
    row_count = 0
    nonempty_row_count = 0
    semantic_keyword_counts: Counter[str] = Counter()
    semantic_examples: dict[str, list[str]] = {keyword: [] for keyword in KEYWORDS}
    sentinel_value_counts: Counter[str] = Counter()
    sentinel_columns: dict[str, Counter[str]] = {}

    with archive.open(path) as handle:
        for event, row in ET.iterparse(handle, events=("end",)):
            if local_name(row.tag) != "row":
                continue
            row_count += 1
            row_values: dict[str, str] = {}
            for child in row:
                if local_name(child.tag) != "c":
                    continue
                ref = child.attrib.get("r", "")
                value = cell_value(child, strings)
                if not ref or value is None or value == "":
                    continue
                column = column_name(ref)
                row_values[column] = value
                info = columns.setdefault(
                    column,
                    {
                        "nonempty_count": 0,
                        "type_counts": Counter(),
                        "distinct_values": set(),
                        "distinct_truncated": False,
                        "numeric_distinct_values": set(),
                        "numeric_distinct_truncated": False,
                        "sentinel_below_count": 0,
                        "sentinel_above_count": 0,
                        "sentinel_equal_count": 0,
                        "numeric_sum": Decimal(0),
                        "numeric_count": 0,
                        "numeric_min": None,
                        "numeric_max": None,
                        "header_labels": [],
                    },
                )
                info["nonempty_count"] = int(info["nonempty_count"]) + 1
                parsed = number(value)
                if parsed is None:
                    info["type_counts"]["text"] += 1
                else:
                    info["type_counts"]["numeric"] += 1
                    if parsed < Decimal("-7.1"):
                        info["sentinel_below_count"] = int(info["sentinel_below_count"]) + 1
                    elif parsed > Decimal("-7.1"):
                        info["sentinel_above_count"] = int(info["sentinel_above_count"]) + 1
                    else:
                        info["sentinel_equal_count"] = int(info["sentinel_equal_count"]) + 1
                    for sentinel in ("-7.1", "7.1"):
                        if parsed == Decimal(sentinel):
                            sentinel_value_counts[sentinel] += 1
                            sentinel_columns.setdefault(column, Counter())[sentinel] += 1
                    if not info["numeric_distinct_truncated"]:
                        info["numeric_distinct_values"].add(parsed)
                        if len(info["numeric_distinct_values"]) > 20000:
                            info["numeric_distinct_truncated"] = True
                            info["numeric_distinct_values"] = set()
                    info["numeric_count"] = int(info["numeric_count"]) + 1
                    info["numeric_sum"] = info["numeric_sum"] + parsed
                    info["numeric_min"] = parsed if info["numeric_min"] is None else min(info["numeric_min"], parsed)
                    info["numeric_max"] = parsed if info["numeric_max"] is None else max(info["numeric_max"], parsed)
                distinct = info["distinct_values"]
                if not info["distinct_truncated"]:
                    distinct.add(value)
                    if len(distinct) > 20000:
                        info["distinct_truncated"] = True
                        info["distinct_values"] = set()

                lowered = value.casefold()
                for keyword in KEYWORDS:
                    if keyword in lowered:
                        semantic_keyword_counts[keyword] += 1
                        examples = semantic_examples[keyword]
                        safe = safe_semantic_text(value)
                        if safe not in examples and len(examples) < 5:
                            examples.append(safe)

            if row_values:
                nonempty_row_count += 1
                if len(header_rows) < 8:
                    header_rows.append(
                        {
                            "row": row.attrib.get("r"),
                            "labels": {column: safe_semantic_text(value) for column, value in sorted(row_values.items())},
                        }
                    )
            row.clear()

    for row in header_rows:
        for column, value in row["labels"].items():
            info = columns.get(column)
            if info is not None and value not in info["header_labels"] and len(info["header_labels"]) < 8:
                info["header_labels"].append(value)

    column_results: list[dict[str, object]] = []
    target_distinct_matches: list[dict[str, object]] = []
    count_like_summaries: list[dict[str, object]] = []
    for column in sorted(columns):
        info = columns[column]
        labels = [
            label
            for label in info["header_labels"]
            if label and not label.startswith("<REDACTED") and not label.startswith("<MASKED")
        ]
        joined = " ".join(labels).casefold()
        distinct_count = "lower_bound_20001" if info["distinct_truncated"] else len(info["distinct_values"])
        numeric_distinct_count = (
            "lower_bound_20001"
            if info["numeric_distinct_truncated"]
            else len(info["numeric_distinct_values"])
        )
        result: dict[str, object] = {
            "column": column,
            "header_labels": info["header_labels"],
            "nonempty_count": info["nonempty_count"],
            "type_counts": dict(info["type_counts"]),
            "distinct_count": distinct_count,
            "numeric_distinct_count": numeric_distinct_count,
        }
        if info["numeric_count"]:
            result["numeric_summary"] = {
                "count": info["numeric_count"],
                "sum": str(info["numeric_sum"]),
                "min": str(info["numeric_min"]),
                "max": str(info["numeric_max"]),
            }
        if info["sentinel_equal_count"]:
            below = int(info["sentinel_below_count"])
            above = int(info["sentinel_above_count"])
            if below == 0 and above > 0:
                direction_consistency = "CONSISTENT_WITH_LOWER_BOUND_FLOOR_NOT_PROOF"
            elif above == 0 and below > 0:
                direction_consistency = "CONSISTENT_WITH_UPPER_BOUND_CAP_NOT_PROOF"
            else:
                direction_consistency = "NOT_DIRECTIONAL_FROM_AGGREGATES"
            result["sentinel_context_minus_7_1"] = {
                "equal_count": int(info["sentinel_equal_count"]),
                "below_count": below,
                "above_count": above,
                "direction_consistency": direction_consistency,
            }
        if isinstance(distinct_count, int) and distinct_count in TARGET_COUNTS:
            target_distinct_matches.append(
                {
                    "column": column,
                    "header_labels": info["header_labels"],
                    "distinct_count": distinct_count,
                    "basis": "all_nonempty_cell_values",
                }
            )
        if isinstance(numeric_distinct_count, int) and numeric_distinct_count in TARGET_COUNTS:
            target_distinct_matches.append(
                {
                    "column": column,
                    "header_labels": info["header_labels"],
                    "distinct_count": numeric_distinct_count,
                    "basis": "numeric_cell_values_only",
                }
            )
        if any(token in joined for token in ("count", "variant", "sublib", "library", "mutation")):
            result["count_like_candidate"] = True
            if "numeric_summary" in result:
                count_like_summaries.append(
                    {
                        "column": column,
                        "header_labels": info["header_labels"],
                        "numeric_summary": result["numeric_summary"],
                    }
                )
        column_results.append(result)

    return {
        "worksheet_path": path,
        "row_element_count": row_count,
        "nonempty_row_count": nonempty_row_count,
        "header_rows_first_eight": header_rows,
        "columns": column_results,
        "target_distinct_count_matches": target_distinct_matches,
        "count_like_numeric_summaries": count_like_summaries,
        "semantic_keyword_counts": dict(sorted(semantic_keyword_counts.items())),
        "semantic_keyword_examples": {
            keyword: examples for keyword, examples in semantic_examples.items() if examples
        },
        "sentinel_value_counts": dict(sorted(sentinel_value_counts.items())),
        "sentinel_columns": {
            column: dict(sorted(counts.items())) for column, counts in sorted(sentinel_columns.items())
        },
        "censor_direction_status": "NOT_ESTABLISHED_SENTINEL_PRESENCE_IS_NOT_DIRECTIONAL_EVIDENCE",
        "raw_values_emitted": False,
        "sequence_values_emitted": False,
        "primary_labels_admitted": False,
    }


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema_version": "phase0-denny-xlsx-semantic-evidence-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "raw_values_emitted": False,
        "sequence_values_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    if not args.input.is_file():
        result.update({"status": "BLOCKED_INPUT_MISSING", "observed_sha256": None})
        write_result(args.output, result)
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return 2

    observed = sha256(args.input)
    result.update({"expected_sha256": args.expected_sha256, "observed_sha256": observed, "size_bytes": args.input.stat().st_size})
    if observed != args.expected_sha256:
        result["status"] = "BLOCKED_HASH_MISMATCH"
        write_result(args.output, result)
        print(json.dumps({"status": result["status"], "observed_sha256": observed}, sort_keys=True))
        return 2

    try:
        with ZipFile(args.input) as archive:
            strings = shared_strings(archive)
            sheets = [
                {"name": name, **audit_sheet(archive, path, strings)}
                for name, path in sheet_targets(archive)
            ]
    except (OSError, KeyError, ET.ParseError, ValueError, IndexError) as exc:
        result.update({"status": "BLOCKED_XLSX_SEMANTIC_READ_ERROR", "error_type": type(exc).__name__})
        write_result(args.output, result)
        print(json.dumps({"status": result["status"], "error_type": result["error_type"]}, sort_keys=True))
        return 2

    result.update(
        {
            "status": "SEMANTIC_EVIDENCE_EXTRACTED_REQUIRES_MANUAL_ACCEPTANCE",
            "shared_string_count": len(strings),
            "sheet_count": len(sheets),
            "sheets": sheets,
            "semantic_acceptance": "NOT_ACCEPTED_NO_EXACT_COUNT_CENSOR_OR_MAPPING_GATE_UNLOCK",
        }
    )
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "sheet_count": len(sheets),
                "shared_string_count": len(strings),
                "raw_values_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
