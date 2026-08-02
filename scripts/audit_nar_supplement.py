#!/usr/bin/env python3
"""Audit the final NAR supplementary package without emitting raw sequences.

This is intentionally stdlib-only.  The S2 workbook is Strict OOXML rather
than the older Transitional OOXML namespace, so a small reader is safer here
than silently depending on an unavailable spreadsheet package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import posixpath
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_nodes(element: ET.Element, wanted: str = "t") -> str:
    return "".join((node.text or "") for node in element.iter() if local_name(node.tag) == wanted)


def relationship_target(rel_path: str, target: str) -> str:
    base = posixpath.dirname(rel_path)
    resolved = posixpath.normpath(posixpath.join(base, target))
    return resolved.lstrip("/")


def col_number(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref.upper())
    if not letters:
        return 0
    value = 0
    for ch in letters.group(1):
        value = value * 26 + ord(ch) - 64
    return value


def split_dimension(ref: str) -> tuple[str, str]:
    if ":" in ref:
        return tuple(ref.split(":", 1))  # type: ignore[return-value]
    return ref, ref


def parse_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [text_nodes(si) for si in root.iter() if local_name(si.tag) == "si"]


def parse_xlsx(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        workbook_path = "xl/workbook.xml"
        rels_path = "xl/_rels/workbook.xml.rels"
        wb = ET.fromstring(zf.read(workbook_path))
        rels = ET.fromstring(zf.read(rels_path))
        relmap: dict[str, str] = {}
        for rel in rels:
            if local_name(rel.tag) == "Relationship":
                relmap[rel.attrib.get("Id", "")] = relationship_target(workbook_path, rel.attrib.get("Target", ""))
        shared = parse_shared_strings(zf)
        sheets: list[dict[str, Any]] = []
        for sheet in wb.iter():
            if local_name(sheet.tag) != "sheet":
                continue
            rid = next((key for key in sheet.attrib if key.endswith("}id")), None)
            target = relmap.get(sheet.attrib.get(rid, "") if rid else "")
            entry: dict[str, Any] = {
                "name": sheet.attrib.get("name"),
                "sheet_id": sheet.attrib.get("sheetId"),
                "target": target,
                "present": bool(target and target in names),
            }
            if not target or target not in names:
                sheets.append(entry)
                continue
            root = ET.fromstring(zf.read(target))
            dimension = next((el.attrib.get("ref") for el in root.iter() if local_name(el.tag) == "dimension"), None)
            rows: list[tuple[int, dict[int, str], int]] = []
            formula_count = 0
            for row in root.iter():
                if local_name(row.tag) != "row":
                    continue
                try:
                    row_num = int(row.attrib.get("r", len(rows) + 1))
                except ValueError:
                    row_num = len(rows) + 1
                values: dict[int, str] = {}
                cell_count = 0
                for cell in row:
                    if local_name(cell.tag) != "c":
                        continue
                    cell_count += 1
                    ref = cell.attrib.get("r", "")
                    column = col_number(ref)
                    cell_type = cell.attrib.get("t", "")
                    if any(local_name(child.tag) == "f" for child in cell):
                        formula_count += 1
                    if cell_type == "inlineStr":
                        value = text_nodes(cell)
                    else:
                        v = next((child for child in cell if local_name(child.tag) == "v"), None)
                        raw = (v.text or "") if v is not None else ""
                        if cell_type == "s" and raw != "":
                            try:
                                value = shared[int(raw)]
                            except (ValueError, IndexError):
                                value = raw
                        elif cell_type == "b":
                            value = "TRUE" if raw == "1" else "FALSE"
                        else:
                            value = raw
                    if column:
                        values[column] = value
                rows.append((row_num, values, cell_count))
            entry.update({
                "dimension": dimension,
                "rows": rows,
                "row_count_xml": len(rows),
                "nonempty_row_count": sum(bool(values) for _, values, _ in rows),
                "formula_count": formula_count,
                "shared_string_count": len(shared),
            })
            sheets.append(entry)
        return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path), "sheets": sheets}


def canonical_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def median_or_none(values: list[int]) -> float | None:
    return statistics.median(values) if values else None


def value_stats(values: list[str]) -> dict[str, Any]:
    nonempty = [v for v in values if v != ""]
    norm = [re.sub(r"\s+", "", v).upper() for v in nonempty]
    seq_like = [v for v in norm if len(v) >= 10 and re.fullmatch(r"[ACGTUN]+", v)]
    dot_like = [v for v in norm if len(v) >= 5 and re.fullmatch(r"[().\[\]{}<>]+", v)]
    accession_like = [v for v in nonempty if re.search(r"\b(?:SRR|ERR|DRR|PRJNA|SAMN)\d+\b", v, re.I)]
    lengths = [len(v) for v in nonempty]
    value_hash = hashlib.sha256("\n".join(nonempty).encode("utf-8")).hexdigest()
    return {
        "nonempty_count": len(nonempty),
        "unique_count": len(set(nonempty)),
        "duplicate_count": len(nonempty) - len(set(nonempty)),
        "length_min": min(lengths) if lengths else None,
        "length_median": median_or_none(lengths),
        "length_max": max(lengths) if lengths else None,
        "numeric_count": sum(numeric(v) for v in nonempty),
        "sequence_like_count": len(seq_like),
        "dot_bracket_like_count": len(dot_like),
        "accession_like_count": len(accession_like),
        "value_sha256_in_row_order": value_hash,
    }


def sequence_like(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value).upper()
    return len(normalized) >= 10 and bool(re.fullmatch(r"[ACGTUN]+", normalized))


def audit_sheet(sheet: dict[str, Any]) -> dict[str, Any]:
    rows = sheet.get("rows", [])
    nonempty_rows = [(num, vals) for num, vals, _ in rows if vals]
    if not nonempty_rows:
        return {k: v for k, v in sheet.items() if k != "rows"}
    # Supplementary Table S2 has a seven-row explanatory preamble and an
    # example sequence before its actual header.  Select a header-like row by
    # field-name evidence instead of assuming row 1.
    def header_score(item: tuple[int, dict[int, str]]) -> tuple[int, int, int]:
        num, values = item
        normalized = [canonical_header(v) for v in values.values() if v and not sequence_like(v)]
        field_hits = sum(bool(re.search(r"(?:sequence|seq|construct|structure|primer|name|id|barcode|sample|condition|filter|position)", v)) for v in normalized)
        return (field_hits, len(normalized), -num)

    header_row_num, header_values = max(nonempty_rows, key=header_score)
    max_col = max(max(values.keys(), default=0) for _, values in nonempty_rows)
    headers = {col: (header_values.get(col, "") or f"column_{col}") for col in range(1, max_col + 1)}
    data = [(num, values) for num, values in nonempty_rows if num != header_row_num]
    col_audit: list[dict[str, Any]] = []
    for col in range(1, max_col + 1):
        header = headers[col]
        values = [values.get(col, "") for _, values in data]
        record = {"column": col, "header": header, "header_normalized": canonical_header(header)}
        record.update(value_stats(values))
        col_audit.append(record)
    candidate_groups = {
        "identifier": r"(?:^|_)(?:id|name|construct|seq|sequence|library|oligo)(?:$|_)",
        "sequence": r"(?:sequence|seq|oligo|construct)",
        "structure": r"(?:structure|dot|bracket|secondary|ss)",
        "motif_or_position": r"(?:motif|position|start|end|boundary|index|junction|stem|loop)",
        "mapping": r"(?:sample|barcode|batch|library|run|sra|accession|fastq|primer|condition|replicate)",
        "filter_or_reason": r"(?:filter|exclude|include|retain|remove|reason|pass|fail|qc|status)",
    }
    candidates = {
        group: [record["header"] for record in col_audit if re.search(pattern, record["header_normalized"], re.I)]
        for group, pattern in candidate_groups.items()
    }
    row_id_hash = hashlib.sha256()
    for _, values in data:
        row_id_hash.update("\x1f".join(values.get(col, "") for col in range(1, max_col + 1)).encode("utf-8"))
        row_id_hash.update(b"\n")
    post_header_data = [(num, values) for num, values in data if num > header_row_num]
    result = {k: v for k, v in sheet.items() if k != "rows"}
    result.update({
        "header_row_number": header_row_num,
        "preamble_row_numbers": [num for num, _ in nonempty_rows if num < header_row_num],
        "headers": [headers[col] for col in range(1, max_col + 1)],
        "data_row_count_after_header": len(data),
        "data_rows_with_any_value": sum(bool(values) for _, values in data),
        "data_row_count_with_all_columns_nonempty": sum(all(values.get(col, "") != "" for col in range(1, max_col + 1)) for _, values in data),
        "post_header_data_row_count": len(post_header_data),
        "row_order_sha256": row_id_hash.hexdigest(),
        "column_audit": col_audit,
        "candidate_fields_by_group": candidates,
        "first_data_row_cell_presence": {headers[col]: bool(data[0][1].get(col, "")) for col in range(1, max_col + 1)} if data else {},
    })
    id_columns = [col for col in range(1, max_col + 1) if re.search(r"(?:^|_)(?:sequence_name|construct_id|sequence_id|id|name)(?:$|_)", canonical_header(headers[col]))]
    if not id_columns:
        id_columns = [col for col in range(1, max_col + 1) if re.search(r"(?:construct|sequence).*name", canonical_header(headers[col]))]
    id_col = id_columns[0] if id_columns else None
    construct_records: list[tuple[int, dict[int, str], int]] = []
    if id_col is not None:
        construct_records = [
            (num, values, int(match.group(1)))
            for num, values in data
            for match in [re.fullmatch(r"construct(\d+)", values.get(id_col, ""), re.I)]
            if match
        ]
    sequence_columns = [col for col in range(1, max_col + 1) if re.search(r"(?:^|_)(?:dna|rna|dna_sequence|rna_sequence|sequence|seq|oligo)(?:$|_)", canonical_header(headers[col])) and not re.search(r"(?:structure|secondary)", canonical_header(headers[col]))]
    structure_columns = [col for col in range(1, max_col + 1) if re.search(r"(?:structure|dot|bracket|secondary|ss)", canonical_header(headers[col]))]
    construct_numbers = sorted(record[2] for record in construct_records)
    expected_numbers = set(range(construct_numbers[0], construct_numbers[-1] + 1)) if construct_numbers else set()
    observed_numbers = set(construct_numbers)
    result["construct_block_audit"] = {
        "identifier_column": headers[id_col] if id_col is not None else None,
        "construct_row_count": len(construct_records),
        "construct_unique_id_count": len(observed_numbers),
        "construct_duplicate_id_count": len(construct_numbers) - len(observed_numbers),
        "construct_numeric_min": min(construct_numbers) if construct_numbers else None,
        "construct_numeric_max": max(construct_numbers) if construct_numbers else None,
        "construct_missing_numeric_ids_in_observed_range": sorted(expected_numbers - observed_numbers),
        "construct_sequence_columns": [headers[col] for col in sequence_columns],
        "construct_structure_columns": [headers[col] for col in structure_columns],
        "construct_rows_with_all_sequence_columns_nonempty": sum(all(values.get(col, "") != "" for col in sequence_columns) for _, values, _ in construct_records) if sequence_columns else None,
        "construct_rows_with_all_structure_columns_nonempty": sum(all(values.get(col, "") != "" for col in structure_columns) for _, values, _ in construct_records) if structure_columns else None,
        "nonconstruct_data_row_count": len(data) - len(construct_records),
        "nonconstruct_post_header_row_count": sum(not re.fullmatch(r"construct\d+", values.get(id_col, ""), re.I) for _, values in post_header_data) if id_col is not None else None,
        "nonconstruct_identifier_value_sha256": hashlib.sha256("\n".join(values.get(id_col, "") if id_col is not None else "" for _, values in post_header_data if id_col is not None and not re.fullmatch(r"construct\d+", values.get(id_col, ""), re.I)).encode("utf-8")).hexdigest() if id_col is not None else None,
    }
    return result


def audit_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    headers = rows[0] if rows else []
    data = rows[1:] if rows else []
    fields = []
    for idx, header in enumerate(headers):
        values = [(row[idx] if idx < len(row) else "") for row in data]
        fields.append({"column": idx + 1, "header": header, **value_stats(values)})
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path), "row_count_after_header": len(data), "headers": headers, "field_audit": fields}


def audit_docx(path: Path) -> dict[str, Any]:
    tokens = ["Table S1", "Table S2", "Table S4", "Table S7", "barcode", "filter", "SRA", "FASTQ", "7500", "17"]
    counts = {token: 0 for token in tokens}
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
        text = " ".join((node.text or "") for node in root.iter() if local_name(node.tag) == "t")
        text = re.sub(r"\s+", " ", text)
        for token in tokens:
            counts[token] = len(re.findall(re.escape(token), text, re.I))
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path), "text_token_counts": counts}


def normalize_sequence(value: str) -> str:
    return re.sub(r"\s+", "", value).upper().replace("U", "T")


def audit_s2_figshare_identity(source_dir: Path, archive: Path) -> dict[str, Any]:
    """Reconcile S2 with the official library table and processed payloads."""
    s2_path = source_dir / "Supplemental_Table_S2.xlsx"
    sequences_sheet = next(sheet for sheet in parse_xlsx(s2_path)["sheets"] if sheet.get("name") == "Sequences")
    s2_rows = {
        values.get(1, "").lower(): values
        for _, values, _ in sequences_sheet.get("rows", [])
        if re.fullmatch(r"construct\d+", values.get(1, ""), re.I)
    }
    s2_sequence_to_record = {
        normalize_sequence(values.get(3, "")): (name, values.get(4, ""))
        for name, values in s2_rows.items()
    }
    identity: dict[str, Any] = {
        "audit_version": "nar-gkag672-s2-figshare-identity-v1",
        "archive": {"path": str(archive), "size": archive.stat().st_size, "sha256": sha256_file(archive)},
        "s2_construct_count": len(s2_rows),
        "normalization": "uppercase, remove whitespace, U-to-T for sequence identity only",
        "raw_sequences_emitted": False,
        "raw_rows_emitted": False,
    }
    with zipfile.ZipFile(archive) as zf:
        library_rows = list(csv.DictReader(io.StringIO(zf.read("data/csvs/library_sequences.csv").decode("utf-8-sig"))))
        library_sequence_to_names: dict[str, list[str]] = {}
        for row in library_rows:
            library_sequence_to_names.setdefault(normalize_sequence(row.get("sequence", "")), []).append(row.get("name", ""))
        library_matches = [row for row in library_rows if normalize_sequence(row.get("sequence", "")) in s2_sequence_to_record]
        name_matches = [
            row for row in library_matches
            if row.get("name", "").lower() == s2_sequence_to_record[normalize_sequence(row.get("sequence", ""))][0]
        ]
        index_name_matches = [
            row for row in library_matches
            if (match := re.fullmatch(r"seq_(\d+)", row.get("name", ""), re.I))
            and s2_sequence_to_record[normalize_sequence(row.get("sequence", ""))][0] == f"construct{match.group(1)}"
        ]
        structure_mismatches: list[dict[str, Any]] = []
        for row in library_matches:
            construct, s2_structure = s2_sequence_to_record[normalize_sequence(row.get("sequence", ""))]
            library_structure = row.get("structure", "")
            if s2_structure != library_structure:
                diffs = [i for i, (a, b) in enumerate(zip(s2_structure, library_structure)) if a != b]
                structure_mismatches.append({
                    "construct": construct,
                    "length_s2": len(s2_structure),
                    "length_library": len(library_structure),
                    "hamming_distance": len(diffs) + abs(len(s2_structure) - len(library_structure)),
                    "differing_positions_zero_based": diffs,
                    "s2_structure_sha256": hashlib.sha256(s2_structure.encode()).hexdigest(),
                    "library_structure_sha256": hashlib.sha256(library_structure.encode()).hexdigest(),
                })
        identity["library_sequences_csv"] = {
            "row_count": len(library_rows),
            "unique_sequence_count": len(library_sequence_to_names),
            "rows_with_s2_sequence_match": len(library_matches),
            "rows_with_s2_sequence_and_name_match": len(name_matches),
            "rows_with_s2_sequence_and_index_corresponding_name_match": len(index_name_matches),
            "s2_sequences_with_library_match": len(set(s2_sequence_to_record) & set(library_sequence_to_names)),
            "rows_with_duplicate_library_sequence": sum(len(names) > 1 for names in library_sequence_to_names.values()),
            "structure_exact_match_count": len(library_matches) - len(structure_mismatches),
            "structure_mismatch_count": len(structure_mismatches),
            "structure_mismatches": structure_mismatches,
        }
        assay_results = []
        for assay in ["pdb_library_1", "pdb_library_2", "pdb_library_3", "pdb_library_37C_2min", "pdb_library_denature", "pdb_library_nomod"]:
            records = json.loads(zf.read(f"data/raw-jsons/constructs/{assay}.json"))
            sequence_matches = 0
            structure_matches = 0
            for index, record in enumerate(records):
                s2_values = s2_rows[f"construct{index}"]
                s2_sequence = normalize_sequence(s2_values.get(3, ""))[12:-20]
                s2_structure = s2_values.get(4, "")[12:-20]
                sequence_matches += normalize_sequence(str(record.get("sequence", ""))) == s2_sequence
                structure_matches += str(record.get("structure", "")) == s2_structure
            assay_results.append({
                "assay": assay,
                "record_count": len(records),
                "sequence_match_after_5prime_12_and_3prime_20_trim": sequence_matches,
                "structure_match_after_5prime_12_and_3prime_20_trim": structure_matches,
            })
        identity["processed_construct_assays"] = assay_results
    return identity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figshare-archive", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.source_dir.iterdir() if p.is_file())
    inventory = []
    for path in files:
        item: dict[str, Any] = {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix.lower() == ".xlsx":
            parsed = parse_xlsx(path)
            item["kind"] = "xlsx"
            item["workbook"] = {**parsed, "sheets": [audit_sheet(sheet) for sheet in parsed["sheets"]]}
        elif path.suffix.lower() == ".csv":
            item["kind"] = "csv"
            item["table"] = audit_csv(path)
        elif path.suffix.lower() == ".docx":
            item["kind"] = "docx"
            item["document"] = audit_docx(path)
        else:
            item["kind"] = path.suffix.lower().lstrip(".") or "file"
        inventory.append(item)
    s2 = next((x for x in inventory if x["name"] == "Supplemental_Table_S2.xlsx"), None)
    s2_sheets = (s2 or {}).get("workbook", {}).get("sheets", [])
    schema: dict[str, Any] = {
        "audit_version": "nar-gkag672-supplement-v1",
        "source_dir": str(args.source_dir),
        "source_files": [{"name": x["name"], "size": x["size"], "sha256": x["sha256"], "kind": x["kind"]} for x in inventory],
        "s2_workbook": {
            "sheet_count": len(s2_sheets),
            "sheets": [
                {
                    "name": sh.get("name"),
                    "dimension": sh.get("dimension"),
                    "header_row_number": sh.get("header_row_number"),
                    "preamble_row_numbers": sh.get("preamble_row_numbers", []),
                    "data_row_count_after_header": sh.get("data_row_count_after_header"),
                    "post_header_data_row_count": sh.get("post_header_data_row_count"),
                    "headers": sh.get("headers", []),
                    "candidate_fields_by_group": sh.get("candidate_fields_by_group", {}),
                    "column_audit": sh.get("column_audit", []),
                    "construct_block_audit": sh.get("construct_block_audit", {}),
                }
                for sh in s2_sheets
            ],
        },
        "raw_sequences_emitted": False,
        "raw_rows_emitted": False,
        "interpretation": {
            "exact_7500_constructs": "ASSERTED_IF_S2_CONSTRUCT_BLOCK_AUDIT_IS_7500_UNIQUE_CONTIGUOUS_IDS_WITH_NONEMPTY_SEQUENCE_AND_STRUCTURE",
            "condition_batch_fastq_sra_mapping": "NOT_ASSERTED_UNTIL_EXPLICIT_FIELDS_ARE_PRESENT",
            "filtered_17_reason": "NOT_ASSERTED_UNTIL_EXPLICIT_REASON_OR_AUDITABLE_ROW_STATE_IS_PRESENT",
            "s2_to_figshare_processed_identity": "NOT_ASSERTED_UNTIL_S2_FIGSHARE_IDENTITY_ARTIFACT_IS_PRESENT",
        },
    }
    (args.out_dir / "supplement_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "table_s2_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    with (args.out_dir / "supplement_sha256.txt").open("w") as fh:
        for item in inventory:
            fh.write(f"{item['sha256']}  {item['name']}\n")
    identity_path = None
    if args.figshare_archive:
        identity_path = args.out_dir / "s2_figshare_identity.json"
        identity_path.write_text(json.dumps(audit_s2_figshare_identity(args.source_dir, args.figshare_archive), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "inventory_path": str(args.out_dir / "supplement_inventory.json"),
        "schema_path": str(args.out_dir / "table_s2_schema.json"),
        "sha256_path": str(args.out_dir / "supplement_sha256.txt"),
        "identity_path": str(identity_path) if identity_path else None,
        "file_count": len(inventory),
        "s2_sheets": [
            {"name": sh.get("name"), "dimension": sh.get("dimension"), "data_rows": sh.get("data_row_count_after_header"), "headers": sh.get("headers", [])}
            for sh in s2_sheets
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
