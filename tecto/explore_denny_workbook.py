#!/usr/bin/env python3
"""T0 exploration: decode Denny workbook structure (sheets, headers, row counts).

Reads the xlsx via stdlib zipfile/xml (no openpyxl). Emits sheet names, headers,
row counts, and column-level value-type sampling. This is a read-only inspection
used to design the T0 canonical builder.
"""
from __future__ import annotations

import argparse
import posixpath
import re
import sys
from collections import Counter
from zipfile import ZipFile
from xml.etree import ElementTree as ET

NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sheet_targets(archive: ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rels_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels if local_name(r.tag) == "Relationship"}
    out = []
    for sheet in wb.iter():
        if local_name(sheet.tag) != "sheet":
            continue
        rid = sheet.attrib.get(f"{{{NS_REL}}}id", "")
        target = rels_map.get(rid, "")
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        out.append((sheet.attrib.get("name", "<unnamed>"), path))
    return out


def shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    vals = []
    for item in root:
        if local_name(item.tag) != "si":
            continue
        vals.append("".join(n.text or "" for n in item.iter() if local_name(n.tag) == "t"))
    return vals


def cell_value(cell: ET.Element, strings: list[str]):
    t = cell.attrib.get("t", "")
    if t == "inlineStr":
        return "".join(n.text or "" for n in cell.iter() if local_name(n.tag) == "t") or None
    v = next((c for c in cell if local_name(c.tag) == "v"), None)
    if v is None or v.text is None:
        return None
    if t == "s":
        try:
            return strings[int(v.text)]
        except (ValueError, IndexError):
            return None
    return v.text


def col_name(ref: str) -> str:
    m = re.match(r"([A-Z]+)", ref)
    return m.group(1) if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--max-rows", type=int, default=5)
    args = ap.parse_args()

    with ZipFile(args.xlsx) as z:
        print("=== SHEETS ===")
        for name, path in sheet_targets(z):
            print(f"  {name!r} -> {path}")
        strings = shared_strings(z)
        print("=== shared strings count:", len(strings))
        for name, path in sheet_targets(z):
            if not path.endswith(".xml") or "sheet" not in path:
                continue
            root = ET.fromstring(z.read(path))
            rows = [r for r in root.iter() if local_name(r.tag) == "row"]
            print(f"\n=== SHEET {name!r}: {len(rows)} rows ===")
            # header row = first row
            for ri, row in enumerate(rows[:args.max_rows]):
                cells = [(col_name(c.attrib.get("r", "")), cell_value(c, strings))
                         for c in row if local_name(c.tag) == "c"]
                print(f"  row {ri}: {cells}")
            # column index -> type distribution over all rows
            col_types = Counter()
            for row in rows:
                for c in row:
                    if local_name(c.tag) != "c":
                        continue
                    col_types[col_name(c.attrib.get("r", ""))] += 1
            print("  non-empty cell counts by column:", dict(sorted(col_types.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())