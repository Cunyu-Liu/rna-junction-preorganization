#!/usr/bin/env python3
"""Audit NCBI SRA run-level metadata without exporting run identifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


GROUP_FIELDS = (
    "LibraryName",
    "LibraryStrategy",
    "LibrarySelection",
    "LibrarySource",
    "LibraryLayout",
    "Platform",
    "Model",
    "SampleType",
    "Consent",
)
NUMERIC_FIELDS = ("spots", "bases", "spots_with_mates", "avgLength", "size_MB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    observed_sha256 = sha256(args.input)
    if observed_sha256 != args.expected_sha256:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_HASH_MISMATCH",
                    "expected_sha256": args.expected_sha256,
                    "observed_sha256": observed_sha256,
                },
                sort_keys=True,
            )
        )
        return 2

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)

    distinct_group_counts: dict[str, int] = {}
    missing_counts: Counter[str] = Counter()
    numeric_summary: dict[str, dict[str, int | float | None]] = {}
    for field in GROUP_FIELDS:
        values = {row.get(field, "").strip() for row in rows if row.get(field, "").strip()}
        distinct_group_counts[field] = len(values)
        missing_counts[field] = sum(not row.get(field, "").strip() for row in rows)

    for field in NUMERIC_FIELDS:
        values: list[float] = []
        for row in rows:
            raw = row.get(field, "").strip()
            if not raw:
                missing_counts[field] += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                missing_counts[field] += 1
                continue
            if math.isfinite(value):
                values.append(value)
        numeric_summary[field] = {
            "finite_value_count": len(values),
            "missing_or_non_numeric_count": missing_counts[field],
            "min_observed": min(values) if values else None,
            "max_observed": max(values) if values else None,
            "sum_observed": sum(values) if values else None,
        }

    result = {
        "status": "SRA_RUNINFO_METADATA_AUDIT_COMPLETE",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "sha256": observed_sha256,
        "size_bytes": args.input.stat().st_size,
        "row_count": len(rows),
        "field_count": len(fields),
        "field_presence": {field: field in fields for field in (*GROUP_FIELDS, *NUMERIC_FIELDS)},
        "distinct_group_counts": distinct_group_counts,
        "missing_counts": dict(missing_counts),
        "numeric_summary": numeric_summary,
        "raw_run_identifiers_emitted": False,
        "read_depth_semantics": "RUN_LEVEL_SPOTS_AND_BASES_ONLY_CONSTRUCT_LEVEL_HIERARCHY_UNRESOLVED",
        "treated_background_hierarchy": "NOT_ESTABLISHED_FROM_RUNINFO_ALONE",
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "sha256": observed_sha256,
                "row_count": len(rows),
                "field_count": len(fields),
                "distinct_group_counts": distinct_group_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
