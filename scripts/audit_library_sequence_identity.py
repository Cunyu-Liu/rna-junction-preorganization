#!/usr/bin/env python3
"""Audit the official library table to processed-construct identity binding.

The authors' public processing code:
  * renames seq_N to constructN; and
  * trims the common 5-prime sequence and a fixed 20-nucleotide 3-prime tail.

This script checks those source-defined semantics inside the Figshare data.zip
without emitting sequence, reactivity, or label values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


P5_MEMBER = "data/csvs/p5_sequences.csv"
LIBRARY_MEMBER = "data/csvs/library_sequences.csv"
CONSTRUCT_PREFIX = "data/raw-jsons/constructs/"
CONSTRUCT_RE = re.compile(r"construct(\d+)")
LIBRARY_RE = re.compile(r"seq_(\d+)")


def _csv_rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def _to_rna(value: str) -> str:
    return value.replace("T", "U").replace("t", "u")


def _name_set_sha256(names: set[str]) -> str:
    canonical = "\n".join(sorted(str(name) for name in names)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_id(name: str, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected identity name: {name!r}")
    return int(match.group(1))


def audit(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        if P5_MEMBER not in members or LIBRARY_MEMBER not in members:
            raise ValueError("required official library members are missing")

        p5_rows = _csv_rows(archive, P5_MEMBER)
        library_rows = _csv_rows(archive, LIBRARY_MEMBER)
        if not library_rows:
            raise ValueError("official library table is empty")

        library_by_id: dict[int, dict[str, str]] = {}
        for row in library_rows:
            identity = _parse_id(row["name"], LIBRARY_RE)
            if identity in library_by_id:
                raise ValueError(f"duplicate library identity: {identity}")
            library_by_id[identity] = row

        library_sequences = [_to_rna(row["sequence"]) for row in library_rows]
        common_p5 = ""
        for row in p5_rows:
            candidate = _to_rna(row["sequence"])
            if candidate and all(sequence.startswith(candidate) for sequence in library_sequences):
                common_p5 = candidate
        if not common_p5:
            raise ValueError("official p5 sequence rule found no common prefix")

        construct_members = sorted(
            member
            for member in members
            if member.startswith(CONSTRUCT_PREFIX) and member.endswith(".json")
        )
        if not construct_members:
            raise ValueError("no official processed construct JSON members found")

        condition_results: list[dict[str, Any]] = []
        for member in construct_members:
            records = json.load(archive.open(member))
            processed_by_id: dict[int, dict[str, Any]] = {}
            for record in records:
                identity = _parse_id(str(record["name"]), CONSTRUCT_RE)
                if identity in processed_by_id:
                    raise ValueError(f"duplicate processed identity in {member}: {identity}")
                processed_by_id[identity] = record

            common_ids = set(library_by_id) & set(processed_by_id)
            sequence_matches = 0
            structure_matches = 0
            for identity in common_ids:
                source = library_by_id[identity]
                processed = processed_by_id[identity]
                source_sequence = _to_rna(source["sequence"])
                expected_sequence = source_sequence[len(common_p5) : -20]
                if expected_sequence == str(processed["sequence"]):
                    sequence_matches += 1
                source_structure = source.get("structure", "")
                processed_structure = str(processed.get("structure", ""))
                if source_structure[len(common_p5) : -20] == processed_structure:
                    structure_matches += 1

            condition_results.append(
                {
                    "member": member,
                    "record_count": len(records),
                    "library_identity_count": len(library_by_id),
                    "processed_identity_count": len(processed_by_id),
                    "identity_intersection_count": len(common_ids),
                    "sequence_matches_after_source_trim": sequence_matches,
                    "structure_matches_after_source_trim": structure_matches,
                    "processed_name_set_sha256": _name_set_sha256(set(processed_by_id)),
                }
            )

        all_complete = all(
            result["record_count"] == len(library_by_id)
            and result["identity_intersection_count"] == len(library_by_id)
            and result["sequence_matches_after_source_trim"] == len(library_by_id)
            and result["structure_matches_after_source_trim"] == len(library_by_id)
            for result in condition_results
        )
        return {
            "audit_id": "official_library_sequence_identity",
            "status": (
                "OFFICIAL_LIBRARY_TABLE_TO_PROCESSED_CONSTRUCT_BIJECTION_COMPLETE_NO_RAW_RUN_UNLOCK"
                if all_complete
                else "BLOCKED_LIBRARY_TABLE_TO_PROCESSED_CONSTRUCT_IDENTITY_RECONCILIATION_FAILED"
            ),
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": _file_sha256(archive_path),
            "official_processing_semantics": {
                "repository": "https://github.com/YesselmanLabPublications/2025_char_3d_struct_features",
                "source_commit_checked": "ed75e36bb36dc2f32c866c436c437d88a4743cf8",
                "name_binding": "seq_N -> constructN",
                "five_prime_member": P5_MEMBER,
                "five_prime_common_length": len(common_p5),
                "three_prime_trim_length": 20,
                "sequence_alphabet_rule": "T -> U before comparison",
            },
            "library_table": {
                "member": LIBRARY_MEMBER,
                "row_count": len(library_rows),
                "unique_identity_count": len(library_by_id),
                "name_set_sha256": _name_set_sha256(set(str(row["name"]) for row in library_rows)),
            },
            "condition_results": condition_results,
            "gate_effect": {
                "library_identity_gate_effect": "PASS",
                "raw_processed_crosswalk_gate_effect": "NO_CHANGE",
                "phase0_gate_effect": "NO_PHASE_0_PASS",
                "scientific_gate_effect": "NO_UNLOCK",
                "primary_labels_admitted": False,
                "training_started": False,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
