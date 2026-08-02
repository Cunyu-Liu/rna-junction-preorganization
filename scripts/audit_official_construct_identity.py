#!/usr/bin/env python3
"""Audit official library/construct identity without emitting scientific values."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from typing import Any, Iterable


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
CONSTRUCT_NAME_RE = re.compile(r"^construct[0-9]+$")


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, name: str) -> str:
    with archive.open(name) as handle:
        return sha256_bytes(iter(lambda: handle.read(8 * 1024 * 1024), b""))


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_bytes(iter(lambda: handle.read(8 * 1024 * 1024), b""))


def sorted_set_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(walk_strings(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(walk_strings(item))
        return output
    return []


def extract_construct_refs(value: Any) -> list[str]:
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        candidate = value["name"]
        if CONSTRUCT_NAME_RE.fullmatch(candidate):
            return [candidate]
    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            if key != "name":
                output.extend(extract_construct_refs(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(extract_construct_refs(item))
        return output
    return []


def load_json_member(archive: zipfile.ZipFile, name: str) -> Any:
    with archive.open(name) as handle:
        return json.load(io.TextIOWrapper(handle, encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = args.payload.resolve()
    contract = args.contract.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {output}")
    if hashlib.sha256(contract.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise SystemExit("contract SHA256 does not match the frozen 1.1 contract")

    started_at = utc_now()
    library_name = "data/csvs/library_sequences.csv"
    with zipfile.ZipFile(payload) as archive:
        construct_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("data/raw-jsons/constructs/")
            and name.lower().endswith(".json")
        )
    if not construct_names:
        raise SystemExit("official construct JSON members are absent")

    with zipfile.ZipFile(payload) as archive:
        with archive.open(library_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            if reader.fieldnames is None or "name" not in reader.fieldnames:
                raise SystemExit("library_sequences.csv has no name field")
            library_names: list[str] = []
            for row in reader:
                value = row.get("name")
                if value:
                    library_names.append(value)

        construct_union: set[str] = set()
        condition_name_sets: list[set[str]] = []
        condition_sequence_sets: list[set[str]] = []
        construct_member_records: list[dict[str, Any]] = []
        for name in construct_names:
            info = archive.getinfo(name)
            payload_json = load_json_member(archive, name)
            if not isinstance(payload_json, list):
                raise SystemExit(f"construct member is not a list: {name}")
            refs = [
                item["name"]
                for item in payload_json
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and CONSTRUCT_NAME_RE.fullmatch(item["name"])
            ]
            sequences = [
                item["sequence"]
                for item in payload_json
                if isinstance(item, dict)
                and isinstance(item.get("sequence"), str)
            ]
            ref_set = set(refs)
            sequence_set = set(sequences)
            construct_union.update(ref_set)
            condition_name_sets.append(ref_set)
            condition_sequence_sets.append(sequence_set)
            construct_member_records.append(
                {
                    "member": name,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256_zip_member(archive, name),
                    "row_count": len(payload_json),
                    "construct_reference_count": len(refs),
                    "construct_reference_unique_count": len(ref_set),
                    "construct_reference_duplicate_count": len(refs) - len(ref_set),
                    "construct_reference_set_sha256": sorted_set_hash(ref_set),
                    "sequence_unique_count": len(sequence_set),
                    "sequence_set_sha256": sorted_set_hash(sequence_set),
                }
            )

        payload_names = archive.namelist()
        direct_sra_tokens = sorted(
            name for name in payload_names if re.search(r"SRR[0-9]+", name, re.IGNORECASE)
        )
        library_info = archive.getinfo(library_name)
        library_sha256 = sha256_zip_member(archive, library_name)

    library_set = set(library_names)
    duplicate_library_name_count = len(library_names) - len(library_set)
    missing_count = len(library_set - construct_union)
    extra_count = len(construct_union - library_set)
    construct_condition_name_sets_equal = bool(condition_name_sets) and all(
        values == condition_name_sets[0] for values in condition_name_sets[1:]
    )
    construct_condition_sequence_sets_equal = bool(condition_sequence_sets) and all(
        values == condition_sequence_sets[0]
        for values in condition_sequence_sets[1:]
    )
    identity_pass = (
        construct_condition_name_sets_equal
        and construct_condition_sequence_sets_equal
        and all(len(values) == 7500 for values in condition_name_sets)
        and all(len(values) == 7500 for values in condition_sequence_sets)
    )
    audit = {
        "schema": "phase0-official-construct-identity-audit-v1",
        "status": (
            "OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY_SET_EQUAL"
            if identity_pass
            else "BLOCKED_OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY_MISMATCH"
        ),
        "created_at_utc": started_at,
        "run_id": args.run_id,
        "payload": str(payload),
        "payload_sha256": sha256_file(payload),
        "contract_sha256": CONTRACT_SHA256,
        "library_reference_member": library_name,
        "library_reference_member_bytes": library_info.file_size,
        "library_reference_member_crc32": f"{library_info.CRC:08x}",
        "library_reference_member_sha256": library_sha256,
        "library_reference_fieldnames": [
            "sequence",
            "name",
            "len",
            "molecular weight",
            "extinction coeff",
            "structure",
            "mfe",
            "ens div",
        ],
        "library_reference_row_count": len(library_names),
        "library_reference_unique_name_count": len(library_set),
        "library_reference_duplicate_name_count": duplicate_library_name_count,
        "library_reference_construct_name_format_count": sum(
            1 for value in library_set if CONSTRUCT_NAME_RE.fullmatch(value)
        ),
        "library_reference_name_set_sha256": sorted_set_hash(library_set),
        "library_reference_relation_status": (
            "NON_EQUIVALENT_STRUCTURAL_LIBRARY_NAME_NAMESPACE_OBSERVED"
        ),
        "construct_json_member_count": len(construct_member_records),
        "construct_json_members": construct_member_records,
        "construct_reference_union_count": len(construct_union),
        "construct_reference_union_sha256": sorted_set_hash(construct_union),
        "construct_condition_name_set_equal": construct_condition_name_sets_equal,
        "construct_condition_sequence_set_equal": construct_condition_sequence_sets_equal,
        "construct_condition_reference_count": (
            len(condition_name_sets[0]) if condition_name_sets else 0
        ),
        "construct_condition_sequence_count": (
            len(condition_sequence_sets[0]) if condition_sequence_sets else 0
        ),
        "library_name_missing_from_construct_union_count": missing_count,
        "library_name_extra_not_in_construct_union_count": extra_count,
        "direct_sra_token_member_count": len(direct_sra_tokens),
        "raw_processed_run_crosswalk_status": (
            "BLOCKED_NO_DIRECT_SRA_RUN_TOKEN_IN_OFFICIAL_ARCHIVE_MEMBER_NAMES"
            if not direct_sra_tokens
            else "DIRECT_SRA_TOKEN_MEMBERS_OBSERVED_REQUIRES_RUN_LEVEL_RECONCILIATION"
        ),
        "identity_binding_status": (
            "PASS_OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY"
            if identity_pass
            else "BLOCKED_OFFICIAL_PROCESSED_CONSTRUCT_CONDITION_IDENTITY"
        ),
        "primary_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "audit_boundary": (
            "Counts, field names, member hashes, and opaque identifier-set hashes "
            "only; no sequence, label, or effect-value content is emitted."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "library_reference_row_count": audit["library_reference_row_count"],
                "construct_reference_union_count": audit["construct_reference_union_count"],
                "missing_count": missing_count,
                "extra_count": extra_count,
                "raw_processed_run_crosswalk_status": audit["raw_processed_run_crosswalk_status"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if identity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
