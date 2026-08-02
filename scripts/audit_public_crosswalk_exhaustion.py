#!/usr/bin/env python3
"""Search bounded public payloads for an accession-preserving crosswalk.

The audit emits member names, counts, hashes, and co-occurrence metadata only.
It never emits sequence content or arbitrary text values. A no-hit result is a
bounded negative search, not proof that an unpublished mapping does not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".tsv", ".yaml", ".yml", ".xml", ".fa", ".fasta"}
MAX_CANDIDATE_TEXT_BYTES = 64 * 1024 * 1024
PATTERNS = {
    "raw_accession": re.compile(rb"\b(?:SRR|ERR|DRR)\d{6,}\b", re.IGNORECASE),
    "project_accession": re.compile(rb"\bPRJNA\d+\b", re.IGNORECASE),
    "trial_name": re.compile(rb"\brna_library_trial[12]\b", re.IGNORECASE),
    "processed_main_library": re.compile(rb"\bpdb_library_[123]\b", re.IGNORECASE),
    "processed_control_library": re.compile(
        rb"\bpdb_library_(?:nomod|denature|37c_2min)\b", re.IGNORECASE
    ),
    "sequence_workbook": re.compile(rb"\bSequences[.]xlsx\b", re.IGNORECASE),
    "sample_manifest": re.compile(rb"\b(?:sample[_ -]?sheet|run[_ -]?manifest|crosswalk)\b", re.IGNORECASE),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_matches(data: bytes) -> dict[str, int]:
    return {name: len(pattern.findall(data)) for name, pattern in PATTERNS.items()}


def scan_zip(path: Path) -> dict[str, object]:
    name_matches: dict[str, list[str]] = {name: [] for name in PATTERNS}
    content_matches: dict[str, list[dict[str, object]]] = {name: [] for name in PATTERNS}
    text_member_count = 0
    skipped_textlike_member_count = 0
    with ZipFile(path) as archive:
        infos = archive.infolist()
        for info in infos:
            name_bytes = info.filename.encode("utf-8", errors="replace")
            name_counts = count_matches(name_bytes)
            for token, count in name_counts.items():
                if count and len(name_matches[token]) < 100:
                    name_matches[token].append(info.filename)
            suffix = Path(info.filename).suffix.casefold()
            lower_name = info.filename.casefold()
            candidate_path = (
                lower_name.startswith("data/csvs/")
                or lower_name.startswith("metadata/")
                or lower_name.startswith("source_metadata/")
                or Path(info.filename).name.casefold() in {"readme", "readme.md", "manifest.txt"}
                or any(token in Path(info.filename).name.casefold() for token in ("sample", "manifest", "metadata", "sequence"))
            )
            if suffix not in TEXT_SUFFIXES or not candidate_path:
                continue
            if info.file_size > MAX_CANDIDATE_TEXT_BYTES:
                skipped_textlike_member_count += 1
                continue
            text_member_count += 1
            with archive.open(info) as handle:
                data = handle.read()
            counts = count_matches(data)
            present = {token: count for token, count in counts.items() if count}
            if present:
                member_record = {
                    "member": info.filename,
                    "size_bytes": info.file_size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "token_counts": present,
                }
                for token in present:
                    if len(content_matches[token]) < 100:
                        content_matches[token].append(member_record)
        return {
            "member_count": len(infos),
            "text_member_count": text_member_count,
            "skipped_textlike_member_count": skipped_textlike_member_count,
            "content_scan_policy": {
                "candidate_directories": ["data/csvs/", "metadata/", "source_metadata/"],
                "candidate_basenames": ["README", "README.md", "manifest.txt"],
                "candidate_basename_tokens": ["sample", "manifest", "metadata", "sequence"],
                "max_uncompressed_member_bytes": MAX_CANDIDATE_TEXT_BYTES,
                "raw_json_and_revision_trees_skipped": True,
            },
            "name_matches": name_matches,
            "content_matches": content_matches,
        }


def scan_source(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    counts = count_matches(data)
    return {
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "token_counts": counts,
        "raw_processed_cooccurrence": bool(
            counts["raw_accession"] and counts["processed_main_library"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema_version": "phase0-public-crosswalk-exhaustion-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": args.contract_sha256,
        "archive": {"path": str(args.archive), "expected_sha256": args.expected_archive_sha256},
        "source_snapshot": {"path": str(args.source_snapshot), "expected_sha256": args.expected_source_sha256},
        "bounded_search_scope": [
            "current Figshare data.zip member names",
            "current Figshare text-like members with suffixes csv/json/md/txt/tsv/yaml/yml/xml/fa/fasta",
            "pinned current DMS PMC/BioC source snapshot bytes",
            "raw accession, project accession, trial name, processed library namespace, Sequences.xlsx, and manifest tokens",
        ],
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    missing = []
    for path, label in ((args.archive, "archive"), (args.source_snapshot, "source_snapshot")):
        if not path.is_file():
            missing.append(label)
    if missing:
        result.update({"status": "BLOCKED_INPUT_MISSING", "missing_inputs": missing})
    else:
        archive_sha = sha256_file(args.archive)
        source_sha = sha256_file(args.source_snapshot)
        result["archive"]["observed_sha256"] = archive_sha  # type: ignore[index]
        result["source_snapshot"]["observed_sha256"] = source_sha  # type: ignore[index]
        if archive_sha != args.expected_archive_sha256 or source_sha != args.expected_source_sha256:
            result["status"] = "BLOCKED_INPUT_HASH_MISMATCH"
        else:
            archive_scan = scan_zip(args.archive)
            source_scan = scan_source(args.source_snapshot)
            result["archive_scan"] = archive_scan
            result["source_scan"] = source_scan
            same_member_cooccurrence = []
            for record in archive_scan["content_matches"]["raw_accession"]:  # type: ignore[index]
                tokens = set(record["token_counts"])  # type: ignore[index]
                if "processed_main_library" in tokens or "processed_control_library" in tokens:
                    same_member_cooccurrence.append(record)
            result["same_member_raw_processed_cooccurrence"] = same_member_cooccurrence
            result["accession_preserving_crosswalk_evidence_found"] = bool(
                same_member_cooccurrence or source_scan["raw_processed_cooccurrence"]
            )
            result["status"] = (
                "PUBLIC_CROSSWALK_CANDIDATES_FOUND_REVIEW_REQUIRED"
                if result["accession_preserving_crosswalk_evidence_found"]
                else "PUBLIC_CROSSWALK_SEARCH_COMPLETE_UNRESOLVED"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "accession_preserving_crosswalk_evidence_found": result.get(
                    "accession_preserving_crosswalk_evidence_found"
                ),
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {
        "PUBLIC_CROSSWALK_SEARCH_COMPLETE_UNRESOLVED",
        "PUBLIC_CROSSWALK_CANDIDATES_FOUND_REVIEW_REQUIRED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
