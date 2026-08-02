#!/usr/bin/env python3
"""Audit an author code release for an explicit DMS source crosswalk.

The audit is deliberately negative-evidence preserving: it records archive
identity, member names, and token counts, but never emits sequence payloads or
whole text files.  A code release is not promoted to a scientific source
crosswalk merely because it contains a parser or an example command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    ".c", ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".r", ".rst",
    ".sh", ".sql", ".toml", ".tsv", ".txt", ".xml", ".yaml", ".yml",
}

TOKENS = {
    "Sequences.xlsx": re.compile(r"sequences\.xlsx", re.I),
    "barcode": re.compile(r"barcode", re.I),
    "rtb": re.compile(r"rtb", re.I),
    "trial1": re.compile(r"trial1", re.I),
    "trial2": re.compile(r"trial2", re.I),
    "SRA_accession": re.compile(r"\b(?:SRR|ERR|DRR|PRJNA|SRP)\d+\b", re.I),
    "FASTQ": re.compile(r"fastq", re.I),
    "sample_sheet": re.compile(r"sample[ _-]?sheet", re.I),
    "condition": re.compile(r"condition", re.I),
    "batch": re.compile(r"batch", re.I),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tokens(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in TOKENS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-doi", required=True)
    args = parser.parse_args()

    archive = args.archive.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        member_names = sorted(zf.namelist())
        member_name_token_counts = count_tokens("\n".join(member_names))
        text_members_scanned: list[str] = []
        text_token_counts = {name: 0 for name in TOKENS}
        unreadable_text_members: list[str] = []
        for member in member_names:
            suffix = Path(member).suffix.lower()
            if suffix not in TEXT_SUFFIXES or member.endswith("/"):
                continue
            try:
                text = zf.read(member).decode("utf-8", "replace")
            except Exception:
                unreadable_text_members.append(member)
                continue
            text_members_scanned.append(member)
            counts = count_tokens(text)
            for name, value in counts.items():
                text_token_counts[name] += value

    expected_presence = {
        "Sequences.xlsx": any(Path(name).name.lower() == "sequences.xlsx" for name in member_names),
        "barcode_or_rtb_payload_name": any(re.search(r"(?:barcode|rtb)", Path(name).name, re.I) for name in member_names),
        "sample_or_crosswalk_payload_name": any(re.search(r"(?:sample|crosswalk|trial|condition|batch)", Path(name).name, re.I) for name in member_names),
        "FASTQ_or_SRA_payload_name": any(re.search(r"(?:fastq|sra|srr|err|drr|prjna|srp)", Path(name).name, re.I) for name in member_names),
    }
    explicit_signal_keys = {
        "Sequences.xlsx",
        "barcode",
        "rtb",
        "trial1",
        "trial2",
        "SRA_accession",
        "FASTQ",
        "sample_sheet",
    }
    explicit_signal_count = sum(
        member_name_token_counts[key] + text_token_counts[key]
        for key in explicit_signal_keys
    )
    # Generic words such as "condition" can occur in ordinary code variable
    # names.  They are recorded but cannot establish a source crosswalk.
    any_crosswalk_evidence = explicit_signal_count > 0
    result: dict[str, Any] = {
        "audit_version": "author-release-crosswalk-v1",
        "source": {
            "url": args.source_url,
            "doi": args.source_doi,
            "archive": str(archive),
            "size": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "archive_member_count": len(member_names),
        "member_names": member_names,
        "expected_payload_presence": expected_presence,
        "member_name_token_counts": member_name_token_counts,
        "text_members_scanned": text_members_scanned,
        "text_token_counts": text_token_counts,
        "explicit_crosswalk_signal_count": explicit_signal_count,
        "unreadable_text_members": unreadable_text_members,
        "crosswalk_evidence_found": any_crosswalk_evidence,
        "status": "AUTHOR_RELEASE_CROSSWALK_PAYLOAD_FOUND_REQUIRES_SEPARATE_VALIDATION" if any_crosswalk_evidence else "AUTHOR_RELEASE_NO_EXPLICIT_CROSSWALK_OR_BARCODE_PAYLOAD",
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "primary_labels_admitted": False,
        "raw_sequences_emitted": False,
        "raw_text_payloads_emitted": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "archive_member_count": len(member_names),
        "expected_payload_presence": expected_presence,
        "text_token_counts": text_token_counts,
        "status": result["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
