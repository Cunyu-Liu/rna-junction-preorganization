#!/usr/bin/env python3
"""Validate the Phase 0 manual matching acceptance component.

The input is an auditable TSV containing opaque record references only. This
validator deliberately does not print or persist sequences, reactivities,
counts, effect values, or reviewer notes. It reports aggregate agreement and
coverage so the project-level Phase 0 gate can remain fail-closed until the
primary payload and the complete evidence chain are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DECISIONS = {"matched", "rejected", "ambiguous"}
REQUIRED_COLUMNS = {
    "audit_id",
    "source_record_ref",
    "candidate_record_ref",
    "automated_decision",
    "manual_decision",
    "reason_code",
    "evidence_ref",
    "evidence_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blocked(reason: str, *, input_path: Path, observed_sha256: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "phase0-manual-matching-v1",
        "status": "BLOCKED_MANUAL_MATCHING_AUDIT",
        "reason": reason,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    if observed_sha256 is not None:
        result["sha256"] = observed_sha256
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        result = blocked("INPUT_TABLE_MISSING", input_path=args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "reason": result["reason"]}, sort_keys=True))
        return 2

    observed = sha256(args.input)
    if observed != args.expected_sha256:
        result = blocked("INPUT_HASH_MISMATCH", input_path=args.input, observed_sha256=observed)
        result["expected_sha256"] = args.expected_sha256
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "reason": result["reason"]}, sort_keys=True))
        return 2

    errors: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with args.input.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                errors.append("MISSING_COLUMNS:" + ",".join(missing))
            for line_number, row in enumerate(reader, start=2):
                if not row or all(value in (None, "") for value in row.values()):
                    continue
                normalized = {key: (value or "").strip() for key, value in row.items() if key is not None}
                rows.append(normalized)
                if not normalized.get("audit_id"):
                    errors.append(f"EMPTY_AUDIT_ID:{line_number}")
                if not normalized.get("source_record_ref") or not normalized.get("candidate_record_ref"):
                    errors.append(f"EMPTY_RECORD_REFERENCE:{line_number}")
                for column in ("automated_decision", "manual_decision"):
                    if normalized.get(column) not in DECISIONS:
                        errors.append(f"INVALID_{column.upper()}:{line_number}")
                if not normalized.get("reason_code"):
                    errors.append(f"EMPTY_REASON_CODE:{line_number}")
                if not normalized.get("evidence_ref") or not normalized.get("evidence_sha256"):
                    errors.append(f"MISSING_EVIDENCE:{line_number}")
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"TABLE_READ_ERROR:{type(exc).__name__}")

    audit_ids = [row.get("audit_id", "") for row in rows]
    if any(key and count > 1 for key, count in Counter(audit_ids).items()):
        errors.append("DUPLICATE_AUDIT_IDS")

    manual_counts = Counter(row.get("manual_decision", "") for row in rows)
    reviewed = len(rows)
    matched = manual_counts["matched"]
    rejected_or_ambiguous = manual_counts["rejected"] + manual_counts["ambiguous"]
    agreements = sum(
        row.get("automated_decision") == row.get("manual_decision")
        for row in rows
    )
    agreement = agreements / reviewed if reviewed else 0.0

    if matched < 50:
        errors.append("INSUFFICIENT_MANUAL_MATCHED_LT_50")
    if rejected_or_ambiguous < 30:
        errors.append("INSUFFICIENT_MANUAL_REJECTED_OR_AMBIGUOUS_LT_30")
    if agreement < 0.95:
        errors.append("MATCHING_ACCURACY_LT_0_95")

    status = "PASS_MANUAL_MATCHING_COMPONENT" if not errors else "BLOCKED_MANUAL_MATCHING_AUDIT"
    result = {
        "schema_version": "phase0-manual-matching-v1",
        "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "sha256": observed,
        "reviewed_row_count": reviewed,
        "manual_decision_counts": {
            "matched": matched,
            "rejected": manual_counts["rejected"],
            "ambiguous": manual_counts["ambiguous"],
        },
        "rejected_or_ambiguous_count": rejected_or_ambiguous,
        "automated_manual_agreement_count": agreements,
        "automated_manual_agreement_fraction": agreement,
        "acceptance_requirements": {
            "manual_matched_at_least": 50,
            "manual_rejected_or_ambiguous_at_least": 30,
            "agreement_at_least": 0.95,
        },
        "errors": errors,
        "raw_sequence_content_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "reviewed_row_count": reviewed,
                "manual_matched": matched,
                "manual_rejected_or_ambiguous": rejected_or_ambiguous,
                "agreement": agreement,
                "error_count": len(errors),
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
