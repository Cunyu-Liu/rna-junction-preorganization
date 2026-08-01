#!/usr/bin/env python3
"""Audit public PMC BioC text for contract-relevant terminology only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


TERMS = ("censor", "censored", "interpol", "-7.1", "limit", "bootstrap", "replicate", "covariance")


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
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()

    observed = sha256(args.input)
    if observed != args.expected_sha256:
        print(json.dumps({"status": "BLOCKED_HASH_MISMATCH", "expected": args.expected_sha256, "observed": observed}, sort_keys=True))
        return 2
    data = json.loads(args.input.read_text(encoding="utf-8"))
    text_parts: list[str] = []
    for collection in data:
        for document in collection.get("documents", []):
            text_parts.extend(passage.get("text", "") for passage in document.get("passages", []))
    text = "\n".join(text_parts)
    term_counts = {term: len(re.findall(re.escape(term), text, flags=re.IGNORECASE)) for term in TERMS}
    result = {
        "schema_version": "phase0-pmc-semantics-audit-v1",
        "status": "PMC_SEMANTICS_TERM_AUDIT_COMPLETE",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": args.source_id,
        "input": str(args.input),
        "sha256": observed,
        "term_counts": term_counts,
        "raw_article_text_emitted": False,
        "scientific_interpretation": "PUBLIC_TEXT_TERM_EVIDENCE_ONLY_CENSOR_DIRECTION_NOT_ESTABLISHED",
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sha256": observed, "term_counts": term_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
