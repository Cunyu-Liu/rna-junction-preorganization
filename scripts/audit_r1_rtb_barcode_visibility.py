#!/usr/bin/env python3
"""Bounded audit of published RTB barcode visibility in raw R1 FASTQ.

The PMC methods give three example RTB barcode sequences and say the barcode
is at the 5' end of read 1.  This audit tests only whether those examples are
systematically visible in a bounded R1 sample.  It never infers a condition,
run, or processed-library label from the result.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PUBLISHED_RTB = {
    "RTB021": "CCAATGGGTGTA",
    "RTB022": "AGCCAAAACTGG",
    "RTB023": "GTGTGTTTGCCC",
}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def sha256_prefix(path: Path, limit: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def redacted_counter(counter: collections.Counter[str], limit: int = 8) -> list[dict[str, Any]]:
    result = []
    for prefix, count in counter.most_common(limit):
        result.append({
            "sha256": hashlib.sha256(prefix.encode("ascii")).hexdigest(),
            "length": len(prefix),
            "count": count,
        })
    return result


def audit_run(path: Path, sample_reads: int, search_window: int) -> dict[str, Any]:
    first12: collections.Counter[str] = collections.Counter()
    known_hits = {
        name: {"forward": 0, "reverse_complement": 0, "forward_positions": collections.Counter(), "reverse_complement_positions": collections.Counter()}
        for name in PUBLISHED_RTB
    }
    records = 0
    malformed = 0
    header_index_like = 0
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as handle:
        while records < sample_reads:
            header = handle.readline()
            sequence = handle.readline().strip().upper()
            plus = handle.readline()
            quality = handle.readline()
            if not header:
                break
            if not plus or not quality or not sequence:
                malformed += 1
                break
            records += 1
            first12[sequence[:12]] += 1
            if re.search(r"(?:#|index=|barcode=)[A-Z0-9]+", header, re.I):
                header_index_like += 1
            window = sequence[:search_window]
            for name, barcode in PUBLISHED_RTB.items():
                reverse = reverse_complement(barcode)
                position = window.find(barcode)
                while position >= 0:
                    known_hits[name]["forward"] += 1
                    known_hits[name]["forward_positions"][position] += 1
                    position = window.find(barcode, position + 1)
                position = window.find(reverse)
                while position >= 0:
                    known_hits[name]["reverse_complement"] += 1
                    known_hits[name]["reverse_complement_positions"][position] += 1
                    position = window.find(reverse, position + 1)
    clean_hits: dict[str, Any] = {}
    total_known_hits = 0
    for name, values in known_hits.items():
        total_known_hits += values["forward"] + values["reverse_complement"]
        clean_hits[name] = {
            "barcode_length": len(PUBLISHED_RTB[name]),
            "forward_hits": values["forward"],
            "reverse_complement_hits": values["reverse_complement"],
            "forward_positions": dict(sorted(values["forward_positions"].items())),
            "reverse_complement_positions": dict(sorted(values["reverse_complement_positions"].items())),
        }
    dominant_count = first12.most_common(1)[0][1] if first12 else 0
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "first_1m_compressed_sha256": sha256_prefix(path),
        "records_read": records,
        "malformed_records": malformed,
        "sample_reads_requested": sample_reads,
        "search_window_bases": search_window,
        "first12_unique_count": len(first12),
        "dominant_first12_count": dominant_count,
        "dominant_first12_fraction": (dominant_count / records) if records else None,
        "dominant_first12_fingerprint": redacted_counter(first12, 8),
        "header_index_like_count": header_index_like,
        "published_rtb_visibility": clean_hits,
        "published_rtb_total_hits": total_known_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastq-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sample-reads", type=int, default=100_000)
    parser.add_argument("--search-window", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        path = args.fastq_root / run / f"{run}_1.fastq.gz"
        if not path.is_file():
            raise SystemExit(f"missing complete R1 FASTQ: {path}")
        results.append(audit_run(path, args.sample_reads, args.search_window))
    total_hits = sum(int(row["published_rtb_total_hits"]) for row in results)
    result: dict[str, Any] = {
        "audit_version": "phase0-r1-rtb-barcode-visibility-candidate-v1",
        "source_method_reference": "PMC11601540 methods: published RTB examples; barcode described at 5-prime R1",
        "runs": results,
        "published_rtb_total_hits_across_runs": total_hits,
        "status": "NO_SYSTEMATIC_KNOWN_RTB_BARCODE_SIGNAL_IN_BOUNDED_R1_SAMPLE_CANDIDATE_ONLY" if total_hits <= len(results) else "KNOWN_RTB_BARCODE_SIGNAL_OBSERVED_REQUIRES_SEPARATE_VALIDATION",
        "interpretation": {
            "barcode_payload_absence_proven": False,
            "demultiplexing_status_proven": False,
            "raw_to_processed_crosswalk_proven": False,
            "primary_labels_admitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
            "notes": [
                "The bounded sample tests only the three published example RTB sequences.",
                "A common R1 prefix is recorded as a fingerprint and is not treated as a barcode identity.",
                "A single low-frequency hit is not treated as systematic barcode evidence.",
            ],
        },
        "raw_sequences_emitted": False,
        "raw_headers_emitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "runs": len(results),
        "sample_reads_per_run": args.sample_reads,
        "published_rtb_total_hits": total_hits,
        "status": result["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
