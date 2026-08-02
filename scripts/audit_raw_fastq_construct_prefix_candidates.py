#!/usr/bin/env python3
"""Compare a bounded raw FASTQ construct-prefix sample with processed counts.

The result is deliberately candidate evidence only.  A sampled read-prefix
distribution cannot prove which raw accession produced a processed artifact,
and this script never promotes a run/condition binding.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


P5_PRIMER = "GGGCTTCGGCCC"
CONSTRUCT_MEMBERS = {
    "pdb_library_1": "data/raw-jsons/constructs/pdb_library_1.json",
    "pdb_library_2": "data/raw-jsons/constructs/pdb_library_2.json",
    "pdb_library_3": "data/raw-jsons/constructs/pdb_library_3.json",
}


def sha256_prefix(path: Path, limit: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_library(archive: Path, prefix_length: int) -> dict[str, str]:
    with zipfile.ZipFile(archive) as handle:
        raw = io.TextIOWrapper(handle.open("data/csvs/library_sequences.csv"), encoding="utf-8")
        rows = list(csv.DictReader(raw))
    prefix_to_name: dict[str, str] = {}
    for row in rows:
        name = row["name"].replace("seq_", "construct", 1)
        sequence = row["sequence"].strip().upper().replace("U", "T")
        prefix = sequence[len(P5_PRIMER) : len(P5_PRIMER) + prefix_length]
        if prefix in prefix_to_name:
            raise ValueError(f"non-unique library prefix: {prefix_length} {prefix}")
        prefix_to_name[prefix] = name
    if len(prefix_to_name) != 7500:
        raise ValueError(f"expected 7500 unique library prefixes, found {len(prefix_to_name)}")
    return prefix_to_name


def load_processed(archive: Path) -> dict[str, dict[str, int]]:
    with zipfile.ZipFile(archive) as handle:
        result = {}
        for library, member in CONSTRUCT_MEMBERS.items():
            rows = json.loads(handle.read(member))
            result[library] = {row["name"]: int(row["num_reads"]) for row in rows}
            if len(result[library]) != 7500:
                raise ValueError(f"{library} expected 7500 processed rows")
    return result


def read_sample(path: Path, prefix_to_name: dict[str, str], prefix_length: int, limit: int) -> dict[str, object]:
    counts: Counter[str] = Counter()
    records = 0
    malformed = 0
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as handle:
        while records < limit:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline().strip().upper()
            plus = handle.readline()
            quality = handle.readline()
            if not plus or not quality or not sequence:
                malformed += 1
                break
            records += 1
            if sequence.startswith(P5_PRIMER):
                prefix = sequence[len(P5_PRIMER) : len(P5_PRIMER) + prefix_length]
                name = prefix_to_name.get(prefix)
                if name is not None:
                    counts[name] += 1
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "first_1m_compressed_sha256": sha256_prefix(path),
        "records_read": records,
        "malformed_records": malformed,
        "primer_prefix_records": sum(counts.values()),
        "assigned_constructs": len(counts),
        "unassigned_records": records - sum(counts.values()),
        "assignment_fraction": (sum(counts.values()) / records) if records else None,
        "counts": dict(sorted(counts.items())),
    }


def rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[ordered[k][0]] = mean_rank
        i = j
    return result


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or not x:
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return (sum(a * b for a, b in zip(dx, dy)) / denom) if denom else None


def cosine(x: list[float], y: list[float]) -> float | None:
    denom = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return (sum(a * b for a, b in zip(x, y)) / denom) if denom else None


def compare(raw: dict[str, object], processed: dict[str, dict[str, int]]) -> dict[str, object]:
    names = [f"construct{i}" for i in range(7500)]
    raw_counts = raw["counts"]
    raw_vector = [float(raw_counts.get(name, 0)) for name in names]
    result = {}
    for library, values in processed.items():
        processed_vector = [float(values[name]) for name in names]
        result[library] = {
            "pearson_count_correlation": correlation(raw_vector, processed_vector),
            "spearman_count_correlation": correlation(rank(raw_vector), rank(processed_vector)),
            "cosine_count_similarity": cosine(raw_vector, processed_vector),
            "processed_num_reads_sum": int(sum(processed_vector)),
            "raw_sample_count_sum": int(sum(raw_vector)),
            "admission": "NOT_ADMITTED",
        }
    return result


def build_audit(archive: Path, fastq: Path, sample_reads: int, prefix_length: int) -> dict[str, object]:
    prefix_to_name = load_library(archive, prefix_length)
    processed = load_processed(archive)
    raw = read_sample(fastq, prefix_to_name, prefix_length, sample_reads)
    return {
        "schema": "phase0_raw_fastq_construct_prefix_candidate_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RAW_FASTQ_PREFIX_SAMPLE_CANDIDATE_ONLY",
        "source_processed_archive": {
            "path": str(archive),
            "sha256": sha256_file(archive),
        },
        "source_raw_fastq": raw,
        "method": {
            "read_file": "R2 FASTQ only",
            "sampling": "first N gzip records only",
            "primer": P5_PRIMER,
            "prefix_length_after_primer": prefix_length,
            "library_identity": "library_sequences.csv seq_N renamed to constructN; U->T",
            "comparison": "raw sampled construct counts versus processed JSON num_reads",
            "not_identity_proof": True,
            "limitations": [
                "bounded early-file sample is not a full-run count",
                "read-prefix assignment is not rna-map reproduction",
                "correlation cannot establish processed artifact provenance",
            ],
        },
        "processed_comparisons": compare(raw, processed),
        "gate": {
            "raw_processed_crosswalk_gate_effect": "NO_CHANGE",
            "primary_labels_admitted": False,
            "phase0_gate_effect": "NO_PHASE_0_PASS",
            "scientific_gate_effect": "NO_UNLOCK",
            "training_started": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fastq", type=Path, required=True)
    parser.add_argument("--sample-reads", type=int, default=250_000)
    parser.add_argument("--prefix-length", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.archive, args.fastq, args.sample_reads, args.prefix_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    print(f"records_read={audit['source_raw_fastq']['records_read']}")
    print(f"primer_prefix_records={audit['source_raw_fastq']['primer_prefix_records']}")
    print(f"assigned_constructs={audit['source_raw_fastq']['assigned_constructs']}")
    for library, row in audit["processed_comparisons"].items():
        print(f"{library} pearson={row['pearson_count_correlation']:.9f} spearman={row['spearman_count_correlation']:.9f}")


if __name__ == "__main__":
    main()
