#!/usr/bin/env python3
"""Audit read-depth candidates without admitting a raw/processed crosswalk.

This script deliberately treats SRA ``spots`` and processed ``num_reads`` as
non-equivalent aggregate quantities.  It emits candidate rankings for review,
but it cannot promote a candidate to an accepted condition binding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RUNS = {
    "SRR31402663": "rna_library_trial2",
    "SRR31402664": "rna_library_trial1",
    "SRR35766784": "rna_library_nomod",
    "SRR35766785": "rna_library_denature",
    "SRR38259812": "rna_lib_37C_2min",
}

PUBLIC_CANDIDATES = {
    "pdb_library_1": ["SRR31402663", "SRR31402664"],
    "pdb_library_2": ["SRR31402663", "SRR31402664"],
    "pdb_library_3": ["SRR31402663", "SRR31402664"],
    "pdb_library_37C_2min": ["SRR38259812"],
    "pdb_library_denature": ["SRR35766785"],
    "pdb_library_nomod": ["SRR35766784"],
}

KNOWN_PUBLIC_BINDINGS = {
    "pdb_library_37C_2min": "SRR38259812",
    "pdb_library_denature": "SRR35766785",
    "pdb_library_nomod": "SRR35766784",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_runinfo(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run = row.get("Run", "")
            if run in RUNS:
                result[run] = {
                    "run": run,
                    "library_name": row.get("LibraryName", ""),
                    "spots": int(row["spots"]),
                    "bases": int(row["bases"]),
                    "library_layout": row.get("LibraryLayout", ""),
                    "platform": row.get("Platform", ""),
                    "model": row.get("Model", ""),
                }
    missing = sorted(set(RUNS) - set(result))
    if missing:
        raise ValueError(f"selected SRA runs missing from runinfo: {missing}")
    return result


def read_processed(archive: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive) as handle:
        members = sorted(
            name
            for name in handle.namelist()
            if name.startswith("data/raw-jsons/constructs/pdb_library_")
            and name.endswith(".json")
        )
        for member in members:
            name = Path(member).stem
            records = json.loads(handle.read(member))
            values = [int(record["num_reads"]) for record in records]
            result[name] = {
                "member": member,
                "records": len(records),
                "num_reads_sum": sum(values),
                "num_reads_median": statistics.median(values),
                "num_reads_min": min(values),
                "num_reads_max": max(values),
            }
    expected = set(PUBLIC_CANDIDATES)
    if set(result) != expected:
        raise ValueError(f"processed construct set mismatch: {sorted(set(result) ^ expected)}")
    return result


def build_audit(runinfo_path: Path, archive_path: Path) -> dict[str, object]:
    runinfo = parse_runinfo(runinfo_path)
    processed = read_processed(archive_path)
    rows = []
    for construct in sorted(processed):
        processed_row = processed[construct]
        candidates = []
        for run in PUBLIC_CANDIDATES[construct]:
            raw_spots = int(runinfo[run]["spots"])
            processed_sum = int(processed_row["num_reads_sum"])
            ratio = processed_sum / raw_spots
            candidates.append(
                {
                    "run": run,
                    "library_name": runinfo[run]["library_name"],
                    "raw_spots": raw_spots,
                    "processed_num_reads_sum": processed_sum,
                    "processed_to_raw_spots_ratio": ratio,
                    "absolute_fractional_difference_from_one": abs(ratio - 1.0),
                }
            )
        candidates.sort(key=lambda item: item["absolute_fractional_difference_from_one"])
        rows.append(
            {
                "processed_construct": construct,
                "processed": processed_row,
                "public_binding": KNOWN_PUBLIC_BINDINGS.get(construct),
                "candidate_basis": (
                    "public_condition_label_plus_read_depth_sanity_check"
                    if construct in KNOWN_PUBLIC_BINDINGS
                    else "read_depth_candidate_only"
                ),
                "candidates_ranked_by_read_depth": candidates,
                "admission": "NOT_ADMITTED",
            }
        )
    return {
        "schema": "phase0_condition_run_read_depth_candidate_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_ONLY_RAW_PROCESSED_CROSSWALK_UNRESOLVED",
        "source_runinfo": {
            "path": str(runinfo_path),
            "sha256": sha256_file(runinfo_path),
            "selected_runs": sorted(runinfo),
        },
        "source_processed_archive": {
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
        },
        "method": {
            "raw_quantity": "NCBI SRA runinfo spots",
            "processed_quantity": "sum of processed construct JSON num_reads",
            "comparison": "processed_num_reads_sum / raw_spots",
            "ranking": "absolute distance of ratio from one",
            "not_identity_proof": True,
            "reason": (
                "processed num_reads may reflect trimming, filtering, pairing, "
                "and rna-map processing; it is not an accession-preserving key"
            ),
        },
        "rows": rows,
        "gate": {
            "raw_processed_crosswalk_gate_effect": "NO_CHANGE",
            "primary_labels_admitted": False,
            "phase0_gate_effect": "NO_PHASE_0_PASS",
            "scientific_gate_effect": "NO_UNLOCK",
            "training_started": False,
        },
        "required_next_evidence": [
            "official accession-preserving sample/condition manifest or equivalent",
            "or independently adjudicated manual review with evidence_ref and evidence_sha256 per row",
            "no inference from read depth alone may satisfy the crosswalk gate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runinfo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.runinfo, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    print(f"output={args.output}")
    for row in audit["rows"]:
        top = row["candidates_ranked_by_read_depth"][0]
        print(
            f"{row['processed_construct']} top={top['run']} "
            f"ratio={top['processed_to_raw_spots_ratio']:.9f} admission={row['admission']}"
        )


if __name__ == "__main__":
    main()
