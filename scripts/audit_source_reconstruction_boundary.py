#!/usr/bin/env python3
"""Audit whether the public source tree can stand in for the frozen construct reference.

This audit records only provenance and executable-boundary metadata.  It does not
emit sequence content, labels, or derived scientific measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if not args.contract.is_file():
        raise SystemExit(f"missing contract: {args.contract}")
    if sha256_file(args.contract) != CONTRACT_SHA256:
        raise SystemExit("contract sha256 mismatch")
    if not args.source_root.is_dir():
        raise SystemExit(f"missing source root: {args.source_root}")

    build_script = args.source_root / "dms_3d_features" / "library_build.py"
    if not build_script.is_file():
        raise SystemExit(f"missing public library-build script: {build_script}")
    build_text = build_script.read_text(encoding="utf-8")

    required_paths = [
        "data/csvs/motif_sequences.csv",
        "data/raw-jsons/constructs",
        "data/jsons/pdb_library.json",
        "data/Sequences.xlsx",
    ]
    input_inventory = []
    for relative in required_paths:
        path = args.source_root / relative
        entry = {"relative_path": relative, "exists": path.exists(), "is_file": path.is_file()}
        if path.is_file():
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
        input_inventory.append(entry)

    markers = {
        "imports_random": bool(re.search(r"^import\s+random\b", build_text, flags=re.MULTILINE)),
        "random_operations": len(re.findall(r"\brandom\.(?:shuffle|choice|randint)\s*\(", build_text)),
        "motif_csv_reference": bool(re.search(r"csvs/motif_sequences\.csv", build_text)),
        "example_desired_sequences_10": "desired_sequences=10" in build_text,
        "writes_pdb_library_json": bool(re.search(r"jsons/pdb_library\.json", build_text)),
    }

    output = {
        "schema_version": "phase0-source-reconstruction-boundary-audit-v1",
        "run_id": args.run_id,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "source_root": str(args.source_root),
        "source_git_head": git_head(args.source_root),
        "library_build_script": str(build_script),
        "library_build_script_sha256": sha256_file(build_script),
        "required_input_inventory": input_inventory,
        "public_source_markers": markers,
        "published_library_size_claim": 7500,
        "official_construct_identity_proven": False,
        "algorithmic_equivalence_proven": False,
        "processed_dms_payload_admitted": False,
        "primary_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "status": "BLOCKED_SOURCE_RECONSTRUCTION_NOT_EQUIVALENT",
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "interpretation_boundary": (
            "The public library-build code is a stochastic generator requiring a missing motif CSV; "
            "its example entry point writes a small generated library.  This is source semantics only "
            "and cannot substitute for the frozen official construct reference or processed-DMS payload."
        ),
        "required_next_evidence": [
            "official Sequences.xlsx or an equivalently frozen construct reference with provenance",
            "processed-DMS payload or a verified official route",
            "construct-level sequence/structure/count/background/depth reconciliation",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
