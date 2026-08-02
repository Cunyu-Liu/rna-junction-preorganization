#!/usr/bin/env python3
"""Audit an isolated RNA-MAP replay probe without emitting controlled payloads.

The probe is engineering/provenance evidence only.  It records tool versions,
input/output hashes, and bounded failure/success metadata; it never promotes a
raw run to a processed condition and never emits sequences, reactivities, or
mutation values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11601540/"
RNA_MAP_URL = "https://github.com/YesselmanLab/rna_map"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_capture(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "version_line": next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            next((line.strip() for line in result.stderr.splitlines() if line.strip()), ""),
        )[:240],
    }


def collect_tools(env_prefix: Path) -> dict[str, object]:
    bin_dir = env_prefix / "bin"
    tools = {}
    for name, args in {
        "python": [str(bin_dir / "python"), "--version"],
        "rna_map": [str(bin_dir / "rna-map"), "--help"],
        "bowtie2": [str(bin_dir / "bowtie2"), "--version"],
        "fastqc": [str(bin_dir / "fastqc"), "--version"],
        "cutadapt": [str(bin_dir / "cutadapt"), "--version"],
        "trim_galore": [str(bin_dir / "trim_galore"), "--version"],
    }.items():
        tools[name] = run_capture(args)

    python = bin_dir / "python"
    if python.is_file():
        probe = (
            "import importlib.metadata as m; "
            "names=['rna-map','numpy','pandas','biopython','PyYAML']; "
            "norm=lambda s: ''.join(c for c in s.lower() if c.isalnum()); "
            "print({n: next((d.version for d in m.distributions() "
            "if norm(d.metadata.get('Name',''))==norm(n)), None) for n in names})"
        )
        result = run_capture([str(python), "-c", probe])
        tools["python_packages"] = result
    return tools


def input_summary(run_root: Path) -> dict[str, object]:
    path = run_root / "inputs_manifest.json"
    if not path.is_file():
        return {"present": False, "status": "INPUT_MANIFEST_MISSING"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "source_fastq_run": data.get("source_fastq_run"),
        "sample_records_per_mate": data.get("sample_records_per_mate"),
        "reference_count": data.get("reference_count"),
        "raw_sequence_content_emitted": data.get("raw_sequence_content_emitted"),
        "input_sha256": data.get("sha256", {}),
    }


def archive_summary(archive: Path) -> dict[str, object]:
    members = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.filename == "data/csvs/library_sequences.csv" or (
                info.filename.startswith("data/mutation-histograms/")
                and info.filename.endswith(".p")
            ):
                members.append(
                    {
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                    }
                )
    return {
        "path": str(archive),
        "sha256": sha256_file(archive),
        "selected_member_count": len(members),
        "selected_members": members,
    }


def log_summary(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    exit_match = re.search(r"exit_code=(\d+)", text)
    errors = []
    for marker in (
        "ValueError: preset",
        "DREEMMissingRequirementsException:",
        "Traceback (most recent call last):",
    ):
        if marker in text:
            errors.append(marker.rstrip(":"))
    if exit_match and exit_match.group(1) == "0":
        status = "REPLAY_PROBE_PASS_ENGINEERING_ONLY"
    elif "DREEMMissingRequirementsException:" in text:
        status = "BLOCKED_REPLAY_ENVIRONMENT"
    elif exit_match:
        status = "REPLAY_PROBE_FAILED"
    else:
        status = "REPLAY_PROBE_INCOMPLETE"
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "status": status,
        "exit_code": int(exit_match.group(1)) if exit_match else None,
        "error_markers": errors,
        "analysis_completed_marker": "Analysis completed successfully" in text,
    }


def output_inventory(run_root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or "inputs" in path.parts:
            continue
        entries.append(
            {
                "relative_path": str(path.relative_to(run_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--env-prefix", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if sha256_file(args.archive) != "241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681":
        raise SystemExit("processed archive hash mismatch")
    result = {
        "schema_version": "phase0-rna-map-replay-probe-audit-v1",
        "run_id": args.run_id,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": CONTRACT_SHA256,
        "run_root": str(args.run_root),
        "sources": {"article_methods": PMC_URL, "processing_software": RNA_MAP_URL},
        "toolchain": collect_tools(args.env_prefix),
        "inputs": input_summary(args.run_root),
        "processed_archive": archive_summary(args.archive),
        "logs": [log_summary(path) for path in args.log],
        "outputs": output_inventory(args.run_root),
        "primary_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "phase0_gate_effect": "NO_PHASE_0_PASS",
        "scientific_gate_effect": "NO_UNLOCK",
        "training_started": False,
        "interpretation_boundary": (
            "This is a CPU engineering replay probe. Even a successful bounded or full replay "
            "would require separate row-level raw-to-processed evidence and the contract manual "
            "review thresholds before any primary labels or scientific gate could be admitted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "log_count": len(args.log), "output_count": len(result["outputs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
