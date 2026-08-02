#!/usr/bin/env python3
"""Fail-closed merge of externally transferred, independently verified chunks.

This tool is for engineering transfer closure only.  It validates the
acquisition manifest, re-hashes every staged chunk, concatenates chunks in
their declared byte order, checks the merged ZIP archive, and refuses to
overwrite an existing payload or report.  It never interprets scientific
labels or admits a phase gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.stage = args.stage.resolve()
    args.manifest = args.manifest.resolve()
    args.contract = args.contract.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    started_at = utc_now()
    assembly: Path | None = None
    verified_chunks: list[dict[str, Any]] = []

    base_record: dict[str, Any] = {
        "schema": "phase0-verified-uploaded-chunk-merge-v1",
        "run_id": args.run_id,
        "started_at_utc": started_at,
        "stage": str(args.stage),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "report": str(args.report),
        "raw_sequence_content_emitted": False,
        "scientific_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "audit_boundary": (
            "Engineering transfer and archive-integrity evidence only; this "
            "record is not construct identity reconciliation, label admission, "
            "model evidence, or a scientific conclusion."
        ),
    }

    try:
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {args.report}")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
        if not args.stage.is_dir():
            raise FileNotFoundError(f"stage directory does not exist: {args.stage}")
        if not args.manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {args.manifest}")
        if not args.contract.is_file():
            raise FileNotFoundError(f"contract does not exist: {args.contract}")
        contract_sha256 = sha256_file(args.contract)
        if contract_sha256 != CONTRACT_SHA256:
            raise ValueError("contract SHA256 does not match the frozen 1.1 contract")

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "LOCAL_CHUNKS_VERIFIED_READY_FOR_BATCH_TRANSFER":
            raise ValueError("transfer manifest is not in the verified-ready state")
        expected_bytes = int(manifest["expected_bytes"])
        chunk_bytes = int(manifest["chunk_bytes"])
        records = manifest["chunks"]
        if expected_bytes <= 0 or chunk_bytes != 128 * 1024 * 1024:
            raise ValueError("manifest does not declare the required 128 MiB chunk size")
        if len(records) != int(manifest["chunk_count"]):
            raise ValueError("manifest chunk_count does not match the chunk records")

        previous_end = -1
        expected_names: set[str] = set()
        for expected_index, record in enumerate(records):
            index = int(record["index"])
            start = int(record["start"])
            end = int(record["end"])
            expected_chunk_bytes = end - start + 1
            name = Path(str(record["path"])).name
            path = args.stage / name
            if index != expected_index:
                raise ValueError(f"chunk index is not sequential: {index}")
            if start != previous_end + 1:
                raise ValueError(f"chunk start is not contiguous: {name}")
            if expected_chunk_bytes <= 0 or expected_chunk_bytes > chunk_bytes:
                raise ValueError(f"chunk byte range is invalid: {name}")
            if end >= expected_bytes:
                raise ValueError(f"chunk exceeds declared object size: {name}")
            if not path.is_file():
                raise FileNotFoundError(f"declared chunk is missing: {path}")
            observed_bytes = path.stat().st_size
            if observed_bytes != expected_chunk_bytes:
                raise ValueError(
                    f"chunk size mismatch for {name}: "
                    f"observed={observed_bytes} expected={expected_chunk_bytes}"
                )
            observed_sha256 = sha256_file(path)
            if observed_sha256 != record["sha256"]:
                raise ValueError(f"chunk SHA256 mismatch for {name}")
            expected_names.add(name)
            verified_chunks.append(
                {
                    "index": index,
                    "name": name,
                    "start": start,
                    "end": end,
                    "bytes": observed_bytes,
                    "sha256": observed_sha256,
                    "status": "REMOTE_CHUNK_REHASH_PASS",
                }
            )
            previous_end = end

        if previous_end + 1 != expected_bytes:
            raise ValueError("declared chunks do not cover the complete object")
        actual_names = {path.name for path in args.stage.glob("chunk-*.bin")}
        if actual_names != expected_names:
            raise ValueError(
                "stage contains undeclared or missing chunk files: "
                f"extra={sorted(actual_names - expected_names)} "
                f"missing={sorted(expected_names - actual_names)}"
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        assembly = args.output.parent / f".{args.output.name}.{args.run_id}.partial"
        if assembly.exists():
            raise FileExistsError(f"refusing to reuse existing assembly partial: {assembly}")

        digest = hashlib.sha256()
        merged_bytes = 0
        with assembly.open("wb") as destination:
            for record in verified_chunks:
                with (args.stage / record["name"]).open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        destination.write(block)
                        digest.update(block)
                        merged_bytes += len(block)
            destination.flush()
            os.fsync(destination.fileno())
        if merged_bytes != expected_bytes:
            raise ValueError(
                f"merged size mismatch: observed={merged_bytes} expected={expected_bytes}"
            )

        archive_members = 0
        bad_member: str | None = None
        with zipfile.ZipFile(assembly) as archive:
            archive_members = len(archive.infolist())
            bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed for member: {bad_member}")

        merged_sha256 = digest.hexdigest()
        os.replace(assembly, args.output)
        finished_at = utc_now()
        report = {
            **base_record,
            "status": "MERGE_COMPLETE_ZIP_INTEGRITY_PASS",
            "finished_at_utc": finished_at,
            "contract_sha256": contract_sha256,
            "manifest_sha256": sha256_file(args.manifest),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "source_file_id": manifest.get("source_file_id"),
            "source_url": manifest.get("source_url"),
            "expected_bytes": expected_bytes,
            "chunk_bytes": chunk_bytes,
            "chunk_count": len(verified_chunks),
            "chunks": verified_chunks,
            "merged_bytes": merged_bytes,
            "merged_sha256": merged_sha256,
            "zip_member_count": archive_members,
            "zip_crc_status": "PASS",
            "output_installed": True,
        }
        atomic_json_write(args.report, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "bytes": merged_bytes,
                    "sha256": merged_sha256,
                    "zip_member_count": archive_members,
                    "report": str(args.report),
                }
            )
        )
        return 0
    except Exception as exc:
        failure = {
            **base_record,
            "status": "MERGE_BLOCKED_FAIL_CLOSED",
            "finished_at_utc": utc_now(),
            "failure": {
                "exception_type": type(exc).__name__,
                "message": str(exc)[:4000],
            },
            "verified_chunks_before_failure": verified_chunks,
            "assembly_partial": str(assembly) if assembly and assembly.exists() else None,
            "output_installed": False,
        }
        if not args.report.exists():
            atomic_json_write(args.report, failure)
        print(json.dumps(failure["failure"], ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
