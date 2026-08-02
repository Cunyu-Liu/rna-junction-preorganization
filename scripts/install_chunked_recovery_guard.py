#!/usr/bin/env python3
"""Install one verified chunked FASTQ recovery without overwriting anything.

The source must already have a terminal chunked-merge/gzip audit.  The target
must not exist.  A preserved ``.partial`` sibling is never deleted or
replaced; it is recorded as independent historical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--chunked-audit", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    audit_path = args.chunked_audit.resolve()
    target = args.target.resolve()
    report = args.report.resolve()
    if report.exists():
        raise SystemExit(f"refusing to overwrite report: {report}")
    if not source.is_file() or not audit_path.is_file():
        raise SystemExit("source and chunked audit must exist")
    if source == target:
        raise SystemExit("source and target must be different")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "CHUNKED_MERGE_COMPLETE_GZIP_OK":
        raise SystemExit("chunked audit is not a terminal gzip-pass result")
    if audit.get("raw_sequence_content_emitted") is not False or audit.get("scientific_labels_admitted") is not False:
        raise SystemExit("chunked audit is not fail-closed")
    expected_bytes = audit.get("merged_bytes")
    expected_sha256 = audit.get("merged_sha256")
    if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
        raise SystemExit("chunked audit lacks merged size/hash")

    partial = Path(str(target) + ".partial")
    partial_record = {
        "path": str(partial),
        "exists": partial.is_file(),
        "size_bytes": partial.stat().st_size if partial.is_file() else None,
        "preserved": True,
    }
    if target.exists():
        target_bytes = target.stat().st_size
        target_sha256 = sha256(target)
        payload = {
            "schema_version": "phase0-chunked-recovery-install-v1",
            "status": "BLOCKED_TARGET_EXISTS_NO_OVERWRITE",
            "installed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "chunked_audit": str(audit_path),
            "chunked_audit_sha256": sha256(audit_path),
            "expected_bytes": expected_bytes,
            "expected_sha256": expected_sha256,
            "target": str(target),
            "target_existed_before": True,
            "target_bytes": target_bytes,
            "target_sha256": target_sha256,
            "target_matches_chunked_audit": target_bytes == expected_bytes and target_sha256 == expected_sha256,
            "target_overwritten": False,
            "partial": partial_record,
            "scientific_labels_admitted": False,
            "raw_sequence_content_emitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
            "deletions": [],
        }
        dump_atomic(report, payload)
        print(json.dumps({"status": payload["status"], "target": str(target), "target_matches_chunked_audit": payload["target_matches_chunked_audit"], "report": str(report)}, sort_keys=True))
        return 2

    source_bytes = source.stat().st_size
    source_sha256 = sha256(source)
    if source_bytes != expected_bytes or source_sha256 != expected_sha256:
        raise SystemExit("source changed or disagrees with terminal chunked audit")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.install.{os.getpid()}.tmp"
    if temporary.exists():
        raise SystemExit(f"refusing to overwrite existing install temporary: {temporary}")
    copied_sha256 = hashlib.sha256()
    copied_bytes = 0
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            for block in iter(lambda: src.read(16 * 1024 * 1024), b""):
                dst.write(block)
                copied_sha256.update(block)
                copied_bytes += len(block)
            dst.flush()
            os.fsync(dst.fileno())
        copied_sha256_hex = copied_sha256.hexdigest()
        if copied_bytes != expected_bytes or copied_sha256_hex != expected_sha256:
            raise SystemExit("copied temporary disagrees with terminal chunked audit")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise SystemExit(f"target appeared during install; refusing overwrite: {target}") from exc
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()

    target_stat = target.stat()
    payload = {
        "schema_version": "phase0-chunked-recovery-install-v1",
        "status": "CHUNKED_RECOVERY_INSTALLED_NO_OVERWRITE",
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_bytes": source_bytes,
        "source_sha256": source_sha256,
        "chunked_audit": str(audit_path),
        "chunked_audit_sha256": sha256(audit_path),
        "target": str(target),
        "target_bytes": target_stat.st_size,
        "target_sha256": copied_sha256_hex,
        "target_existed_before": False,
        "target_overwritten": False,
        "partial": partial_record,
        "scientific_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "deletions": [],
    }
    dump_atomic(report, payload)
    print(json.dumps({"status": payload["status"], "target": str(target), "target_sha256": copied_sha256_hex, "report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
