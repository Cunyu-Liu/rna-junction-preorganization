#!/usr/bin/env python3
"""Safely continue explicitly stopped project downloader processes in place."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from typing import Any


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
PROJECT_TOKEN = "rna_junction_preorganization_v1_1_20260801"
DOWNLOADER_TOKEN = "scripts/download_ena_fastq.py"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_proc(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    status_path = proc / "status"
    cmdline_path = proc / "cmdline"
    status: dict[str, str] = {}
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    wchan_path = proc / "wchan"
    wchan = wchan_path.read_text(encoding="utf-8", errors="replace").strip() if wchan_path.is_file() else None
    return {
        "pid": pid,
        "ppid": status.get("PPid"),
        "state": status.get("State"),
        "name": status.get("Name"),
        "tracer_pid": status.get("TracerPid"),
        "wchan": wchan,
        "cmdline": cmdline,
    }


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", action="append", dest="pids", required=True, type=int)
    parser.add_argument("--project-token", default=PROJECT_TOKEN)
    parser.add_argument("--downloader-token", default=DOWNLOADER_TOKEN)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    contract = args.contract.resolve()
    output = args.output.resolve()
    if not contract.is_file():
        parser.error(f"missing contract: {contract}")
    observed_contract_sha256 = sha256_file(contract)
    if observed_contract_sha256 != CONTRACT_SHA256:
        parser.error(f"contract hash mismatch: {observed_contract_sha256}")
    if output.exists():
        parser.error(f"refusing to overwrite existing audit artifact: {output}")
    if len(set(args.pids)) != len(args.pids):
        parser.error("duplicate PID is not allowed")
    if not args.pids:
        parser.error("at least one PID is required")

    before: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    requested_pids = {str(pid) for pid in args.pids}
    for pid in args.pids:
        try:
            observed = read_proc(pid)
        except FileNotFoundError:
            validation_errors.append(f"pid {pid} does not exist")
            continue
        before.append(observed)
        if not str(observed.get("state", "")).startswith("T"):
            validation_errors.append(f"pid {pid} is not stopped: {observed.get('state')}")
        if args.project_token not in observed.get("cmdline", ""):
            validation_errors.append(f"pid {pid} does not match project token")
        command = observed.get("cmdline", "")
        is_python_downloader = args.downloader_token in command
        is_scoped_ena_curl = (
            observed.get("name") == "curl"
            and "ftp.sra.ebi.ac.uk" in command
            and ".fastq.gz.partial" in command
            and "/phase0/source_payloads/" in command
            and observed.get("ppid") in requested_pids
        )
        if not is_python_downloader and not is_scoped_ena_curl:
            validation_errors.append(f"pid {pid} does not match project downloader or scoped ENA child curl")
        if observed.get("tracer_pid") not in {None, "0"}:
            validation_errors.append(f"pid {pid} has a tracer: {observed.get('tracer_pid')}")

    report: dict[str, Any] = {
        "schema_version": "phase0-safe-continue-stopped-downloader-v1",
        "status": "BLOCKED_VALIDATION" if validation_errors else "VALIDATION_PASSED",
        "run_id": args.run_id,
        "checked_at_utc": utc_now(),
        "contract_path": str(contract),
        "contract_sha256": observed_contract_sha256,
        "pids": args.pids,
        "before": before,
        "validation_errors": validation_errors,
        "signal": "SIGCONT",
        "signal_sent": False,
        "new_process_started": False,
        "final_files_overwritten": False,
        "partial_files_deleted": False,
        "raw_sequence_content_emitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    if validation_errors:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(output, report)
        print(json.dumps({"status": report["status"], "output": str(output), "validation_errors": validation_errors}, ensure_ascii=False))
        return 2

    try:
        for pid in args.pids:
            os.kill(pid, signal.SIGCONT)
        report["signal_sent"] = True
        report["status"] = "SIGCONT_SENT"
    except ProcessLookupError as exc:
        report["status"] = "BLOCKED_PROCESS_EXITED_DURING_CONTINUE"
        report["validation_errors"].append(str(exc))
    except PermissionError as exc:
        report["status"] = "BLOCKED_PERMISSION_DENIED"
        report["validation_errors"].append(str(exc))
    time.sleep(0.25)
    after: list[dict[str, Any]] = []
    for pid in args.pids:
        try:
            after.append(read_proc(pid))
        except FileNotFoundError:
            after.append({"pid": pid, "state": "EXITED_AFTER_SIGCONT"})
    report["after"] = after
    report["finished_at_utc"] = utc_now()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output, report)
    print(json.dumps({"status": report["status"], "signal_sent": report["signal_sent"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["signal_sent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
