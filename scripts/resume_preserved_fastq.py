#!/usr/bin/env python3
"""Safely resume preserved ENA partial downloads after the original job exits.

This wrapper is intentionally not an autonomous retry loop. It refuses to
start while any ``download_ena_fastq.py`` process is active, delegates to the
existing atomic/no-overwrite downloader for explicitly named runs, and keeps
all stdout/stderr in the caller-selected artifact log. It never deletes or
renames an existing final payload.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def event(status: str, **extra: object) -> dict[str, object]:
    return {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def active_downloader_pids() -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        try:
            tokens = shlex.split(text)
        except ValueError:
            continue
        if not tokens:
            continue
        try:
            pid = int(tokens[0])
        except ValueError:
            continue
        interpreter_names = {"python", "python3", "python3.10"}
        is_downloader = any(
            Path(tokens[index]).name in interpreter_names
            and index + 1 < len(tokens)
            and Path(tokens[index + 1]).name == "download_ena_fastq.py"
            for index in range(1, len(tokens) - 1)
        )
        if not is_downloader:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return sorted(set(pids))


def write_event(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-script", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run", action="append", dest="runs", required=True)
    parser.add_argument("--expected-pid", required=True, type=int)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    active = active_downloader_pids()
    if active:
        payload = event(
            "BLOCKED_ACTIVE_DOWNLOADER",
            expected_pid=args.expected_pid,
            active_downloader_pids=active,
            runs=args.runs,
            scientific_gate_effect="NO_PHASE_0_PASS",
        )
        write_event(args.log, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2

    if not args.download_script.is_file():
        payload = event(
            "BLOCKED_DOWNLOAD_SCRIPT_MISSING",
            download_script=str(args.download_script),
            scientific_gate_effect="NO_PHASE_0_PASS",
        )
        write_event(args.log, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2

    start = event(
        "RESUME_START",
        expected_pid=args.expected_pid,
        runs=args.runs,
        output_root=str(args.output_root),
        final_files_not_overwritten=True,
        partial_files_deleted=False,
        scientific_gate_effect="NO_PHASE_0_PASS",
    )
    write_event(args.log, start)

    command = [
        sys.executable,
        str(args.download_script),
        "--manifest",
        str(args.manifest),
        "--output-root",
        str(args.output_root),
    ]
    for run in args.runs:
        command.extend(["--run", run])

    with args.log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, check=False, stdout=handle, stderr=subprocess.STDOUT)

    status = "RESUME_COMPLETE" if completed.returncode == 0 else "RESUME_PARTIAL_FAILURES_PRESERVED"
    finish = event(
        status,
        returncode=completed.returncode,
        runs=args.runs,
        final_files_not_overwritten=True,
        partial_files_deleted=False,
        scientific_gate_effect="NO_PHASE_0_PASS",
    )
    write_event(args.log, finish)
    print(json.dumps(finish, ensure_ascii=False, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
