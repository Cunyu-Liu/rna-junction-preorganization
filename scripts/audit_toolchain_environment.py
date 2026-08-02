#!/usr/bin/env python3
"""Record isolated toolchain readiness without reading scientific inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rna-junction-preorganization/toolchain-environment-audit/v1"
REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "biopython": "Bio",
    "viennarna": "RNA",
    "rna_map": "rna_map",
    "torch": "torch",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"argv": args, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "argv": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def import_probe(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # keep the exact failure as evidence
        return {"module": module_name, "ok": False, "error": repr(exc)}
    return {
        "module": module_name,
        "ok": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "file": str(getattr(module, "__file__", "unknown")),
    }


def cuda_probe(torch_probe: dict[str, Any]) -> dict[str, Any]:
    if not torch_probe["ok"]:
        return {"checked": False, "cuda_available": False, "reason": "torch_import_failed"}
    import torch  # type: ignore

    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices = []
        for index in range(count):
            devices.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        return {
            "checked": True,
            "cuda_available": available,
            "device_count": count,
            "torch_cuda_version": str(torch.version.cuda),
            "devices": devices,
        }
    except Exception as exc:  # keep CUDA failure fail-closed
        return {"checked": True, "cuda_available": False, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--environment-yml", type=Path, required=True)
    parser.add_argument("--install-log", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install-pid", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    imports = {name: import_probe(module) for name, module in REQUIRED_IMPORTS.items()}
    torch_probe = imports["torch"]
    cuda = cuda_probe(torch_probe)
    process = run_checked(["ps", "-o", "pid=,stat=,etime=,cmd=", "-p", str(args.install_pid)])
    pip = run_checked([str(args.prefix / "bin" / "pip"), "freeze"])
    conda = run_checked([str(args.prefix.parent.parent / "bin" / "conda"), "list", "--explicit", "-p", str(args.prefix)])

    all_imports_ok = all(item["ok"] for item in imports.values())
    ready = all_imports_ok and cuda.get("cuda_available", False) and cuda.get("device_count", 0) > 0
    record = {
        "schema": SCHEMA,
        "audit_id": datetime.now(timezone.utc).strftime("toolchain-%Y%m%dT%H%M%SZ"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {"path": str(args.contract), "sha256": sha256_file(args.contract)},
        "source": {"commit": args.source_commit, "environment_yml": {"path": str(args.environment_yml), "sha256": sha256_file(args.environment_yml)}},
        "environment": {"prefix": str(args.prefix), "python": sys.executable, "python_version": platform.python_version()},
        "install": {"pid": args.install_pid, "log": str(args.install_log), "log_sha256": sha256_file(args.install_log), "process_evidence": process},
        "imports": imports,
        "cuda_probe": cuda,
        "package_lock_evidence": {"pip_freeze": pip, "conda_explicit": conda},
        "training_started": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
        "status": "READY_FOR_GPU_ONLY_TOOLCHAIN_USE" if ready else "BLOCKED_TOOLCHAIN_NOT_READY",
        "readiness_rule": "all required imports succeed and torch reports at least one CUDA device; this is not a scientific or Phase 0 acceptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "output": str(args.output), "sha256": sha256_file(args.output)}, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
