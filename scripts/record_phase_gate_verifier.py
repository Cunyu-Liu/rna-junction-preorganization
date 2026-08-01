#!/usr/bin/env python3
"""Record a completed phase-governance verifier without changing science gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", default="PASS_GOVERNANCE_INVARIANTS")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    manifest = args.code_root / "manifests/project_manifest.json"
    history = args.code_root / "manifests/history"
    history.mkdir(parents=True, exist_ok=True)
    backup = history / f"project_manifest_{args.run_id}.json"
    if not backup.exists():
        shutil.copy2(manifest, backup)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["phase_gate_verifier"] = {
        "status": args.status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": args.exit_code,
        "violations": [],
        "log": str(args.log.relative_to(args.code_root)),
        "log_sha256": sha256(args.log),
        "scientific_gate_effect": "NO_UNLOCK",
    }
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VERIFIER_RECORD_UPDATED", "manifest": str(manifest), "log_sha256": data["phase_gate_verifier"]["log_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
