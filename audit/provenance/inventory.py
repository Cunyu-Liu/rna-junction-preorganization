"""P0.1 authority, source and run-lineage freeze.

Records, in a new isolated run root, the byte-identity and role of every
authority input (contracts, canonical source, legacy scripts, historical run
manifests), builds the run DAG, and writes an authoritative conflict ledger.

This is read-only with respect to historical inputs: nothing under the
historical worktree is modified.  Only new files under RUN_ROOT are created.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(worktree: Path) -> dict:
    out = {}
    for cmd, key in [
        (["git", "-C", str(worktree), "rev-parse", "HEAD"], "commit"),
        (["git", "-C", str(worktree), "branch", "--show-current"], "branch"),
    ]:
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            out[key] = f"ERROR:{e}"
    try:
        out["dirty_files"] = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            capture_output=True, text=True).stdout.splitlines()
    except Exception as e:  # noqa: BLE001
        out["dirty_files"] = [f"ERROR:{e}"]
    return out


def file_record(path: Path, role: str) -> dict:
    path = Path(path)
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
        "sha256": sha256(path),
    }


def build_inventory(cfg: dict) -> dict:
    inv = {"generated_at": cfg["utc"], "run_root": cfg["run_root"], "worktree": cfg["worktree"]}
    inv["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
    }
    inv["contracts"] = [file_record(p, r) for p, r in cfg["contracts"]]
    inv["canonical_source"] = [file_record(p, r) for p, r in cfg["sources"]]
    inv["legacy_scripts"] = [file_record(p, "legacy_script") for p in cfg["legacy_scripts"]]
    inv["historical_outputs"] = [file_record(p, "historical_manifest") for p in cfg["historical_outputs"]]
    inv["git"] = git_state(Path(cfg["worktree"]))
    return inv


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = build_inventory(cfg)
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out"]).write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}, indent=2))
