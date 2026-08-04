"""Runtime binding for an isolated RNA Junction v1.2 execution.

All builders/finalizers read paths and the contract binding from environment
variables supplied by the run launcher.  Defaults resolve to this checkout and
an explicitly unbound local run root; they never point at the historical
2026-08-03 run.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKTREE = os.path.abspath(os.environ.get("RNA_V12_WORKTREE", str(PROJECT_ROOT)))
RUN_ID = os.environ.get("RNA_V12_RUN_ID", "v1_2_tecto_qmap_unbound")
RUN_ROOT = os.path.abspath(os.environ.get("RNA_V12_RUN_ROOT", str(PROJECT_ROOT / "runroot")))
QDATA = os.path.join(RUN_ROOT, "qmap")
MANIFEST_PATH = os.path.abspath(os.environ.get(
    "RNA_V12_MANIFEST_PATH",
    os.path.join(WORKTREE, "manifests", f"canonical_manifest_{RUN_ID}.json"),
))
CONTRACT_PATH = os.path.abspath(os.environ.get(
    "RNA_V12_CONTRACT_PATH", os.path.join(WORKTREE, "contract", "1.2.docx")
))
CONTRACT_SHA256 = os.environ.get(
    "RNA_V12_CONTRACT_SHA256",
    "3ad0c9997cdea8e510f80424c4b011062f0f95a8bf8879a4659a847adcab22a0",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256() -> str | None:
    if not os.path.isfile(CONTRACT_PATH):
        return None
    return sha256_file(CONTRACT_PATH)


def verify_contract() -> bool:
    return contract_sha256() == CONTRACT_SHA256

GENERATED_PREFIXES = (
    "manifests/", "specs/", "docs/", "reports/", "Sentinel_",
)


def source_tree_dirty(status: str) -> bool:
    """Return whether tracked source/config code is dirty.

    Run-local manifests, frozen specs, reports, docs, and sentinels are
    expected outputs and are checked separately by each finalizer.  A source
    edit, contract edit, or unrelated worktree change still fails closed.
    """
    for line in status.splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if path.startswith(GENERATED_PREFIXES):
            continue
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            return True
    return False
