"""Shared pytest fixtures for the audit test suite (P0.2 repair).

The durable, read-only data artifacts (CleaningLedger.jsonl, DataProfile.json)
live in the parent audit run root on the server.  The ``data_dir`` fixture
points at that directory so tests that validate ledger/count conservation can
read the real, frozen artifacts without copying them into the repo.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Frozen parent audit run root (read-only).  Contains data/CleaningLedger.jsonl
# and data/DataProfile.json.  Tests only read these files.
DATA_ROOT = Path("/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/data")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    ledger = DATA_ROOT / "CleaningLedger.jsonl"
    profile = DATA_ROOT / "DataProfile.json"
    if not ledger.exists() or not profile.exists():
        pytest.skip(f"frozen data artifacts not present under {DATA_ROOT}")
    return DATA_ROOT