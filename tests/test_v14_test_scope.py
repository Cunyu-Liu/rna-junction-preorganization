"""Verify the v1.4 acceptance test scope and legacy-test isolation.

Guards the conftest.py isolation so a future change cannot silently re-admit the
legacy v1.2 parent-contract tests (which fail without the v1.2 run_root env) or
drop one of the v1.4 acceptance test files.
"""
from pathlib import Path

TESTS = Path(__file__).resolve().parent

LEGACY_V12_FILES = ["test_contract_compliance.py", "test_schema_validation.py"]

# One v1.4 acceptance test file per gate / contract invariant (B01-B10 mapping).
V14_ACCEPTANCE_FILES = [
    "test_manifest_seal.py",            # B01 manifest self-hash paradox
    "test_manifest_freshness.py",       # B02 freshness / source_commit
    "test_manifest_completeness.py",    # B03 recursive inventory
    "test_status_consistency.py",       # B04 single authority / conflict ledger
    "test_estimand_non_null_hash.py",   # B05 tecto estimand binding
    "test_estimand_data_metric_trace.py",
    "test_q6_reconstruction.py",        # B06/B07/B08 source membership
    "test_q7_transfer.py",              # B09 component / group integrity
    "test_n0_prior_art.py",
    "test_b0_benchmark.py",
    "test_b1_failure_mode.py",
    "test_b2_sensitivity.py",
    "test_r1_release.py",
    "test_m1_manuscript.py",
    "test_e1_review.py",
]


def test_legacy_v12_files_excluded_from_conftest_collect_ignore():
    conftest = (TESTS / "conftest.py").read_text()
    for f in LEGACY_V12_FILES:
        assert f in conftest, f"{f} must be listed in conftest.collect_ignore"


def test_v14_acceptance_files_present():
    missing = [f for f in V14_ACCEPTANCE_FILES if not (TESTS / f).exists()]
    assert not missing, f"missing v1.4 acceptance test files: {missing}"
