"""v1.4 contract test-scope isolation.

The repository carries forward legacy test files that validate the **v1.2 parent
contract** (section 21) against the v1.2 run_root:

- tests/test_contract_compliance.py  -> needs RNA_V12_RUN_ROOT / RNA_V12_MANIFEST_PATH / RNA_V12_PARENT_RUN_ID
- tests/test_schema_validation.py    -> targets manifests/canonical_manifest_v1_2_unbound.json

These are NOT part of the v1.4 acceptance DAG
(C0/T6/Q6/Q7/N0/B0/B1/B2/R1/M1/E1). Running them from the v1.4 worktree without
the v1.2 environment fails with FileNotFoundError. They are excluded from
collection here so that `pytest tests/` reflects the v1.4 acceptance suite; the
v1.2 parent contract is verified in its own worktree.

The v1.4 acceptance suite is the gate/invariant test files enumerated in
tests/test_v14_test_scope.py.
"""
import pytest

# Legacy v1.2 parent-contract test files (excluded from v1.4 collection).
collect_ignore = ["test_contract_compliance.py", "test_schema_validation.py"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "v14_contract: v1.4 acceptance test (C0/T6/Q6/Q7/N0/B0/B1/B2/R1/M1/E1)",
    )
