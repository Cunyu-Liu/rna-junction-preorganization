"""Tests for the CanonicalStateManifest governance module."""
import os
import sys
import tempfile
import unittest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "governance"))
from canonical_manifest import (  # noqa: E402
    CanonicalStateManifest,
    finalize_gate,
    validate_schema,
    sha256_text,
)


def make_manifest():
    return CanonicalStateManifest.new(
        contract_version="v1.2",
        contract_sha256="x" * 64,
        code_commit="abc123",
        worktree_path="/tmp/foo",
        run_id="r1",
        parent_run_id="p1",
        host="host",
    )


class TestManifest(unittest.TestCase):
    def test_initial_state(self):
        m = make_manifest()
        self.assertEqual(m.data["current_operational_state"], "BLOCKED_AT_TECTO_DATA_ADMISSION")
        self.assertEqual(m.data["scientific_unlock"], "NO_UNLOCK")
        self.assertEqual(m.data["claim_class"], "NOT_ADJUDICATED")
        self.assertEqual(m.data["gate_statuses"]["T0"], "BLOCKED")
        self.assertEqual(m.data["gate_statuses"]["Q0"], "NOT_STARTED")
        self.assertEqual(m.data["qmap_terminal_disposition"], "NOT_ADJUDICATED")
        self.assertEqual(m.data["sentinel_status"], "NOT_STARTED")

    def test_schema_valid(self):
        m = make_manifest()
        self.assertEqual(validate_schema(m.data), [])

    def test_non_finalizer_cannot_write_pass(self):
        m = make_manifest()
        with self.assertRaises(ValueError):
            m.write_gate("T0", "PASS", finalizer=False)

    def test_finalizer_can_write_pass(self):
        m = make_manifest()
        m.write_gate("T0", "PASS", decision={}, finalizer=True)
        self.assertEqual(m.data["gate_statuses"]["T0"], "PASS")

    def test_finalize_gate_partial_when_missing_artifact(self):
        m = make_manifest()
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "missing.json")
            status = finalize_gate(
                m, "T0", {"gate_id": "T0"},
                required_artifacts=[art], checksum_valid=True,
                tests_passed=True, contract_hash_ok=True, schema_ok=True,
            )
        self.assertEqual(status, "PARTIAL_ENGINEERING_EVIDENCE")
        self.assertEqual(m.data["gate_statuses"]["T0"], "BLOCKED")

    def test_finalize_gate_pass_when_all_ok(self):
        m = make_manifest()
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "ok.json")
            with open(art, "w") as f:
                f.write("{}")
            status = finalize_gate(
                m, "T0", {"gate_id": "T0"},
                required_artifacts=[art], checksum_valid=True,
                tests_passed=True, contract_hash_ok=True, schema_ok=True,
            )
        self.assertEqual(status, "PASS")
        self.assertEqual(m.data["gate_statuses"]["T0"], "PASS")

    def test_save_load_roundtrip(self):
        m = make_manifest()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            h = m.save(p)
            m2 = CanonicalStateManifest.load(p)
            self.assertEqual(m2.data["run_id"], "r1")
            self.assertEqual(sha256_text(open(p, encoding="utf-8").read()) != "", True)
            self.assertEqual(len(h), 64)

    def test_invalid_gate_state_rejected(self):
        m = make_manifest()
        with self.assertRaises(ValueError):
            m.write_gate("T0", "BOGUS", finalizer=True)


if __name__ == "__main__":
    unittest.main()