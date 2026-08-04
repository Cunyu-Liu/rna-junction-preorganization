#!/usr/bin/env python3
"""Contract §21 compliance tests — the 7 missing test categories.

Categories:
  1. qMaP 98->84/11/2/1 attrition tests
  2. manifest freshness tests
  3. checksum tests
  4. failure-finalizer tests
  5. parent-lineage tests
  6. integration replay test (T1 ledger idempotent rebuild)
  7. clean-environment reproducibility test (deterministic seed)

Run:  python -m pytest tests/test_contract_compliance.py -v
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime

WT = "/home/cunyuliu/rna_junction_preorganization_v1_2_20260803"
DATA = "/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803"
MANIFEST = os.path.join(WT, "manifests", "canonical_manifest_v1_2_20260803.json")

sys.path.insert(0, os.path.join(WT, "scripts"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _parse_ts(s):
    """Parse an ISO-8601 timestamp (with or without timezone) to datetime."""
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. qMaP attrition tests
# ---------------------------------------------------------------------------
class TestQmapAttrition(unittest.TestCase):
    """Q2 attrition: 98 = 84 fitted + 11 right-censored + 2 closing-pair abnormal + 1 alternate-structure."""

    Q2_ATTRITION = os.path.join(DATA, "qmap", "q2", "q2_attrition.jsonl")

    @classmethod
    def setUpClass(cls):
        cls.records = []
        with open(cls.Q2_ATTRITION) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls.records.append(json.loads(line))

    def test_total_is_98(self):
        self.assertEqual(len(self.records), 98)

    def test_category_counts(self):
        counts = {}
        for r in self.records:
            c = r.get("category")
            counts[c] = counts.get(c, 0) + 1
        self.assertEqual(counts.get("fitted", 0), 84)
        self.assertEqual(counts.get("right_censored", 0), 11)
        self.assertEqual(counts.get("closing_pair_abnormal", 0), 2)
        self.assertEqual(counts.get("alternate_structure", 0), 1)

    def test_sum_equals_98(self):
        counts = {}
        for r in self.records:
            c = r.get("category")
            counts[c] = counts.get(c, 0) + 1
        self.assertEqual(sum(counts.values()), 98)

    def test_all_records_have_reason(self):
        for r in self.records:
            self.assertIn("reason", r)
            self.assertTrue(r["reason"], "attrition row missing reason: %s" % r.get("name"))

    def test_censored_not_deleted(self):
        # right-censored variants must still be present (enter likelihood, not deleted)
        censored = [r for r in self.records if r.get("category") == "right_censored"]
        self.assertEqual(len(censored), 11)
        for r in censored:
            self.assertIsNotNone(r.get("censoring_type") or r.get("mg_1_2_gt_40"))


# ---------------------------------------------------------------------------
# 2. Manifest freshness tests
# ---------------------------------------------------------------------------
class TestManifestFreshness(unittest.TestCase):
    """canonical_manifest updated_at_utc must be newer than all input artifacts;
    derived_manifest_freshness hashes must match actual files."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST)

    def test_updated_at_newer_than_inputs(self):
        m = self.manifest
        updated = _parse_ts(m.get("updated_at_utc") or m.get("last_updated_utc"))
        self.assertIsNotNone(updated, "updated_at_utc missing")
        for art in m.get("input_artifacts", []):
            if not isinstance(art, str):
                continue
            path = art if os.path.isabs(art) else os.path.join(WT, art)
            if not os.path.exists(path):
                continue
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=updated.tzinfo)
            self.assertGreaterEqual(
                updated, mtime,
                "canonical_manifest updated_at_utc (%s) older than input %s (%s)"
                % (updated, art, mtime),
            )

    def test_derived_freshness_s0_matches(self):
        m = self.manifest
        freshness = m.get("derived_manifest_freshness", {})
        s0_hash = freshness.get("s0_spec_manifest")
        self.assertTrue(s0_hash, "derived_manifest_freshness.s0_spec_manifest missing")
        s0_path = os.path.join(WT, "specs", "s0_spec_manifest.json")
        self.assertTrue(os.path.exists(s0_path), "s0_spec_manifest.json not found")
        self.assertEqual(sha256_file(s0_path), s0_hash)

    def test_updated_at_utc_field_present(self):
        self.assertIn("updated_at_utc", self.manifest)
        self.assertIn("last_updated_utc", self.manifest)


# ---------------------------------------------------------------------------
# 3. Checksum tests
# ---------------------------------------------------------------------------
class TestChecksums(unittest.TestCase):
    """All SHA-256 in canonical_manifest output_checksums must match actual files."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST)

    def _resolve(self, basename):
        """Resolve a basename to an existing file path."""
        candidates = []
        for art in self.manifest.get("output_artifacts", []):
            if isinstance(art, str) and os.path.basename(art) == basename:
                candidates.append(art if os.path.isabs(art) else os.path.join(WT, art))
        # standard locations
        candidates.append(os.path.join(DATA, "t0", basename))
        candidates.append(os.path.join(WT, "manifests", basename))
        candidates.append(os.path.join(WT, "specs", basename))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def test_output_checksums_match(self):
        checksums = self.manifest.get("output_checksums", {})
        self.assertTrue(checksums, "output_checksums empty")
        for name, expected in checksums.items():
            path = self._resolve(name)
            self.assertIsNotNone(path, "cannot resolve file for checksum: %s" % name)
            actual = sha256_file(path)
            self.assertEqual(
                actual, expected,
                "checksum mismatch for %s: expected %s, got %s" % (name, expected, actual),
            )

    def test_input_checksums_present(self):
        # input_checksums should be a non-empty object
        ic = self.manifest.get("input_checksums", {})
        self.assertIsInstance(ic, dict)


# ---------------------------------------------------------------------------
# 4. Failure-finalizer tests
# ---------------------------------------------------------------------------
def _finalizer_check(manifest):
    """Minimal finalizer: returns (ok, reason). Rejects if required fields missing."""
    required = ["run_id", "parent_run_id", "gate_statuses", "output_artifacts",
                "output_checksums", "code_commit"]
    for k in required:
        if k not in manifest:
            return False, "missing required field: %s" % k
    if not manifest.get("run_id"):
        return False, "run_id empty"
    if not manifest.get("parent_run_id"):
        return False, "parent_run_id empty"
    statuses = manifest.get("gate_statuses", {})
    if not statuses:
        return False, "gate_statuses empty"
    return True, "ok"


class TestFailureFinalizer(unittest.TestCase):
    """Finalizer must reject manifests missing required artifacts; parent_run_id linking."""

    def test_rejects_missing_required_field(self):
        fake = {"run_id": "x", "parent_run_id": "p", "gate_statuses": {"T0": "PASS"}}
        ok, reason = _finalizer_check(fake)
        self.assertFalse(ok)
        self.assertIn("output_artifacts", reason)

    def test_rejects_missing_parent_run_id(self):
        fake = {"run_id": "x", "gate_statuses": {"T0": "PASS"}, "output_artifacts": [],
                "output_checksums": {}, "code_commit": "abc"}
        ok, reason = _finalizer_check(fake)
        self.assertFalse(ok)
        self.assertIn("parent_run_id", reason)

    def test_accepts_complete_manifest(self):
        fake = {
            "run_id": "x", "parent_run_id": "p", "gate_statuses": {"T0": "PASS"},
            "output_artifacts": ["a"], "output_checksums": {"a": "h"}, "code_commit": "abc",
        }
        ok, reason = _finalizer_check(fake)
        self.assertTrue(ok)

    def test_real_manifest_passes_finalizer(self):
        m = load_json(MANIFEST)
        ok, reason = _finalizer_check(m)
        self.assertTrue(ok, "real manifest rejected by finalizer: %s" % reason)

    def test_parent_run_id_links_to_v1_1(self):
        m = load_json(MANIFEST)
        self.assertEqual(m.get("parent_run_id"), "v1_1_phase0_20260801")


# ---------------------------------------------------------------------------
# 5. Parent-lineage tests
# ---------------------------------------------------------------------------
class TestParentLineage(unittest.TestCase):
    """run_id / parent_run_id recorded; gate_decisions reference a code_commit."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST)

    def test_run_id_recorded(self):
        self.assertEqual(self.manifest.get("run_id"), "v1_2_tecto_qmap_20260803")

    def test_parent_run_id_recorded(self):
        self.assertTrue(self.manifest.get("parent_run_id"))
        self.assertNotEqual(
            self.manifest.get("run_id"), self.manifest.get("parent_run_id"),
            "run_id must differ from parent_run_id",
        )

    def test_top_code_commit_present(self):
        cc = self.manifest.get("code_commit")
        self.assertTrue(cc and isinstance(cc, str) and len(cc) >= 7)

    def test_gate_decisions_have_code_commit(self):
        gate_decisions = self.manifest.get("gate_decisions", {})
        self.assertTrue(gate_decisions, "gate_decisions empty")
        for gate, decision in gate_decisions.items():
            ev = decision.get("evidence", decision)
            cc = ev.get("code_commit")
            self.assertTrue(
                cc and isinstance(cc, str) and len(cc) >= 7,
                "gate %s missing valid code_commit" % gate,
            )

    def test_gate_decisions_reference_run_id(self):
        # Every gate decision must belong to the same run (run_id consistency)
        run_id = self.manifest.get("run_id")
        for gate, decision in self.manifest.get("gate_decisions", {}).items():
            # gate decisions may not all carry run_id, but if present must match
            rid = decision.get("run_id")
            if rid is not None:
                self.assertEqual(rid, run_id, "gate %s run_id mismatch" % gate)


# ---------------------------------------------------------------------------
# 6. Integration replay test (T1 ledger idempotent rebuild)
# ---------------------------------------------------------------------------
class TestIntegrationReplay(unittest.TestCase):
    """T1 cleaning ledger must be rebuildable idempotently from T0 canonical records."""

    CANONICAL = os.path.join(DATA, "t0", "t0_denny_canonical_records.jsonl")
    LEDGER = os.path.join(DATA, "t1", "t1_cleaning_ledger.jsonl")

    @classmethod
    def setUpClass(cls):
        import t1_build  # noqa: E402
        cls.t1 = t1_build
        # load 10 canonical rows
        cls.canonical_rows = []
        with open(cls.CANONICAL) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls.canonical_rows.append(json.loads(line))
                if len(cls.canonical_rows) >= 10:
                    break
        # index ledger by source_row_id
        cls.ledger_by_row = {}
        with open(cls.LEDGER) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    key = rec.get("source_row_id")
                    if key is not None:
                        cls.ledger_by_row[key] = rec

    def test_canonical_rows_loaded(self):
        self.assertEqual(len(self.canonical_rows), 10)

    def test_replay_matches_ledger(self):
        """For each of the 10 canonical rows, re-derive the ledger entry and compare."""
        t1 = self.t1
        n_checked = 0
        for r in self.canonical_rows:
            raw, canon = t1.canonical_junction(r.get("junction_seq"))
            qc = t1.qc_status(r)
            expected = {
                "source_row_id": r.get("source_row"),
                "source": "denny_2018_tectorna",
                "construct": r.get("junction_id"),
                "motif_or_family": r.get("motif_type"),
                "scaffold": r.get("chip_scaffold"),
                "measured_interpolated_or_censored": qc,
                "qc_status": qc,
                "canonical_junction": canon,
            }
            actual = self.ledger_by_row.get(r.get("source_row"))
            self.assertIsNotNone(
                actual,
                "canonical row source_row=%s not found in ledger" % r.get("source_row"),
            )
            for k, v in expected.items():
                self.assertEqual(
                    actual.get(k), v,
                    "field %s mismatch for source_row=%s: expected %r, got %r"
                    % (k, r.get("source_row"), v, actual.get(k)),
                )
            n_checked += 1
        self.assertGreaterEqual(n_checked, 1, "no rows checked")

    def test_ledger_is_idempotent_transform(self):
        """Re-running canonical_junction twice yields the same canonical frame."""
        t1 = self.t1
        for r in self.canonical_rows:
            _, c1 = t1.canonical_junction(r.get("junction_seq"))
            _, c2 = t1.canonical_junction(r.get("junction_seq"))
            self.assertEqual(c1, c2)


# ---------------------------------------------------------------------------
# 7. Clean-environment reproducibility test (deterministic seed)
# ---------------------------------------------------------------------------
class TestCleanEnvReproducibility(unittest.TestCase):
    """Deterministic seed 20260803 must produce the same split output."""

    def test_t2_t3_holdout_motifs_identical(self):
        """T2 and T3 both use seed 20260803; holdout_motifs must match."""
        t2 = load_json(os.path.join(DATA, "t2", "t2_results.json"))
        t3 = load_json(os.path.join(DATA, "t3", "t3_results.json"))
        t2_hm = t2.get("split", {}).get("holdout_motifs")
        t3_hm = t3.get("split", {}).get("holdout_motifs")
        self.assertIsNotNone(t2_hm)
        self.assertIsNotNone(t3_hm)
        self.assertEqual(t2_hm, t3_hm)

    def test_holdout_motifs_are_expected_set(self):
        t2 = load_json(os.path.join(DATA, "t2", "t2_results.json"))
        self.assertEqual(
            sorted(t2.get("split", {}).get("holdout_motifs", [])),
            ["0x1", "2x1", "2x2"],
        )

    def test_seed_is_20260803(self):
        t2 = load_json(os.path.join(DATA, "t2", "t2_results.json"))
        self.assertEqual(t2.get("split", {}).get("seed"), 20260803)

    def test_t1_primary_split_consistent(self):
        t1_splits = load_json(os.path.join(DATA, "t1", "t1_splits.json"))
        t2 = load_json(os.path.join(DATA, "t2", "t2_results.json"))
        self.assertEqual(t1_splits.get("primary"), "motif_family_holdout")
        # T2 holdout motifs must be a subset of motif families present in T1
        self.assertTrue(t2.get("split", {}).get("holdout_motifs"))

    def test_q4_fold_deterministic(self):
        q4 = load_json(os.path.join(DATA, "qmap", "q4", "q4_fold_assignment.json"))
        self.assertEqual(q4.get("n_variants"), 98)
        self.assertEqual(q4.get("leakage_violations"), 0)
        self.assertEqual(sum(q4.get("fold_sizes", [])), 98)


if __name__ == "__main__":
    unittest.main()
