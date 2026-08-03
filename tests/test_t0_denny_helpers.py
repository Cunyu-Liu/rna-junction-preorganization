"""Unit tests for the T0 Denny canonical builder helper functions.

These tests cover the pure helper functions (is_junctionmat detection,
sublibrary normalization, and designed/crystal partition) that the T0
three-set reconstruction (1687 / 1713 / 1636) depends on. They do not read
the large workbook; they exercise the semantics directly.
"""
import os
import sys
import unittest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "tecto"))
from t0_build_denny_canonical import (  # noqa: E402
    _is_jm,
    _sublib,
    _partition,
)


class TestIsJunctionmat(unittest.TestCase):
    def test_numeric_one_is_true(self):
        self.assertTrue(_is_jm("1"))

    def test_zero_is_false(self):
        self.assertFalse(_is_jm("0"))

    def test_yes_variants(self):
        for v in ("yes", "Yes", "YES", "True", "TRUE", "true", "Y", "y"):
            self.assertTrue(_is_jm(v), v)

    def test_none_and_blank_are_false(self):
        self.assertFalse(_is_jm(None))
        self.assertFalse(_is_jm(""))
        self.assertFalse(_is_jm("  "))


class TestSublib(unittest.TestCase):
    def test_strips_curly_quotes(self):
        self.assertEqual(_sublib("\u201cjunction_conformations\u201d"), "junction_conformations")

    def test_strips_leading_whitespace(self):
        self.assertEqual(_sublib(" \u201cjunction_conformations\u201d"), "junction_conformations")

    def test_none_returns_empty(self):
        self.assertEqual(_sublib(None), "")


class TestPartition(unittest.TestCase):
    def test_partitions_designed_and_crystal(self):
        records = [
            {"junction_id": "1", "sublibrary": "\u201cjunction_conformations\u201d"},
            {"junction_id": "2", "sublibrary": "\u201cjunction_conformations_pdb\u201d"},
            {"junction_id": "3", "sublibrary": "\u201cjunction_conformations\u201d"},
        ]
        designed, crystal = _partition(records)
        self.assertEqual(designed, {"1", "3"})
        self.assertEqual(crystal, {"2"})

    def test_ignores_missing_junction_id(self):
        records = [
            {"junction_id": None, "sublibrary": "junction_conformations"},
            {"junction_id": "1", "sublibrary": "junction_conformations"},
        ]
        designed, crystal = _partition(records)
        self.assertEqual(designed, {"1"})
        self.assertEqual(crystal, set())


if __name__ == "__main__":
    unittest.main()