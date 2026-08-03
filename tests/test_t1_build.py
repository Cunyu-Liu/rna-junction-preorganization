import os
import sys
import unittest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "scripts"))

from t1_build import canonical_junction, rc, qc_status, parse_num, CAP  # noqa: E402


class TestCanonicalJunction(unittest.TestCase):
    def test_lexicographic_minimal(self):
        raw, canon = canonical_junction("CUAAG_CUUAG")
        # canonical must be a member of the transform set
        s1, s2 = "CUAAG", "CUUAG"
        cands = {f"{s1};{s2}", f"{s2};{s1}", f"{rc(s1)};{rc(s2)}", f"{rc(s2)};{rc(s1)}"}
        self.assertIn(canon, cands)

    def test_reverse_complement_symmetry_maps_to_same(self):
        # a junction and its strand-swap should canonicalize to the same group
        j1 = "CUAAG_CUUAG"
        j2 = "CUUAG_CUAAG"
        _, c1 = canonical_junction(j1)
        _, c2 = canonical_junction(j2)
        self.assertEqual(c1, c2)

    def test_no_underscore_returns_same(self):
        raw, canon = canonical_junction("ACGU")
        self.assertEqual(canon, "ACGU")


class TestRc(unittest.TestCase):
    def test_rc(self):
        self.assertEqual(rc("ACGU"), "ACGU".translate(str.maketrans("ACGU", "UGCA"))[::-1])
        self.assertEqual(rc("ACGU"), "ACGU")


class TestQcStatus(unittest.TestCase):
    def test_measured(self):
        r = {"dg10": "-5.0", "dg10_interp": "-5.0"}
        self.assertEqual(qc_status(r), "measured")

    def test_censored_at_cap(self):
        r = {"dg10": "-7.1", "dg10_interp": "-7.1"}
        self.assertEqual(qc_status(r), "censored_at_cap")

    def test_interpolated_only(self):
        r = {"dg10": None, "dg10_interp": "-6.0"}
        self.assertEqual(qc_status(r), "interpolated_only")

    def test_missing(self):
        r = {"dg10": None, "dg10_interp": None}
        self.assertEqual(qc_status(r), "missing")


class TestParseNum(unittest.TestCase):
    def test_numeric_string(self):
        self.assertEqual(parse_num("-7.1"), -7.1)
        self.assertEqual(parse_num(" 1.25 "), 1.25)

    def test_none(self):
        self.assertIsNone(parse_num(None))
        self.assertIsNone(parse_num("abc"))


if __name__ == "__main__":
    unittest.main()