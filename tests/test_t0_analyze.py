"""Unit tests for the T0 data-admission analysis helpers."""
import os
import sys
import unittest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "tecto"))
from t0_analyze import _norm_num, _load_records  # noqa: E402


class TestNormNum(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(_norm_num(None))

    def test_float_and_int(self):
        self.assertEqual(_norm_num(3.5), 3.5)
        self.assertEqual(_norm_num(3), 3.0)

    def test_numeric_string(self):
        self.assertEqual(_norm_num("-7.1"), -7.1)
        self.assertEqual(_norm_num(" 1.25 "), 1.25)

    def test_parse_failure(self):
        self.assertIsNone(_norm_num("abc"))
        self.assertIsNone(_norm_num(""))


class TestLoadRecords(unittest.TestCase):
    def test_loads_and_normalizes_numeric_fields(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"junction_id":"1","dg10":"-7.1","dg9":"-6.0","motif_type":"0x0"}\n')
            f.write('{"junction_id":"2","dg10":null,"dg10_interp":"-5.0"}\n')
            path = f.name
        try:
            recs = _load_records(path)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0]["dg10"], -7.1)
            self.assertEqual(recs[0]["dg9"], -6.0)
            self.assertIsNone(recs[1]["dg10"])
            self.assertEqual(recs[1]["dg10_interp"], -5.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()