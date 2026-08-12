"""Unit tests for the join-local-context feature builder."""
import numpy as np

from audit.benchmark.vienna_local_context_features import (
    ALPHABET, N_FEAT, W, build_raw_by_jid, fit_scaler, raw_features, transform,
)


def test_feature_dim_and_binary():
    x = raw_features("CUAG_CUAAG")
    assert x.shape == (N_FEAT,)
    assert np.all((x == 0) | (x == 1))
    # each window position has at most one active base
    for i in range(2 * W):
        assert int(x[i * 4:(i + 1) * 4].sum()) <= 1


def test_window_is_join_anchored():
    # "AAAC_GGGC" -> tail of left (last W of AAAC= AAC) + head of right (first W of GGGC=GGG)
    x = raw_features("AAAC_GGGC")
    window = "AAC" + "GGG"
    for i, b in enumerate(window):
        assert x[i * 4 + ALPHABET.index(b)] == 1.0


def test_short_arm_left_all_zero():
    # left arm shorter than W -> those positions stay all-zero
    x = raw_features("AC_GGC")  # left len 2 < W=3
    tail = "AC"
    head = "GGC"[:W]
    window = tail + head
    for i, b in enumerate(window):
        assert x[i * 4 + ALPHABET.index(b)] == 1.0
    # remaining window positions empty
    assert int(x[len(window) * 4:].sum()) == 0


def test_scaler_is_identity():
    rows = [{"jid": "1", "junction_seq": "CUAG_CUAAG"},
            {"jid": "2", "junction_seq": "AAAC_GGGC"}]
    by = build_raw_by_jid(rows)
    mean, sd = fit_scaler(["1", "2"], by)
    assert np.allclose(mean, 0) and np.allclose(sd, 1)
    t = transform(["1", "2"], by, mean, sd)
    assert np.allclose(t[0], raw_features("CUAG_CUAAG"))
    assert np.allclose(t[1], raw_features("AAAC_GGGC"))


if __name__ == "__main__":
    tests = [test_feature_dim_and_binary, test_window_is_join_anchored,
             test_short_arm_left_all_zero, test_scaler_is_identity]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("vienna_local_context tests", "PASS" if failed == 0 else f"{failed} FAILURES")
