"""Unit tests for the R2 pairing-null and decision module."""
import numpy as np
import pandas as pd
import pytest

from audit import r2_null_and_decision as r2


def _pair_df(nll_full, nll_ns):
    return pd.DataFrame({
        "fold": [f"f{i}" for i in range(len(nll_full))],
        "pooled_junction_macro_nll_full": nll_full,
        "pooled_junction_macro_nll_ns": nll_ns,
    })


def test_axis_stat_positive_gain():
    m = _pair_df([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    nll_full, nll_ns, rel = r2.axis_stat(m)
    assert rel == pytest.approx(0.5)   # (2-1)/2


def test_axis_stat_zero_gain():
    m = _pair_df([1.0, 1.0], [1.0, 1.0])
    _, _, rel = r2.axis_stat(m)
    assert rel == pytest.approx(0.0)


def test_pairing_null_centered_and_symmetric():
    m = _pair_df([1.0] * 8, [1.0] * 8)   # null true: full == no-sequence
    nulls = r2.pairing_null(m, n_null=2000, seed=3)
    assert len(nulls) == 2000
    # all per-fold differences are exactly 0 -> point mass at 0
    assert nulls.mean() == 0.0
    # observed 0 gain gives p ~ 1.0 (no separation under the null)
    p = float((np.sum(nulls >= 0.0) + 1) / (2000 + 1))
    assert p > 0.9


def test_pairing_null_separates_strong_signal():
    # strong, consistent positive gain on MANY folds -> fine null resolution
    m = _pair_df([1.0] * 20, [2.0] * 20)
    nulls = r2.pairing_null(m, n_null=2000, seed=7)
    p = float((np.sum(nulls >= 0.5) + 1) / (2000 + 1))
    assert p < 0.01   # observed gain 0.5 is far in the right tail


def test_pairing_null_deterministic_with_seed():
    m = _pair_df([1.0] * 6, [1.5] * 6)
    a = r2.pairing_null(m, n_null=100, seed=1)
    b = r2.pairing_null(m, n_null=100, seed=1)
    np.testing.assert_array_equal(a, b)
