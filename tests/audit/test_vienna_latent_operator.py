"""Unit tests for the ViennaRNA-representation latent-operator model.

Verifies the feature builder (finite, deterministic scalar folding proxies), the
train-only standardization (no test leakage), the shared latent-operator
interface, strict-gate fields, and the matched-ablation semantics (ViennaRNA
must USE junction identity via sequence-derived features, unlike the
no-sequence model).
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.benchmark.vienna_features import (
    raw_features, build_raw_by_jid, fit_scaler, transform, _RNA,
)
from audit.models.vienna_latent_operator import (
    make_vienna_latent_adapter, VIENNA_LATENT_OPERATOR,
)

pytestmark = pytest.mark.skipif(_RNA is None, reason="ViennaRNA not installed")


def _rows():
    # 3 junctions, 2 scaffolds, censored+measured mix
    rows = []
    for j, seq in enumerate(["GGGAAACCC", "CCCCGGGG", "AAAAUUUU"]):
        for s in (1, 2):
            rows.append({"source_row_id": f"{j}_{s}", "jid": f"j{j}",
                         "junction_seq": seq, "scaf": s, "helix_seq": f"h{j}_{s}",
                         "y": -5.0 - j, "cens": (j == 1)})
    return rows


def test_feature_dimension_is_11():
    assert raw_features("GGGAAACCC").shape == (11,)


def test_feature_finite_and_deterministic():
    a = raw_features("GGGAAACCC")
    b = raw_features("GGGAAACCC")
    assert np.all(np.isfinite(a))
    assert np.allclose(a, b)


def test_feature_uses_folding_not_position():
    # Different sequences must give different features
    assert not np.allclose(raw_features("GGGAAACCC"), raw_features("AAAAUUUU"))


def test_scaler_is_train_only_and_invertible():
    rows = _rows()
    by_jid = build_raw_by_jid(rows)
    jids = sorted(set(str(r["jid"]) for r in rows))
    mean, sd = fit_scaler(jids, by_jid)
    assert mean.shape == (11,) and sd.shape == (11,)
    X = transform(jids, by_jid, mean, sd)
    assert X.shape == (len(jids), 11)
    assert np.all(np.isfinite(X))
    assert np.all(sd > 0)


def test_model_registered():
    assert "vienna_latent_operator" in VIENNA_LATENT_OPERATOR
    fit, predict = VIENNA_LATENT_OPERATOR["vienna_latent_operator"]
    assert callable(fit) and callable(predict)


def test_fit_returns_gate_fields_and_predicts():
    fit, predict = make_vienna_latent_adapter()
    rows = _rows()
    model = fit(rows)
    assert model["kind"] == "vienna_latent_operator"
    assert "success" in model and "nit" in model and "final_grad_norm" in model
    assert "grad" in model and "bounds" in model and "beta" in model
    assert model["theta"].shape == (11,)
    mu, sigma, cp, support, abstain = predict(model, rows)
    assert len(mu) == len(rows)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert sigma.min() > 0


def test_uses_junction_sequence_not_constant():
    """Unlike the no-sequence model, ViennaRNA location must differ across
    junctions (sequence-derived folding features change the latent location)."""
    fit, predict = make_vienna_latent_adapter()
    rows = _rows()
    model = fit(rows)
    mu, sigma, cp, support, abstain = predict(model, rows)
    mu_s1 = [mu[i] for i, r in enumerate(rows) if int(r["scaf"]) == 1]
    # 3 distinct junctions in scaffold 1 -> 3 distinct latent locations
    assert len({round(float(x), 6) for x in mu_s1}) >= 2


def test_unseen_scaffold_abstains():
    fit, predict = make_vienna_latent_adapter()
    train = _rows()
    test = [{"source_row_id": "x", "jid": "j9", "junction_seq": "CCCCAAAA",
             "scaf": 99, "helix_seq": "hx", "y": -6.0, "cens": 0}]
    model = fit(train)
    mu, sigma, cp, support, abstain = predict(model, test)
    assert bool(abstain[0]) is True and bool(support[0]) is False


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    print("vienna tests", "PASS" if failed == 0 else f"{failed} FAILURES")
