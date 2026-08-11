"""Unit tests for the k-mer-representation latent-operator model."""
from __future__ import annotations

import numpy as np

from audit.benchmark.kmer_features import (
    raw_features, build_raw_by_jid, fit_scaler, transform,
)
from audit.models.kmer_latent_operator import (
    make_kmer_latent_adapter, KMER_LATENT_OPERATOR, K,
)

N_FEAT = 4 ** K + 3  # kmer freqs + length + 2 part lengths


def _rows():
    rows = []
    for j, seq in enumerate(["GGGAAACCC", "CCCCGGGG", "AAAAUUUU"]):
        for s in (1, 2):
            rows.append({"source_row_id": f"{j}_{s}", "jid": f"j{j}",
                         "junction_seq": seq, "scaf": s, "helix_seq": f"h{j}_{s}",
                         "y": -5.0 - j, "cens": (j == 1)})
    return rows


def test_feature_dimension():
    assert raw_features("GGGAAACCC").shape == (N_FEAT,)


def test_feature_finite_and_deterministic():
    a = raw_features("GGGAAACCC")
    b = raw_features("GGGAAACCC")
    assert np.all(np.isfinite(a))
    assert np.allclose(a, b)


def test_feature_uses_sequence():
    assert not np.allclose(raw_features("GGGAAACCC"), raw_features("AAAAUUUU"))


def test_frequency_normalization():
    # A pure homopolymer has frequency concentrated in one k-mer; the freq
    # sub-vector (first 4**k) must sum to 1.0.
    f = raw_features("CCCCCCCC", k=3)
    assert abs(f[:64].sum() - 1.0) < 1e-6


def test_scaler_train_only():
    rows = _rows()
    by_jid = build_raw_by_jid(rows, k=K)
    jids = sorted(set(str(r["jid"]) for r in rows))
    mean, sd = fit_scaler(jids, by_jid, k=K)
    assert mean.shape == (N_FEAT,) and sd.shape == (N_FEAT,)
    X = transform(jids, by_jid, mean, sd, k=K)
    assert X.shape == (len(jids), N_FEAT)
    assert np.all(np.isfinite(X)) and np.all(sd > 0)


def test_model_registered():
    assert "kmer_latent_operator" in KMER_LATENT_OPERATOR
    fit, predict = KMER_LATENT_OPERATOR["kmer_latent_operator"]
    assert callable(fit) and callable(predict)


def test_fit_predict_and_gate_fields():
    fit, predict = make_kmer_latent_adapter()
    rows = _rows()
    model = fit(rows)
    assert model["kind"] == "kmer_latent_operator"
    assert "success" in model and "nit" in model and "final_grad_norm" in model
    assert "grad" in model and "bounds" in model
    assert model["theta"].shape == (N_FEAT,)
    mu, sigma, cp, support, abstain = predict(model, rows)
    assert len(mu) == len(rows)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and sigma.min() > 0


def test_unseen_scaffold_abstains():
    fit, predict = make_kmer_latent_adapter()
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
    print("kmer tests", "PASS" if failed == 0 else f"{failed} FAILURES")