"""Unit tests for the frozen RNA-FM baseline (contract §9.4, GPU-aware)."""
from __future__ import annotations

import numpy as np
import pytest

from audit.benchmark.frozen_lm import _contig, _add_intercept, fit_frozen_head, predict_frozen_head


def test_contig_strips_separators():
    assert _contig("CUAG_CUAAG") == "CUAGCUAAG"
    assert _contig("AC_GG&CC_GU") == "ACGGCCGU"
    assert _contig(None) == ""


def test_add_intercept_columns():
    X = np.zeros((3, 4))
    Xb = _add_intercept(X)
    assert Xb.shape == (3, 5)
    assert np.all(Xb[:, 0] == 1.0)


def test_head_fit_predict_shapes():
    embs = {}
    rng = np.random.default_rng(0)
    for i, s in enumerate(["SEQ%d" % i for i in range(20)]):
        embs[s] = rng.standard_normal(32)
    train = [{"junction_seq": "SEQ%d" % i, "y": float(rng.standard_normal()),
              "cens": bool(i % 3 == 0)} for i in range(20)]
    test = [{"junction_seq": "SEQ%d" % i, "y": 0.0, "cens": False} for i in range(20)]
    model = fit_frozen_head(train, embs)
    assert "beta" in model and model["gate"]["converged"] is not None
    mu, sigma, cp, support, abstain = predict_frozen_head(model, test, embs)
    assert mu.shape == (len(test),) and sigma.shape == (len(test),)
    assert set(support.tolist()) == {True} and set(abstain.tolist()) == {False}
    assert np.all(np.isfinite(mu))


def test_embed_sequences_needs_gpu_tokenizer():
    """embed_sequences requires a real frozen model; without one we at least
    assert the pooling helper contract via a minimal fake."""
    import torch
    from audit.benchmark.frozen_lm import embed_sequences

    class FakeTok:
        def __call__(self, seqs, padding=None, return_tensors=None):
            n = len(seqs)
            return {"attention_mask": torch.ones((n, 4))}

    class FakeOut:
        last_hidden_state = torch.ones((3, 4, 5))

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **kw):
            return FakeOut()

    embs = embed_sequences(["ACGU", "AAAA", "GGGG"], FakeTok(), FakeModel(), "cpu")
    assert set(embs) == {"ACGU", "AAAA", "GGGG"}
    for v in embs.values():
        assert v.shape == (5,)
        assert np.allclose(v, 1.0)  # mean-pool of all-ones hidden = 1.0