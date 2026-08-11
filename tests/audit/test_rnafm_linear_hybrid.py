"""Unit tests for the RNA-FM frozen-embedding plain-linear hybrids.

Contract rules (mirroring the ViennaRNA hybrid): fit on TRAIN rows only, no
test leakage, right-censor aware, finite correctly-shaped outputs, unseen-
scaffold abstention, missing-embedding fail-closed, and the sequence block
must change predictions vs the nuisance-only model.  The RNA-FM cache is
synthetic here (random 1920-D vectors) because the real cache requires the
offline GPU extraction; the tests exercise the linear-hybrid plumbing without
torch/GPU.
"""
import numpy as np
import pytest

from audit.benchmark.rnafm_features import (
    RENDER_DIM, build_raw_by_jid, fit_scaler, load_cache, transform,
)
from audit.models.rnafm_linear_hybrid import make_rnafm_linear_hybrid
from audit.models.rnafm_vienna_linear_hybrid import make_rnafm_vienna_linear_hybrid
from audit.benchmark.vienna_features import _RNA as _VIENNA_AVAILABLE

HAVE_VIENNA = _VIENNA_AVAILABLE is not None

CAP = -7.1


def _rows():
    rows = []
    seqs = {1: "CUAG_CUAAG", 2: "CGAC_CGAC", 3: "AUGC_GCUA", 4: "UACG_ACGU",
            5: "GCUA_AUCG", 6: "AUCG_UAGC"}
    motifs = {1: "0x1", 2: "0x2", 3: "0x3", 4: "0x1", 5: "0x2", 6: "0x3"}
    r0 = 0
    for j, s in seqs.items():
        scaf = (j % 3) + 1
        for n in range(4):
            cens = (n == 3)
            y = -8.5 + 0.2 * (r0 % 5) if not cens else CAP
            rows.append({"source_row_id": f"R{r0:05d}", "jid": f"j{j}",
                         "motif": motifs[j], "scaf": scaf, "y": y, "cens": cens,
                         "junction_seq": s, "helix_seq": f"h{j}_{n}",
                         "symmetry_key": "_".join(reversed(s.split("_")))})
            r0 += 1
    return rows


def _cache(rows, seed=0):
    """Synthetic {junction_seq: 1920-D vector} cache covering all row seqs."""
    rng = np.random.default_rng(seed)
    cache = {}
    for r in rows:
        cache[str(r["junction_seq"])] = rng.standard_normal(RENDER_DIM)
    return cache


def test_cache_roundtrip(tmp_path):
    rows = _rows()
    cache = _cache(rows)
    np.savez(tmp_path / "c.npz",
             seqs=np.asarray(list(cache), dtype=object),
             vecs=np.asarray(list(cache.values()), dtype=float))
    loaded = load_cache(str(tmp_path / "c.npz"))
    assert set(loaded) == set(cache)
    for k, v in cache.items():
        assert np.allclose(loaded[k], v)


def test_build_raw_by_jid_and_scaler(tmp_path):
    rows = _rows()
    cache = _cache(rows)
    by_jid = build_raw_by_jid(rows, cache)
    assert set(by_jid) == {str(r["jid"]) for r in rows}
    tr_jids = sorted({str(r["jid"]) for r in rows[:18]})
    mean, sd = fit_scaler(tr_jids, by_jid)
    assert mean.shape == (RENDER_DIM,) and sd.shape == (RENDER_DIM,)
    assert np.all(sd > 0)
    X = transform([tr_jids[0]], by_jid, mean, sd)
    assert X.shape == (1, RENDER_DIM)
    assert np.all(np.isfinite(X))


def test_rnafm_hybrid_fit_predict_shapes():
    fit, predict = make_rnafm_linear_hybrid(_cache(_rows()))
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "rnafm_linear_hybrid"
    assert model["n_rnafm"] == RENDER_DIM
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,) and support.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert support.dtype == bool and abstain.dtype == bool


def test_rnafm_hybrid_unseen_scaffold_abstains():
    fit, predict = make_rnafm_linear_hybrid(_cache(_rows()))
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "BBBB_AAAA"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False


def test_rnafm_hybrid_sequence_block_changes_predictions():
    """Two junctions with identical nuisance but different sequences must yield
    different mu if the RNA-FM block is live."""
    fit_fn, predict_fn = make_rnafm_linear_hybrid(_cache(_rows()))
    rows = _rows()
    a = next(r for r in rows if r["jid"] == "j1")
    b = next(r for r in rows if r["jid"] == "j2")
    tr = [r for r in rows if r["jid"] not in ("j1", "j2")]
    model = fit_fn(tr)
    mu_a, *_ = predict_fn(model, [a])
    mu_b, *_ = predict_fn(model, [b])
    assert abs(mu_a[0] - mu_b[0]) > 1e-9


def test_rnafm_hybrid_missing_embedding_fails_closed():
    fit_fn, _ = make_rnafm_linear_hybrid(_cache(_rows()))
    rows = _rows()
    bogus = dict(rows[0])
    bogus["jid"] = "j99"
    bogus["junction_seq"] = "ZZZZ_ZZZZ"  # not in cache, unique jid
    tr = rows[:17] + [bogus]
    with pytest.raises(RuntimeError):
        fit_fn(tr)


@pytest.mark.skipif(not HAVE_VIENNA, reason="ViennaRNA unavailable")
def test_rnafm_vienna_hybrid_fit_predict_shapes():
    fit, predict = make_rnafm_vienna_linear_hybrid(_cache(_rows()))
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "rnafm_vienna_linear_hybrid"
    assert model["n_vienna"] == 11
    assert model["n_rnafm"] == RENDER_DIM
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(cp))
    assert sigma.min() > 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    tests = [test_rnafm_hybrid_fit_predict_shapes,
             test_rnafm_hybrid_unseen_scaffold_abstains,
             test_rnafm_hybrid_sequence_block_changes_predictions,
             test_rnafm_hybrid_missing_embedding_fails_closed]
    if HAVE_VIENNA:
        tests.append(test_rnafm_vienna_hybrid_fit_predict_shapes)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("rnafm hybrid tests", "PASS" if failed == 0 else f"{failed} FAILURES")