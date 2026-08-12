"""Unit tests for the richer-feature nonlinear (shallow-MLP) hybrids.

Contract rules mirror nonlinear_mlp_hybrid: fit on TRAIN rows only, no test
leakage, right-censor aware, finite correctly-shaped outputs, unseen-scaffold
abstention, deterministic training, and a recorded convergence gate.  The
RNA-FM variants additionally require a non-empty embedding cache and PCA fit
strictly on TRAIN embeddings.
"""
import numpy as np
import pytest

from audit.models.nonlinear_mlp_hybrid import HAVE_TORCH
from audit.models.nonlinear_mlp_rich_hybrid import (
    make_nonlinear_mlp_extended_hybrid,
    make_nonlinear_mlp_extended_hybrid_reg,
    make_nonlinear_mlp_extended_hybrid_reg_strong,
    make_nonlinear_mlp_extended_hybrid_reg_light,
    make_nonlinear_mlp_extended_hybrid_reg_wider,
    make_nonlinear_mlp_extended_hybrid_reg_deep,
    make_nonlinear_mlp_rnafm_pca_hybrid,
    make_nonlinear_mlp_rnafm_only_pca_hybrid,
)

try:
    import RNA as _RNA  # noqa: F401
    HAVE_VIENNA = True
except Exception:  # noqa: BLE001
    HAVE_VIENNA = False

CAP = -7.1
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch unavailable")
needs_vienna = pytest.mark.skipif(not HAVE_VIENNA, reason="ViennaRNA unavailable")
needs_torch_vienna = pytest.mark.skipif(not (HAVE_TORCH and HAVE_VIENNA),
                                        reason="torch and ViennaRNA required")


def _rows():
    rows = []
    seqs = {1: "CUAG_CUAAG", 2: "CGAC_CGAC", 3: "AUGC_GCUA", 4: "UACG_ACGU",
            5: "GCUA_AUCG", 6: "AUCG_UAGC"}
    motifs = {1: "0x1", 2: "0x2", 3: "0x3", 4: "0x1", 5: "0x2", 6: "0x3"}
    r0 = 0
    for k, s in seqs.items():
        scaf = (k % 3) + 1
        for n in range(4):
            cens = (n == 3)
            y = -8.5 + 0.2 * (r0 % 5) if not cens else CAP
            rows.append({"source_row_id": f"R{r0:05d}", "jid": f"j{k}",
                         "motif": motifs[k], "scaf": scaf, "y": y, "cens": cens,
                         "junction_seq": s, "helix_seq": f"h{k}_{n}",
                         "symmetry_key": "_".join(reversed(s.split("_")))})
            r0 += 1
    return rows


def _rnafm_cache(rows):
    """Build a fake RNA-FM cache (1920-D) keyed by junction_seq."""
    rng = np.random.default_rng(0)
    cache = {}
    for r in rows:
        cache[str(r["junction_seq"])] = rng.normal(size=1920).astype(np.float32)
    return cache


@needs_torch_vienna
def test_extended_shapes_and_finiteness():
    fit, predict = make_nonlinear_mlp_extended_hybrid()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid"
    assert model["n_vienna"] == 21
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,) and support.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_extended_train_only_scaling_no_leakage():
    from audit.benchmark.vienna_extended_features import build_raw_by_jid, fit_scaler
    fit_fn, _ = make_nonlinear_mlp_extended_hybrid()
    rows = _rows()
    tr = rows[:18]
    model = fit_fn(tr)
    tr_jids = sorted({str(r["jid"]) for r in tr})
    by_jid = build_raw_by_jid(tr)
    mean, sd = fit_scaler(tr_jids, by_jid)
    assert np.allclose(model["mean"], mean)
    assert np.allclose(model["sd"], sd)


@needs_torch_vienna
def test_extended_unseen_scaffold_abstains():
    fit, predict = make_nonlinear_mlp_extended_hybrid()
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False


@needs_torch_vienna
def test_reg_variant_shapes_and_dropout():
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid"
    assert model["n_vienna"] == 21
    assert model.get("dropout", 0.0) > 0.0
    assert model.get("weight_decay", 0.0) >= 1e-2
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert sigma.min() > 0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_reg_architecture_variants_shapes():
    """All regularization/architecture probes must fit and predict cleanly.

    Each probe is the same 21-D extended-Vienna MLP under a different
    (dropout, weight_decay, hidden) budget; all must yield finite, correctly
    shaped, support/abstain-typed outputs with a recorded convergence gate.
    """
    factories = [
        make_nonlinear_mlp_extended_hybrid_reg_strong,
        make_nonlinear_mlp_extended_hybrid_reg_light,
        make_nonlinear_mlp_extended_hybrid_reg_wider,
        make_nonlinear_mlp_extended_hybrid_reg_deep,
    ]
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    for make in factories:
        fit, predict = make()
        model = fit(tr)
        assert model["n_vienna"] == 21
        assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
        assert model["hidden"] == list(model["hidden"])
        mu, sigma, cp, support, abstain = predict(model, te)
        assert mu.shape == (len(te),) and sigma.shape == (len(te),)
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
        assert sigma.min() > 0
        assert cp.min() >= 0 and cp.max() <= 1.0
        assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_rnafm_pca_shapes_and_finiteness():
    rows = _rows()
    cache = _rnafm_cache(rows)
    fit, predict = make_nonlinear_mlp_rnafm_pca_hybrid(cache, k=8)
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_rnafm_pca_hybrid"
    assert model["n_vienna"] == 11
    # PCA rank is capped by n_unique_train_junctions (6 here) -> effective k < 8.
    assert 0 < model["n_rnafm_pca"] <= 8
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert sigma.min() > 0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_rnafm_pca_train_only_pca_no_leakage():
    from audit.models.rnafm_pca_linear_hybrid import _fit_pca, _apply_pca
    from audit.benchmark.rnafm_features import build_raw_by_jid
    rows = _rows()
    cache = _rnafm_cache(rows)
    fit, _ = make_nonlinear_mlp_rnafm_pca_hybrid(cache, k=8)
    tr = rows[:18]
    model = fit(tr)
    k_eff = model["n_rnafm_pca"]
    tr_jids = sorted({str(r["jid"]) for r in tr})
    by_jid = build_raw_by_jid(tr, cache)
    Xraw = np.asarray([by_jid[j] for j in tr_jids], dtype=float)
    pca_mean, comps, scale = _fit_pca(Xraw, k_eff)
    assert np.allclose(model["pca_mean"], pca_mean)
    assert np.allclose(model["comps"], comps)
    assert np.allclose(model["scale"], scale)


@needs_torch_vienna
def test_rnafm_only_pca_shapes():
    rows = _rows()
    cache = _rnafm_cache(rows)
    fit, predict = make_nonlinear_mlp_rnafm_only_pca_hybrid(cache, k=8)
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_rnafm_only_pca_hybrid"
    assert model.get("n_vienna", 0) == 0 and 0 < model["n_rnafm_pca"] <= 8
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu))
    assert support.dtype == bool and abstain.dtype == bool


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    tests = [test_extended_shapes_and_finiteness,
             test_extended_train_only_scaling_no_leakage,
             test_extended_unseen_scaffold_abstains,
             test_reg_variant_shapes_and_dropout,
             test_reg_architecture_variants_shapes,
             test_rnafm_pca_shapes_and_finiteness,
             test_rnafm_pca_train_only_pca_no_leakage,
             test_rnafm_only_pca_shapes]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("nonlinear_mlp_rich tests", "PASS" if failed == 0 else f"{failed} FAILURES")
