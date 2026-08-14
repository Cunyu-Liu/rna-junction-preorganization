"""Unit tests for the right-censored XGBoost hybrid (new model family).

Contract rules mirror the other hybrid families: fit on TRAIN rows only, no test
leakage, right-censor aware, finite correctly-shaped outputs, unseen-scaffold
abstention, deterministic training, and a recorded convergence gate.
"""
import numpy as np
import pytest

from audit.models.xgboost_censored_hybrid import (
    make_xgboost_censored_hybrid,
    _censored_nll_grad_hess,
    HAVE_XGB,
)

try:
    import RNA as _RNA  # noqa: F401
    HAVE_VIENNA = True
except Exception:  # noqa: BLE001
    HAVE_VIENNA = False

needs_xgb_vienna = pytest.mark.skipif(not (HAVE_XGB and HAVE_VIENNA),
                                      reason="xgboost and ViennaRNA required")


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
            y = -8.5 + 0.2 * (r0 % 5) if not cens else -7.1
            rows.append({"source_row_id": f"R{r0:05d}", "jid": f"j{k}",
                         "motif": motifs[k], "scaf": scaf, "y": y, "cens": cens,
                         "junction_seq": s, "helix_seq": f"h{k}_{n}",
                         "symmetry_key": "_".join(reversed(s.split("_")))})
            r0 += 1
    return rows


def test_censored_grad_hess_monotone():
    """A censored row further below the cap must have a larger gradient magnitude."""
    mu = np.array([-3.0, 3.0])
    y = np.array([-7.1, -7.1])
    cens = np.array([True, True])
    g, h = _censored_nll_grad_hess(mu, y, cens)
    assert g[0] < g[1]
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(h))
    assert np.all(h > 0)


def test_censored_hess_matches_finite_diff():
    """Numeric check of the censored gradient against the analytic formula."""
    from audit.evaluation.metrics import row_nll
    y, cens = -7.1, True
    eps = 1e-4
    mu = np.array([-1.0, 0.0, 1.0])
    for m in mu:
        nll_p = row_nll([y], [cens], [m + eps], [0.7])[0]
        nll_m = row_nll([y], [cens], [m - eps], [0.7])[0]
        fd = (nll_p - nll_m) / (2 * eps)
        g, _ = _censored_nll_grad_hess(np.array([m]), np.array([y]), np.array([True]))
        assert abs(fd - g[0]) < 1e-3, f"mu={m}: fd={fd} analytic={g[0]}"


@needs_xgb_vienna
def test_xgb_shapes_and_finiteness():
    fit, predict = make_xgboost_censored_hybrid(n_estimators=100, max_depth=3)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "xgboost_censored_hybrid"
    assert "eligible" in model["gate"] and model["gate"]["eligible"] is True
    assert model["best_iteration"] >= 1
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool


@needs_xgb_vienna
def test_xgb_unseen_scaffold_abstains():
    fit, predict = make_xgboost_censored_hybrid(n_estimators=50, max_depth=3)
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False
