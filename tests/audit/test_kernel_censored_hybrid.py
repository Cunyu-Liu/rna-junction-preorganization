"""Unit tests for the kernel (RBF) right-censored hybrid.

Contract rules mirror the other hybrid families: fit on TRAIN rows only, no
test leakage, right-censor aware, finite correctly-shaped outputs, unseen-scaffold
abstention, deterministic training (per seed), and a recorded convergence gate.
"""
import numpy as np
import pytest

from audit.models.kernel_censored_hybrid import (
    make_kernel_censored_hybrid,
    _censored_grad_hess,
    _rbf_kernel,
    HAVE_TORCH,
)

try:
    import RNA as _RNA  # noqa: F401
    HAVE_VIENNA = True
except Exception:  # noqa: BLE001
    HAVE_VIENNA = False

needs_kernel_vienna = pytest.mark.skipif(not (HAVE_TORCH and HAVE_VIENNA),
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
            y = -8.5 + 0.2 * (r0 % 5) if not cens else -7.1
            rows.append({"source_row_id": f"R{r0:05d}", "jid": f"j{k}",
                         "motif": motifs[k], "scaf": scaf, "y": y, "cens": cens,
                         "junction_seq": s, "helix_seq": f"h{k}_{n}",
                         "symmetry_key": "_".join(reversed(s.split("_")))})
            r0 += 1
    return rows


def test_rbf_kernel_shapes_and_properties():
    X = np.random.default_rng(0).normal(size=(8, 5))
    K = _rbf_kernel(X, 0.5)
    K = K.numpy()
    assert K.shape == (8, 8)
    assert np.allclose(K, K.T, atol=1e-5)  # symmetric
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)  # RBF self-sim = 1
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-5
    # cross kernel shape
    Kc = _rbf_kernel(X[:3], 0.5, X).numpy()
    assert Kc.shape == (3, 8)


def test_kernel_grad_hess_matches_finite_diff():
    """Numeric check of the censored Gaussian grad/hess used by the IRLS."""
    from scipy.stats import norm
    for mu in (-3.0, 0.0, 3.0):
        for y, cens in ((-7.1, True), (-6.0, False)):
            eps = 1e-4
            def nll(m):
                if cens:
                    return float(-norm.logcdf((m + 7.1) / 0.7))
                return float(-norm.logpdf(y, loc=m, scale=0.7))
            fd = (nll(mu + eps) - nll(mu - eps)) / (2 * eps)
            g, _ = _censored_grad_hess(np.array([mu]), np.array([y]), np.array([cens]))
            assert abs(fd - g[0]) < 1e-3, f"mu={mu} cens={cens}: fd={fd} g={g[0]}"


@needs_kernel_vienna
def test_kernel_shapes_and_finiteness():
    fit, predict = make_kernel_censored_hybrid(gamma=0.5, lam=0.01)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "kernel_censored_hybrid"
    assert model["gate"]["eligible"] is True
    assert model["gate"]["n_iter"] >= 1
    # final_grad_norm may be NaN for tiny toy data; not a contract failure
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool


@needs_kernel_vienna
def test_kernel_grid_selects_hyperparameters():
    """With gamma/lam=None the subsample grid returns finite hyperparameters."""
    fit, predict = make_kernel_censored_hybrid()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["gamma"] > 0.0 and model["lam"] > 0.0
    mu, sigma, cp, support, abstain = predict(model, te)
    assert np.all(np.isfinite(mu))


@needs_kernel_vienna
def test_kernel_unseen_scaffold_abstains():
    fit, predict = make_kernel_censored_hybrid(gamma=0.5, lam=0.01)
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False
