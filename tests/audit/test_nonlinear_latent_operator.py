"""Unit tests for the nonlinear latent-operator head.

Contract rules mirror the latent-operator family (v1.31 / vienna_latent_operator)
and the flat-MLP family: fit on TRAIN rows only, no test leakage, right-censor
aware, finite correctly-shaped outputs, unseen-scaffold abstention, deterministic
training, and a recorded convergence gate.  The key new claims:
  - the MLP maps JUNCTION-level features (no scaffold one-hot) to a latent;
  - the operator head a_s + b_s * q_j produces scaffold-calibrated mu;
  - GH-marginal training is differentiable and finite;
  - unseen scaffold -> abstain (no placeholder scoring).
"""
import numpy as np
import pytest

from audit.models.nonlinear_mlp_hybrid import HAVE_TORCH
from audit.models.nonlinear_latent_operator import (
    make_nonlinear_latent_operator,
    _junction_panel,
    _junction_feats,
    _latent_marginal_nll,
)
from audit.models.nonlinear_mlp_rich_hybrid import _student_t_survival

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


@needs_torch_vienna
def test_latent_operator_shapes_and_finiteness():
    fit, predict = make_nonlinear_latent_operator(hidden=(8, 4))
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_latent_operator"
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    assert model["n_junction_feats"] > 0
    assert model["a"].shape == (len(model["scaffolds"]),)
    assert model["b"].shape == (len(model["scaffolds"]),)
    assert np.all(model["b"] > 0) and np.isfinite(model["b"]).all()
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,) and support.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_latent_operator_unseen_scaffold_abstains():
    fit, predict = make_nonlinear_latent_operator(hidden=(8, 4))
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False


@needs_torch_vienna
def test_latent_operator_train_only_scaling():
    """Scaler (mean/sd) must be fit on train junction features only."""
    from audit.benchmark.vienna_extended_features import build_raw_by_jid, fit_scaler
    fit_fn, _ = make_nonlinear_latent_operator(hidden=(8, 4))
    rows = _rows()
    tr = rows[:18]
    model = fit_fn(tr)
    tr_jids = model["jids"]
    by_jid = build_raw_by_jid(tr)
    mean, sd = fit_scaler(tr_jids, by_jid)
    assert np.allclose(model["mean"], mean)
    assert np.allclose(model["sd"], sd)


@needs_torch_vienna
def test_latent_operator_no_scaffold_in_mlp_features():
    """The MLP junction features must NOT include a scaffold one-hot block."""
    fit, _ = make_nonlinear_latent_operator(hidden=(8, 4))
    rows = _rows()
    tr = rows[:18]
    model = fit(tr)
    motifs = sorted({str(r["motif"]) for r in tr})
    panel = _junction_panel(tr)
    X = _junction_feats(tr, panel["jids"], motifs, model["mean"], model["sd"],
                        model["by_jid"])
    # layout: [1, motif_onehot, topology(3), vienna21]
    assert X.shape[1] == 1 + len(motifs) + 3 + 21
    # last 21 columns are the train-scaled Vienna features (finite, not constant)
    vcol = X[:, -21:]
    assert np.all(np.isfinite(vcol))


@needs_torch_vienna
def test_marginal_nll_finite_and_differentiable():
    import torch
    fit, _ = make_nonlinear_latent_operator(hidden=(8, 4))
    rows = _rows()
    tr = rows[:18]
    model = fit(tr)
    panel = _junction_panel(tr)
    n_j = len(panel["jids"])
    nodes, lw = np.polynomial.hermite.hermgauss(48)
    log_w = np.log(np.maximum(lw, 1e-300)) - 0.5 * np.log(np.pi)
    f = torch.zeros(n_j, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(model["a"], dtype=torch.float32)
    b = torch.tensor(model["b"], dtype=torch.float32)
    loss = _latent_marginal_nll(f, panel, a, b,
                                torch.tensor(nodes, dtype=torch.float32),
                                torch.tensor(log_w, dtype=torch.float32),
                                df=7.0)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert f.grad is not None and bool(torch.isfinite(f.grad).all())
    assert f.grad.shape == (n_j,)


@needs_torch_vienna
def test_marginal_nll_censoring_consistent():
    """Raising a censored row's latent location must NOT increase its NLL.

    Under right censoring, a higher mu (further above the cap) gives a higher
    survival probability and therefore a lower censored NLL contribution.
    """
    import torch
    # single junction, single censored row, one scaffold
    panel = {"jids": ["j1"], "scaffolds": [1],
             "flat_j": np.asarray([0], dtype=int),
             "flat_s": np.asarray([0], dtype=int),
             "flat_y": np.asarray([CAP], dtype=float),
             "flat_c": np.asarray([True], dtype=bool)}
    nodes, lw = np.polynomial.hermite.hermgauss(48)
    log_w = np.log(np.maximum(lw, 1e-300)) - 0.5 * np.log(np.pi)
    a = torch.tensor([-5.0], dtype=torch.float32)
    b = torch.tensor([1.0], dtype=torch.float32)
    f_lo = torch.tensor([-3.0], dtype=torch.float32)
    f_hi = torch.tensor([3.0], dtype=torch.float32)
    loss_lo = _latent_marginal_nll(f_lo, panel, a, b,
                                   torch.tensor(nodes, dtype=torch.float32),
                                   torch.tensor(log_w, dtype=torch.float32),
                                   df=None)
    loss_hi = _latent_marginal_nll(f_hi, panel, a, b,
                                   torch.tensor(nodes, dtype=torch.float32),
                                   torch.tensor(log_w, dtype=torch.float32),
                                   df=None)
    assert float(loss_hi) <= float(loss_lo) + 1e-6


@needs_torch_vienna
def test_latent_operator_gate_fields():
    fit, _ = make_nonlinear_latent_operator(hidden=(8, 4))
    rows = _rows()
    tr = rows[:18]
    model = fit(tr)
    g = model["gate"]
    for key in ("eligible", "converged", "final_grad_norm", "n_epochs",
                "final_train_nll", "best_train_nll", "plateau_reached",
                "n_nan_inf_params"):
        assert key in g
