"""Unit tests for the nonlinear (shallow-MLP) hybrid.

Contract rules: fit on TRAIN rows only, no test leakage, right-censor aware,
finite correctly-shaped outputs, unseen-scaffold abstention, deterministic
training (fixed seed), and a convergence gate that is recorded on the model.
"""
import numpy as np
import pytest

from audit.models.nonlinear_mlp_hybrid import (
    NONLINEAR_MLP_HYBRID, make_nonlinear_mlp_hybrid, HAVE_TORCH)

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
    for j, (k, s) in enumerate(seqs.items()):
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


@needs_torch
def test_model_registered():
    assert "nonlinear_mlp_hybrid" in NONLINEAR_MLP_HYBRID
    fit, predict = NONLINEAR_MLP_HYBRID["nonlinear_mlp_hybrid"]
    assert callable(fit) and callable(predict)


@needs_torch_vienna
def test_fit_predict_shapes_and_finiteness():
    fit, predict = make_nonlinear_mlp_hybrid()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_hybrid"
    assert model["n_nuisance"] >= 1 and model["n_vienna"] == 11
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,) and support.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert cp.min() >= 0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_unseen_scaffold_abstains():
    fit, predict = make_nonlinear_mlp_hybrid()
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False


@needs_torch_vienna
def test_sequence_block_changes_predictions():
    fit_fn, predict_fn = make_nonlinear_mlp_hybrid()
    rows = _rows()
    a = next(r for r in rows if r["jid"] == "j1")
    b = next(r for r in rows if r["jid"] == "j2")
    assert a["junction_seq"] != b["junction_seq"]
    tr = [r for r in rows if r["jid"] not in ("j1", "j2")]
    model = fit_fn(tr)
    mu_a, *_ = predict_fn(model, [a])
    mu_b, *_ = predict_fn(model, [b])
    assert abs(mu_a[0] - mu_b[0]) > 1e-9


@needs_torch_vienna
def test_deterministic_seed():
    fit_fn, predict_fn = make_nonlinear_mlp_hybrid()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    m1 = fit_fn(tr)
    m2 = fit_fn(tr)
    mu1, *_ = predict_fn(m1, te)
    mu2, *_ = predict_fn(m2, te)
    assert np.allclose(mu1, mu2, atol=1e-5)


@needs_torch_vienna
def test_train_only_scaling_no_leakage():
    from audit.benchmark.vienna_features import build_raw_by_jid, fit_scaler
    fit_fn, _ = make_nonlinear_mlp_hybrid()
    rows = _rows()
    tr = rows[:18]
    model = fit_fn(tr)
    tr_jids = sorted({str(r["jid"]) for r in tr})
    by_jid = build_raw_by_jid(tr)
    mean, sd = fit_scaler(tr_jids, by_jid)
    assert np.allclose(model["mean"], mean)
    assert np.allclose(model["sd"], sd)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    tests = [test_model_registered, test_fit_predict_shapes_and_finiteness,
             test_unseen_scaffold_abstains, test_sequence_block_changes_predictions,
             test_deterministic_seed, test_train_only_scaling_no_leakage]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("nonlinear_mlp tests", "PASS" if failed == 0 else f"{failed} FAILURES")
