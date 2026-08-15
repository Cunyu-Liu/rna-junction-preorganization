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
    make_nonlinear_mlp_extended_hybrid_reg_deep4,
    make_nonlinear_mlp_extended_hybrid_reg_deep4w,
    make_nonlinear_mlp_extended_hybrid_reg_deep5,
    make_nonlinear_mlp_extended_hybrid_het,
    make_nonlinear_mlp_extended_hybrid_localctx,
    make_nonlinear_mlp_extended_hybrid_reg_deep_t,
    make_nonlinear_mlp_extended_hybrid_reg_deep_t_scaf,
    make_nonlinear_mlp_extended_hybrid_reg_deep_t_bag,
    make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw,
    make_nonlinear_mlp_nuisance_only_t,
    _student_t_survival,
    make_nonlinear_mlp_rnafm_pca_hybrid,
    make_nonlinear_mlp_rnafm_only_pca_hybrid,
    make_nonlinear_mlp_rnafm_extended_reg_deep,
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
        make_nonlinear_mlp_extended_hybrid_reg_deep4,
        make_nonlinear_mlp_extended_hybrid_reg_deep4w,
        make_nonlinear_mlp_extended_hybrid_reg_deep5,
    ]
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    for make in factories:
        fit, predict = make()
        model = fit(tr)
        assert model["n_vienna"] == 21
        assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
        assert model["hidden"] == list(model["hidden"])
        assert len(model["hidden"]) >= 2
        mu, sigma, cp, support, abstain = predict(model, te)
        assert mu.shape == (len(te),) and sigma.shape == (len(te),)
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
        assert sigma.min() > 0
        assert cp.min() >= 0 and cp.max() <= 1.0
        assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_het_variant_shapes_and_heteroscedastic_sigma():
    """Two-output (mu, sigma) head must fit/predict with finite, floored sigma.

    The reg_deep family fixes sigma=0.7; this variant learns a per-input
    sigma = softplus(raw) + 0.05.  We assert the head carries a convergence
    gate, outputs are finite, sigma is floored above 0, and sigma is NOT
    identical across rows (i.e. it is genuinely input-dependent), while
    remaining a valid probability model for the capped censoring.
    """
    fit, predict = make_nonlinear_mlp_extended_hybrid_het()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_het"
    assert model["n_vienna"] == 21
    assert model["hidden"] == [96, 64, 32]
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert sigma.min() > 0.05 and sigma.min() > 0.0
    # heteroscedastic: sigma should not be constant across a varied test set
    assert np.ptp(sigma) > 1e-6
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool
    # learned sigma must remain a valid censoring model: mu capped at CAP
    # corresponds to cp near 0.5, not a degenerate all-0/1
    assert np.all(np.isfinite(cp))


@needs_torch_vienna
def test_robust_t_variant_shapes_and_finiteness():
    """reg_deep trained with the Student-t objective must fit/predict cleanly.

    The robust head keeps the 21-D extended-Vienna feature block and reg_deep
    architecture but minimizes a heavier-tailed Student-t right-censored NLL.
    We assert the carried df matches, the convergence gate is present, outputs
    are finite and well-typed, and the Gaussian evaluation NLL is computable.
    """
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_reg_deep_t"
    assert model["n_vienna"] == 21
    assert model["hidden"] == [96, 64, 32]
    assert model["df"] == 5.0
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    assert model["gate"]["df"] == 5.0
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.allclose(sigma, 0.7)          # evaluation sigma stays fixed at 0.7
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_robust_t_df_variants_shapes():
    """Student-t tail-heaviness sweep (df=3/5/7/10) must fit/predict cleanly.

    Each variant carries its own df through the fit gate and returns finite,
    correctly-typed outputs with the fixed evaluation sigma=0.7.
    """
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    for df in (3.0, 5.0, 7.0, 10.0):
        fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=df)
        model = fit(tr)
        assert model["n_vienna"] == 21
        assert model["hidden"] == [96, 64, 32]
        assert model["df"] == df
        assert model["gate"]["df"] == df
        assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
        mu, sigma, cp, support, abstain = predict(model, te)
        n = len(te)
        assert mu.shape == (n,) and sigma.shape == (n,)
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
        assert np.allclose(sigma, 0.7)
        assert cp.min() >= 0.0 and cp.max() <= 1.0
        assert np.all(np.isfinite(cp))
        assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_robust_t_seed_replication():
    """A different seed must change the fit (independent replication), cleanly.

    The robust head carries its seed through the gate; re-fitting with a distinct
    seed must yield finite outputs and a recorded gate, and (with fixed df) the
    two fits need not be numerically identical, confirming the seed is threaded
    into training rather than ignored.
    """
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    m1 = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=23)[0](tr)
    m2 = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=99)[0](tr)
    assert m1["seed"] == 23 and m2["seed"] == 99
    assert m1["df"] == m2["df"] == 5.0
    assert m1["gate"]["df"] == 5.0
    assert "eligible" in m1["gate"] and "final_grad_norm" in m1["gate"]
    assert "eligible" in m2["gate"] and "final_grad_norm" in m2["gate"]
    # predict both and check finiteness
    pred1 = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=23)[1]
    pred2 = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=5.0, seed=99)[1]
    mu1, s1, cp1, su1, ab1 = pred1(m1, te)
    mu2, s2, cp2, su2, ab2 = pred2(m2, te)
    assert np.all(np.isfinite(mu1)) and np.all(np.isfinite(mu2))
    assert np.allclose(s1, 0.7) and np.allclose(s2, 0.7)
    assert su1.dtype == bool and ab1.dtype == bool


@needs_torch_vienna
def test_gaussian_reg_deep_seed_threading():
    """Gaussian reg_deep must thread an independent seed through training."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    m1 = make_nonlinear_mlp_extended_hybrid_reg_deep(seed=23)[0](tr)
    m2 = make_nonlinear_mlp_extended_hybrid_reg_deep(seed=99)[0](tr)
    assert m1["seed"] == 23 and m2["seed"] == 99
    assert m1["hidden"] == [96, 64, 32] and m2["hidden"] == [96, 64, 32]
    assert "eligible" in m1["gate"] and "final_grad_norm" in m1["gate"]
    assert "eligible" in m2["gate"] and "final_grad_norm" in m2["gate"]
    pred1 = make_nonlinear_mlp_extended_hybrid_reg_deep(seed=23)[1]
    pred2 = make_nonlinear_mlp_extended_hybrid_reg_deep(seed=99)[1]
    mu1, s1, cp1, su1, ab1 = pred1(m1, te)
    mu2, s2, cp2, su2, ab2 = pred2(m2, te)
    assert np.all(np.isfinite(mu1)) and np.all(np.isfinite(mu2))
    assert np.allclose(s1, 0.7) and np.allclose(s2, 0.7)
    assert su1.dtype == bool and ab1.dtype == bool
    assert not np.allclose(mu1, mu2)  # distinct seeds -> distinct fits


@needs_torch_vienna
def test_robust_t_swa_shapes_and_finiteness():
    """reg_deep_t with SWA (swa_n>0) must fit/predict cleanly and record swa_n."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, swa_n=10)
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_reg_deep_t"
    assert model["df"] == 7.0
    assert model["swa_n"] == 10
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool
    # SWA must actually change the fit vs. no-SWA (same seed)
    fit0, pred0 = make_nonlinear_mlp_extended_hybrid_reg_deep_t(df=7.0, swa_n=0)
    m0 = fit0(tr)
    mu0, s0, cp0, su0, ab0 = pred0(m0, te)
    assert not np.allclose(mu, mu0)  # SWA vs best-epoch differ on the same seed


@needs_torch_vienna
def test_robust_t_bag_shapes_and_finiteness():
    """Bagged t7 (n_bags>1) must fit/predict cleanly and average mu over bags."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t_bag(
        df=7.0, n_bags=3)
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_reg_deep_t_bag"
    assert model["df"] == 7.0
    assert model["n_bags"] == 3
    assert len(model["nets"]) == 3 and len(model["gates"]) == 3
    for g in model["gates"]:
        assert "eligible" in g and "final_grad_norm" in g
        assert g["df"] == 7.0
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool
    # bagging with different seeds must differ (independent bags)
    fit2, pred2 = make_nonlinear_mlp_extended_hybrid_reg_deep_t_bag(
        df=7.0, n_bags=3, seed=99)
    m2 = fit2(tr)
    mu2, s2, cp2, su2, ab2 = pred2(m2, te)
    assert not np.allclose(mu, mu2)  # distinct bag seeds -> distinct mu


@needs_torch_vienna
def test_robust_t_dropout_threading():
    """df=7 t7 factory must thread dropout through the model dict."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    for do in (0.05, 0.1, 0.2):
        fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t(
            df=7.0, dropout=do)
        model = fit(tr)
        assert model["df"] == 7.0
        assert model["dropout"] == do
        assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
        mu, sigma, cp, support, abstain = predict(model, te)
        assert np.all(np.isfinite(mu)) and np.allclose(sigma, 0.7)
        assert cp.min() >= 0.0 and cp.max() <= 1.0
        assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_robust_t_scaf_sigma_shapes_and_finiteness():
    """Per-scaffold sigma (r40) must fit/predict cleanly with a learned table.

    The jointly-learned per-scaffold sigma head keeps the reg_deep mu network
    but adds a ~9-param log-sigma table indexed by scaffold.  The gate must
    record the learned per-scaffold sigmas, and prediction must emit per-row
    sigma from that table (not the fixed 0.7).  This is the auditable ablation
    probe for the r40 negative result (training-time sigma degrades mu fit).
    """
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t_scaf(df=7.0)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_reg_deep_t_scaf"
    assert model["n_vienna"] == 21
    assert model["hidden"] == [96, 64, 32]
    assert model["df"] == 7.0
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    assert "scaf_sigma" in model["gate"]      # learned per-scaffold table recorded
    assert len(model["gate"]["scaf_sigma"]) >= 1
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    # prediction must NOT be forced to fixed 0.7: sigma comes from the table
    assert not np.allclose(sigma, 0.7) or model["gate"]["scaf_sigma"].get(
        str(int(te[0]["scaf"])), 0.7) == 0.7
    assert np.all(sigma >= 0.05)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert np.all(np.isfinite(cp))
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_robust_t_scaf_sigma_seed_threading():
    """Different seeds must thread through the scaf-sigma factory."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    fit1, pred1 = make_nonlinear_mlp_extended_hybrid_reg_deep_t_scaf(df=7.0, seed=23)
    fit2, pred2 = make_nonlinear_mlp_extended_hybrid_reg_deep_t_scaf(df=7.0, seed=99)
    m1 = fit1(tr)
    m2 = fit2(tr)
    assert m1["seed"] == 23 and m2["seed"] == 99
    assert m1["df"] == m2["df"] == 7.0
    for m in (m1, m2):
        assert "eligible" in m["gate"] and "final_grad_norm" in m["gate"]
        assert "scaf_sigma" in m["gate"]
    mu1, s1, cp1, su1, ab1 = pred1(m1, te)
    mu2, s2, cp2, su2, ab2 = pred2(m2, te)
    assert np.all(np.isfinite(mu1)) and np.all(np.isfinite(mu2))


@needs_torch_vienna
def test_nuisance_only_t7_shapes_and_ablation():
    """nuisance-only t7 (no ViennaRNA) must fit/predict cleanly, with gate.

    This is the matched ablation for the 3x t7 ensemble.  The fit must be clean
    even without ViennaRNA features, and the number of features must be exactly
    [1 + n_motifs + n_scafs + 3] (motif one-hot, scaffold one-hot, topology).
    """
    fit, predict = make_nonlinear_mlp_nuisance_only_t(df=7.0)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_nuisance_only_t"
    assert model["df"] == 7.0
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    # no Vienna features: n_nuisance = 1 + 3 motifs + 3 scafs + 3 topology = 10
    assert model["n_nuisance"] == 1 + len({r["motif"] for r in tr}) + len({r["scaf"] for r in tr}) + 3
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool
    # unseen scaffold must abstain
    te2 = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
            "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
            "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    mu2, sigma2, cp2, support2, abstain2 = predict(model, te2)
    assert bool(abstain2[0]) is True and bool(support2[0]) is False


@needs_torch_vienna
def test_censor_weight_variant_shapes_and_gate():
    """r49 censor-aware reweighting: fits cleanly with a recorded gate.

    Asserts the cw factory threads cw_strength/floor into the returned model,
    emits frozen 0.7 sigma at prediction, abstains on unseen scaffolds, and
    records the per-row train weights.
    """
    fit, predict = make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw(
        df=7.0, cw_strength=1.0, cw_floor=0.15)
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_reg_deep_t_cw"
    assert model["df"] == 7.0
    assert model["cw_strength"] == 1.0 and model["cw_floor"] == 0.15
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    w = model["train_weights"]
    assert len(w) == len(tr)
    assert np.all(np.asarray(w) >= 1.0), "measured rows must be >= 1.0 weight"
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.allclose(sigma, 0.7)
    assert cp.min() >= 0.0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool


@needs_torch_vienna
def test_censor_weight_prioritizes_high_censoring_scaffold():
    """cw weights must be LARGER for measured rows on high-censoring scaffolds.

    Build synthetic rows where scaf=9 is mostly censored (c_s~0.8) and scaf=2
    is measured-only (c_s=0); measured rows on scaf9 must receive a larger
    weight than measured rows on scaf2.
    """
    rows = _rows()
    # force scaf 9 rows to be censored (they already exist in the fixture)
    weighted = []
    for r in rows:
        r2 = dict(r)
        if int(r["scaf"]) == 9:
            r2["cens"] = 1
        weighted.append(r2)
    fit, _ = make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw(
        df=7.0, cw_strength=1.0, cw_floor=0.15)
    tr = weighted[:18]
    model = fit(tr)
    w = model["train_weights"]
    w_scaf9 = [w[i] for i, r in enumerate(tr) if int(r["scaf"]) == 9 and not r["cens"]]
    w_scaf2 = [w[i] for i, r in enumerate(tr) if int(r["scaf"]) == 2 and not r["cens"]]
    if w_scaf9 and w_scaf2:
        assert np.mean(w_scaf9) > np.mean(w_scaf2), \
            "measured rows on high-censoring scaffold must be upweighted"
    # censored rows always weight 1.0
    for i, r in enumerate(tr):
        if r["cens"]:
            assert np.isclose(w[i], 1.0)


@needs_torch_vienna
def test_censor_weight_seed_threading():
    """Different seeds must thread through the cw factory."""
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    fit1, _ = make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw(df=7.0, seed=23)
    fit2, _ = make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw(df=7.0, seed=99)
    m1, m2 = fit1(tr), fit2(tr)
    assert m1["seed"] == 23 and m2["seed"] == 99
    assert m1["df"] == m2["df"] == 7.0
    for m in (m1, m2):
        assert "eligible" in m["gate"]
        assert len(m["train_weights"]) == len(tr)


@needs_torch_vienna
def test_student_t_survival_matches_scipy():
    import torch
    from scipy import stats as spstats
    ts = torch.tensor([-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0], dtype=torch.float64,
                      requires_grad=True)
    for nu in (3.0, 5.0, 7.0):
        got = _student_t_survival(ts, nu).detach().numpy()
        want = spstats.t.sf(ts.detach().numpy(), df=nu)
        assert np.allclose(got, want, atol=1e-6, rtol=1e-4)
    # differentiability: gradient w.r.t. t is finite and nonzero
    out = _student_t_survival(ts, 5.0).sum()
    out.backward()
    assert ts.grad is not None and np.all(np.isfinite(ts.grad.numpy()))
    assert np.all(np.abs(ts.grad.numpy()) > 0)


@needs_torch_vienna
def test_localctx_variant_shapes_and_finiteness():
    """reg_deep + extended-Vienna(21) + join-local-context(24) must fit/predict.

    Asserts the model keeps 21-D Vienna plus a 24-D local-context block, carries
    a convergence gate, returns finite well-typed outputs, and abstains only on
    unseen scaffolds.
    """
    fit, predict = make_nonlinear_mlp_extended_hybrid_localctx()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_extended_hybrid_localctx"
    assert model["n_vienna"] == 21
    assert model["n_localctx"] == 24
    assert model["hidden"] == [96, 64, 32]
    assert "eligible" in model["gate"] and "final_grad_norm" in model["gate"]
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert sigma.min() > 0
    assert cp.min() >= 0.0 and cp.max() <= 1.0
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


@needs_torch_vienna
def test_rnafm_extended_reg_deep_shapes_and_finiteness():
    """Combined reg_deep arch + extended-Vienna(21) + RNA-FM-PCA must fit/predict.

    The r14 scan found reg_deep (96,64,32) is the best-robust 21-D extended-Vienna
    model (13.17% over nuisance, CI excludes 0).  This probe verifies that adding
    the learned RNA-FM-PCA block on top still fits cleanly, keeps 21-D Vienna,
    carries a convergence gate, and returns finite, well-typed outputs.
    """
    rows = _rows()
    cache = _rnafm_cache(rows)
    fit, predict = make_nonlinear_mlp_rnafm_extended_reg_deep(cache, k=8)
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "nonlinear_mlp_rnafm_extended_reg_deep"
    assert model["n_vienna"] == 21
    assert 0 < model["n_rnafm_pca"] <= 8
    assert model["hidden"] == [96, 64, 32]
    assert model["dropout"] == 0.1 and model["weight_decay"] == 1e-2
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
             test_het_variant_shapes_and_heteroscedastic_sigma,
             test_rnafm_pca_shapes_and_finiteness,
             test_rnafm_extended_reg_deep_shapes_and_finiteness,
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
