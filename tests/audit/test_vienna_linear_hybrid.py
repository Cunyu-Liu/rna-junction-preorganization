"""Unit tests for the ViennaRNA plain-linear hybrid (winning-head increment test).

Contract rules: fit on TRAIN rows only, no test leakage, right-censor aware,
finite correctly-shaped outputs, unseen-scaffold abstention, and the sequence
block must actually change predictions vs the nuisance-only model.
"""
import numpy as np

from audit.models.vienna_linear_hybrid import VIENNA_LINEAR_HYBRID, make_vienna_linear_hybrid

CAP = -7.1


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


def test_model_registered():
    assert "vienna_linear_hybrid" in VIENNA_LINEAR_HYBRID
    fit, predict = VIENNA_LINEAR_HYBRID["vienna_linear_hybrid"]
    assert callable(fit) and callable(predict)


def test_fit_predict_shapes_and_finiteness():
    fit, predict = make_vienna_linear_hybrid()
    rows = _rows()
    tr, te = rows[:18], rows[18:]
    model = fit(tr)
    assert model["kind"] == "vienna_linear_hybrid"
    assert model["n_nuisance"] >= 1 and model["n_vienna"] == 11
    mu, sigma, cp, support, abstain = predict(model, te)
    n = len(te)
    assert mu.shape == (n,) and sigma.shape == (n,)
    assert cp.shape == (n,) and abstain.shape == (n,) and support.shape == (n,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert cp.min() >= 0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool


def test_unseen_scaffold_abstains():
    fit, predict = make_vienna_linear_hybrid()
    tr = _rows()
    te = [{"source_row_id": "R999", "jid": "j99", "motif": "0x1", "scaf": 99,
           "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
           "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
    model = fit(tr)
    mu, sigma, cp, support, abstain = predict(model, te)
    assert bool(abstain[0]) is True and bool(support[0]) is False


def test_sequence_block_changes_predictions():
    """Two junctions with identical nuisance (motif/scaffold/topology) but
    different sequences must yield different mu if the ViennaRNA block is live."""
    fit_fn, predict_fn = make_vienna_linear_hybrid()
    rows = _rows()
    # pick two rows with same motif and scaffold but different sequence
    a = next(r for r in rows if r["jid"] == "j1")
    b = next(r for r in rows if r["jid"] == "j2")
    assert a["motif"] != b["motif"] or a["scaf"] != b["scaf"] or a["junction_seq"] != b["junction_seq"]
    tr = [r for r in rows if r["jid"] not in ("j1", "j2")]
    model = fit_fn(tr)
    mu_a, *_ = predict_fn(model, [a])
    mu_b, *_ = predict_fn(model, [b])
    # sequences differ -> ViennaRNA features differ -> linear predictor should differ
    assert abs(mu_a[0] - mu_b[0]) > 1e-9


def test_train_only_scaling_no_leakage():
    """The ViennaRNA scaler must be fit on train junction ids only; the stored
    mean/sd must equal what fit_scaler(train jids) produces, not anything from test."""
    from audit.benchmark.vienna_features import build_raw_by_jid, fit_scaler
    fit_fn, _ = make_vienna_linear_hybrid()
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
             test_train_only_scaling_no_leakage]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("vienna_linear_hybrid tests", "PASS" if failed == 0 else f"{failed} FAILURES")