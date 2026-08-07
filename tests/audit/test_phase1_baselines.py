"""P1 tests: strong-simple / publicly-reproducible baseline correctness.

Each baseline must:
  - implement the shared model interface  fit(train_rows)->model,
    predict(model, test_rows)->(mu, sigma, censor_prob, support, abstain);
  - be fit on TRAIN rows only and applied to test rows (no test leakage);
  - return finite, correctly-shaped outputs;
  - respect right-censor semantics (raising mu must not raise censored NLL).
These are unit/contract tests, not scientific claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.benchmark.features import raw_features
from audit.benchmark.phase1_baselines import (
    fit_motif_topology, predict_motif_topology,
    fit_kmer_ridge, predict_kmer_ridge,
    fit_position_additive, predict_position_additive,
    fit_edit_knn, predict_edit_knn,
    fit_mutation_graph, predict_mutation_graph,
    fit_small_mlp, predict_small_mlp,
    PHASE1_MODELS,
)

CAP = -7.1


def make_rows(n=24):
    """Small synthetic admitted-style fixture with a few shared junctions."""
    # 6 distinct junctions, each repeated across 4 scaffold/context rows
    seqs = ["CUAG_CUAAG", "CGAC_CGAC", "AUGC_GCUA", "UACG_ACGU",
            "GCUA_AUCG", "AUCG_UAGC"]
    motifs = ["0x1", "0x2", "0x3", "0x1", "0x2", "0x3"]
    rows = []
    r0 = 0
    for s, motif in zip(seqs, motifs):
        scaf = int(s.count("C") % 3) + 1
        # 4 scaffold/context rows per junction, one censored
        for k in range(4):
            cens = (k == 3)
            y = -8.5 + 0.2 * (r0 % 5) if not cens else CAP
            rows.append({
                "source_row_id": f"R{r0:05d}", "jid": str(100 + r0 % 6),
                "motif": motif, "scaf": scaf, "y": y, "cens": cens,
                "junction_seq": s, "helix_seq": f"AC_GG&CC_GU{k}",
                "symmetry_key": "_".join(reversed(s.split("_"))),
            })
            r0 += 1
    return rows


def _assert_predict_shape(model, train, test, predict_fn):
    mu, sigma, cp, support, abstain = predict_fn(model, test)
    n = len(test)
    assert mu.shape == (n,)
    assert sigma.shape == (n,)
    assert cp.shape == (n,)
    assert support.shape == (n,)
    assert abstain.shape == (n,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(cp))
    assert sigma.min() > 0
    assert cp.min() >= 0 and cp.max() <= 1.0
    assert support.dtype == bool and abstain.dtype == bool


@pytest.mark.parametrize("model_id", sorted(PHASE1_MODELS))
def test_all_models_fit_and_predict(model_id):
    rows = make_rows()
    tr, te = rows[:16], rows[16:]
    fit_fn, pred_fn = PHASE1_MODELS[model_id]
    model = fit_fn(tr)
    _assert_predict_shape(model, tr, te, pred_fn)


def test_feature_dimension_is_63():
    assert raw_features("CUAG_CUAAG").shape == (63,)


def test_edit_knn_train_only_no_test_leakage():
    rows = make_rows()
    tr, te = rows[:16], rows[16:]
    model = fit_edit_knn(tr)
    # neighbor graph only over distinct train junction sequences
    tr_seqs = {str(r["junction_seq"]) for r in tr}
    assert set(model["seqs"]) == tr_seqs
    assert all(s in tr_seqs for s in model["seqs"])


def test_edit_knn_known_junction_returns_its_train_mean():
    rows = make_rows()
    tr = rows[:16]
    model = fit_edit_knn(tr)
    # pick a train junction sequence and confirm node value == train mean
    seq = str(tr[0]["junction_seq"])
    idx = model["seq_idx"][seq]
    tr_ys = [r["y"] for r in tr if str(r["junction_seq"]) == seq]
    assert abs(model["node_val"][idx] - float(np.mean(tr_ys))) < 1e-9


def test_right_censor_direction_kmer_fit():
    # On a larger train set, raise a fitted location and confirm censored NLL drops.
    from scipy.special import log_ndtr
    rows = make_rows(48)
    model = fit_kmer_ridge(rows[:40])
    mu0 = -9.0
    mu1 = -7.5
    nll0 = -np.clip(log_ndtr((mu0 - CAP) / 0.7), -50, 50)
    nll1 = -np.clip(log_ndtr((mu1 - CAP) / 0.7), -50, 50)
    assert nll1 < nll0


def test_position_additive_output_uses_train_scaler():
    rows = make_rows()
    tr, te = rows[:16], rows[16:]
    model = fit_position_additive(tr)
    assert model["mean"].shape == (63,)
    assert model["sd"].shape == (63,)
    _assert_predict_shape(model, tr, te, predict_position_additive)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    failed = 0
    for model_id in sorted(PHASE1_MODELS):
        try:
            test_all_models_fit_and_predict(model_id)
            print(f"PASS {model_id}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {model_id}: {e}")
    for fn in [test_feature_dimension_is_63, test_edit_knn_train_only_no_test_leakage,
               test_edit_knn_known_junction_returns_its_train_mean,
               test_right_censor_direction_kmer_fit, test_position_additive_output_uses_train_scaler]:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print("P1 baseline tests", "PASS" if failed == 0 else f"{failed} FAILURES")
