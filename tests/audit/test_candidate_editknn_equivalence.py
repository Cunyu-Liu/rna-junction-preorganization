"""R0 regression: Candidate C (support_aware_mixture) is an edit-KNN predictor.

Contract §13.8 #5 / §6.3: on every supported row, Candidate C's base predictor
(support_aware_mixture.fit_local + predict_gated) must be pointwise identical to
the P1 edit-KNN baseline (phase1_baselines.fit_edit_knn + predict_edit_knn) in
mu and sigma.  This isomorphism is what makes Candidate C a duplicate base
predictor with a distance-abstention policy, not an independent method, and
triggers its permanent retirement.

The supported rows are those where min_edit_dist <= d_thresh and n_neighbors>=1;
there Candidate C does NOT abstain and returns _local_mu (KNN mean) + sigma=TAU,
which is exactly edit-KNN's output.  Unsupported rows are excluded from the
equality check because Candidate C abstains there (edit-KNN still emits a value).
"""
import numpy as np

from audit.models.support_aware_mixture import (
    fit_local, predict_gated, support_features, SUPPORT_DIST)
from audit.benchmark.phase1_baselines import fit_edit_knn, predict_edit_knn


def _rows():
    # 6 distinct junctions forming a neighborhood + a far one.
    train = []
    seqs = ["AAAAAC", "AAAAAG", "AAAAAU", "CCCCCC", "CCCCCG", "GGGGGG"]
    for i, s in enumerate(seqs):
        train.append({"source_row_id": f"t{i}", "jid": f"j{i}",
                      "junction_seq": s, "scaf": "1", "helix_seq": "h1",
                      "y": -6.0 - (i % 3) * 0.5, "cens": 0})
    # add a second scaffold/context row for a near junction to inflate rows
    train.append({"source_row_id": "t6", "jid": "j0", "junction_seq": seqs[0],
                  "scaf": "2", "helix_seq": "h2", "y": -5.0, "cens": 0})
    # tests: j0 in train (exact), a near-neighbor, and a far junction
    test = [
        {"source_row_id": "s0", "jid": "j0", "junction_seq": seqs[0],
         "scaf": "1", "helix_seq": "h1", "y": -6.1, "cens": 0},
        {"source_row_id": "s1", "jid": "jN", "junction_seq": "AAAAAU",
         "scaf": "1", "helix_seq": "h1", "y": -5.0, "cens": 0},
        {"source_row_id": "s2", "jid": "jF", "junction_seq": "UUUUUU",
         "scaf": "1", "helix_seq": "h1", "y": -4.0, "cens": 0},
    ]
    return train, test


def test_candidate_is_pointwise_identical_to_edit_knn_on_supported_rows():
    train, test = _rows()
    # Candidate C
    cand = fit_local(train)
    feats = support_features(train, test, dist=SUPPORT_DIST)
    c_mu, c_sigma, c_cp, c_support, c_abstain = predict_gated(
        cand, feats, test, d_thresh=SUPPORT_DIST, k_thresh=1)
    # P1 edit-KNN baseline
    knn = fit_edit_knn(train)
    k_mu, k_sigma, k_cp, k_support, k_abstain = predict_edit_knn(knn, test)

    assert not np.all(c_support), "fixture must contain abstained rows"
    supported = np.where(c_support)[0]
    assert len(supported) >= 1, "fixture must contain supported rows"

    # mu / sigma identical on every supported row
    max_dmu = float(np.max(np.abs(c_mu[supported] - k_mu[supported])))
    max_dsig = float(np.max(np.abs(c_sigma[supported] - k_sigma[supported])))
    assert max_dmu <= 1e-9, f"supported mu differs by {max_dmu}"
    assert max_dsig <= 1e-9, f"supported sigma differs by {max_dsig}"

    # Candidate C abstains exactly where edit-KNN emits a raw (non-exact) value,
    # i.e. the only added behaviour is a distance gate on top of the same KNN.
    for i, r in enumerate(test):
        if bool(c_abstain[i]):
            assert bool(c_support[i]) is False
            # abstained row must be far from all train junctions
            d = _min_edit_to_train(train, str(r["junction_seq"]))
            assert d > SUPPORT_DIST


def _min_edit_to_train(train_rows, tseq):
    from audit.models.support_aware_mixture import _lev
    return min(_lev(str(r["junction_seq"]), tseq) for r in train_rows)
