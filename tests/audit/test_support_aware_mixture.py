"""Unit tests for Phase 3 Candidate C (support-aware gated mixture + abstention)."""
import pytest

from audit.models import support_aware_mixture as sam
from audit.models.support_aware_mixture import (
    support_features, fit_local, predict_gated, supported_metrics,
    _lev, SUPPORT_DIST, GATE_GRID,
)


@pytest.fixture(autouse=True)
def _reset_global_distance_cache():
    """Reset the module-level Levenshtein distance cache between tests.

    `support_aware_mixture` keeps a process-global distance matrix/seq index
    (`_DIST_MAT`/`_SEQ_LIST`) that is built lazily on first use.  Because pytest
    runs every test file in one process, a cache built by another test file for
    a different junction universe would otherwise corrupt these fixtures.
    """
    sam._DIST_MAT = None
    sam._SEQ_LIST = []
    sam._DIST_CACHE = {}
    yield
    sam._DIST_MAT = None
    sam._SEQ_LIST = []
    sam._DIST_CACHE = {}


def _rows():
    # 2 train junctions far apart, 1 test junction near junction0, 1 far test
    train = []
    for i, seq in enumerate(["AAAA", "CCCC"]):
        train.append({"source_row_id": f"t{i}", "jid": f"j{i}",
                      "junction_seq": seq, "scaf": "1", "helix_seq": "h1",
                      "y": -5.0 - i, "cens": 0})
    test = [
        {"source_row_id": "s0", "jid": "j0", "junction_seq": "AAAA",
         "scaf": "1", "helix_seq": "h1", "y": -5.2, "cens": 0},
        {"source_row_id": "s1", "jid": "jN", "junction_seq": "GGGG",
         "scaf": "2", "helix_seq": "h9", "y": -4.0, "cens": 0},
    ]
    return train, test


def test_lev_basic():
    assert _lev("AAAA", "AAAA") == 0
    assert _lev("AAAA", "AAAC") == 1
    assert _lev("AAAA", "CCCC") == 4


def test_support_features_near_and_far():
    train, test = _rows()
    feats = support_features(train, test, dist=SUPPORT_DIST)
    assert feats["j0"]["min_edit_dist"] == 0.0      # in train
    assert feats["j0"]["n_neighbors"] >= 1
    assert feats["jN"]["min_edit_dist"] == 4.0      # far from both
    assert feats["jN"]["n_neighbors"] == 0
    assert feats["jN"]["scaffold_seen"] is False    # scaffold 2 unseen


def test_predict_gated_abstains_unsupported():
    train, test = _rows()
    feats = support_features(train, test, dist=SUPPORT_DIST)
    local = fit_local(train)
    mu, sigma, cp, support, abstain = predict_gated(local, feats, test, d_thresh=3)
    assert bool(support[0]) is True    # j0 supported (min dist 0)
    assert bool(abstain[1]) is True    # jN unsupported (min dist 4 > 3)
    assert bool(support[1]) is False
    assert mu[0] == local["node_val"][local["seq_idx"]["AAAA"]]


def test_supported_metrics_coverage():
    train, test = _rows()
    feats = support_features(train, test, dist=SUPPORT_DIST)
    local = fit_local(train)
    mu, sigma, cp, support, abstain = predict_gated(local, feats, test, d_thresh=3)
    sm = supported_metrics(test, mu, sigma, support)
    assert sm["coverage"] == 0.5   # 1 of 2 supported
    assert sm["n_supported"] == 1
    assert sm["supported_nll"] is not None


def test_gate_grid_includes_no_abstention():
    assert GATE_GRID[-1] == 1000  # ~inf => score everything (no abstention ablation)
