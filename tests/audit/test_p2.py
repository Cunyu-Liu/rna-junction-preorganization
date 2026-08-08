"""Phase 2 unit tests (contract Phase 2: all new code requires unit tests)."""
from __future__ import annotations

import json
import numpy as np
import pytest

from audit.p2.hypothesis import HYPOTHESIS_REGISTRY, REFERENCE_BASELINE, CANDIDATE
from audit.p2.bootstrap import bootstrap_ci, run_fold_bootstrap
from audit.p2.common import (
    axis_gain, permute_labels_within_junction, permute_sequence_pairing,
)
from audit.p2.effect_decomposition import decompose
from audit.p2.decision import evaluate


def test_hypothesis_registry_preregistered():
    assert HYPOTHESIS_REGISTRY["hypotheses"]["H0"]
    assert HYPOTHESIS_REGISTRY["hypotheses"]["H1"]
    contrasts = HYPOTHESIS_REGISTRY["contrasts"]
    assert "C1_sequence_vs_operator_null" in contrasts
    assert "C3_genuine_vs_permutation_nulls" in contrasts
    assert REFERENCE_BASELINE == "train_only_scaffold"
    assert CANDIDATE == "corrected_v1_31"


def test_bootstrap_ci_positive_inputs_gives_positive_lower_bound():
    res = bootstrap_ci([0.1, 0.2, 0.15, 0.11, 0.18], n_boot=2000, seed=0)
    assert res["ci_low"] > 0


def test_axis_gain_from_fold_metrics():
    fm = {("sym", "train_only_scaffold", 0): 2.0, ("sym", "corrected_v1_31", 0): 1.5,
          ("sym", "train_only_scaffold", 1): 2.2, ("sym", "corrected_v1_31", 1): 1.8}
    gain, per = axis_gain(fm, "sym", [0, 1])
    assert gain == pytest.approx(0.45, abs=1e-9)
    assert per[0] == pytest.approx(0.5)


def _make_rows():
    rows = []
    k = 0
    for jid in ["1", "2"]:
        for scaf in [1, 2]:
            rows.append({"source_row_id": f"{k:06d}", "jid": jid, "junction_seq": f"J{jid}",
                         "scaf": scaf, "y": -8.0, "cens": False, "helix_seq": f"c{k}"})
            k += 1
    return rows


def test_permute_labels_preserves_junction_label_multiset_and_sequence():
    rows = _make_rows()
    rng = np.random.default_rng(1)
    out = permute_labels_within_junction(rows, rng)
    # sequence unchanged per row
    for r, o in zip(rows, out):
        assert o["junction_seq"] == r["junction_seq"]
        assert o["jid"] == r["jid"]
    # per-junction label multiset preserved
    for jid in ["1", "2"]:
        y_in = sorted(r["y"] for r in rows if r["jid"] == jid)
        y_out = sorted(r["y"] for r in out if r["jid"] == jid)
        assert y_in == y_out


def test_permute_sequence_pairing_breaks_pairs():
    rows = _make_rows()
    rng = np.random.default_rng(2)
    out = permute_sequence_pairing(rows, rng)
    # sequence multiset preserved (all J1/J2 appear)
    seqs_in = sorted({r["junction_seq"] for r in rows})
    seqs_out = sorted({r["junction_seq"] for r in out})
    assert seqs_in == seqs_out
    # labels stay in place
    for r, o in zip(rows, out):
        assert o["y"] == r["y"] and o["scaf"] == r["scaf"]


def test_effect_decomposition_telescopes():
    fm = {}
    nf = 3
    for f in range(nf):
        fm[("ax", "global_censor_intercept", f)] = 3.0
        fm[("ax", "train_only_scaffold", f)] = 2.5
        fm[("ax", "scaffold_context_hierarchy", f)] = 2.4
        fm[("ax", "motif_topology_hierarchy", f)] = 2.3
        fm[("ax", "corrected_v1_31", f)] = 2.0
    rows = decompose(fm, [("ax", nf)])
    row = rows[0]
    msum = (row["margin_operator"] + row["margin_context"] +
            row["margin_motif_topo"] + row["margin_sequence"])
    assert msum == pytest.approx(row["total_gain"], abs=1e-9)


def test_decision_evaluate_pass_case():
    fb = {"symmetry_5fold": {"observed_mean_gain": 0.2,
                             "all_folds_positive": True, "n_folds_positive": 5,
                             "per_fold_gain": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
                             "fold_unit_bootstrap": {"ci_low": 0.05}},
          "edit_5fold": {"observed_mean_gain": 0.2, "all_folds_positive": True,
                         "n_folds_positive": 5,
                         "per_fold_gain": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
                         "fold_unit_bootstrap": {"ci_low": 0.05}},
          "context_lomo": {"observed_mean_gain": 0.2, "all_folds_positive": True,
                           "n_folds_positive": 234,
                           "per_fold_gain": {f: 0.2 for f in range(234)},
                           "fold_unit_bootstrap": {"ci_low": 0.05}},
          "scaffold_lomo": {"observed_mean_gain": 0.0, "all_folds_positive": False,
                            "n_folds_positive": 0,
                            "per_fold_gain": {f: 0.0 for f in range(9)},
                            "fold_unit_bootstrap": {"ci_low": -0.01}}}
    ns = {}
    for a in ["symmetry_5fold", "edit_5fold", "context_lomo", "scaffold_lomo"]:
        ns[a] = {"label": {"p975_gain": -0.01, "n_perms": 1000},
                 "sequence": {"p975_gain": -0.01, "n_perms": 1000}}
    axes = [("symmetry_5fold", 5), ("edit_5fold", 5), ("context_lomo", 234), ("scaffold_lomo", 9)]
    crit = evaluate(fb, ns, [], axes)
    assert crit["A1_gain_ci_lower_bound_gt_0"]["symmetry_5fold"]["pass"] is True
    assert crit["A3_genuine_gt_null_975"]["symmetry_5fold"]["label"]["pass"] is True
    assert crit["A4_blocked_context_positive"]["pass"] is True
    assert crit["A6_no_catastrophic_fold"]["symmetry_5fold"]["pass"] is True
