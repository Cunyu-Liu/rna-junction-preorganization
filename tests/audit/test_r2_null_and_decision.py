"""Unit tests for the R2 final CoreHypothesisDecision_v3 gate logic."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit import r2_null_and_decision as r2


def _fake_run(axis_results):
    """Return a fake multiway_cluster.run() result shaped like the real one."""
    return {"axes": axis_results}


def _ax(axis, gain=0.05, boot_lo=0.0, null_p=0.5, n975=0.2):
    """An 'available' axis dict with controllable gates.  theta>n975 for the
    null_975_upper_lt_genuine check to pass."""
    theta = max(n975 + 0.01, gain)
    return {
        "axis": axis, "available": True, "n_junctions": 30, "n_contexts": 20,
        "n_rows": 300, "theta": theta, "relative_gain": gain,
        "junction_boot_ci": [boot_lo, boot_lo + 0.2],
        "junction_boot_lower_gt_0": bool(boot_lo > 0),
        "two_way_ci": None, "two_way_lower_gt_0": None,
        "null_p_value": null_p, "null_975_upper": n975,
        "null_975_upper_lt_genuine": bool(n975 < theta),
    }


def _make_r2(run_root, axis_results, fold_all_positive=True):
    (run_root / "r05_v131").mkdir(parents=True, exist_ok=True)
    (run_root / "r1").mkdir(parents=True, exist_ok=True)
    import pandas as pd
    full = pd.DataFrame([
        {"axis": "symmetry_5fold", "fold": "0", "model_id": "corrected_v1_31",
         "eligible_full_coverage": True, "pooled_junction_macro_nll": 1.0},
        {"axis": "symmetry_5fold", "fold": "1", "model_id": "corrected_v1_31",
         "eligible_full_coverage": True, "pooled_junction_macro_nll": 1.0},
    ])
    ns = pd.DataFrame([
        {"axis": "symmetry_5fold", "fold": "0", "model_id": "no_sequence_latent_operator",
         "eligible_full_coverage": True,
         "pooled_junction_macro_nll": 2.0 if fold_all_positive else 0.5},
        {"axis": "symmetry_5fold", "fold": "1", "model_id": "no_sequence_latent_operator",
         "eligible_full_coverage": True,
         "pooled_junction_macro_nll": 2.0 if fold_all_positive else 2.0},
    ])
    full.to_csv(run_root / "r05_v131" / "Leaderboard_v1_31.csv", index=False)
    ns.to_csv(run_root / "r1" / "Leaderboard_v2.csv", index=False)
    # monkeypatch the data-driven multiway_cluster pieces
    r2.mw.run = lambda rr, axes, out: _fake_run(axis_results)
    r2.mw.load_axis_rows = lambda rr, axis: []
    r2.mw.junction_pairing_null = lambda rows, n_null=1000, seed=17: []
    import numpy as np
    r2.pd.DataFrame([]).to_parquet  # ensure pd imported path is fine


def test_decision_not_supported_without_gain(tmp_path):
    # gain below 0.10, boot_lo <= 0 -> NOT_SUPPORTED even though p small
    ax = _ax("symmetry_5fold", gain=0.05, boot_lo=-0.01, null_p=0.001, n975=0.0)
    _make_r2(tmp_path, [ax])
    status = r2.main({"run_id": "x", "run_root": str(tmp_path)})
    dec = json.loads((tmp_path / "r2" / "CoreHypothesisDecision_v3.json").read_text())
    assert dec["axes"][0]["verdict"] == "NOT_SUPPORTED_OR_INCONCLUSIVE"
    assert status["n_axes_supported"] == 0


def test_decision_supported_when_all_gates_pass(tmp_path):
    ax = _ax("symmetry_5fold", gain=0.20, boot_lo=0.05, null_p=0.001, n975=0.02)
    _make_r2(tmp_path, [ax], fold_all_positive=True)
    r2.main({"run_id": "x", "run_root": str(tmp_path)})
    dec = json.loads((tmp_path / "r2" / "CoreHypothesisDecision_v3.json").read_text())
    assert dec["axes"][0]["verdict"] == "SEQUENCE_INCREMENT_SUPPORTED"


def test_decision_blocks_when_fold_not_all_positive(tmp_path):
    # all stat gates pass but one fold is not positive -> NOT_SUPPORTED
    ax = _ax("symmetry_5fold", gain=0.20, boot_lo=0.05, null_p=0.001, n975=0.02)
    _make_r2(tmp_path, [ax], fold_all_positive=False)
    r2.main({"run_id": "x", "run_root": str(tmp_path)})
    dec = json.loads((tmp_path / "r2" / "CoreHypothesisDecision_v3.json").read_text())
    assert dec["axes"][0]["verdict"] == "NOT_SUPPORTED_OR_INCONCLUSIVE"
