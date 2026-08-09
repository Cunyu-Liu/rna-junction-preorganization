"""Unit tests for the R4 Track A evidence-closure module."""
import numpy as np
import pytest

from audit.r4_track_a import (effective_n, noise_ceiling, power_analysis,
                              TARGET_REL_GAIN)


def _rows(n_junctions=8, n_contexts=4, cens=False):
    """Synthetic eligible rows (all measured unless cens=True)."""
    rows = []
    for j in range(n_junctions):
        for c in range(n_contexts):
            y = float(j)
            rows.append({"source_row_id": f"{j}_{c}", "jid": str(j),
                         "context": f"ctx{c}", "scaf": j % 3,
                         "y": y + (0.0 if cens else 0.1 * (c % 2)),
                         "cens": cens, "d": 0.1, "nll_ns": 2.0})
    return rows


def test_effective_n_counts_units():
    rows = _rows(n_junctions=8, n_contexts=4)
    eff = effective_n(rows)
    assert eff["n_rows"] == 32
    assert eff["n_junctions"] == 8
    assert eff["n_contexts"] == 4
    assert eff["n_scaffolds"] == 3
    assert eff["mean_rows_per_junction"] == 4.0


def test_effective_n_icc_design_effect_present():
    rows = _rows(n_junctions=20, n_contexts=5)
    eff = effective_n(rows)
    assert eff["icc_junction"] is not None
    assert eff["design_effect"] is not None and eff["design_effect"] >= 1.0
    assert eff["effective_junctions"] <= eff["n_junctions"]


def test_noise_ceiling_measured_only():
    rows = _rows(n_junctions=8, n_contexts=4, cens=False)
    nc = noise_ceiling(rows)
    assert nc["n_measured"] == 32
    assert nc["n_junctions_with_gt1_measurement"] == 8
    assert nc["operator_exposure_spread_sigma_kcal"] is not None
    assert nc["junction_mean_baseline_pooled_junction_macro_nll"] is not None


def test_power_analysis_available_and_scaled():
    rows = _rows(n_junctions=100, n_contexts=4)
    pw = power_analysis(rows)
    assert pw["available"] is True
    assert pw["n_junctions"] == 100
    assert pw["target_relative_gain"] == TARGET_REL_GAIN
    assert pw["power_at_target_gain"] is not None
    assert 0.0 <= pw["power_at_target_gain"] <= 1.0
    assert pw["negative_result_interpretation"] in (
        "POWER_BOUNDARY", "ADEQUATE_POWER_TO_EXCLUDE_TARGET")
