"""Unit tests for the R4 Track A evidence-closure module."""
import numpy as np
import pytest

from audit.r4_track_a import (effective_n, noise_ceiling, power_analysis,
                              model_coverage, TARGET_REL_GAIN)


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


# ---------------------------------------------------------------------------
# model coverage: mutation_graph family is run via mutation_graph_smoother
# ---------------------------------------------------------------------------
def _write_leaderboard(tmp_path, model_ids):
    """Write a minimal R1 leaderboard with the given model_id rows."""
    lb = tmp_path / "r1"
    lb.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in model_ids:
        rows.append(f"symmetry_5fold,0,{mid},1,True")
        rows.append(f"edit_5fold,0,{mid},1,True")
    (lb / "Leaderboard_v2.csv").write_text(
        "axis,fold,model_id,coverage,eligible_full_coverage\n" + "\n".join(rows)
        + "\n")


def test_coverage_maps_mutation_graph_smoother(tmp_path):
    """mutation_graph_smoother maps to the mutation_graph class."""
    _write_leaderboard(tmp_path, ["edit_knn", "mutation_graph_smoother"])
    cov = model_coverage(tmp_path)
    by_id = {c["model_id"]: c for c in cov}
    assert by_id["mutation_graph_smoother"]["class"] == "mutation_graph"
    assert by_id["mutation_graph_smoother"].get("status", "RUN") != "NOT_RUN"


def test_coverage_skips_mutation_graph_not_run_when_smoother_present(tmp_path):
    """When mutation_graph_smoother is in R1, mutation_graph_propagation is
    NOT flagged as NOT_RUN (family already covered)."""
    _write_leaderboard(tmp_path, ["edit_knn", "mutation_graph_smoother"])
    cov = model_coverage(tmp_path)
    ids = {c["model_id"] for c in cov}
    # mutation_graph family covered -> no NOT_RUN entry for it
    assert "mutation_graph_propagation" not in ids
    # physical_prior / frozen_lm still genuinely NOT_RUN
    assert "physical_ensemble_prior" in ids
    assert "frozen_rna_lm" in ids
    for c in cov:
        if c["model_id"] in ("physical_ensemble_prior", "frozen_rna_lm"):
            assert c["status"] == "NOT_RUN"


def test_coverage_flags_mutation_graph_not_run_when_absent(tmp_path):
    """If no mutation_graph member ran, mutation_graph_propagation is NOT_RUN."""
    _write_leaderboard(tmp_path, ["edit_knn"])
    cov = model_coverage(tmp_path)
    by_id = {c["model_id"]: c for c in cov}
    assert by_id["mutation_graph_propagation"]["status"] == "NOT_RUN"