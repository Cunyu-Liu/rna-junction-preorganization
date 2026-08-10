"""Unit tests for the R1 supplemental mutation-graph runner (audit/r1_graph_run.py).

The runner is a thin orchestrator over the already-tested
fit_mutation_graph / predict_mutation_graph.  Its one non-trivial correctness
property is idempotency: it must refuse to re-merge mutation_graph_smoother into
the R1 leaderboard once already present (prevents duplicated-primary-key
corruption).  These tests cover that guard.
"""
from __future__ import annotations

from pathlib import Path

from audit.r1_graph_run import MODEL_ID, mutation_graph_already_in_leaderboard


def test_guard_false_when_absent(tmp_path):
    lib = tmp_path / "Leaderboard_v2.csv"
    lib.write_text("axis,fold,model_id,coverage\nsymmetry_5fold,0,edit_knn,0.5\n")
    assert mutation_graph_already_in_leaderboard(lib) is False


def test_guard_true_when_present(tmp_path):
    lib = tmp_path / "Leaderboard_v2.csv"
    lib.write_text(f"axis,fold,model_id,coverage\nsymmetry_5fold,0,{MODEL_ID},0.5\n")
    assert mutation_graph_already_in_leaderboard(lib) is True


def test_guard_false_when_missing_file():
    assert mutation_graph_already_in_leaderboard(Path("/nonexistent/leaderboard.csv")) is False