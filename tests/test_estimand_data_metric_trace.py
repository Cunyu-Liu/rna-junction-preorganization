#!/usr/bin/env python3
"""test_estimand_data_metric_trace — data columns, metric, censoring and score direction
must be traceable from the bound EstimandSpec through the data_metric_trace.tsv."""
import json, os, sys

RUN_ROOT = os.environ.get("V14_RUN_ROOT", "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z")

def test_trace_exists():
    path = f"{RUN_ROOT}/tecto/t6/data_metric_trace.tsv"
    assert os.path.exists(path), "data_metric_trace.tsv missing"
    rows = [l.split("\t") for l in open(path).read().strip().splitlines()]
    assert len(rows) >= 3, "trace should have header + data rows"
    return rows

def test_trace_covers_required_fields():
    rows = test_trace_exists()
    fields = {r[1] for r in rows[1:]}
    for req in ["data_column", "censoring", "primary_metric", "direction", "baseline", "primary_split"]:
        assert req in fields, f"trace missing field {req}"

def test_trace_consistent_with_decision():
    binding = json.load(open(f"{RUN_ROOT}/tecto/t6/estimand_binding.json"))
    assert binding["preserve_rerun_rule"]["rule"] == "PRESERVE"
    assert binding["result_preserved"] is True

def test_censoring_direction_left():
    verification = json.load(open(f"{RUN_ROOT}/tecto/t6/t6_verification.json"))
    assert verification["censoring_direction_left"] is True
    assert verification["censoring_threshold"] == -7.1

def test_score_direction_lower_better():
    verification = json.load(open(f"{RUN_ROOT}/tecto/t6/t6_verification.json"))
    assert verification["score_direction_lower_better"] is True

if __name__ == "__main__":
    test_trace_exists()
    test_trace_covers_required_fields()
    test_trace_consistent_with_decision()
    test_censoring_direction_left()
    test_score_direction_lower_better()
    print("test_estimand_data_metric_trace: PASS")