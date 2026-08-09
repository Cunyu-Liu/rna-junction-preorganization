"""Unit tests for the label-null structural diagnostic (P2 re-adjudication)."""
import numpy as np
from audit.p2.label_null_diagnostic import junction_seq_uniformity, label_null_preserves_mean

def _rows(n_junctions=3, rows_per=4):
    out = []
    i = 0
    for j in range(n_junctions):
        seq = "ACGU" * (j + 1)
        for _ in range(rows_per):
            out.append({
                "source_row_id": str(i),
                "jid": str(j),
                "junction_seq": seq,
                "y": -5.0 - j,
                "cens": 0,
            })
            i += 1
    return out

def test_all_junctions_single_distinct_seq():
    rows = _rows()
    stats = junction_seq_uniformity(rows)
    assert stats["n_junctions"] == 3
    assert stats["n_junctions_single_distinct_seq"] == 3
    assert stats["frac_junctions_single_distinct_seq"] == 1.0
    assert stats["max_distinct_seq_per_junction"] == 1

def test_label_null_preserves_junction_mean():
    rows = _rows()
    shift = label_null_preserves_mean(rows, seed=0)
    assert shift < 1e-9  # within-junction permutation cannot move junction-mean

def test_mixed_seq_junction_detected():
    # if a junction had distinct sequences, uniformity should drop
    rows = _rows()
    rows[0]["junction_seq"] = "GGGG"  # split first junction's seq
    stats = junction_seq_uniformity(rows)
    assert stats["max_distinct_seq_per_junction"] >= 2
