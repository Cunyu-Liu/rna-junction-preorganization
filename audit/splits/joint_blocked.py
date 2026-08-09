"""R0.3 joint-blocked split feasibility and construction (contract §7.2, R0.3).

Two decisive joint splits, each requiring ZERO overlap in BOTH blocked
dimensions (not just one):

  edit_x_nested_context : unseen sequence family (edit component) AND unseen
                          nested context, while scaffolds remain SEEN (other
                          contexts of the same scaffold stay in train).
  seq_x_scaffold_bundle : unseen sequence family (edit component) AND unseen
                          scaffold+context bundle.

Because contexts are strictly nested within scaffolds, these joint cells are
the honest test of transferable sequence increment.  If a joint fold has
insufficient support or empty train/test, we report INFEASIBLE for that axis
rather than constructing a leaky split (contract R0.3 credible criterion).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_TEST_ROWS = 1
MIN_TRAIN_ROWS = 200


def _tkey(*parts):
    return "::".join(str(p) for p in parts)


def edit_context_contingency(admitted):
    """Rows per (edit_component, helix_seq) cell (string keys for JSON)."""
    tab = defaultdict(int)
    for r in admitted:
        tab[_tkey(r["edit_component"], r["helix_seq"])] += 1
    return dict(tab)


def edit_scaffold_contingency(admitted):
    """Rows per (edit_component, scaffold) cell (string keys for JSON)."""
    tab = defaultdict(int)
    for r in admitted:
        tab[_tkey(r["edit_component"], r["scaf"])] += 1
    return dict(tab)


def _rows_by(admitted):
    by_edit = defaultdict(list)
    by_ctx = defaultdict(list)
    by_scaf = defaultdict(list)
    for r in admitted:
        by_edit[str(r["edit_component"])].append(r)
        by_ctx[str(r["helix_seq"])].append(r)
        by_scaf[int(r["scaf"])].append(r)
    return by_edit, by_ctx, by_scaf


def build_joint_edit_context(admitted):
    """Leave-one-edit-out, and remove from train every row sharing a context
    with the held-out edit component.  Returns folds or INFEASIBLE."""
    by_edit, by_ctx, _ = _rows_by(admitted)
    edits = sorted(by_edit)
    folds = []
    for e in edits:
        test_rows = by_edit[e]
        test_ctxs = {str(r["helix_seq"]) for r in test_rows}
        train_rows = [r for r in admitted
                      if str(r["edit_component"]) != e
                      and str(r["helix_seq"]) not in test_ctxs]
        if len(test_rows) < MIN_TEST_ROWS or len(train_rows) < MIN_TRAIN_ROWS:
            folds.append({"edit_component": e, "n_test": len(test_rows),
                          "n_train": len(train_rows), "feasible": False})
            continue
        # zero-overlap checks
        train_edits = {str(r["edit_component"]) for r in train_rows}
        train_ctxs = {str(r["helix_seq"]) for r in train_rows}
        zero_seq = not ({str(r["edit_component"]) for r in test_rows} & train_edits)
        zero_ctx = not ({str(r["helix_seq"]) for r in test_rows} & train_ctxs)
        folds.append({"edit_component": e, "n_test": len(test_rows),
                      "n_train": len(train_rows), "feasible": bool(zero_seq and zero_ctx),
                      "zero_overlap_sequence": zero_seq,
                      "zero_overlap_context": zero_ctx})
    n_feasible = sum(1 for f in folds if f["feasible"])
    return {"axis": "edit_x_nested_context", "n_folds": len(folds),
            "n_feasible": n_feasible, "feasible": n_feasible == len(folds),
            "folds": folds}


def build_joint_seq_scaffold(admitted):
    """Leave-one-scaffold-out combined with leave-one-edit-out:
    test = rows in scaffold s with edit component e; train = rows neither in
    scaffold s nor in edit component e.  Zero overlap in both dimensions."""
    by_scaf = defaultdict(list)
    by_edit = defaultdict(list)
    for r in admitted:
        by_scaf[int(r["scaf"])].append(r)
        by_edit[str(r["edit_component"])].append(r)
    scafs = sorted(by_scaf)
    edits = sorted(by_edit)
    folds = []
    for s in scafs:
        for e in edits:
            test_rows = [r for r in admitted
                         if int(r["scaf"]) == s and str(r["edit_component"]) == e]
            train_rows = [r for r in admitted
                          if int(r["scaf"]) != s and str(r["edit_component"]) != e]
            feasible = (len(test_rows) >= MIN_TEST_ROWS and len(train_rows) >= MIN_TRAIN_ROWS)
            folds.append({"scaf": s, "edit_component": e, "n_test": len(test_rows),
                          "n_train": len(train_rows), "feasible": feasible})
    n_feasible = sum(1 for f in folds if f["feasible"])
    return {"axis": "seq_x_scaffold_bundle", "n_candidate_folds": len(folds),
            "n_feasible": n_feasible, "feasible": n_feasible == len(folds),
            "folds": folds}


def write_feasibility(admitted, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    edit_ctx = build_joint_edit_context(admitted)
    seq_scaf = build_joint_seq_scaffold(admitted)
    report = {
        "edit_x_context_contingency": edit_context_contingency(admitted),
        "edit_x_scaffold_contingency": edit_scaffold_contingency(admitted),
        "joint_edit_x_nested_context": edit_ctx,
        "joint_seq_x_scaffold_bundle": seq_scaf,
        "interpretation": ("joint split feasibility depends on non-empty, "
                           "zero-overlap test cells with adequate train; "
                           "infeasible axes are reported, not silently leaked"),
    }
    (out_dir / "SplitFeasibility.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report
