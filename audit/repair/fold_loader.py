"""Shared fold loader: builds typed FoldSpec for every axis and ensures the
exact same row set reaches fit for every model (full, no-sequence, baselines).

The strict audit's decisive P0.2 defect was that corrected v1.31 discarded the
joint ``train_ids`` it computed.  This loader is the single source of FoldSpec
objects so the full model and its matched no-sequence comparator consume the
same blocked, zero-overlap train set.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from audit.repair.foldspec import FoldSpec, fold_to_spec


def load_single_axis_folds(manifest_path: Path) -> list[FoldSpec]:
    """Build one FoldSpec per fold from a SplitManifest_*.jsonl file.

    For the single-axis grouped splits the train set is everything NOT in the
    test fold (within that axis).  No sequence/context/operator blocking beyond
    the grouped test set is applied here.
    """
    by_fold = defaultdict(set)
    axis = None
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        axis = o["axis"]
        by_fold[o["fold"]].add(str(o["source_row_id"]))
    if axis is None:
        return []
    all_ids = set().union(*by_fold.values()) if by_fold else set()
    specs = []
    for fold in sorted(by_fold):
        test = by_fold[fold]
        train = all_ids - test
        specs.append(fold_to_spec(axis, fold, test, train))
    return specs


def build_joint_edit_context_folds(admitted) -> list[FoldSpec]:
    """Typed FoldSpec for the decisive edit_x_nested_context joint split.

    Zero overlap in BOTH dimensions: test rows of the held-out edit component
    are removed from train, AND every train row sharing a nested context with
    the held-out component is removed too.  This is identical to the builder
    logic in audit.splits.joint_blocked but attached to FoldSpec so the full
    model can no longer drop the train ids.
    """
    from audit.splits.joint_blocked import build_joint_edit_context
    rep = build_joint_edit_context(admitted)
    by_edit = defaultdict(list)
    for r in admitted:
        by_edit[str(r["edit_component"])].append(r)
    specs = []
    for f in rep["folds"]:
        e = f["edit_component"]
        test_rows = by_edit.get(e, [])
        test_ctxs = {str(r["helix_seq"]) for r in test_rows}
        test_ids = {str(r["source_row_id"]) for r in test_rows}
        train_ids = {str(r["source_row_id"]) for r in admitted
                     if str(r["edit_component"]) != e
                     and str(r["helix_seq"]) not in test_ctxs}
        blocked_ctx = {str(r["helix_seq"]) for r in admitted}
        specs.append(fold_to_spec(
            "edit_x_nested_context", f"e:{e}", test_ids, train_ids,
            blocked_seq={str(r["edit_component"]) for r in test_rows},
            blocked_ctx=test_ctxs,
        ))
    return specs


def all_axis_folds(manifest_dir: Path, admitted) -> dict[str, list[FoldSpec]]:
    """Return {axis: [FoldSpec,...]} for the four single axes + the joint axis."""
    out = {}
    for axis in ("symmetry_5fold", "edit_5fold", "context_lomo", "scaffold_lomo"):
        mp = manifest_dir / f"SplitManifest_{axis}.jsonl"
        if mp.exists():
            out[axis] = load_single_axis_folds(mp)
    out["edit_x_nested_context"] = build_joint_edit_context_folds(admitted)
    return out


def write_foldspec_manifest(specs: list[FoldSpec], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for s in specs:
            fh.write(json.dumps(s.to_manifest(), sort_keys=True) + "\n")