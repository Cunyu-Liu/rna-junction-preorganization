"""P0.4 overlap audit: train/test leakage across dependency axes.

For each split axis, for each fold, computes the overlap between train and
test on multiple dependency axes: row ID, junction, exact sequence
(symmetry key), edit component, helix context, scaffold/operator.  Also
computes a neighbor-edge overlap using the one-edit graph.
All dependency keys are taken from the admitted row universe (which carries
jid/symmetry/edit/helix/scaf) joined to fold assignment from the manifest.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evaluation.build_splits import build_manifests


def neighbor_overlap(admitted):
    """Build one-edit neighbor adjacency over symmetry keys -> set of neighbor keys."""
    sym_keys = sorted({str(r["symmetry_key"]) for r in admitted})
    parsed = [tuple(k.split("_")) for k in sym_keys]
    adj = defaultdict(set)
    for i in range(len(sym_keys)):
        for j in range(i):
            ap, bp = parsed[i], parsed[j]
            if len(ap) != len(bp):
                continue
            if any(len(a) != len(b) for a, b in zip(ap, bp)):
                continue
            dist = sum(ca != cb for a, b in zip(ap, bp) for ca, cb in zip(a, b))
            if dist <= 1:
                adj[sym_keys[i]].add(sym_keys[j])
                adj[sym_keys[j]].add(sym_keys[i])
    return adj


def _row_key_map(admitted):
    """source_row_id -> dict of dependency keys from the admitted universe."""
    return {str(r["source_row_id"]): r for r in admitted}


def _key_sets(admitted, fold_of_rid, fold):
    """Return sets of each dependency key among test rows (fold) and train rows."""
    test = {"rids": set(), "jids": set(), "sym": set(), "edit": set(),
            "ctx": set(), "scaf": set()}
    train = {"rids": set(), "jids": set(), "sym": set(), "edit": set(),
             "ctx": set(), "scaf": set()}
    for r in admitted:
        rid = str(r["source_row_id"])
        f = fold_of_rid.get(rid)
        target = test if f == fold else train
        target["rids"].add(rid)
        target["jids"].add(str(r["jid"]))
        target["sym"].add(str(r["symmetry_key"]))
        target["edit"].add(str(r["edit_component"]))
        target["ctx"].add(str(r["helix_seq"]))
        target["scaf"].add(int(r["scaf"]))
    return test, train


def audit_overlap(admitted, axis_rows):
    adj = neighbor_overlap(admitted)
    n_folds = {axis: max(r["fold"] for r in rows) + 1 for axis, rows in axis_rows.items()}
    rows = []
    for axis in axis_rows:
        fold_of_rid = {str(r["source_row_id"]): r["fold"] for r in axis_rows[axis]}
        for fold in range(n_folds[axis]):
            test, train = _key_sets(admitted, fold_of_rid, fold)
            neighbor_touches = set()
            for tkey in test["sym"]:
                for nb in adj.get(tkey, ()):
                    if nb in train["sym"]:
                        neighbor_touches.add(tkey)
                        break
            rows.append({
                "axis": axis, "fold": fold,
                "n_test_rows": len(test["rids"]), "n_train_rows": len(train["rids"]),
                "row_id_overlap": len(test["rids"] & train["rids"]),
                "junction_overlap": len(test["jids"] & train["jids"]),
                "symmetry_overlap": len(test["sym"] & train["sym"]),
                "edit_overlap": len(test["edit"] & train["edit"]),
                "context_overlap": len(test["ctx"] & train["ctx"]),
                "scaffold_overlap": len(test["scaf"] & train["scaf"]),
                "neighbor_edge_touching_test_sym_keys": len(neighbor_touches),
                "n_test_sym_keys": len(test["sym"]),
            })
    return rows


def run(records: Path, out_dir: Path):
    from audit.data.audit_dataset import audit_dataset
    _, admitted, *_ = audit_dataset(records)
    axis_rows, meta = build_manifests(admitted)
    overlap = audit_overlap(admitted, axis_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "SplitOverlapMatrix.csv").open("w") as fh:
        cols = ["axis", "fold", "n_test_rows", "n_train_rows", "row_id_overlap",
                "junction_overlap", "symmetry_overlap", "edit_overlap",
                "context_overlap", "scaffold_overlap", "neighbor_edge_touching_test_sym_keys"]
        fh.write(",".join(cols) + "\n")
        for r in overlap:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    (out_dir / "SplitAxisMeta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    forbidden_ok = all(
        (r["row_id_overlap"] == 0 and r["junction_overlap"] == 0 and
         r["symmetry_overlap"] == 0 and r["edit_overlap"] == 0)
        for r in overlap
    )
    ctx_ok = all(r["context_overlap"] == 0 for r in overlap if r["axis"] == "context_lomo")
    scaf_ok = all(r["scaffold_overlap"] == 0 for r in overlap if r["axis"] == "scaffold_lomo")
    status = {
        "phase": "P0.4", "sub": "overlap", "state": "PASS" if (forbidden_ok and ctx_ok and scaf_ok) else "FAIL",
        "checks": {
            "no_row_junction_symmetry_edit_overlap": forbidden_ok,
            "context_lomo_context_overlap_zero": ctx_ok,
            "scaffold_lomo_scaffold_overlap_zero": scaf_ok,
        },
        "n_overlap_rows": len(overlap),
    }
    (out_dir / "STATUS_overlap.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    run(Path(sys.argv[1]), Path(sys.argv[2]))
