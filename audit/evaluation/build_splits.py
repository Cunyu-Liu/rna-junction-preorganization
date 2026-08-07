"""P0.4 split manifest builder.

Freezes four grouped outer-split axes over the admitted row universe:
  - symmetry_5fold     : group by exact-sequence reciprocal symmetry key
  - edit_5fold         : group by edit-component (source-level one-edit graph)
  - context_lomo       : helix-context LOMO (each context is one held-out fold)
  - scaffold_lomo      : scaffold/operator LOMO (each scaffold is one held-out fold)

Each admitted row is assigned to exactly one test fold per axis (or marked
excluded).  The manifest is defined by row IDs (source_row_id) + grouping keys.
Group assignment is seeded so that repeated runs are byte-identical.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260807


def freeze_group_folds(groups, n_folds, seed):
    """Deterministic grouped 5-fold: shuffle group ids, chunk into n_folds."""
    rng = np.random.default_rng(int(seed))
    keys = sorted(groups)
    perm = keys.copy()
    rng.shuffle(perm)
    chunks = [set(x.tolist()) for x in np.array_split(np.asarray(perm, dtype=object), n_folds) if len(x)]
    fold_of_group = {}
    for fi, chunk in enumerate(chunks):
        for g in chunk:
            fold_of_group[str(g)] = fi
    return fold_of_group


def build_manifests(admitted):
    """Return {axis: manifest_rows} where each manifest row is one admitted row."""
    # grouping keys per row
    sym_groups = sorted({str(r["symmetry_key"]) for r in admitted})
    edit_groups = sorted({str(r["edit_component"]) for r in admitted})
    ctx_groups = sorted({str(r["helix_seq"]) for r in admitted})
    scaf_groups = sorted({int(r["scaf"]) for r in admitted})

    sym_fold = freeze_group_folds(sym_groups, 5, SEED)
    edit_fold = freeze_group_folds(edit_groups, 5, SEED + 1)
    ctx_fold = {str(g): i for i, g in enumerate(ctx_groups)}      # LOMO
    scaf_fold = {int(g): i for i, g in enumerate(scaf_groups)}    # LOMO

    axis_rows = {k: [] for k in ("symmetry_5fold", "edit_5fold", "context_lomo", "scaffold_lomo")}
    for r in admitted:
        rid = str(r["source_row_id"])
        axis_rows["symmetry_5fold"].append({
            "source_row_id": rid, "jid": str(r["jid"]), "scaf": int(r["scaf"]),
            "symmetry_key": str(r["symmetry_key"]), "fold": sym_fold[str(r["symmetry_key"])],
            "group": str(r["symmetry_key"]), "axis": "symmetry_5fold",
        })
        axis_rows["edit_5fold"].append({
            "source_row_id": rid, "jid": str(r["jid"]), "scaf": int(r["scaf"]),
            "edit_component": str(r["edit_component"]), "fold": edit_fold[str(r["edit_component"])],
            "group": str(r["edit_component"]), "axis": "edit_5fold",
        })
        axis_rows["context_lomo"].append({
            "source_row_id": rid, "jid": str(r["jid"]), "scaf": int(r["scaf"]),
            "helix_seq": str(r["helix_seq"]), "fold": ctx_fold[str(r["helix_seq"])],
            "group": str(r["helix_seq"]), "axis": "context_lomo",
        })
        axis_rows["scaffold_lomo"].append({
            "source_row_id": rid, "jid": str(r["jid"]), "scaf": int(r["scaf"]),
            "scaf": int(r["scaf"]), "fold": scaf_fold[int(r["scaf"])],
            "group": int(r["scaf"]), "axis": "scaffold_lomo",
        })
    return axis_rows, {
        "symmetry_5fold": {"n_folds": 5, "n_groups": len(sym_groups)},
        "edit_5fold": {"n_folds": 5, "n_groups": len(edit_groups)},
        "context_lomo": {"n_folds": len(ctx_groups), "n_groups": len(ctx_groups)},
        "scaffold_lomo": {"n_folds": len(scaf_groups), "n_groups": len(scaf_groups)},
    }


def write_manifest(axis_rows, axis, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in axis_rows[axis]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.audit_dataset import audit_dataset
    records = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    _, admitted, *_ = audit_dataset(records)
    axis_rows, meta = build_manifests(admitted)
    for axis in axis_rows:
        write_manifest(axis_rows, axis, out_dir / f"SplitManifest_{axis}.jsonl")
    (out_dir / "SplitAxisMeta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"axes": {k: len(v) for k, v in axis_rows.items()}, "meta": meta}, indent=2))
