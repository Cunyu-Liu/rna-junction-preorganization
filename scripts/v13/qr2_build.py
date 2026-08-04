"""QR2 build: split feasibility before outcome (v1.3).

Per v1.3 9.1: the existing mutation graph forms 4 connected components
(83/11/2/2), which must NOT be treated as four equal-weight independent
replicates. QR2 compares feasible split schemes WITHOUT viewing any new
transfer outcome, and states clearly which generalization question each scheme
answers and its limitations.
"""
from __future__ import annotations
import json
import os
import sys
import datetime
import hashlib

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
PARENT_ROOT = os.environ.get("RNA_V12_RUN_ROOT", "/mnt/cunyuliu/v1_2_tecto_qmap_codex_20260804T074900Z")

Q4G = os.path.join(PARENT_ROOT, "qmap", "q4", "q4_mutation_graph.json")
Q4F = os.path.join(PARENT_ROOT, "qmap", "q4", "q4_fold_assignment.json")
Q4S = os.path.join(PARENT_ROOT, "qmap", "q4", "q4_freeze_summary.json")
Q2 = os.path.join(PARENT_ROOT, "qmap", "q2", "q2_attrition.jsonl")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    graph = json.load(open(Q4G))
    fold = json.load(open(Q4F))
    freeze = json.load(open(Q4S))
    q2 = load_jsonl(Q2)
    q2_ids = {r["name"] for r in q2}

    components = graph.get("components", [])
    comp_sizes = [len(c) for c in components]
    fold_sizes = fold.get("fold_sizes", [])
    fold_of_variant = fold.get("fold_of_variant", {})

    # Map each q2 variant to its component and fold
    comp_index = {}
    for ci, comp in enumerate(components):
        for v in comp:
            comp_index[v] = ci

    # Cross-tab: component membership vs fold assignment
    cross = {}
    for v in q2_ids:
        ci = comp_index.get(v)
        fi = fold_of_variant.get(v)
        key = (ci, fi)
        cross[key] = cross.get(key, 0) + 1

    # Feasible schemes (outcome-blind comparison)
    schemes = [
        {
            "id": "S1_connected_component_holdout",
            "description": "Hold out one whole connected component (or a union of components) for test; train on the rest.",
            "generalization_question": "Does the model generalize to a NEW connected component of the mutation graph (i.e., to variants not mutation-reachable from training variants)? This is the strongest and most honest structural generalization.",
            "unit": "connected component (vertex set of mutation graph)",
            "n_units": len(components),
            "component_sizes": comp_sizes,
            "limitation": "Strongly imbalanced: the 83-vertex component dominates; small components are not equal-weight independent replicates and cannot be pooled as if they were 4 i.i.d. folds.",
            "verdict": "RECOMMENDED as the primary honest generalization; must NOT be reported as 4-fold i.i.d. CV.",
        },
        {
            "id": "S2_leave_one_component_out",
            "description": "Leave out each component in turn; report per-component held-out score and group-weighted score.",
            "generalization_question": "Per-component held-out assessment; explicitly reports where generalization fails (e.g., the small components).",
            "unit": "connected component",
            "n_units": len(components),
            "component_sizes": comp_sizes,
            "limitation": "The 83-component train/test is near-degenerate (train set barely changes). Small components give very high-variance estimates.",
            "verdict": "Valid as a SENSITIVITY analysis, not as the primary adjudication.",
        },
        {
            "id": "S3_original_fold_assignment_83_11_2_2",
            "description": "The parent run's 4-fold split whose fold sizes are the 4 connected components (83/11/2/2).",
            "generalization_question": "Parent run treats the 4 components as 4 equal-weight independent repeats, using unweighted macro and n=4 paired-t CI.",
            "unit": "component (treated as fold by parent)",
            "n_units": 4,
            "fold_sizes": fold_sizes,
            "limitation": "REJECTED as primary: the 4 components are NOT equal-weight independent replicates; 83/11/2/2 is badly imbalanced and the n=4 paired-t CI is invalid. This is the parent run's inference defect (A0 finding D).",
            "verdict": "REJECTED for primary adjudication; may be retained only as a historical comparison object.",
        },
        {
            "id": "S4_random_or_row_split",
            "description": "Random split of rows/reads into folds.",
            "generalization_question": "None valid for cross-variant generalization; answers only within-distribution memorization.",
            "unit": "row",
            "limitation": "PROHIBITED by v1.3 (no random row/replicate/condition/titration/nucleotide split).",
            "verdict": "REJECTED.",
        },
    ]

    summary = {
        "schema_version": "1.0",
        "gate": "QR2",
        "run_id": RUN_ID,
        "built_at_utc": ts,
        "frozen_before_viewing_transfer_outcome": True,
        "mutation_graph": {
            "n_vertices": graph.get("n_vertices"),
            "n_edges": graph.get("n_edges"),
            "n_connected_components": graph.get("n_connected_components"),
            "component_sizes": comp_sizes,
        },
        "parent_fold": {
            "k_folds": fold.get("k_folds"),
            "fold_sizes": fold_sizes,
            "split_method": fold.get("split_method"),
            "leakage_violations": fold.get("leakage_violations"),
        },
        "component_vs_fold_crosstab": {f"{k}": v for k, v in cross.items()},
        "feasible_schemes": schemes,
        "recommended_primary": "S1_connected_component_holdout",
        "note": (
            "QR2 freezes the split feasibility BEFORE viewing any new transfer outcome. "
            "The 4 components are not equal-weight independent replicates; the primary "
            "adjudication must use component-level holdout (S1) with the small "
            "components reported as sensitivity strata, not as i.i.d. folds."
        ),
        "source_files": {
            "mutation_graph": {"path": Q4G, "sha256": sha256_file(Q4G)},
            "fold_assignment": {"path": Q4F, "sha256": sha256_file(Q4F)},
            "freeze_summary": {"path": Q4S, "sha256": sha256_file(Q4S)},
            "q2_attrition": {"path": Q2, "sha256": sha256_file(Q2), "rows": len(q2)},
        },
    }

    outdir = os.path.join(RUN_ROOT, "qmap", "qr2")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr2_split_feasibility.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[QR2] components=%d sizes=%s" % (graph.get("n_connected_components"), comp_sizes))
    print("[QR2] parent folds=%d sizes=%s" % (fold.get("k_folds"), fold_sizes))
    print("[QR2] cross component-vs-fold=%s" % {str(k): v for k, v in cross.items()})
    print("[QR2] recommended_primary=S1_connected_component_holdout")
    return 0


if __name__ == "__main__":
    sys.exit(main())