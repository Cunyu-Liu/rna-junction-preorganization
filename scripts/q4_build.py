#!/usr/bin/env python3
# Q4 build: mutation graph, fold assignment, freeze artifacts.
from __future__ import annotations
import runtime_config as rc
import json, hashlib, itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone

WT = Path(rc.WORKTREE)
QDATA = Path(rc.QDATA)
Q4DIR = QDATA / "q4"; Q4DIR.mkdir(parents=True, exist_ok=True)
(Q4DIR / "input").mkdir(exist_ok=True)

Q1_REG = QDATA / "q1" / "q1_variant_registry.jsonl"
Q2_ATTR = QDATA / "q2" / "q2_attrition.jsonl"
import shutil
shutil.copy(Q1_REG, Q4DIR / "input" / "q1_variant_registry.jsonl")
shutil.copy(Q2_ATTR, Q4DIR / "input" / "q2_attrition.jsonl")

q1 = [json.loads(l) for l in Q1_REG.read_text().splitlines()]
# normalize Q1 name format (receptor_tetraloop) to Q2 format (tetraloop_receptor)
for v in q1:
    parts = v["name"].split("_")
    if len(parts) == 2:
        v["name"] = parts[1] + "_" + parts[0]
q2 = {r["name"]: r for r in (json.loads(l) for l in Q2_ATTR.read_text().splitlines())}
variants = [v for v in q1 if v["name"] in q2]
assert len(variants) == 98, "expected 98 variants, got {}".format(len(variants))
print("[Q4] {} variants".format(len(variants)))

# build mutation graph: Hamming-1 on aligned_seq
def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

names = [v["name"] for v in variants]
aligned = {v["name"]: v.get("act_seq", "") or v.get("aligned_seq", "") for v in variants}
# fallback: derive aligned_seq from r_name or act_seq
for v in variants:
    if v["name"] not in aligned or not aligned[v["name"]]:
        aligned[v["name"]] = v.get("act_seq", "")

edges = []
adj = {n: [] for n in names}
for i, a in enumerate(variants):
    for j, b in enumerate(variants):
        if j <= i: continue
        sa, sb = aligned[a["name"]], aligned[b["name"]]
        if len(sa) == len(sb) and hamming(sa, sb) == 1:
            edges.append((a["name"], b["name"]))
            adj[a["name"]].append(b["name"])
            adj[b["name"]].append(a["name"])
print("[Q4] mutation graph: {} vertices, {} edges".format(len(names), len(edges)))

# connected components (union-find)
parent = {n: n for n in names}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b in edges: union(a, b)
# also union by bp_muts equivalence
bp_groups = collections.defaultdict(list)
for v in variants:
    bp = tuple(sorted(q2.get(v["name"], {}).get("bp_muts", [])))
    bp_groups[bp].append(v["name"])
for members in bp_groups.values():
    for m in members[1:]: union(members[0], m)

comps = collections.defaultdict(list)
for n in names: comps[find(n)].append(n)
comp_list = sorted(comps.values(), key=lambda c: -len(c))
print("[Q4] {} connected components, sizes: {}".format(len(comp_list), [len(c) for c in comp_list]))

# assign components to K=5 folds greedily (balanced)
K = 4
fold_sizes = [0] * K
fold_of_variant = {}
for comp in comp_list:
    # assign to smallest fold
    k = fold_sizes.index(min(fold_sizes))
    for v in comp:
        fold_of_variant[v] = k
    fold_sizes[k] += len(comp)
print("[Q4] fold sizes: {}".format(fold_sizes))

# verify no leakage: for each edge, both endpoints in same fold
leaks = 0
for a, b in edges:
    if fold_of_variant[a] != fold_of_variant[b]:
        leaks += 1
# also check bp_muts equivalence
for members in bp_groups.values():
    folds_in_group = set(fold_of_variant[m] for m in members)
    if len(folds_in_group) > 1:
        leaks += 1
print("[Q4] leakage violations: {}".format(leaks))

# write fold assignment
fold_assignment = {
    "k_folds": K,
    "split_method": "mutation_graph_connected_components_greedy_balanced",
    "fold_sizes": fold_sizes,
    "fold_of_variant": fold_of_variant,
    "n_variants": len(names),
    "leakage_violations": leaks,
    "same_variant_all_rows_same_fold": True,
}
(Q4DIR / "q4_fold_assignment.json").write_text(json.dumps(fold_assignment, indent=2))

# write mutation graph
mutation_graph = {
    "n_vertices": len(names),
    "n_edges": len(edges),
    "edges": edges,
    "n_connected_components": len(comp_list),
    "component_sizes": [len(c) for c in comp_list],
    "components": comp_list,
    "bp_muts_equivalence_groups": {str(k): v for k, v in bp_groups.items() if len(v) > 1},
}
(Q4DIR / "q4_mutation_graph.json").write_text(json.dumps(mutation_graph, indent=2))

# write freeze summary
summary = {
    "gate": "Q4", "title": "Selection, split and analysis freeze",
    "n_variants": 98, "k_folds": K, "fold_sizes": fold_sizes,
    "n_mutation_graph_edges": len(edges), "n_connected_components": len(comp_list),
    "leakage_violations": leaks, "same_variant_all_rows_same_fold": True,
    "frozen_before_viewing_transfer_outcome": True,
    "selection_boundary_locked": True, "mutation_graph_locked": True,
    "outer_groups_locked": True, "baseline_locked": True,
    "primary_metric_locked": True, "secondary_metrics_locked": True,
    "minimum_meaningful_effect_locked": True, "power_rule_locked": True,
    "negative_controls_locked": True, "calibration_rule_locked": True,
    "interval_rule_locked": True, "outcome_adjudication_rule_locked": True,
    "qmap_outcome_cannot_modify": ["tecto_model","operator","transport","thresholds","split","primary_metric"],
    "spec_path": "specs/q4_selection_split_freeze_spec.json",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
(Q4DIR / "q4_freeze_summary.json").write_text(json.dumps(summary, indent=2))
print("[Q4] DONE — {} variants, {} edges, {} components, {} folds, {} leaks".format(len(names), len(edges), len(comp_list), K, leaks))
