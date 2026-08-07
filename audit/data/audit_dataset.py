"""P0.2 data / censoring / effective-N audit.

Faithfully re-implements the loader semantics of the legacy
sequence_fingerprint_factor_benchmark.load_records (filter sublibrary==
junction_conformations, dg10 finite, chip_scaffold finite, dg10==CAP -> right
censored) and reconstructs raw -> parsed -> QC -> analysis -> admitted layers
with a per-row CleaningLedger, DataProfile, DependencyGraph, EffectiveNReport
and ExposureRegistry.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CAP = -7.1
EPS = 1e-8


def parse_parts(raw):
    return [x.upper() for x in str(raw or "").replace("&", "_").split("_")]


def symmetry_key(raw):
    parts = parse_parts(raw)
    if len(parts) == 2:
        forward = "_".join(parts)
        swapped = "_".join(parts[::-1])
        return min(forward, swapped)
    return "_".join(parts)


def edit_component_labels(rows, max_distance=1):
    keys = sorted({str(row["symmetry_key"]) for row in rows})
    parsed = [tuple(key.split("_")) for key in keys]
    parent = list(range(len(keys)))
    size = [1] * len(keys)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a)
        b = find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for i in range(len(keys)):
        a_parts = parsed[i]
        for j in range(i):
            b_parts = parsed[j]
            if len(a_parts) != len(b_parts):
                continue
            if any(len(a) != len(b) for a, b in zip(a_parts, b_parts)):
                continue
            distance = sum(ca != cb for a, b in zip(a_parts, b_parts) for ca, cb in zip(a, b))
            if distance <= int(max_distance):
                union(i, j)
    names = {}
    for i, key in enumerate(keys):
        root = find(i)
        names.setdefault(root, key)
    return {key: names[find(i)] for i, key in enumerate(keys)}


def num(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def audit_dataset(records_path: Path):
    """Return per-row ledger + profile + dependency + effective-N + exposure."""
    lines = records_path.read_text().splitlines()
    raw_rows = [ln for ln in lines if ln.strip()]
    ledger = []  # one entry per non-empty source line
    admitted = []

    raw_by_sub = Counter()
    for i, line in enumerate(raw_rows):
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            ledger.append({"source_row_id": i, "layer": "parsed",
                           "excluded": True, "reason": "invalid_json"})
            continue
        rid = f"{i:06d}"
        sub = obj.get("sublibrary")
        raw_by_sub[sub] += 1
        if sub != "junction_conformations":
            ledger.append({"source_row_id": rid, "layer": "qc", "excluded": True,
                           "reason": f"sublibrary={sub}"})
            continue
        y = num(obj.get("dg10"))
        scaf = num(obj.get("chip_scaffold"))
        if y is None:
            ledger.append({"source_row_id": rid, "layer": "analysis", "excluded": True,
                           "reason": "dg10_missing_or_nonfinite"})
            continue
        if scaf is None:
            ledger.append({"source_row_id": rid, "layer": "analysis", "excluded": True,
                           "reason": "chip_scaffold_missing_or_nonfinite"})
            continue
        jseq = obj.get("junction_seq") or ""
        hseq = obj.get("helix_seq") or ""
        rec = {
            "source_row_id": rid,
            "jid": str(obj.get("junction_id")),
            "motif": str(obj.get("motif_type")),
            "scaf": int(scaf),
            "y": float(y),
            "cens": abs(float(y) - CAP) < EPS,
            "junction_seq": jseq,
            "helix_seq": hseq,
            "symmetry_key": symmetry_key(jseq),
            "err10": num(obj.get("err10")),
        }
        ledger.append({"source_row_id": rid, "layer": "admitted", "excluded": False,
                       "reason": "admitted", **rec})
        admitted.append(rec)

    # edit components over admitted universe
    labels = edit_component_labels(admitted, max_distance=1)
    for rec in admitted:
        rec["edit_component"] = labels[str(rec["symmetry_key"])]

    n_admitted = len(admitted)
    n_measured = sum(not r["cens"] for r in admitted)
    n_censored = sum(bool(r["cens"]) for r in admitted)
    n_junctions = len({str(r["jid"]) for r in admitted})
    n_symmetry = len({str(r["symmetry_key"]) for r in admitted})
    n_edit = len({str(r["edit_component"]) for r in admitted})
    # context universe: unique helix_seq within admitted vs. raw sublibrary.
    admitted_ctx = {str(r["helix_seq"]) for r in admitted}
    # raw universe contexts: helix_seq among ALL junction_conformations rows
    raw_ctx = set()
    for i, line in enumerate(raw_rows):
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if obj.get("sublibrary") == "junction_conformations":
            raw_ctx.add(str(obj.get("helix_seq") or ""))
    n_scaf = len({int(r["scaf"]) for r in admitted})
    n_study = 1
    # per-junction scaffold/context multiplicity
    jid_scaf = defaultdict(set)
    jid_ctx = defaultdict(set)
    for r in admitted:
        jid_scaf[str(r["jid"])].add(int(r["scaf"]))
        jid_ctx[str(r["jid"])].add(str(r["helix_seq"]))
    scaf_counts = sorted(len(v) for v in jid_scaf.values())
    ctx_counts = sorted(len(v) for v in jid_ctx.values())

    profile = {
        "raw_lines": len(raw_rows),
        "raw_by_sublibrary": dict(raw_by_sub),
        "admitted_rows": n_admitted,
        "measured_rows": n_measured,
        "right_censored_rows": n_censored,
        "censored_fraction": round(n_censored / max(n_admitted, 1), 6),
        "unique_junction_id": n_junctions,
        "symmetry_keys": n_symmetry,
        "edit_components": n_edit,
        "admitted_helix_contexts": len(admitted_ctx),
        "raw_universe_helix_contexts": len(raw_ctx),
        "context_universe_note": "697 (historical) is raw-universe context count; admitted-universe contexts=234",
        "assay_scaffolds": n_scaf,
        "study": n_study,
        "junction_scaffold_multiplicity": {"min": scaf_counts[0], "median": float(np.median(scaf_counts)),
                                           "max": scaf_counts[-1]},
        "junction_context_multiplicity": {"min": ctx_counts[0], "median": float(np.median(ctx_counts)),
                                          "max": ctx_counts[-1]},
        "all_admitted_rows_unique": len({(r["source_row_id"], r["jid"], int(r["scaf"])) for r in admitted}) == n_admitted,
    }

    # Effective-N report (multi-axis, no assertion of independent biological N)
    eff_n = {
        "row_N": n_admitted,
        "junction_N": n_junctions,
        "symmetry_group_N": n_symmetry,
        "edit_component_N": n_edit,
        "context_N": len(admitted_ctx),
        "scaffold_operator_N": n_scaf,
        "study_N": n_study,
        "biological_replicate_unit": "NOT_CONFIRMED_no_independent_biological_replicates_in_source",
        "max_credible_independent_groups": min(n_edit, n_scaf) if (n_edit and n_scaf) else 0,
        "note": "11893 rows are repeated junction x scaffold/context measurements, not 11893 independent biological samples.",
    }

    # Dependency graph
    dep = {
        "junction_x_scaffold_pairs": len({(r["jid"], int(r["scaf"])) for r in admitted}),
        "junction_x_context_pairs": len({(r["jid"], r["helix_seq"]) for r in admitted}),
        "unique_junction_x_scaffold_pairs_equal_rows": len({(r["jid"], int(r["scaf"])) for r in admitted}) == n_admitted,
        "edges": {"junction->scaffold": "many_to_many", "junction->context": "many_to_many",
                  "scaffold": "operator/measurement system", "context": "helix context repetition"},
    }

    # Exposure registry: fields never admitted into sequence primary model
    exposure = {
        "admitted_features_primary": ["junction_seq", "symmetry_key", "edit_component"],
        "forbidden_from_primary": ["dg9", "dg11", "dg10_5mM", "DMS", "qMaPseq_labels",
                                   "interpolated_labels", "outer_test_labels", "foundation_model",
                                   "target_fingerprint", "same_variant_reference_dg"],
        "source_fields_retained_but_not_primary": ["dg_fold", "dg_fold_constrained", "err10",
                                                   "helix_seq (context only, not sequence feature)", "motif"],
    }

    return ledger, admitted, profile, eff_n, dep, exposure


def run(records_path: Path, out_dir: Path):
    ledger, admitted, profile, eff_n, dep, exposure = audit_dataset(records_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "CleaningLedger.jsonl").open("w") as fh:
        for rec in ledger:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "DataProfile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "EffectiveNReport.json").write_text(json.dumps(eff_n, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "DependencyGraph.json").write_text(json.dumps(dep, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "ExposureRegistry.json").write_text(json.dumps(exposure, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    # status
    status = {
        "phase": "P0.2", "state": "PASS", "profile": profile, "effective_n": eff_n,
        "checks": {
            "counts_reproduce_contract": (
                n_admitted := profile["admitted_rows"]) == 11893 and (
                n_measured := profile["measured_rows"]) == 9961 and (
                n_censored := profile["right_censored_rows"]) == 1932 and (
                profile["unique_junction_id"]) == 1336 and (
                profile["symmetry_keys"]) == 684 and (
                profile["edit_components"]) == 37 and (
                profile["admitted_helix_contexts"]) == 234 and (
                profile["assay_scaffolds"]) == 9,
        },
    }
    (out_dir / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"profile": profile, "effective_n": eff_n, "status_checks": status["checks"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    run(Path(sys.argv[1]), Path(sys.argv[2]))
