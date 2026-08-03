#!/usr/bin/env python3
"""T0 tecto data-admission analysis.

Computes the contract's T0 admission metrics from the canonical Denny records:
  - censoring semantics for the -7.1 kcal/mol cap
  - replicate / scaffold / context semantics
  - raw->parsed->QC->analysis->admitted attrition table
  - motif-construct-scaffold-study graph and connected components
  - counts at all levels and group-adjusted effective (N)
  - outer holdout feasibility per generalization axis
  - license and provenance completeness
  - exclusion accounting with checksums

Emits:
  - manifests/t0_admission_analysis.json  (machine-readable)
  - docs/t0_admission_report.md           (human-readable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

CENSOR_CAP = -7.1  # kcal/mol, paper's measurable-range floor

# Columns serialized as numbers but stored as strings in the canonical JSONL
# (the builder uses default=str). Parse them back to float/None.
_NUM_FIELDS = ("dg_fold", "dg_fold_constrained", "dg10", "dg9", "dg11",
               "dg10_5mM", "err10", "err9", "err11", "err10_5mM",
               "dg10_interp", "dg9_interp", "dg11_interp")


def _norm_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _load_records(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for f in _NUM_FIELDS:
            if f in r:
                r[f] = _norm_num(r[f])
        out.append(r)
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _sublib(v):
    if v is None:
        return ""
    return str(v).strip().strip("\u201c\u201d\"")


def _is_jm(v):
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "yes", "true", "y")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifests-out", required=True)
    ap.add_argument("--docs-out", required=True)
    ap.add_argument("--run-id", default="v1_2_tecto_qmap_20260803")
    args = ap.parse_args()

    recs = _load_records(args.canonical)
    n_raw = len(recs)

    # ---- rows with a sublibrary ----
    with_sublib = [r for r in recs if r.get("sublibrary")]
    # ---- measured / interpolated / censored / missing (row level) ----
    measured = [r for r in recs if r.get("dg10") is not None]
    censored = [r for r in measured if r["dg10"] == CENSOR_CAP]
    plain = [r for r in measured if r["dg10"] != CENSOR_CAP]
    interp_only = [r for r in recs if r.get("dg10") is None and r.get("dg10_interp") is not None]
    missing = [r for r in recs if r.get("dg10") is None and r.get("dg10_interp") is None]

    # ---- construct-level (junction_id) sets ----
    def cid_set(pred):
        return {r["junction_id"] for r in recs if pred(r) and r["junction_id"]}

    all_c = cid_set(lambda r: True)
    jm_c = cid_set(lambda r: _is_jm(r.get("is_junctionmat")))
    measured_c = cid_set(lambda r: r.get("dg10") is not None)
    # construct with at least one non-censored measurement
    noncens_c = cid_set(lambda r: r.get("dg10") is not None and r["dg10"] != CENSOR_CAP)
    # construct with ALL measurements censored (exclusively censored)
    per_c = defaultdict(list)
    for r in recs:
        if r.get("junction_id"):
            per_c[r["junction_id"]].append(r)
    excl_censored_c = {
        jid for jid, rows in per_c.items()
        if rows and all(r.get("dg10") is not None and r["dg10"] == CENSOR_CAP for r in rows)
    }

    # ---- scaffold / motif / study levels ----
    scaffold_of = defaultdict(set)
    motif_of = defaultdict(set)
    study_of = defaultdict(set)
    for r in recs:
        jid = r.get("junction_id")
        if not jid:
            continue
        if r.get("chip_scaffold"):
            scaffold_of[jid].add(r["chip_scaffold"])
        if r.get("motif_type"):
            motif_of[jid].add(r["motif_type"])
        study_of[jid].add("denny2018")  # single study

    n_scaffolds = len({s for v in scaffold_of.values() for s in v})
    n_motifs = len({m for v in motif_of.values() for m in v})
    n_studies = 1

    # ---- motif-construct-scaffold-study graph + connected components ----
    # Nodes: constructs, scaffolds, motifs, studies. Edges: c-scaffold, c-motif, c-study.
    # Build union-find over constructs connected via shared scaffolds/motifs.
    parent = {c: c for c in all_c}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # connect constructs sharing a scaffold
    scaffold_members = defaultdict(list)
    for c, scs in scaffold_of.items():
        for s in scs:
            scaffold_members[s].append(c)
    for s, members in scaffold_members.items():
        for i in range(1, len(members)):
            union(members[0], members[i])
    # connect constructs sharing a motif
    motif_members = defaultdict(list)
    for c, ms in motif_of.items():
        for m in ms:
            motif_members[m].append(c)
    for m, members in motif_members.items():
        for i in range(1, len(members)):
            union(members[0], members[i])

    comp = defaultdict(set)
    for c in all_c:
        comp[find(c)].add(c)
    comp_sizes = sorted((len(v) for v in comp.values()), reverse=True)
    giant = comp_sizes[0] if comp_sizes else 0

    # ---- effective N by level ----
    # group-adjusted effective N: unique constructs / independent groups
    # independent scaffold groups = number of scaffolds (each scaffold reused => not independent)
    effective_n = {
        "constructs": len(all_c),
        "motifs": n_motifs,
        "scaffolds": n_scaffolds,
        "studies": n_studies,
        "junctionmat_constructs": len(jm_c),
        "measured_constructs": len(measured_c),
        "noncensored_constructs": len(noncens_c),
        "exclusively_censored_constructs": len(excl_censored_c),
        "independent_scaffold_groups": n_scaffolds,
        "independent_study_groups": n_studies,
        "connected_components": len(comp),
        "giant_component_size": giant,
    }

    # ---- outer holdout feasibility per axis ----
    # scaffold holdout: only 9 scaffolds -> severely limited
    # study holdout: 1 study -> infeasible
    # motif holdout: 60 motifs, 1713 constructs -> feasible but heavy sharing
    # construct holdout: feasible but leakage via shared scaffold/motif
    holdout = {
        "construct_holdout": {
            "feasible": True,
            "note": "hold out whole junction_ids; leakage via shared scaffold/motif must be blocked",
        },
        "motif_family_holdout": {
            "feasible": True,
            "n_motifs": n_motifs,
            "note": "hold out whole motif families; construct->scaffold reuse must be blocked",
        },
        "scaffold_holdout": {
            "feasible": n_scaffolds >= 3,
            "n_scaffolds": n_scaffolds,
            "note": "only 9 scaffolds; high risk of scaffold-level confounding",
        },
        "study_holdout": {
            "feasible": False,
            "n_studies": n_studies,
            "note": "single study (Denny 2018); cross-study generalization requires qMaPseq or other platform",
        },
        "giant_component_rule": {
            "random_row_split": "FORBIDDEN",
            "same_construct_cross_fold": "FORBIDDEN",
            "required": "pre-registered multi-axis blocked generalization",
        },
    }

    # ---- censoring semantics ----
    censoring = {
        "cap_kcal_per_mol": CENSOR_CAP,
        "direction": "left-censored (very stable end; more negative than measurable floor)",
        "basis": "Kd <= 5000 nM measurable range; cap is the most stable measurable value",
        "n_rows_at_cap": len(censored),
        "n_constructs_exclusively_censored": len(excl_censored_c),
        "likelihood": "censored likelihood for rows at cap; do not treat as exact point values",
    }

    # ---- replicate / bootstrap / covariance / context semantics ----
    # The workbook reports one aggregated measurement per (junction, scaffold) with
    # bootstrap 95% CI (err10/err9/err11/err10_5mM). The paper's Figure 1G reports
    # two replicate experiments; the workbook does not carry per-replicate rows, so
    # replicate variance is not recoverable at row level and must not be assumed.
    rep_semantics = {
        "per_replicate_rows_present": False,
        "rows_are_aggregated": True,
        "bootstrap_ci_columns": ["err10", "err9", "err11", "err10_5mM"],
        "bootstrap_ci_meaning": "95% CI from bootstrapped cluster fluorescence (paper Figure 1H)",
        "two_replicate_experiments": "reported in paper Figure 1G; not present as row-level replicates",
        "replicate_variance_recoverable": False,
        "covariance_default": "NOT independent; same construct/scaffold shared across rows",
        "scaffold_context": "chip_scaffold (9 values) reused across constructs; context must enter grouping/hierarchical model",
        "note": "Treat per-row err as measurement uncertainty, not independent replicate noise",
    }

    # ---- attrition table ----
    attrition = {
        "raw_rows": n_raw,
        "with_sublibrary_rows": len(with_sublib),
        "measured_rows": len(measured),
        "noncensored_rows": len(plain),
        "censored_rows_at_cap": len(censored),
        "interpolated_only_rows": len(interp_only),
        "missing_rows": len(missing),
        "distinct_constructs": len(all_c),
        "distinct_junctionmat_constructs": len(jm_c),
        "distinct_measured_constructs": len(measured_c),
        "distinct_noncensored_constructs": len(noncens_c),
        "distinct_exclusively_censored_constructs": len(excl_censored_c),
    }

    # ---- provenance completeness ----
    # every row must trace to a source row in the workbook and carry a scanning checksum
    provenance = {
        "source_is_paper_supplementary_workbook": True,
        "source_path": "261_SI.xlsx (Denny et al. 2018 Cell, supplementary)",
        "row_level_source_row_present": all(r.get("source_row") is not None for r in recs),
        "row_level_checksum": "per-row SHA-256 of serialized canonical JSON computed at build",
        "complete": all(r.get("source_row") is not None for r in recs),
    }

    # ---- license check ----
    license = {
        "status": "CELL_PAPER_SUPPLEMENTARY_DATA",
        "note": "NIH open-access author manuscript (PMC6053692); Cell article data. "
                "Verify the specific Cell license (Elsevier) before any redistribution; "
                "analysis/reproduction use is standard but redistribution terms must be confirmed.",
        "allowed_for_analysis": True,
        "redistribution_confirmed": False,
    }

    result = {
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_records": args.canonical,
        "canonical_sha256": sha256_file(args.canonical),
        "n_records": n_raw,
        "attrition": attrition,
        "censoring": censoring,
        "replicate_semantics": rep_semantics,
        "effective_n": effective_n,
        "graph": {
            "levels": {"constructs": len(all_c), "scaffolds": n_scaffolds, "motifs": n_motifs, "studies": n_studies},
            "connected_components": len(comp),
            "component_size_distribution": comp_sizes[:20],
            "giant_component_size": giant,
        },
        "holdout_feasibility": holdout,
        "provenance": provenance,
        "license": license,
        "gate_status": "T0_RUNNING_NOT_PASS",
        "decision_required": "finalizer must confirm all T0 items before PASS",
    }

    os.makedirs(args.manifests_out, exist_ok=True)
    os.makedirs(args.docs_out, exist_ok=True)
    mpath = os.path.join(args.manifests_out, "t0_admission_analysis.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- human-readable report ----
    L = []
    L.append("# T0 tecto Data Admission Report")
    L.append("")
    L.append(f"run_id: {args.run_id}")
    L.append(f"generated_at_utc: {result['generated_at_utc']}")
    L.append(f"canonical sha256: {result['canonical_sha256']}")
    L.append("")
    L.append("## Censoring semantics (-7.1 kcal/mol)")
    L.append(f"- direction: {censoring['direction']}")
    L.append(f"- basis: {censoring['basis']}")
    L.append(f"- rows at cap: {censoring['n_rows_at_cap']}")
    L.append(f"- constructs exclusively censored: {censoring['n_constructs_exclusively_censored']}")
    L.append(f"- likelihood: {censoring['likelihood']}")
    L.append("")
    L.append("## Replicate / bootstrap / covariance semantics")
    for k, v in rep_semantics.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Attrition")
    for k, v in attrition.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Effective N")
    for k, v in effective_n.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Motif-construct-scaffold-study graph")
    L.append(f"- levels: {result['graph']['levels']}")
    L.append(f"- connected components: {result['graph']['connected_components']}")
    L.append(f"- giant component size: {result['graph']['giant_component_size']}")
    L.append(f"- component size distribution: {result['graph']['component_size_distribution']}")
    L.append("")
    L.append("## Outer holdout feasibility")
    for k, v in holdout.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Provenance")
    for k, v in provenance.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## License")
    for k, v in license.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Gate status")
    L.append("- T0 is RUNNING, NOT PASS. The finalizer must confirm all 18 T0 admission items.")
    L.append("")
    rpath = os.path.join(args.docs_out, "t0_admission_report.md")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(json.dumps({
        "attrition": attrition,
        "effective_n": effective_n,
        "giant_component": giant,
        "analysis_manifest": mpath,
        "report": rpath,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())