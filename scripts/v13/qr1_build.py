"""QR1 build: source-authored 84/11/2/1 reconstruction + endpoint namespace audit (v1.3).

Per v1.3 8.2/8.3: categories are decided by the paper/supplementary/archive
explicit identity, NOT by fit outliers, file order, or "making the count work".
Paper-named structural-QC variants enter the registry first by their original
identity; any data-driven extra anomalies are only secondary sensitivity flags.

This build also records the endpoint namespace audit (QR1): qMaP midpoint,
RNA-MaP delta-G, DMS reactivity, sequence embedding, and junction geometry are
distinct endpoint namespaces that do NOT by default share a latent truth.
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

Q2 = os.path.join(PARENT_ROOT, "qmap", "q2", "q2_attrition.jsonl")
QR0 = os.path.join(RUN_ROOT, "qmap", "qr0", "qr0_denominator_truth_table.jsonl")


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
    q2 = load_jsonl(Q2)
    qr0 = load_jsonl(QR0)

    # Paper-named structural-QC candidates (audit candidates, not yet final identity)
    paper_named_structural_qc = [
        "UCUAAA_CAUGA", "CCUACA_UACGG", "CUUAAC_UAUGG",
    ]
    # Parent-run alternative outliers used as substitutes
    parent_outlier_alternatives = [
        "GCUAAA_UACGC", "GCUUAA_CAUGC", "CCUAAG_CACGG",
    ]

    # Reconstruct categories from q2's own source-authored fields
    from collections import Counter
    cat_counts = Counter()
    censored_rows = []
    structural_sensitivity = []
    fitted_gt40 = []
    for r in q2:
        cat = r["category"]
        cat_counts[cat] += 1
        if cat == "right_censored":
            censored_rows.append({
                "name": r["name"], "sub_reason": r["sub_reason"],
                "censoring_type": r["censoring_type"],
                "mg_1_2": r["mg_1_2"], "mg_1_2_gt_40": r["mg_1_2_gt_40"],
                "in_published_exclude_11": r["in_published_exclude_11"],
            })
        if cat in ("closing_pair_abnormal", "alternate_structure"):
            structural_sensitivity.append({
                "name": r["name"], "category": cat, "sub_reason": r["sub_reason"],
                "mg_1_2": r["mg_1_2"], "n": r.get("n"), "a_0": r.get("a_0"),
            })
        if cat == "fitted" and r.get("mg_1_2_gt_40"):
            fitted_gt40.append({"name": r["name"], "mg_1_2": r["mg_1_2"]})

    # Which paper-named candidates are present, and in which category?
    paper_found = {}
    for cand in paper_named_structural_qc:
        row = next((r for r in q2 if r["name"] == cand), None)
        paper_found[cand] = {
            "present_in_q2_98": row is not None,
            "category": row["category"] if row else None,
            "sub_reason": row["sub_reason"] if row else None,
            "mg_1_2": row["mg_1_2"] if row else None,
            "mg_1_2_gt_40": row["mg_1_2_gt_40"] if row else None,
            "in_published_exclude_11": row["in_published_exclude_11"] if row else None,
        }

    # Parent alternative outliers present?
    parent_found = {}
    for cand in parent_outlier_alternatives:
        row = next((r for r in q2 if r["name"] == cand), None)
        parent_found[cand] = {
            "present_in_q2_98": row is not None,
            "category": row["category"] if row else None,
            "sub_reason": row["sub_reason"] if row else None,
        }

    # Endpoint namespace audit (QR1 8.3)
    endpoint_namespace = {
        "qmap_midpoint_Mg2plus": "log10([Mg2+]1/2) from Hill fit; qMaPseq measured endpoint",
        "rna_map_deltaG": "RNA-MaP reference delta-G (kcal/mol); distinct endpoint",
        "dms_reactivity": "DMS chemical-mapping reactivity; distinct endpoint",
        "sequence_embedding": "sequence-derived embedding; NOT a thermodynamic truth",
        "junction_geometry": "geometric feature; distinct endpoint",
        "rule": "endpoints may correlate/calibrate/transport but do NOT by default share the same latent truth (v1.3 3.1)",
    }

    summary = {
        "schema_version": "1.0",
        "gate": "QR1",
        "run_id": RUN_ID,
        "built_at_utc": ts,
        "q2_category_counts": dict(cat_counts),
        "paper_named_structural_qc_candidates": paper_found,
        "parent_outlier_alternatives": parent_found,
        "right_censored_rows": censored_rows,
        "structural_qc_sensitivity_rows": structural_sensitivity,
        "fitted_gt40_rows": fitted_gt40,
        "endpoint_namespace": endpoint_namespace,
        "source_files": {
            "q2": {"path": Q2, "sha256": sha256_file(Q2), "rows": len(q2)},
            "qr0_truth_table": {"path": QR0, "sha256": sha256_file(QR0), "rows": len(qr0)},
        },
        "note": (
            "QR1 reconstructs categories from q2's own source-authored fields and "
            "records the endpoint namespace audit. It does NOT finalize category "
            "identity; the paper-named candidates are still audit candidates and "
            "require source evidence (paper/supplementary/archive) before QR3. "
            "Data-driven extra anomalies are only secondary sensitivity flags."
        ),
    }

    outdir = os.path.join(RUN_ROOT, "qmap", "qr1")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr1_category_reconstruction.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[QR1] category_counts=%s" % dict(cat_counts))
    print("[QR1] paper_named candidates:")
    for k, v in paper_found.items():
        print("   %s -> present=%s category=%s" % (k, v["present_in_q2_98"], v["category"]))
    print("[QR1] endpoint namespace audit recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())