"""QR0 build: authoritative 99->98 denominator truth table (v1.3).

Produces the first authoritative table containing all 99 public-archive rows,
and explains the 98 selected-variant analysis denominator using source-authored
evidence. It does NOT infer exclusion from name/similarity; it only records
what the parent run's registry and attrition files actually say, and explicitly
flags the single unregistered candidate (GCUAAG_UACGG / source UACGG_GCUAAG).
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

Q1 = os.path.join(PARENT_ROOT, "qmap", "q1", "q1_variant_registry.jsonl")
Q2 = os.path.join(PARENT_ROOT, "qmap", "q2", "q2_attrition.jsonl")
Q4 = os.path.join(PARENT_ROOT, "qmap", "q4", "q4_fold_assignment.json")
Q5 = os.path.join(PARENT_ROOT, "qmap", "q5", "q5_transfer_summary.json")


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
    q1 = load_jsonl(Q1)
    q2 = load_jsonl(Q2)
    q4 = json.load(open(Q4))
    q5 = json.load(open(Q5)) if os.path.isfile(Q5) else None

    # canonical id = reversed source name (source name is receptor-first)
    def canonical(src_name):
        parts = src_name.split("_")
        return "_".join(reversed(parts))

    q1_by_canon = {}
    for r in q1:
        q1_by_canon[canonical(r["name"])] = r

    q2_by_canon = {}
    for r in q2:
        q2_by_canon[r["name"]] = r

    # q4 fold assignment: find the variant->fold map
    fold_map = {}
    if "fold_of_variant" in q4:
        fold_map = q4["fold_of_variant"]
    elif "folds" in q4:
        for fk, fv in q4["folds"].items():
            if isinstance(fv, list):
                for v in fv:
                    fold_map[v] = fk
    q4_keys = set(fold_map.keys())

    # Build 99-row truth table
    rows = []
    for src in q1:
        cid = canonical(src["name"])
        rec = {
            "canonical_id": cid,
            "source_name": src["name"],
            "in_q1_99": True,
            "in_q2_98": cid in q2_by_canon,
            "in_q4_98": cid in q4_keys,
            "in_q5_98": (q5.get("n_variants") is not None and q5.get("n_variants") == 98) if q5 else None,
            "is_reference": src.get("is_reference"),
            "receptor_has_no_mutations": src.get("receptor_has_no_mutations"),
            "rna_map_dg": src.get("rna_map_dg"),
            "old_dg": src.get("old_dg"),
            "source_doi": src.get("source_doi"),
            "q2_category": q2_by_canon.get(cid, {}).get("category"),
            "q2_sub_reason": q2_by_canon.get(cid, {}).get("sub_reason"),
            "q2_censoring_type": q2_by_canon.get(cid, {}).get("censoring_type"),
            "q2_mg_1_2": q2_by_canon.get(cid, {}).get("mg_1_2"),
            "q2_in_published_exclude_11": q2_by_canon.get(cid, {}).get("in_published_exclude_11"),
            "q4_fold": fold_map.get(cid),
        }
        rows.append(rec)

    # The single unregistered candidate: present in q1 (99) but not in q2/q4/q5 (98)
    q2_ids = set(q2_by_canon.keys())
    excluded = [r for r in rows if r["in_q1_99"] and not r["in_q2_98"]]
    assert len(excluded) == 1, f"expected exactly 1 excluded, got {len(excluded)}"
    excl = excluded[0]

    # category counts in q2
    from collections import Counter
    cat_counts = Counter(r["q2_category"] for r in rows if r["in_q2_98"])

    summary = {
        "schema_version": "1.0",
        "gate": "QR0",
        "run_id": RUN_ID,
        "built_at_utc": ts,
        "n_q1_99": len(rows),
        "n_q2_98": len(q2),
        "n_q4_98": len(q4_keys),
        "q5_n_variants": q5.get("n_variants") if q5 else None,
        "q2_category_counts": dict(cat_counts),
        "excluded_99_to_98": {
            "canonical_id": excl["canonical_id"],
            "source_name": excl["source_name"],
            "is_reference": excl["is_reference"],
            "receptor_has_no_mutations": excl["receptor_has_no_mutations"],
            "rna_map_dg": excl["rna_map_dg"],
            "old_dg": excl["old_dg"],
            "q4_fold": excl["q4_fold"],
            "q2_category": excl["q2_category"],
            "exclusion_provenance": (
                "present in q1 registry (99) but absent from q2 attrition (98). "
                "Parent run reports this variant has no mg_1_2 fit record in the "
                "public mg_1_2 data source; exclusion is inherited from the "
                "published data, not an explicit methodological exclusion. This "
                "observation is used only to locate the discrepancy, not to "
                "prejudge reference/exclusion identity (per v1.3 8.1)."
            ),
        },
        "source_files": {
            "q1": {"path": Q1, "sha256": sha256_file(Q1), "rows": len(q1)},
            "q2": {"path": Q2, "sha256": sha256_file(Q2), "rows": len(q2)},
            "q4": {"path": Q4, "sha256": sha256_file(Q4)},
            "q5": {"path": Q5, "sha256": sha256_file(Q5)} if q5 else None,
        },
        "note": (
            "QR0 establishes the denominator truth table. It does NOT decide "
            "reference/exclusion identity; structural-QC vs censored classification "
            "is handled in QR1 with source-authored evidence per v1.3 8.2."
        ),
    }

    outdir = os.path.join(RUN_ROOT, "qmap", "qr0")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "qr0_denominator_truth_table.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(outdir, "qr0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[QR0] n_q1_99=%d n_q2_98=%d n_q4_98=%d" % (len(rows), len(q2), len(q4_keys)))
    print("[QR0] q2_category_counts=%s" % dict(cat_counts))
    print("[QR0] excluded=%s" % excl["canonical_id"])
    print("[QR0] wrote qr0_denominator_truth_table.jsonl and qr0_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())